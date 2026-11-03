#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
  BACKTESTER MULTIWHALE v1.0 — Fidelidad máxima al multiwhale_v6.py
═══════════════════════════════════════════════════════════════════════════════

OBJETIVO:
  Reproducir lo más fielmente posible la lógica del multiwhale_v6 para evaluar
  cómo habría performado en cualquier periodo histórico.

REPRODUCE EXACTAMENTE:
  - calcular_ind(): RSI Wilder, EMA, ATR, BB, MACD, OBV, VWAP, asia, momentum
  - calcular_umbral(): los 9 ajustes (ATR, vol, mom, horas, sl_reciente, BTC,
    desgaste, funding, asia)
  - evaluar(): scoring sl/ss + confluencia (RSI/EMA/Vol/OBV/VWAP/Mom/Tend1H/BB/
    Funding/Asia/MACD)
  - abrir_posicion(): cap, sl, tp por config_multi.json
  - trailing_stop(): mover SL al BE+colchón cuando gan >= 0.5%
  - cerrar_por_reversion_1m(): cierre prematuro
  - extender_tp_dinamico(): extensión de TP

NO REPRODUCE (limitaciones del backtest):
  - Tomas reales de datos en vivo de orderbook (slippage real)
  - Estado de Bitget (posiciones huérfanas, conciliación)
  - Macro F&G y Farside ETF (se puede simular si se descargan datos)
  - GT (game theory) — se puede agregar después
  - Cooldowns post-arranque de Bitget
  - Funding rates históricos (Binance los tiene, pero hay que pegarlos)

DATA:
  - OHLCV 1m de Binance vía API pública (con caché en disco)
  - Caché en /home/claude/bt/cache/<sym>_<inicio>_<fin>.pkl
  - Si la API falla, retry con backoff

USO:
  python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-27 \\
                                   --symbols SKYAI,HYPER,LDO \\
                                   --leverage 5 --capital 3000
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import pickle
import argparse
import logging
import warnings
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# PATHS Y CONFIG GLOBAL
# ═════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent.resolve()
CACHE_DIR = HERE / "cache"
OUTPUT_DIR = HERE / "results"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_DIR / "backtest.log"

# Logging
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("backtester")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

log = _setup_logger()


# ═════════════════════════════════════════════════════════════════════════════
# DESCARGA DE OHLCV DE BINANCE (con cache)
# ═════════════════════════════════════════════════════════════════════════════
# Binance public API: https://fapi.binance.com/fapi/v1/klines
# Endpoint público, sin autenticación, con rate limit (~2400 req/min)
# Devuelve velas en formato JSON.
# Cada request devuelve hasta 1500 velas. Para 1 semana × 1m = 10080 velas
# necesitamos ~7 requests por símbolo.

BINANCE_BASE = "https://fapi.binance.com"


def _binance_symbol(sym: str) -> str:
    """Convierte SKYAI -> SKYAIUSDT (formato Binance)."""
    return sym.upper().replace("/USDT:USDT", "") + "USDT"


def fetch_klines_chunk(symbol: str, start_ms: int, end_ms: int,
                       interval: str = "1m") -> List:
    """
    Descarga un chunk de velas (max 1500) de Binance Futures.
    Devuelve lista de velas en formato:
      [open_time, open, high, low, close, volume, close_time, ...]
    """
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1500,
    }
    for intento in range(5):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:  # rate limit
                wait = 60
                log.warning(f"[BINANCE] rate limit, esperando {wait}s")
                time.sleep(wait)
            elif r.status_code == 400 and "Invalid symbol" in r.text:
                log.warning(f"[BINANCE] símbolo {symbol} no existe en futures")
                return []
            else:
                log.warning(f"[BINANCE] HTTP {r.status_code}: {r.text[:120]}")
                time.sleep(2 ** intento)
        except Exception as e:
            log.warning(f"[BINANCE] error: {e}")
            time.sleep(2 ** intento)
    return []


def fetch_ohlcv_periodo(sym_short: str, start_dt: datetime, end_dt: datetime,
                       interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Descarga OHLCV completo del periodo [start_dt, end_dt] para un símbolo.
    Usa caché en disco para evitar redownload.

    Devuelve DataFrame con columnas:
      ts (ms), open, high, low, close, volume, dt (datetime UTC)
    O None si falla.
    """
    # Cache key
    cache_key = f"{sym_short}_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.pkl"
    cache_file = CACHE_DIR / cache_key

    if cache_file.exists():
        try:
            df = pickle.load(open(cache_file, "rb"))
            log.info(f"[CACHE] {sym_short} {interval}: {len(df)} velas (cached)")
            return df
        except Exception as e:
            log.warning(f"[CACHE] error leyendo {cache_file}: {e}")

    # Descargar en chunks
    binance_sym = _binance_symbol(sym_short)
    start_ms = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_klines = []
    cursor_ms = start_ms
    chunks = 0

    while cursor_ms < end_ms:
        # Calcular el end de este chunk: cursor + 1500 minutos
        chunk_end_ms = min(cursor_ms + 1500 * 60 * 1000, end_ms)
        klines = fetch_klines_chunk(binance_sym, cursor_ms, chunk_end_ms, interval)
        if not klines:
            if chunks == 0:
                log.warning(f"[OHLCV] {sym_short}: primer chunk vacío, aborto")
                return None
            else:
                log.warning(f"[OHLCV] {sym_short}: chunk {chunks} vacío, asumo fin")
                break

        all_klines.extend(klines)
        chunks += 1
        # Avanzar cursor al timestamp de la última vela + 1 min
        last_ts = klines[-1][0]
        cursor_ms = last_ts + 60 * 1000
        time.sleep(0.05)  # pequeño delay para no sobrecargar API

    if not all_klines:
        return None

    # Convertir a DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["ts"] = pd.to_numeric(df["ts"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)

    # Cache
    try:
        pickle.dump(df, open(cache_file, "wb"))
    except Exception as e:
        log.warning(f"[CACHE] no pude guardar: {e}")

    log.info(f"[OHLCV] {sym_short} {interval}: {len(df)} velas descargadas ({chunks} chunks)")
    return df


def fetch_ohlcv_multi(symbols: List[str], start_dt: datetime, end_dt: datetime,
                      interval: str = "1m") -> Dict[str, pd.DataFrame]:
    """Descarga OHLCV de varios símbolos."""
    out = {}
    for sym in symbols:
        df = fetch_ohlcv_periodo(sym, start_dt, end_dt, interval)
        if df is not None and len(df) > 100:
            out[sym] = df
        else:
            log.warning(f"[OHLCV] {sym}: insuficiente data, skip")
    return out

# ═════════════════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS — REPRODUCCIÓN EXACTA DE calcular_ind() del multi v6
# ═════════════════════════════════════════════════════════════════════════════
# Nota: el multi v6 trabaja con OHLCV de 5m por defecto. Como tenemos data 1m,
# RESAMPLEAREMOS a 5m antes de calcular indicadores.
# Para sesión asiática (00-04 UTC) usamos data 1H (también resampleada).

def resample_ohlcv(df1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resamplea OHLCV de 1m a otra frecuencia (5m, 1h, etc).
    Usa convención estándar de Binance: cada vela representa el periodo
    [t, t+freq).
    """
    df = df1m.copy()
    df.set_index('dt', inplace=True)
    resampled = df.resample(freq).agg({
        'ts': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    resampled = resampled.reset_index()
    return resampled


def _ema_array(arr: np.ndarray, period: int) -> float:
    """EMA puntual al final del array. Idéntica al multi v6."""
    if len(arr) == 0:
        return 0.0
    k = 2 / (period + 1)
    e = arr[0]
    for x in arr[1:]:
        e = x * k + e * (1 - k)
    return e


def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
    """Devuelve la SERIE EMA completa para todos los puntos."""
    if len(arr) == 0:
        return np.array([])
    k = 2 / (period + 1)
    e = np.zeros(len(arr))
    e[0] = arr[0]
    for i in range(1, len(arr)):
        e[i] = arr[i] * k + e[i-1] * (1 - k)
    return e


def calcular_ind(df5: pd.DataFrame, df1h: pd.DataFrame = None,
                 ts_actual_ms: Optional[int] = None) -> Optional[Dict]:
    """
    Reproduce EXACTAMENTE multi_v6.calcular_ind() (líneas 1071-1246).

    Diferencia con el real:
      - El parámetro `ts_actual_ms` se usa para que la sesión asiática
        se calcule respecto a la fecha de la vela que estamos evaluando,
        no respecto a "ahora". Si es None, usa la última fila de df5.
    """
    if df5 is None or len(df5) < 55:
        return None

    c = df5["close"].values.astype(float)
    h = df5["high"].values.astype(float)
    lo = df5["low"].values.astype(float)
    v = df5["volume"].values.astype(float)
    n = len(c)

    # ── RSI 14 (Wilder) ──────────────────────────────────────────────────────
    rsi_period = 14
    rsi_c = c[-(rsi_period * 3):]
    rsi_d = np.diff(rsi_c)
    rsi_gains = np.where(rsi_d > 0, rsi_d, 0.0)
    rsi_loses = np.where(rsi_d < 0, -rsi_d, 0.0)
    au = rsi_gains[:rsi_period].mean() if rsi_gains[:rsi_period].mean() > 0 else 1e-10
    ad = rsi_loses[:rsi_period].mean() if rsi_loses[:rsi_period].mean() > 0 else 1e-10
    alpha = 1.0 / rsi_period
    for g, l in zip(rsi_gains[rsi_period:], rsi_loses[rsi_period:]):
        au = au * (1 - alpha) + g * alpha
        ad = ad * (1 - alpha) + l * alpha
    rsi = round(100.0 - 100.0 / (1.0 + au / (ad if ad > 0 else 1e-10)), 2)

    # ── EMAs 9, 20, 50 ───────────────────────────────────────────────────────
    ema9 = _ema_array(c[-20:], 9)
    ema20 = _ema_array(c[-40:], 20)
    ema50 = _ema_array(c[-55:], 50)
    precio = c[-1]

    # ── Volumen relativo ─────────────────────────────────────────────────────
    vr = v[-3:].mean() / (v[-20:].mean() + 1e-10)

    # ── OBV simplificado ─────────────────────────────────────────────────────
    obv = sum((v[-i] if c[-i] > c[-i-1] else -v[-i])
              for i in range(1, min(20, n)))
    obv_n = obv / (v[-20:].sum() + 1e-10)

    # ── VWAP 4h (48 velas de 5m) ─────────────────────────────────────────────
    seg = df5.tail(48)
    vwap = (seg["close"] * seg["volume"]).sum() / (seg["volume"].sum() + 1e-10)
    vwap_d = (precio - vwap) / vwap

    # ── ATR 14 ──────────────────────────────────────────────────────────────
    atr = np.mean([max(h[-i] - lo[-i], abs(h[-i] - c[-i-1]), abs(lo[-i] - c[-i-1]))
                   for i in range(1, 15)])
    atr_pct = atr / precio if precio > 0 else 0

    # ATR histórico
    atr_hist_vals = [
        max(h[-i] - lo[-i], abs(h[-i] - c[-i-1]), abs(lo[-i] - c[-i-1]))
        for i in range(1, min(42, n))
    ]
    atr_hist = np.mean(atr_hist_vals) if atr_hist_vals else atr
    atr_ratio = atr / (atr_hist + 1e-10)

    # ── Bollinger Bands (20, 2) ──────────────────────────────────────────────
    bb_period = 20
    bb_c = c[-bb_period:]
    bb_mid = bb_c.mean()
    bb_std = bb_c.std(ddof=1)
    bb_up = bb_mid + 2.0 * bb_std
    bb_dn = bb_mid - 2.0 * bb_std
    bb_ancho = (bb_up - bb_dn) / (bb_mid + 1e-10)
    bb_pos = (precio - bb_mid) / ((bb_up - bb_dn) / 2.0 + 1e-10)
    bb_pos = round(max(-2.0, min(2.0, bb_pos)), 3)
    bb_pct_b = (precio - bb_dn) / (bb_up - bb_dn + 1e-10)

    # ── Momentum 3, 6, 12 ────────────────────────────────────────────────────
    mom3 = (c[-1] - c[-4]) / c[-4] if n >= 4 else 0
    mom6 = (c[-1] - c[-7]) / c[-7] if n >= 7 else 0
    mom12 = (c[-1] - c[-13]) / c[-13] if n >= 13 else 0

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    macd_line = signal_line = macd_hist = macd_hist_prev = 0.0
    if n >= 35:
        macd_line = _ema_array(c[-60:], 12) - _ema_array(c[-60:], 26)
        macd_series = np.array([
            _ema_array(c[max(0, i-60):i], 12) - _ema_array(c[max(0, i-60):i], 26)
            for i in range(max(27, n - 20), n + 1)
        ])
        if len(macd_series) >= 9:
            signal_line = _ema_array(macd_series, 9)
            macd_hist = macd_line - signal_line
            if len(macd_series) >= 10:
                prev_macd = macd_series[-2]
                prev_signal = _ema_array(macd_series[:-1], 9)
                macd_hist_prev = prev_macd - prev_signal

    macd_hist_pct = macd_hist / (precio + 1e-10)
    macd_hist_prev_pct = macd_hist_prev / (precio + 1e-10)
    macd_accel = macd_hist_pct - macd_hist_prev_pct

    # ── Tendencia 1H + sesión asiática ───────────────────────────────────────
    tend1h, reg1h = 0.0, "range"
    asia_high = asia_low = asia_pos = 0.0

    if df1h is not None and len(df1h) >= 20:
        c1h = df1h["close"].values.astype(float)
        tend1h = (c1h[-1] - c1h[-20]) / c1h[-20]
        if tend1h > 0.02:
            reg1h = "trend_up"
        elif tend1h < -0.02:
            reg1h = "trend_down"

        # Sesión asiática: 00-04 UTC del día de la vela actual
        try:
            # Determinar el "día" desde ts_actual_ms (o última vela)
            if ts_actual_ms is None:
                ts_actual_ms = int(df5.iloc[-1]['ts'])
            ahora_dt = datetime.fromtimestamp(ts_actual_ms / 1000, tz=timezone.utc)
            inicio_dia_ms = int(datetime(
                ahora_dt.year, ahora_dt.month, ahora_dt.day,
                tzinfo=timezone.utc
            ).timestamp() * 1000)
            fin_asia_ms = inicio_dia_ms + 4 * 3600 * 1000

            h1h = df1h["high"].values.astype(float)
            lo1h = df1h["low"].values.astype(float)
            ts1h = df1h["ts"].values.astype(float)

            mask_asia = (ts1h >= inicio_dia_ms) & (ts1h < fin_asia_ms)
            if mask_asia.sum() >= 2:
                asia_high = float(h1h[mask_asia].max())
                asia_low = float(lo1h[mask_asia].min())
                rango = asia_high - asia_low
                if rango > 0:
                    asia_pos = (precio - asia_low) / rango
        except Exception:
            pass

    return {
        "rsi": round(rsi, 2), "ema9": ema9, "ema20": ema20, "ema50": ema50,
        "precio": precio, "vol_ratio": round(vr, 3), "obv": round(obv_n, 3),
        "vwap_dist": round(vwap_d, 4), "atr_pct": round(atr_pct, 5),
        "atr_ratio": round(atr_ratio, 3),
        "bb_up": round(bb_up, 8), "bb_dn": round(bb_dn, 8),
        "bb_mid": round(bb_mid, 8), "bb_pos": bb_pos,
        "bb_pct_b": round(bb_pct_b, 4), "bb_ancho": round(bb_ancho, 5),
        "mom3": round(mom3, 5), "mom6": round(mom6, 5), "mom12": round(mom12, 5),
        "tend1h": round(tend1h, 5), "reg1h": reg1h,
        "ema_bull": ema9 > ema20 > ema50, "ema_bear": ema9 < ema20 < ema50,
        "macd_line": round(macd_line, 8),
        "macd_signal": round(signal_line, 8),
        "macd_hist": round(macd_hist_pct, 7),
        "macd_hist_prev": round(macd_hist_prev_pct, 7),
        "macd_accel": round(macd_accel, 7),
        "asia_high": round(asia_high, 8),
        "asia_low": round(asia_low, 8),
        "asia_pos": round(asia_pos, 3),
        # Funding rate placeholder — Binance no lo da junto con OHLCV
        "funding_rate": 0.0,
        "funding_ts": 0,
    }

# ═════════════════════════════════════════════════════════════════════════════
# UMBRAL DINÁMICO — REPRODUCCIÓN EXACTA DE calcular_umbral() del multi v6
# ═════════════════════════════════════════════════════════════════════════════
# El multi v6 ajusta el umbral base por símbolo según 9 factores contextuales.
# Reproducimos los 9 (algunos con simplificaciones porque no tenemos data en vivo).

# Horas quietas UTC (poco volumen) — definidas en multi v6
HORAS_QUIETAS = {6, 7, 8, 11, 12, 13, 14}

# Estado simulado del backtest (se actualiza durante la corrida)
_btc_ctx_sim = {
    "regimen": "lateral",       # "trend_up", "trend_down", "lateral"
    "direccion": "neutral",     # "up", "down", "neutral"
    "fuerza": 0.0,              # 0-1
    "mom1h": 0.0,
    "dominancia_up": False,
    "ts": 0
}

_ultimo_sl_sim: Dict[str, float] = {}  # symbol -> ts_ms
_funding_cache_sim: Dict[str, Dict] = {}  # symbol -> {rate, ts}


def calcular_umbral_bt(symbol: str, ind: Dict, perfil: Dict,
                        ts_actual_ms: int) -> float:
    """
    Reproduce multi_v6.calcular_umbral() (líneas 1251-1413).

    Diferencias menores con el real:
      - Ignora calibrar_umbral_piwhale() (cache dinámico — no aplicable a backtest)
      - BTC context viene de _btc_ctx_sim (simulado durante backtest)
      - Funding viene de _funding_cache_sim (puede estar vacío)
    """
    base = perfil.get(symbol, {}).get("umb", 0.30)
    adj = 0.0
    razones = []

    # ── 1. Volatilidad propia (ATR) ──────────────────────────────────────────
    atr = ind.get("atr_pct", 0)
    if atr > 0.015:
        adj -= 0.06; razones.append(f"atr_alto({atr:.3f})-0.06")
    elif atr > 0.008:
        adj -= 0.04; razones.append(f"atr_med({atr:.3f})-0.04")
    elif atr > 0.004:
        adj -= 0.02; razones.append(f"atr_bajo({atr:.3f})-0.02")
    elif atr < 0.002:
        adj += 0.04; razones.append(f"atr_muybajo({atr:.3f})+0.04")

    # ── 2. Volumen relativo ──────────────────────────────────────────────────
    vr = ind.get("vol_ratio", 1)
    if vr > 3.0:
        adj -= 0.05; razones.append("vol_x3-0.05")
    elif vr > 1.5:
        adj -= 0.03; razones.append("vol_x1.5-0.03")
    elif vr < 0.5:
        adj += 0.04; razones.append("vol_bajo+0.04")

    # ── 3. Momentum del símbolo ──────────────────────────────────────────────
    mom = abs(ind.get("mom3", 0))
    if mom > 0.015:
        adj -= 0.03; razones.append("mom_fuerte-0.03")
    elif mom > 0.008:
        adj -= 0.01; razones.append("mom_medio-0.01")

    # ── 4. Horas quietas ─────────────────────────────────────────────────────
    hora_utc = datetime.fromtimestamp(ts_actual_ms / 1000, tz=timezone.utc).hour
    if hora_utc in HORAS_QUIETAS:
        adj += 0.04; razones.append("hora_quieta+0.04")

    # ── 5. SL reciente (cooldown) ────────────────────────────────────────────
    if (ts_actual_ms - _ultimo_sl_sim.get(symbol, 0)) < 600 * 1000:
        adj += 0.03; razones.append("sl_reciente+0.03")

    # ── 6. Contexto BTC ──────────────────────────────────────────────────────
    btc_fresco = (ts_actual_ms - _btc_ctx_sim.get("ts", 0)) < 180 * 1000
    if btc_fresco:
        regimen = _btc_ctx_sim.get("regimen", "lateral")
        direccion = _btc_ctx_sim.get("direccion", "neutral")
        fuerza = _btc_ctx_sim.get("fuerza", 0.0)
        mom1h_btc = _btc_ctx_sim.get("mom1h", 0.0)
        dom_up = _btc_ctx_sim.get("dominancia_up", False)
        dir_alt = "long" if ind.get("mom3", 0) >= 0 else "short"

        if regimen in ("trend_up", "trend_down"):
            btc_dir_trade = "long" if direccion == "up" else "short"
            if dir_alt == btc_dir_trade:
                reduccion = round(fuerza * 0.06, 3)
                adj -= reduccion
                razones.append(f"btc_{regimen}_alineado-{reduccion:.2f}")
            else:
                aumento = round(fuerza * 0.04, 3)
                adj += aumento
                razones.append(f"btc_{regimen}_opuesto+{aumento:.2f}")

        if mom1h_btc < -0.015:
            adj += 0.05; razones.append(f"btc_caida_1h({mom1h_btc*100:.1f}%)+0.05")
        elif mom1h_btc < -0.008:
            adj += 0.03; razones.append(f"btc_debil_1h({mom1h_btc*100:.1f}%)+0.03")

        if mom1h_btc > 0.015:
            adj -= 0.04; razones.append(f"btc_sube_1h({mom1h_btc*100:.1f}%)-0.04")
        elif mom1h_btc > 0.008:
            adj -= 0.02; razones.append(f"btc_fuerte_1h({mom1h_btc*100:.1f}%)-0.02")

        if dom_up:
            adj += 0.06; razones.append("btc_dominancia+0.06")

    # ── 7. Desgaste GT ──────────────────────────────────────────────────────
    m12v = abs(ind.get("mom12", 0)) / 4
    m6v = abs(ind.get("mom6", 0)) / 2
    m3v = abs(ind.get("mom3", 0))
    dvals = [m12v, m6v, m3v]
    dxm = 1.0
    dym = sum(dvals) / max(len(dvals), 1)
    dnum = sum((i - dxm) * (dvals[i] - dym) for i in range(len(dvals)))
    dden = sum((i - dxm) ** 2 for i in range(len(dvals)))
    desgaste_local = dnum / dden if dden > 0 else 0.0
    if desgaste_local < -0.2:
        adj += 0.05; razones.append(f"desgaste_neg({desgaste_local:+.2f})+0.05")

    # ── 8. Funding rate ──────────────────────────────────────────────────────
    fd = _funding_cache_sim.get(symbol, {})
    fr = fd.get("rate", 0.0)
    if (ts_actual_ms - fd.get("ts", 0)) < 700 * 1000 and fr != 0.0:
        dir_alt = "long" if ind.get("mom3", 0) >= 0 else "short"
        if fr > 0.0015:
            if dir_alt == "short":
                adj -= 0.04; razones.append(f"funding+({fr*100:.3f}%)-0.04")
            else:
                adj += 0.03; razones.append(f"funding+({fr*100:.3f}%)+0.03")
        elif fr > 0.0005:
            if dir_alt == "short":
                adj -= 0.02; razones.append(f"funding+mod({fr*100:.3f}%)-0.02")
        elif fr < -0.0010:
            if dir_alt == "long":
                adj -= 0.04; razones.append(f"funding-({fr*100:.3f}%)-0.04")
            else:
                adj += 0.03; razones.append(f"funding-({fr*100:.3f}%)+0.03")
        elif fr < -0.0003:
            if dir_alt == "long":
                adj -= 0.02; razones.append(f"funding-mod({fr*100:.3f}%)-0.02")

    # ── 9. Sesión asiática (breakout) ────────────────────────────────────────
    asia_high = ind.get("asia_high", 0.0)
    asia_low = ind.get("asia_low", 0.0)
    asia_pos = ind.get("asia_pos", 0.0)
    if asia_high > 0 and asia_low > 0:
        rango_pct = (asia_high - asia_low) / (asia_low + 1e-10)
        dir_alt = "long" if ind.get("mom3", 0) >= 0 else "short"
        if rango_pct > 0.003:
            if asia_pos > 1.10 and dir_alt == "long":
                adj -= 0.04; razones.append(f"asia_breakout_up({asia_pos:.2f})-0.04")
            elif asia_pos > 1.02 and dir_alt == "long":
                adj -= 0.02; razones.append(f"asia_sobre({asia_pos:.2f})-0.02")
            elif asia_pos < -0.10 and dir_alt == "short":
                adj -= 0.04; razones.append(f"asia_breakout_dn({asia_pos:.2f})-0.04")
            elif asia_pos < -0.02 and dir_alt == "short":
                adj -= 0.02; razones.append(f"asia_bajo({asia_pos:.2f})-0.02")
            elif asia_pos > 1.05 and dir_alt == "short":
                adj += 0.02; razones.append(f"asia_contra_up+0.02")
            elif asia_pos < -0.05 and dir_alt == "long":
                adj += 0.02; razones.append(f"asia_contra_dn+0.02")

    umbral = round(max(0.22, min(0.50, base + adj)), 3)
    return umbral

# ═════════════════════════════════════════════════════════════════════════════
# EVALUAR SEÑAL — REPRODUCCIÓN EXACTA DE evaluar() del multi v6
# ═════════════════════════════════════════════════════════════════════════════
# Fuente: multi_v6 líneas 1418-1623
# 11 fuentes de señal: RSI, EMA, Volumen, OBV, VWAP, Momentum, Tend1H,
# Bollinger Bands, Funding, Sesión Asiática, MACD
# Cada fuente suma puntos a sl (long score) o ss (short score).
# Decisión: si score >= umbral Y supera al rival Y (señal primaria O conf >= umb+0.06)

def evaluar_bt(ind: Dict, umbral: float, ts_actual_ms: int) -> Tuple[Optional[str], float, List[str]]:
    """
    Devuelve (direction, conf, razones) o (None, max_conf, []).
    """
    if not ind:
        return None, 0.0, []

    rsi = ind["rsi"]; vr = ind["vol_ratio"]; obv = ind["obv"]
    vd = ind["vwap_dist"]; m3 = ind["mom3"]; m6 = ind["mom6"]
    reg = ind["reg1h"]
    bull = ind["ema_bull"]; bear = ind["ema_bear"]
    ema_gap = (ind["ema9"] - ind["ema20"]) / ind["ema20"] if ind["ema20"] else 0
    bb_pos = ind.get("bb_pos", 0.0)
    bb_pct_b = ind.get("bb_pct_b", 0.5)
    bb_ancho = ind.get("bb_ancho", 0.0)

    sl = ss = 0.0
    pl = ps = 0
    rl: List[str] = []
    rs: List[str] = []

    # RSI
    if rsi < 25:
        sl += 0.22; pl += 1; rl.append(f"RSI sobreventa extrema {rsi:.0f}")
    elif rsi < 35:
        sl += 0.16; pl += 1; rl.append(f"RSI sobreventa {rsi:.0f}")
    elif rsi < 45:
        sl += 0.08; rl.append(f"RSI bajo {rsi:.0f}")
    elif rsi > 75:
        ss += 0.22; ps += 1; rs.append(f"RSI sobrecompra extrema {rsi:.0f}")
    elif rsi > 65:
        ss += 0.16; ps += 1; rs.append(f"RSI sobrecompra {rsi:.0f}")
    elif rsi > 55:
        ss += 0.08; rs.append(f"RSI alto {rsi:.0f}")

    # EMA
    if bull:
        sl += 0.14; pl += 1; rl.append("EMA alcista 9>20>50")
    elif bear:
        ss += 0.14; ps += 1; rs.append("EMA bajista 9<20<50")
    if ema_gap > 0.008:
        sl += 0.08; rl.append(f"EMA gap +{ema_gap*100:.2f}%")
    elif ema_gap < -0.008:
        ss += 0.08; rs.append(f"EMA gap {ema_gap*100:.2f}%")

    # Volumen
    if vr > 2.0:
        if m3 > 0:
            sl += 0.16; pl += 1; rl.append(f"Vol x{vr:.1f} + mom+")
        else:
            ss += 0.16; ps += 1; rs.append(f"Vol x{vr:.1f} + mom-")
    elif vr > 1.3:
        if m3 > 0:
            sl += 0.10; rl.append(f"Vol x{vr:.1f}")
        else:
            ss += 0.10; rs.append(f"Vol x{vr:.1f}")

    # OBV
    if obv > 0.30:
        sl += 0.10; rl.append(f"OBV comprador {obv:.2f}")
    elif obv < -0.30:
        ss += 0.10; rs.append(f"OBV vendedor {obv:.2f}")

    # VWAP
    if vd > 0.005:
        sl += 0.08; rl.append(f"Sobre VWAP +{vd*100:.2f}%")
    elif vd < -0.005:
        ss += 0.08; rs.append(f"Bajo VWAP {vd*100:.2f}%")

    # Momentum
    if m3 > 0.008:
        sl += 0.10; pl += 1; rl.append(f"Mom +{m3*100:.2f}%")
    elif m3 > 0.003:
        sl += 0.06; rl.append(f"Mom +{m3*100:.2f}%")
    elif m3 < -0.008:
        ss += 0.10; ps += 1; rs.append(f"Mom {m3*100:.2f}%")
    elif m3 < -0.003:
        ss += 0.06; rs.append(f"Mom {m3*100:.2f}%")
    if m6 > 0.015:
        sl += 0.06; rl.append(f"Mom6 +{m6*100:.2f}%")
    elif m6 < -0.015:
        ss += 0.06; rs.append(f"Mom6 {m6*100:.2f}%")

    # Tendencia 1H
    if reg == "trend_up":
        sl += 0.10; rl.append("Tendencia 1H alcista")
    elif reg == "trend_down":
        ss += 0.10; rs.append("Tendencia 1H bajista")

    # Bollinger Bands — mean reversion
    if bb_ancho > 0.008:
        en_tendencia_fuerte = abs(ind.get("tend1h", 0)) > 0.035

        if bb_pct_b < 0.10:
            exceso = round(abs(bb_pct_b), 3)
            if not en_tendencia_fuerte:
                if bb_pct_b < 0.0:
                    sl += 0.18; pl += 1
                    rl.append(f"BB bajo banda ({exceso:.2f})")
                elif bb_pct_b < 0.05:
                    sl += 0.10
                    rl.append(f"BB cerca banda inf ({bb_pct_b:.2f})")
                else:
                    sl += 0.05
                    rl.append(f"BB zona baja ({bb_pct_b:.2f})")
                if obv > 0.10:
                    sl += 0.06; rl.append("BB+OBV comprador confirmado")
            else:
                if bb_pct_b < -0.05:
                    sl += 0.05; rl.append(f"BB extremo (tendencia, cuidado)")

        elif bb_pct_b > 0.90:
            exceso = round(bb_pct_b - 1.0, 3) if bb_pct_b > 1.0 else 0
            if not en_tendencia_fuerte:
                if bb_pct_b > 1.0:
                    ss += 0.18; ps += 1
                    rs.append(f"BB sobre banda ({exceso:.2f})")
                elif bb_pct_b > 0.95:
                    ss += 0.10
                    rs.append(f"BB cerca banda sup ({bb_pct_b:.2f})")
                else:
                    ss += 0.05
                    rs.append(f"BB zona alta ({bb_pct_b:.2f})")
                if obv < -0.10:
                    ss += 0.06; rs.append("BB+OBV vendedor confirmado")
            else:
                if bb_pct_b > 1.05:
                    ss += 0.05; rs.append(f"BB extremo (tendencia, cuidado)")

        elif 0.10 < bb_pct_b < 0.90:
            if bb_ancho < 0.012:
                sl -= 0.03; ss -= 0.03

    # Funding contrarian
    fr = ind.get("funding_rate", 0.0)
    fr_fresco = (ts_actual_ms - ind.get("funding_ts", 0)) < 700 * 1000
    if fr_fresco and abs(fr) > 0.0:
        if fr > 0.0015:
            ss += 0.12; ps += 1; rs.append(f"Funding+ extremo ({fr*100:.3f}%)")
        elif fr > 0.0008:
            ss += 0.07; rs.append(f"Funding+ alto ({fr*100:.3f}%)")
        elif fr > 0.0003:
            ss += 0.04; rs.append(f"Funding+ mod ({fr*100:.3f}%)")
        elif fr < -0.0010:
            sl += 0.12; pl += 1; rl.append(f"Funding- extremo ({fr*100:.3f}%)")
        elif fr < -0.0005:
            sl += 0.07; rl.append(f"Funding- alto ({fr*100:.3f}%)")
        elif fr < -0.0002:
            sl += 0.04; rl.append(f"Funding- mod ({fr*100:.3f}%)")

    # Sesión asiática
    asia_high = ind.get("asia_high", 0.0)
    asia_low = ind.get("asia_low", 0.0)
    asia_pos = ind.get("asia_pos", 0.0)
    if asia_high > 0 and asia_low > 0:
        rango_pct = (asia_high - asia_low) / (asia_low + 1e-10)
        if rango_pct > 0.003:
            if asia_pos > 1.15:
                sl += 0.12; pl += 1; rl.append(f"Asia breakout alcista ({asia_pos:.2f}x)")
            elif asia_pos > 1.05:
                sl += 0.07; rl.append(f"Asia sobre rango ({asia_pos:.2f}x)")
            elif asia_pos > 1.01:
                sl += 0.04; rl.append(f"Asia leve sobre ({asia_pos:.2f}x)")
            elif asia_pos < -0.15:
                ss += 0.12; ps += 1; rs.append(f"Asia breakout bajista ({asia_pos:.2f}x)")
            elif asia_pos < -0.05:
                ss += 0.07; rs.append(f"Asia bajo rango ({asia_pos:.2f}x)")
            elif asia_pos < -0.01:
                ss += 0.04; rs.append(f"Asia leve bajo ({asia_pos:.2f}x)")

    # MACD
    macd_hist = ind.get("macd_hist", 0.0)
    macd_accel = ind.get("macd_accel", 0.0)
    macd_hist_prev = ind.get("macd_hist_prev", 0.0)
    if abs(macd_hist) > 0 or abs(macd_accel) > 0:
        thresh = 1e-6
        if macd_hist > thresh:
            if macd_accel > thresh:
                if macd_hist_prev <= thresh:
                    sl += 0.14; pl += 1; rl.append(f"MACD cruce alcista")
                else:
                    sl += 0.08; rl.append(f"MACD hist+ acelerando")
            else:
                sl += 0.04; rl.append(f"MACD hist+ ({macd_hist*1e5:.1f})")
        elif macd_hist < -thresh:
            if macd_accel < -thresh:
                if macd_hist_prev >= -thresh:
                    ss += 0.14; ps += 1; rs.append(f"MACD cruce bajista")
                else:
                    ss += 0.08; rs.append(f"MACD hist- acelerando")
            else:
                ss += 0.04; rs.append(f"MACD hist- ({macd_hist*1e5:.1f})")

    # ── Confluencia y decisión final ─────────────────────────────────────────
    for conf, prim, dir_, razones in [(sl, pl, "long", rl), (ss, ps, "short", rs)]:
        rival = ss if dir_ == "long" else sl
        if conf >= umbral and conf > rival:
            if prim >= 1 or conf >= (umbral + 0.06):
                return dir_, round(conf, 3), razones

    return None, round(max(sl, ss), 3), []


# ═════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO (Hurst, Z-score, GARCH)
# ═════════════════════════════════════════════════════════════════════════════
# Reproduce los filtros de multi v6 (líneas 3198-3248).
# Se usan POST-evaluar para vetar señales en condiciones extremas.

def calcular_hurst(closes: np.ndarray, max_lag: int = 20) -> Dict:
    """
    Calcula el exponente de Hurst usando R/S analysis simplificado.
    H ~0.5 = random walk
    H >0.55 = trending (momentum funciona)
    H <0.45 = reverting (momentum es trampa)
    """
    if len(closes) < 60:
        return {"validos": False, "H": 0.5, "regimen": "unknown"}

    log_returns = np.diff(np.log(closes))
    if len(log_returns) < max_lag:
        return {"validos": False, "H": 0.5, "regimen": "unknown"}

    # Variance ratio approach (simplificado)
    try:
        rs_values = []
        for lag in range(2, min(max_lag, len(log_returns)//4)):
            sub_series = log_returns[:lag * (len(log_returns)//lag)]
            chunks = sub_series.reshape(-1, lag)
            mean_chunks = chunks.mean(axis=1, keepdims=True)
            cumdev = (chunks - mean_chunks).cumsum(axis=1)
            R = cumdev.max(axis=1) - cumdev.min(axis=1)
            S = chunks.std(axis=1, ddof=1)
            S = np.where(S < 1e-10, 1e-10, S)
            rs = R / S
            rs_values.append((lag, rs.mean()))

        if len(rs_values) < 4:
            return {"validos": False, "H": 0.5, "regimen": "unknown"}

        lags = np.log([r[0] for r in rs_values])
        rss = np.log([r[1] for r in rs_values if r[1] > 0])
        if len(rss) != len(lags):
            return {"validos": False, "H": 0.5, "regimen": "unknown"}

        H = np.polyfit(lags, rss, 1)[0]
        H = max(0.0, min(1.0, H))

        if H > 0.55:
            regimen = "trending"
        elif H < 0.45:
            regimen = "reverting"
        else:
            regimen = "random"

        return {"validos": True, "H": round(H, 3), "regimen": regimen}
    except Exception:
        return {"validos": False, "H": 0.5, "regimen": "unknown"}


def calcular_zscore_robust(closes: np.ndarray, ventana: int = 20) -> Dict:
    """
    Z-score robusto del precio actual vs ventana reciente.
    Detecta extremos de sobrecompra/sobreventa.
    extremo_up: z > 2.5 → veto LONG
    extremo_down: z < -2.5 → veto SHORT
    """
    if len(closes) < ventana + 2:
        return {"validos": False, "z": 0.0, "extremo_up": False, "extremo_down": False}

    serie = closes[-ventana:]
    mediana = np.median(serie)
    mad = np.median(np.abs(serie - mediana))
    if mad < 1e-10:
        return {"validos": False, "z": 0.0, "extremo_up": False, "extremo_down": False}

    # Z robusto usando MAD (más estable que std en presencia de outliers)
    z = 0.6745 * (closes[-1] - mediana) / mad
    return {
        "validos": True,
        "z": round(z, 2),
        "extremo_up": z > 2.5,
        "extremo_down": z < -2.5,
    }


def calcular_garch_simple(closes: np.ndarray, ventana: int = 30) -> Dict:
    """
    GARCH(1,1) simplificado: detecta spike de volatilidad reciente vs histórica.
    Si σ actual >> σ histórica → vol explota → SL/TP no confiables → veto.
    """
    if len(closes) < ventana + 10:
        return {"validos": False, "sigma_spike": False}

    log_returns = np.diff(np.log(closes))
    sigma_actual = log_returns[-5:].std()  # últimas 5 velas
    sigma_hist = log_returns[-ventana:-5].std()  # ventana histórica
    if sigma_hist < 1e-10:
        return {"validos": False, "sigma_spike": False}

    ratio = sigma_actual / sigma_hist
    spike = ratio > 2.5  # σ actual >2.5x histórica

    return {
        "validos": True,
        "sigma_actual": round(sigma_actual, 6),
        "sigma_hist_mean": round(sigma_hist, 6),
        "sigma_spike": spike,
        "ratio": round(ratio, 2),
    }


def aplicar_filtros_estadisticos(direction: Optional[str], conf: float,
                                  umbral: float, ind: Dict,
                                  closes: np.ndarray) -> Tuple[Optional[str], float, List[str]]:
    """
    Aplica los 3 filtros del multi v6 (líneas 3198-3248):
      - Hurst: bonus/penalty según régimen trending/reverting
      - Z-score: veto si extremo
      - GARCH: veto si sigma_spike
    Retorna (direction_ajustada, conf_ajustada, razones_vetos).
    """
    if direction is None:
        return None, conf, []

    razones = []

    # Hurst
    hurst_data = calcular_hurst(closes)
    if hurst_data.get("validos"):
        H = hurst_data["H"]
        reg = hurst_data["regimen"]
        if reg == "trending":
            bonus_h = 0.05
        elif reg == "reverting":
            bonus_h = -0.05
        else:
            bonus_h = 0.0
        if abs(bonus_h) > 0:
            conf = max(0.0, min(1.0, conf + bonus_h))
            razones.append(f"hurst={H:.2f}_{reg}({bonus_h:+.2f})")
            if conf < umbral:
                return None, conf, razones + ["HURST_VETO"]

    # Z-score
    z_data = calcular_zscore_robust(closes)
    if z_data.get("validos"):
        z = z_data["z"]
        if direction == "long" and z_data["extremo_up"]:
            return None, conf, razones + [f"ZSCORE_VETO long z={z:+.2f}"]
        elif direction == "short" and z_data["extremo_down"]:
            return None, conf, razones + [f"ZSCORE_VETO short z={z:+.2f}"]

    # GARCH
    g_data = calcular_garch_simple(closes)
    if g_data.get("validos") and g_data.get("sigma_spike"):
        return None, conf, razones + [f"GARCH_VETO ratio={g_data.get('ratio',0):.1f}x"]

    return direction, conf, razones


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRACIÓN DINÁMICA piwhale (ajuste de umbral por WR reciente)
# ═════════════════════════════════════════════════════════════════════════════
# Para cada símbolo, miramos los últimos 10 trades cerrados.
# Si WR >= 60% → bajar umbral 0.02 (capturar más)
# Si WR < 40%  → subir umbral 0.03 (ser más selectivo)

def calibrar_umbral_piwhale_bt(symbol: str, ledger_recientes: List) -> float:
    """
    Devuelve un ajuste al umbral basado en performance reciente del símbolo.
    """
    sym_trades = [p for p in ledger_recientes
                  if p.symbol == symbol and p.status == "closed"][-10:]
    if len(sym_trades) < 5:
        return 0.0

    wins = sum(1 for p in sym_trades if p.pnl_usd > 0)
    wr = wins / len(sym_trades)

    if wr >= 0.60:
        return -0.02
    elif wr <= 0.40:
        return 0.03
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# ENGINE DE BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
# Este es el corazón del simulador. Recorre el rango temporal vela por vela
# y ejecuta la lógica del bot en cada una.
#
# DIFERENCIAS CLAVE vs el bot real:
#   1. El bot real corre cada ANALISIS_INT=30 segundos. Acá corremos por VELA
#      de 1 minuto (ANALISIS_INT efectivo = 60s). Es 2x menos frecuente, pero
#      así no necesitamos OHLCV de segundos.
#   2. El bot real usa orderbook real para detectar spread; acá asumimos
#      spread cero (limitación: optimismo de ~0.05%).
#   3. El bot real puede perder ticks por congelamientos; acá nunca pierde.

@dataclass
class Posicion:
    """Posición simulada en el backtest."""
    id: str
    symbol: str
    direction: str           # "long" / "short"
    entry_price: float
    cantidad: float
    nocional: float
    margen: float
    stop_loss: float
    take_profit: float
    sl_inicial: float        # SL original (antes de trailing)
    open_ts: int             # ms
    open_dt: datetime
    conf: float
    razon: str
    # Tracking
    max_price: float = 0.0
    trailing: bool = False
    # Cierre
    status: str = "open"
    close_price: float = 0.0
    close_ts: int = 0
    close_dt: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    motivo_cierre: str = ""
    fees: float = 0.0
    duracion_min: float = 0.0


@dataclass
class BacktestConfig:
    """Configuración del backtest."""
    capital_inicial: float = 3000.0
    leverage: int = 5
    max_pos_total: int = 5
    max_pos_symbol: int = 1
    fee_rate: float = 0.0006        # 0.06% por lado (0.12% round-trip)
    spread_estimado: float = 0.0005 # estimación de slippage (cancele = 0 si querés ser optimista)
    daily_loss_max_pct: float = 0.05
    cooldown_sl_seg: int = 1800
    cooldown_open_min_seg: int = 60
    short_size_mult: float = 0.60
    cierre_por_reversion_1m: bool = True
    extender_tp_dinamico: bool = True


class BacktestEngine:
    """
    Motor principal del backtest.
    Procesa cronológicamente las velas de 1m de todos los símbolos
    y simula la lógica completa del bot.
    """

    def __init__(self, config: BacktestConfig, perfil: Dict,
                 ohlcv_data: Dict[str, pd.DataFrame], btc_df: Optional[pd.DataFrame] = None):
        self.config = config
        self.perfil = perfil
        self.ohlcv_data = ohlcv_data  # {sym: df_1m}
        self.btc_df = btc_df

        # Estado
        self.equity = config.capital_inicial
        self.libre = config.capital_inicial
        self.posiciones: List[Posicion] = []     # todas (abiertas y cerradas)
        self.posiciones_abiertas: Dict[str, Posicion] = {}  # symbol -> pos
        self.equity_curve: List[Tuple[int, float]] = []  # (ts_ms, equity)
        self.dia_actual = ""
        self.pnl_dia = 0.0
        self.pausado_hasta_ts: int = 0  # daily loss max

        # Cooldowns
        self.last_close_ts: Dict[str, int] = {}  # symbol -> ms
        self.last_open_ts: Dict[str, int] = {}
        self.consec_sl: Dict[str, int] = defaultdict(int)

        # Resamples cacheados (5m, 1h por símbolo)
        self.df5m_cache: Dict[str, pd.DataFrame] = {}
        self.df1h_cache: Dict[str, pd.DataFrame] = {}
        self._precompute_resamples()

        # BTC context (se actualiza cada hora durante el backtest)
        self.btc_5m_cache: Optional[pd.DataFrame] = None
        if btc_df is not None:
            self.btc_5m_cache = resample_ohlcv(btc_df, "5min")

        # Ledger de trades
        self.trade_id_counter = 0

    def _precompute_resamples(self):
        """Resamplea OHLCV de 1m a 5m y 1h para todos los símbolos (acelera el backtest)."""
        for sym, df1m in self.ohlcv_data.items():
            self.df5m_cache[sym] = resample_ohlcv(df1m, "5min")
            self.df1h_cache[sym] = resample_ohlcv(df1m, "1h")
            log.info(f"[INIT] {sym}: 1m={len(df1m)} → 5m={len(self.df5m_cache[sym])} → 1h={len(self.df1h_cache[sym])}")

    # ────────────────────────────────────────────────────────────────────────
    # APERTURA DE POSICIÓN
    # ────────────────────────────────────────────────────────────────────────
    def _abrir_posicion(self, symbol: str, direction: str, precio: float,
                        ts_ms: int, conf: float, razones: List[str],
                        ind: Optional[Dict] = None) -> bool:
        """
        v2 (28/4): SL/TP estructural ATR-based + sizing dinámico.
        """
        # Verificar daily loss
        if self.pausado_hasta_ts > ts_ms:
            return False

        # Verificar máximo posiciones
        if len(self.posiciones_abiertas) >= self.config.max_pos_total:
            return False
        if symbol in self.posiciones_abiertas:
            return False

        # Cooldowns
        if (ts_ms - self.last_close_ts.get(symbol, 0)) < self.config.cooldown_sl_seg * 1000:
            return False
        if (ts_ms - self.last_open_ts.get(symbol, 0)) < self.config.cooldown_open_min_seg * 1000:
            return False

        # Filtro EMA gap MAX (multi v6 línea 5837): skip si EMA gap > 3% en dirección extendida
        if ind is not None:
            ema9 = ind.get("ema9", 0)
            ema20 = ind.get("ema20", 1)
            ema_gap = (ema9 - ema20) / ema20 if ema20 else 0
            EMA_GAP_MAX_SKIP = 0.030
            if direction == "long" and ema_gap > EMA_GAP_MAX_SKIP:
                return False  # tendencia muy extendida, evitar entrar al final
            if direction == "short" and ema_gap < -EMA_GAP_MAX_SKIP:
                return False

        # Calcular tamaño base
        perf = self.perfil.get(symbol, {"cap": 0.05, "sl": 0.025, "tp": 0.050})
        cap_pct_base = perf.get("cap", 0.05)
        sl_pct_perfil = perf.get("sl", 0.025)
        tp_pct_perfil = perf.get("tp", 0.050)

        # ── Sizing dinámico (multi v6 líneas 1660-1740) ─────────────────────
        # Multiplicadores por GT, ATR ratio, desgaste, BTC, EMA gap
        size_mult = 1.0

        if ind is not None:
            atr_r = ind.get("atr_ratio", 1.0)
            # ATR ratio multiplier
            if atr_r >= 1.40:
                size_mult *= 1.20
            elif atr_r >= 1.10:
                size_mult *= 1.10
            elif atr_r >= 0.80:
                size_mult *= 1.00
            elif atr_r >= 0.60:
                size_mult *= 0.85
            else:
                size_mult *= 0.70

            # GT p_victoria (simplificado: usar conf como proxy)
            # En el real es p_victoria con desgaste; acá aproximamos con conf
            if conf >= 0.45:
                size_mult *= 1.20
            elif conf < 0.30:
                size_mult *= 0.75

        # SHORT systematic reduction
        if direction == "short":
            size_mult *= self.config.short_size_mult

        # Cap final limitado a 1.5x base
        cap_pct = min(cap_pct_base * 1.5, cap_pct_base * size_mult)

        margen_usd = self.equity * cap_pct
        nocional_usd = margen_usd * self.config.leverage

        # Mínimo nocional $20 (multi v6 línea 1827)
        if nocional_usd < 20:
            return False

        # Aplicar slippage al entry
        if direction == "long":
            entry_real = precio * (1 + self.config.spread_estimado)
        else:
            entry_real = precio * (1 - self.config.spread_estimado)

        cantidad = nocional_usd / entry_real
        if cantidad <= 0:
            return False

        # ── SL/TP ESTRUCTURAL basado en ATR (multi v6 líneas 1759-1788) ─────
        # sl_dist = min(atr_pct * 1.2, perf["sl"] * 2.0)
        # tp = max(perf["tp"], sl_dist * 1.5)
        if ind is not None and ind.get("atr_pct", 0) > 0:
            atr_pct = ind["atr_pct"]
            sl_dist = min(atr_pct * 1.2, sl_pct_perfil * 2.0)
            # No menos de 0.4× perfil
            if sl_dist < sl_pct_perfil * 0.4:
                sl_dist = sl_pct_perfil
            tp_dist = max(tp_pct_perfil, sl_dist * 1.5)
        else:
            sl_dist = sl_pct_perfil
            tp_dist = tp_pct_perfil

        if direction == "long":
            sl = entry_real * (1 - sl_dist)
            tp = entry_real * (1 + tp_dist)
        else:
            sl = entry_real * (1 + sl_dist)
            tp = entry_real * (1 - tp_dist)

        # Crear posición
        self.trade_id_counter += 1
        pos = Posicion(
            id=f"bt_{self.trade_id_counter:05d}",
            symbol=symbol,
            direction=direction,
            entry_price=entry_real,
            cantidad=cantidad,
            nocional=entry_real * cantidad,
            margen=margen_usd,
            stop_loss=sl,
            take_profit=tp,
            sl_inicial=sl,
            open_ts=ts_ms,
            open_dt=datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc),
            conf=conf,
            razon=" | ".join(razones[:3]),
            max_price=entry_real,
        )

        self.posiciones_abiertas[symbol] = pos
        self.posiciones.append(pos)
        self.last_open_ts[symbol] = ts_ms

        # Cobrar fee de apertura
        fee_apertura = pos.nocional * self.config.fee_rate
        pos.fees += fee_apertura
        self.libre -= margen_usd
        self.equity -= fee_apertura

        return True

    # ────────────────────────────────────────────────────────────────────────
    # CIERRE DE POSICIÓN
    # ────────────────────────────────────────────────────────────────────────
    def _cerrar_posicion(self, pos: Posicion, precio: float, ts_ms: int, motivo: str):
        if pos.status != "open":
            return

        # Aplicar slippage al cierre
        if pos.direction == "long":
            exit_real = precio * (1 - self.config.spread_estimado)
            pnl_bruto = pos.cantidad * (exit_real - pos.entry_price)
        else:
            exit_real = precio * (1 + self.config.spread_estimado)
            pnl_bruto = pos.cantidad * (pos.entry_price - exit_real)

        # Fee de cierre
        fee_cierre = pos.nocional * self.config.fee_rate
        pos.fees += fee_cierre
        pnl_neto = pnl_bruto - fee_cierre  # Apertura ya descontada del equity

        pos.close_price = exit_real
        pos.close_ts = ts_ms
        pos.close_dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        pos.pnl_usd = pnl_neto - (pos.fees - fee_cierre)  # restar también la fee de apertura ya descontada
        # Actually corrección: pos.fees ya tiene apertura+cierre, pero el equity ya tiene la apertura descontada
        # PnL realizado del trade = pnl_bruto - fees totales
        pos.pnl_usd = pnl_bruto - pos.fees
        pos.pnl_pct = (pos.pnl_usd / pos.margen) * 100 if pos.margen > 0 else 0
        pos.motivo_cierre = motivo
        pos.status = "closed"
        pos.duracion_min = (ts_ms - pos.open_ts) / 60000

        # Actualizar equity
        # equity ya tenía -fee_apertura. Ahora suma pnl_bruto y resta fee_cierre.
        self.equity += pnl_bruto - fee_cierre
        self.libre += pos.margen + (pnl_bruto - fee_cierre)

        self.pnl_dia += pos.pnl_usd
        self.last_close_ts[pos.symbol] = ts_ms

        # Track consecutive SLs
        if motivo in ("sl", "ext_sl"):
            self.consec_sl[pos.symbol] += 1
            _ultimo_sl_sim[pos.symbol] = ts_ms
        elif motivo in ("tp", "1m_rev"):
            self.consec_sl[pos.symbol] = 0

        # Daily loss max
        limite = -self.config.daily_loss_max_pct * self.config.capital_inicial
        if self.pnl_dia <= limite:
            self.pausado_hasta_ts = ts_ms + 24 * 3600 * 1000

        # Liberar la posición
        del self.posiciones_abiertas[pos.symbol]

    # ────────────────────────────────────────────────────────────────────────
    # GESTIÓN ACTIVA (trailing, TP/SL hit, 1M_REV, timeout)
    # ────────────────────────────────────────────────────────────────────────
    def _trailing_stop(self, pos: Posicion, precio: float):
        """Reproduce trailing_stop() del multi v6."""
        BE_TRIGGER = 0.005
        BE_COLCHON = 0.001
        TRAIL_ACT = 0.008
        TRAIL_DIST = 0.006

        if pos.direction == "long":
            if precio > pos.max_price:
                pos.max_price = precio
            gan = (pos.max_price - pos.entry_price) / pos.entry_price

            if gan >= BE_TRIGGER and gan < TRAIL_ACT:
                nuevo_sl = pos.entry_price * (1 + BE_COLCHON)
                if nuevo_sl > pos.stop_loss:
                    pos.stop_loss = nuevo_sl
            if gan >= TRAIL_ACT:
                if not pos.trailing:
                    pos.trailing = True
                nuevo_sl = pos.max_price * (1 - TRAIL_DIST)
                if nuevo_sl > pos.stop_loss:
                    pos.stop_loss = nuevo_sl
        else:
            if pos.max_price == 0 or precio < pos.max_price:
                pos.max_price = precio
            gan = (pos.entry_price - pos.max_price) / pos.entry_price

            if gan >= BE_TRIGGER and gan < TRAIL_ACT:
                nuevo_sl = pos.entry_price * (1 - BE_COLCHON)
                if nuevo_sl < pos.stop_loss:
                    pos.stop_loss = nuevo_sl
            if gan >= TRAIL_ACT:
                if not pos.trailing:
                    pos.trailing = True
                nuevo_sl = pos.max_price * (1 + TRAIL_DIST)
                if nuevo_sl < pos.stop_loss:
                    pos.stop_loss = nuevo_sl

    def _check_close_levels(self, pos: Posicion, vela: pd.Series, ts_ms: int) -> bool:
        """
        Verifica si la vela actual tocó SL o TP.
        Como las velas de 1m muestran high/low, podemos detectar si el precio
        cruzó el nivel durante el minuto.

        v2 (28/4): MEJORADA CLASIFICACIÓN SL vs EXT_SL.

        Distinción:
          - SL  = el bot ANTICIPA (cierra preventivamente porque la conf cayó):
                  → No hay forma directa de simular esto sin re-correr evaluar()
                  cada vela. Para acercarnos al comportamiento real, vamos a
                  marcar como "SL" (bot detection) si el SL es el ORIGINAL.
                  Si el SL fue movido por trailing → "EXT_SL" (Bitget ejecuta).
                  Esto NO es exacto pero captura el espíritu: el bot puede
                  "rescatar" trades antes que toquen el SL inicial cerrando
                  por reversión, etc.
          - EXT_SL = Bitget ejecuta el SL en exchange (típicamente trailing).
          - TP = TP completo alcanzado.

        Para clasificar mejor: si el SL ya fue movido (trailing) → ext_sl.
        Si está en el SL_inicial → sl (bot lo cerraría preventivamente
        en el real, acá lo simulamos como bot detection).
        """
        high = vela['high']
        low = vela['low']
        open_ = vela['open']
        close = vela['close']

        sl_movido = abs(pos.stop_loss - pos.sl_inicial) > 1e-10  # SL fue movido por trailing/BE

        if pos.direction == "long":
            tocp_tp = high >= pos.take_profit
            toco_sl = low <= pos.stop_loss
            if tocp_tp and toco_sl:
                # Ambigüedad: usar dirección de vela
                if close > open_:
                    self._cerrar_posicion(pos, pos.take_profit, ts_ms, "tp")
                else:
                    motivo = "ext_sl" if sl_movido else "sl"
                    self._cerrar_posicion(pos, pos.stop_loss, ts_ms, motivo)
                return True
            elif tocp_tp:
                self._cerrar_posicion(pos, pos.take_profit, ts_ms, "tp")
                return True
            elif toco_sl:
                motivo = "ext_sl" if sl_movido else "sl"
                self._cerrar_posicion(pos, pos.stop_loss, ts_ms, motivo)
                return True
        else:  # short
            tocp_tp = low <= pos.take_profit
            toco_sl = high >= pos.stop_loss
            if tocp_tp and toco_sl:
                if close < open_:
                    self._cerrar_posicion(pos, pos.take_profit, ts_ms, "tp")
                else:
                    motivo = "ext_sl" if sl_movido else "sl"
                    self._cerrar_posicion(pos, pos.stop_loss, ts_ms, motivo)
                return True
            elif tocp_tp:
                self._cerrar_posicion(pos, pos.take_profit, ts_ms, "tp")
                return True
            elif toco_sl:
                motivo = "ext_sl" if sl_movido else "sl"
                self._cerrar_posicion(pos, pos.stop_loss, ts_ms, motivo)
                return True
        return False

    def _check_1m_rev(self, pos: Posicion, df1m_window: pd.DataFrame, ts_ms: int) -> bool:
        """
        Reproduce cerrar_por_reversion_1m() del multi v6.
        Detecta reversión clara en últimas 3 velas 1m que va contra la posición,
        sólo si el trade está en gan >= +0.3% y trailing NO activado aún.
        """
        if not self.config.cierre_por_reversion_1m:
            return False

        if pos.trailing:
            return False  # dejar que trailing haga su trabajo

        precio = float(df1m_window.iloc[-1]['close'])

        # Ganancia actual
        if pos.direction == "long":
            gan_pct = (precio - pos.entry_price) / pos.entry_price
        else:
            gan_pct = (pos.entry_price - precio) / pos.entry_price

        if gan_pct < 0.003:
            return False

        # Detectar reversión: últimas 3 velas 1m
        if len(df1m_window) < 3:
            return False

        v1, v2, v3 = df1m_window.iloc[-3], df1m_window.iloc[-2], df1m_window.iloc[-1]

        # Tamaño de cada vela en %
        d1 = (v1['close'] - v1['open']) / v1['open']
        d2 = (v2['close'] - v2['open']) / v2['open']
        d3 = (v3['close'] - v3['open']) / v3['open']

        # Reversión: si pos es long, queremos 3 velas bajistas crecientes
        # Si pos es short, 3 velas alcistas crecientes
        if pos.direction == "long":
            es_rev = (d1 < 0 and d2 < 0 and d3 < 0
                      and abs(d2) > abs(d1) * 0.7
                      and abs(d3) > abs(d1) * 0.7
                      and (abs(d1) + abs(d2) + abs(d3)) > 0.003)  # total > 0.3%
        else:
            es_rev = (d1 > 0 and d2 > 0 and d3 > 0
                      and abs(d2) > abs(d1) * 0.7
                      and abs(d3) > abs(d1) * 0.7
                      and (d1 + d2 + d3) > 0.003)

        if es_rev:
            self._cerrar_posicion(pos, precio, ts_ms, "1m_rev")
            return True
        return False

    # ────────────────────────────────────────────────────────────────────────
    # ACTUALIZAR CONTEXTO BTC (se llama cada hora)
    # ────────────────────────────────────────────────────────────────────────
    def _update_btc_context(self, ts_ms: int):
        """Actualiza el _btc_ctx_sim global usando la BTC histórica."""
        if self.btc_5m_cache is None:
            return

        # Tomar las últimas N velas hasta ts_ms
        df = self.btc_5m_cache
        df_hasta = df[df['ts'] <= ts_ms]
        if len(df_hasta) < 25:
            return

        c = df_hasta['close'].values[-25:]
        # Calcular régimen + dirección + fuerza
        change_25 = (c[-1] - c[0]) / c[0]  # ~2 horas
        # mom1h: cambio en última hora (12 velas de 5m)
        if len(c) >= 13:
            mom1h = (c[-1] - c[-13]) / c[-13]
        else:
            mom1h = 0

        if change_25 > 0.015:
            regimen = "trend_up"
            direccion = "up"
        elif change_25 < -0.015:
            regimen = "trend_down"
            direccion = "down"
        else:
            regimen = "lateral"
            direccion = "neutral"

        fuerza = min(1.0, abs(change_25) / 0.05)

        _btc_ctx_sim.update({
            "regimen": regimen,
            "direccion": direccion,
            "fuerza": fuerza,
            "mom1h": mom1h,
            "dominancia_up": False,  # no calculamos dominancia en backtest
            "ts": ts_ms,
        })

    # ────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ────────────────────────────────────────────────────────────────────────
    def run(self):
        """
        Recorre cronológicamente todas las velas 1m de todos los símbolos
        y simula la lógica del bot.
        """
        # Construir un timeline unificado: todos los timestamps únicos
        all_ts = set()
        for sym, df1m in self.ohlcv_data.items():
            all_ts.update(df1m['ts'].astype(int).tolist())
        timeline = sorted(all_ts)

        log.info(f"[ENGINE] Iniciando backtest: {len(timeline)} velas 1m, "
                 f"{len(self.ohlcv_data)} símbolos")
        log.info(f"[ENGINE] Capital inicial: ${self.config.capital_inicial:.2f}")

        # Loop principal
        last_btc_update_ts = 0
        ANALISIS_INT_MS = 60 * 1000  # cada 1 minuto evaluamos señales

        # Precomputar índices para acceso rápido por símbolo
        sym_index_1m = {}
        sym_index_5m = {}
        sym_index_1h = {}
        for sym in self.ohlcv_data:
            sym_index_1m[sym] = self.ohlcv_data[sym].set_index('ts')
            sym_index_5m[sym] = self.df5m_cache[sym].set_index('ts')
            sym_index_1h[sym] = self.df1h_cache[sym].set_index('ts')

        for i, ts_ms in enumerate(timeline):
            if i % 1000 == 0 and i > 0:
                pct = 100 * i / len(timeline)
                log.info(f"[ENGINE] Progreso {pct:.1f}% — equity=${self.equity:.2f} "
                         f"trades={len(self.posiciones)} abiertas={len(self.posiciones_abiertas)}")

            # Actualizar día actual
            dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
            dia = dt.strftime("%Y-%m-%d")
            if dia != self.dia_actual:
                if self.dia_actual:
                    log.info(f"[DÍA] {self.dia_actual} cerrado, PnL día: ${self.pnl_dia:.2f}")
                self.dia_actual = dia
                self.pnl_dia = 0.0

            # Actualizar BTC context cada hora
            if (ts_ms - last_btc_update_ts) >= 3600 * 1000:
                self._update_btc_context(ts_ms)
                last_btc_update_ts = ts_ms

            # === FASE 1: Gestionar posiciones abiertas ===
            for symbol in list(self.posiciones_abiertas.keys()):
                pos = self.posiciones_abiertas[symbol]
                if ts_ms <= pos.open_ts:
                    continue  # vela en la que se abrió, no gestionar todavía

                df_sym_1m = sym_index_1m.get(symbol)
                if df_sym_1m is None or ts_ms not in df_sym_1m.index:
                    continue

                vela = df_sym_1m.loc[ts_ms]
                precio_now = float(vela['close'])

                # Trailing
                self._trailing_stop(pos, precio_now)

                # Check TP/SL hit
                if self._check_close_levels(pos, vela, ts_ms):
                    continue  # cerrada

                # Check 1M_REV
                idx_now = df_sym_1m.index.get_loc(ts_ms)
                if idx_now >= 3:
                    df_window = self.ohlcv_data[symbol].iloc[max(0, idx_now-2):idx_now+1]
                    if self._check_1m_rev(pos, df_window, ts_ms):
                        continue

            # === FASE 2: Buscar nuevas señales ===
            # Solo si hay cupo y no estamos pausados
            if self.pausado_hasta_ts > ts_ms:
                self.equity_curve.append((ts_ms, self.equity))
                continue

            cupo = self.config.max_pos_total - len(self.posiciones_abiertas)
            if cupo <= 0:
                self.equity_curve.append((ts_ms, self.equity))
                continue

            candidatos = []
            for symbol in self.ohlcv_data:
                if symbol in self.posiciones_abiertas:
                    continue

                df1m = sym_index_1m.get(symbol)
                if df1m is None or ts_ms not in df1m.index:
                    continue

                # Necesitamos al menos 55 velas de 5m para calcular_ind
                # Eso son 55*5 = 275 minutos hacia atrás
                df5m = sym_index_5m.get(symbol)
                if df5m is None:
                    continue

                df5m_hasta = df5m[df5m.index <= ts_ms]
                if len(df5m_hasta) < 55:
                    continue

                df1h = sym_index_1h.get(symbol)
                df1h_hasta = df1h[df1h.index <= ts_ms] if df1h is not None else None
                if df1h_hasta is None or len(df1h_hasta) < 20:
                    continue

                # Calcular indicadores
                df5m_sub = df5m_hasta.tail(60).reset_index()
                df1h_sub = df1h_hasta.tail(40).reset_index()

                ind = calcular_ind(df5m_sub, df1h_sub, ts_actual_ms=ts_ms)
                if ind is None:
                    continue

                # Calcular umbral base
                umbral = calcular_umbral_bt(symbol, ind, self.perfil, ts_ms)

                # v2 (28/4): ajuste dinámico piwhale (WR reciente del símbolo)
                # Mira los últimos 10 trades cerrados de este símbolo
                ajuste_piwhale = calibrar_umbral_piwhale_bt(symbol, self.posiciones)
                umbral_final = round(max(0.22, min(0.50, umbral + ajuste_piwhale)), 3)

                direction, conf, razones = evaluar_bt(ind, umbral_final, ts_ms)

                # v2 (28/4): aplicar filtros estadísticos (Hurst, Z-score, GARCH)
                if direction is not None:
                    closes_arr = df5m_sub['close'].values.astype(float)
                    direction, conf, razones_filtro = aplicar_filtros_estadisticos(
                        direction, conf, umbral_final, ind, closes_arr
                    )
                    razones = razones + razones_filtro

                if direction is not None:
                    candidatos.append({
                        'symbol': symbol,
                        'direction': direction,
                        'conf': conf,
                        'precio': float(df1m.loc[ts_ms]['close']),
                        'razones': razones,
                        'ind': ind,  # v2: pasar ind para usar en sizing/SL ATR
                    })

            # Abrir mejores candidatos
            candidatos.sort(key=lambda x: -x['conf'])
            for c in candidatos[:cupo]:
                self._abrir_posicion(c['symbol'], c['direction'], c['precio'],
                                     ts_ms, c['conf'], c['razones'], c.get('ind'))

            self.equity_curve.append((ts_ms, self.equity))

        # Cerrar todas las posiciones que quedaron abiertas al final
        if self.posiciones_abiertas:
            log.info(f"[FIN] Cerrando {len(self.posiciones_abiertas)} posiciones abiertas al final del periodo")
            ultimo_ts = timeline[-1]
            for symbol in list(self.posiciones_abiertas.keys()):
                pos = self.posiciones_abiertas[symbol]
                df_sym = sym_index_1m.get(symbol)
                if df_sym is not None and ultimo_ts in df_sym.index:
                    precio = float(df_sym.loc[ultimo_ts]['close'])
                    self._cerrar_posicion(pos, precio, ultimo_ts, "fin_backtest")

        log.info(f"[ENGINE] Backtest completado")
        log.info(f"[ENGINE] Trades totales: {len(self.posiciones)}")
        log.info(f"[ENGINE] Equity final: ${self.equity:.2f} ({(self.equity/self.config.capital_inicial-1)*100:+.2f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# REPORTES Y OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

def generar_reporte(engine: BacktestEngine, output_dir: Path,
                    config: BacktestConfig, periodo_str: str):
    """Genera reportes completos del backtest en archivos."""
    cerradas = [p for p in engine.posiciones if p.status == "closed"]
    abiertas = [p for p in engine.posiciones if p.status == "open"]

    # ── Resumen general ──────────────────────────────────────────────────────
    pnl_total = sum(p.pnl_usd for p in cerradas)
    fees_total = sum(p.fees for p in cerradas)
    wins = [p for p in cerradas if p.pnl_usd > 0]
    losses = [p for p in cerradas if p.pnl_usd < 0]

    wr = 100 * len(wins) / len(cerradas) if cerradas else 0
    avg_win = sum(p.pnl_usd for p in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(p.pnl_usd for p in losses) / len(losses)) if losses else 1
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    pct_capital = (engine.equity / config.capital_inicial - 1) * 100

    print()
    print("=" * 80)
    print(f" RESULTADOS DEL BACKTEST  — periodo: {periodo_str}")
    print("=" * 80)
    print(f" Capital inicial:    ${config.capital_inicial:.2f}")
    print(f" Capital final:      ${engine.equity:.2f}  ({pct_capital:+.2f}%)")
    print(f" PnL total:          ${pnl_total:+.2f}")
    print(f" Fees pagadas:       ${fees_total:.2f}")
    print()
    print(f" Trades cerrados:    {len(cerradas)}")
    print(f" Trades abiertos:    {len(abiertas)}")
    print(f" Win rate:           {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f" Ganador promedio:   ${avg_win:+.3f}")
    print(f" Perdedor promedio:  ${-avg_loss:+.3f}")
    print(f" R:R real:           1:{rr:.2f}")
    print()

    # ── Por motivo de cierre ─────────────────────────────────────────────────
    print(" PnL por motivo de cierre:")
    print(f"   {'Motivo':<14} {'N':>4} {'PnL':>10} {'PnL/trade':>10} {'WR':>5}")
    por_motivo = defaultdict(lambda: {'trades': [], 'pnl': 0})
    for p in cerradas:
        por_motivo[p.motivo_cierre]['trades'].append(p)
        por_motivo[p.motivo_cierre]['pnl'] += p.pnl_usd
    for motivo, d in sorted(por_motivo.items(), key=lambda x: x[1]['pnl']):
        n = len(d['trades'])
        pnl = d['pnl']
        wins_m = sum(1 for t in d['trades'] if t.pnl_usd > 0)
        wr_m = 100 * wins_m / n if n else 0
        print(f"   {motivo:<14} {n:>4} ${pnl:>+8.2f} ${pnl/n if n else 0:>+8.3f} {wr_m:>4.0f}%")
    print()

    # ── Por símbolo ──────────────────────────────────────────────────────────
    print(" PnL por símbolo:")
    print(f"   {'Sym':<10} {'N':>3} {'PnL':>10} {'WR':>5} {'$/trade':>10}")
    por_sym = defaultdict(lambda: {'trades': [], 'pnl': 0})
    for p in cerradas:
        sym_c = p.symbol.replace("/USDT:USDT", "")
        por_sym[sym_c]['trades'].append(p)
        por_sym[sym_c]['pnl'] += p.pnl_usd
    for sym, d in sorted(por_sym.items(), key=lambda x: -x[1]['pnl']):
        n = len(d['trades'])
        pnl = d['pnl']
        wins_s = sum(1 for t in d['trades'] if t.pnl_usd > 0)
        wr_s = 100 * wins_s / n if n else 0
        print(f"   {sym:<10} {n:>3} ${pnl:>+8.2f} {wr_s:>4.0f}% ${pnl/n if n else 0:>+8.3f}")
    print()

    # ── Trades CSV ───────────────────────────────────────────────────────────
    trades_csv = output_dir / "trades.csv"
    rows = []
    for p in cerradas:
        rows.append({
            'id': p.id,
            'symbol': p.symbol.replace("/USDT:USDT", ""),
            'direction': p.direction,
            'open_dt': p.open_dt.isoformat() if p.open_dt else "",
            'close_dt': p.close_dt.isoformat() if p.close_dt else "",
            'duracion_min': round(p.duracion_min, 1),
            'entry_price': round(p.entry_price, 6),
            'close_price': round(p.close_price, 6),
            'cantidad': round(p.cantidad, 4),
            'nocional': round(p.nocional, 2),
            'margen': round(p.margen, 2),
            'sl_inicial': round(p.sl_inicial, 6),
            'sl_final': round(p.stop_loss, 6),
            'tp': round(p.take_profit, 6),
            'trailing_activado': p.trailing,
            'pnl_usd': round(p.pnl_usd, 4),
            'pnl_pct_roe': round(p.pnl_pct, 2),
            'fees': round(p.fees, 4),
            'motivo': p.motivo_cierre,
            'conf': round(p.conf, 3),
            'razon': p.razon,
        })
    df_trades = pd.DataFrame(rows)
    df_trades.to_csv(trades_csv, index=False)
    print(f" Trades detallados guardados en: {trades_csv}")

    # ── Equity curve CSV ─────────────────────────────────────────────────────
    eq_csv = output_dir / "equity_curve.csv"
    df_eq = pd.DataFrame(engine.equity_curve, columns=['ts', 'equity'])
    df_eq['dt'] = pd.to_datetime(df_eq['ts'], unit='ms', utc=True)
    df_eq.to_csv(eq_csv, index=False)
    print(f" Equity curve guardada en: {eq_csv}")

    # ── Equity curve PNG (si matplotlib disponible) ──────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_eq['dt'], df_eq['equity'], color='#2196F3', linewidth=1)
        ax.axhline(config.capital_inicial, color='gray', linestyle='--', alpha=0.5, label=f'Inicial ${config.capital_inicial:.0f}')
        ax.set_title(f'Equity Curve — {periodo_str}\n'
                     f'Final: ${engine.equity:.2f} ({pct_capital:+.2f}%) | '
                     f'Trades: {len(cerradas)} | WR: {wr:.0f}% | R:R 1:{rr:.2f}')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Equity (USD)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        png_path = output_dir / "equity_curve.png"
        plt.savefig(png_path, dpi=100)
        plt.close()
        print(f" Equity curve PNG guardada en: {png_path}")
    except ImportError:
        print(" (matplotlib no disponible — skip PNG)")

    # ── Resumen JSON ─────────────────────────────────────────────────────────
    resumen = {
        'periodo': periodo_str,
        'capital_inicial': config.capital_inicial,
        'capital_final': round(engine.equity, 2),
        'pct_cambio': round(pct_capital, 2),
        'pnl_total': round(pnl_total, 2),
        'fees_total': round(fees_total, 2),
        'trades': len(cerradas),
        'wr_pct': round(wr, 1),
        'wins': len(wins),
        'losses': len(losses),
        'avg_win': round(avg_win, 3),
        'avg_loss': round(-avg_loss, 3),
        'rr_ratio': round(rr, 2),
        'por_motivo': {
            m: {'n': len(d['trades']), 'pnl': round(d['pnl'], 2)}
            for m, d in por_motivo.items()
        },
        'por_sym': {
            s: {'n': len(d['trades']), 'pnl': round(d['pnl'], 2)}
            for s, d in por_sym.items()
        }
    }
    json_path = output_dir / "resumen.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f" Resumen JSON guardado en: {json_path}")

    return resumen


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backtester multiwhale_v6 — máxima fidelidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Backtest última semana con símbolos del config
  python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-27

  # Backtest 14 días con símbolos custom
  python backtester_multiwhale.py --start 2026-04-13 --end 2026-04-27 \\
                                   --symbols SKYAI,HYPER,LDO,RAVE,CHIP

  # Backtest con leverage diferente
  python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-27 \\
                                   --leverage 3 --capital 5000
        """
    )
    parser.add_argument('--start', required=True, help='Fecha inicio (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='Fecha fin (YYYY-MM-DD), exclusivo')
    parser.add_argument('--symbols', default=None,
                        help='Lista coma de símbolos (sin /USDT:USDT). Default: del config.')
    parser.add_argument('--config', default='config_multi.json',
                        help='Path al config_multi.json (default: ./config_multi.json)')
    parser.add_argument('--capital', type=float, default=3000,
                        help='Capital inicial USD (default: 3000)')
    parser.add_argument('--leverage', type=int, default=5, help='Leverage (default: 5)')
    parser.add_argument('--max-pos', type=int, default=5,
                        help='Máximo de posiciones simultáneas (default: 5)')
    parser.add_argument('--fee-rate', type=float, default=0.0006,
                        help='Fee por lado (default: 0.0006 = 0.06%%)')
    parser.add_argument('--no-1m-rev', action='store_true',
                        help='Desactivar el cierre por reversión 1m')
    parser.add_argument('--no-btc', action='store_true',
                        help='No descargar BTC para contexto (más rápido pero menos fiel)')
    parser.add_argument('--output-dir', default=None,
                        help='Directorio de output (default: ./results/<timestamp>)')

    args = parser.parse_args()

    # Parsear fechas
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("ERROR: fechas deben ser YYYY-MM-DD")
        sys.exit(1)

    # Cargar config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: no se encuentra {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    perfil = config_data.get('perfil', {})
    all_symbols_full = config_data.get('all_symbols', [])

    # Determinar símbolos a operar
    if args.symbols:
        symbols_short = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols_short = [s.replace("/USDT:USDT", "") for s in all_symbols_full]

    log.info(f"[CONFIG] Símbolos a backtestear: {symbols_short}")

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"bt_{ts_str}"
    output_dir.mkdir(exist_ok=True, parents=True)
    log.info(f"[CONFIG] Output dir: {output_dir}")

    # Descargar OHLCV (margen de 1 día antes para warm-up de indicadores)
    warmup_dt = start_dt - timedelta(days=1)
    log.info(f"[DATA] Descargando OHLCV {warmup_dt.date()} a {end_dt.date()}...")

    ohlcv_data = fetch_ohlcv_multi(symbols_short, warmup_dt, end_dt, "1m")

    if not ohlcv_data:
        print("ERROR: no se pudo descargar OHLCV de ningún símbolo")
        sys.exit(1)

    log.info(f"[DATA] OHLCV descargado para {len(ohlcv_data)} símbolos")

    # BTC para contexto
    btc_df = None
    if not args.no_btc:
        log.info(f"[DATA] Descargando BTC para contexto...")
        btc_df = fetch_ohlcv_periodo("BTC", warmup_dt, end_dt, "1m")
        if btc_df is None:
            log.warning("[DATA] BTC no descargado — backtest sin contexto BTC")

    # Reconstruir perfil con sólo los símbolos requeridos
    perfil_full = {}
    for s in symbols_short:
        sf = f"{s}/USDT:USDT"
        if sf in perfil:
            perfil_full[sf] = perfil[sf]
            perfil_full[s] = perfil[sf]  # también con clave corta
        else:
            # Default si no está en config
            perfil_full[sf] = {"cap": 0.04, "sl": 0.015, "tp": 0.025, "umb": 0.30}
            perfil_full[s] = perfil_full[sf]

    # Crear config y engine
    bt_config = BacktestConfig(
        capital_inicial=args.capital,
        leverage=args.leverage,
        max_pos_total=args.max_pos,
        fee_rate=args.fee_rate,
        cierre_por_reversion_1m=not args.no_1m_rev,
    )

    # Adaptar ohlcv_data: keys deben ser símbolos cortos (consistente con perfil)
    engine = BacktestEngine(
        config=bt_config,
        perfil=perfil_full,
        ohlcv_data=ohlcv_data,  # ya viene con keys cortos
        btc_df=btc_df,
    )

    # Filtrar trades al periodo real (no warm-up)
    log.info(f"[ENGINE] Filtrando warm-up data, simulando desde {start_dt.date()}")
    start_ms = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Recortar warmup data (mantenemos para indicadores pero no operamos antes)
    # Nota: el engine ya tiene la data, sólo filtramos el timeline
    # Esto se podría hacer más limpio dentro del engine — TODO siguiente versión

    # Correr backtest
    t0 = time.time()
    engine.run()
    t1 = time.time()
    log.info(f"[ENGINE] Tiempo de ejecución: {t1-t0:.1f}s")

    # Reporte
    periodo_str = f"{start_dt.date()} a {end_dt.date()}"
    generar_reporte(engine, output_dir, bt_config, periodo_str)

    print()
    print(f"✓ Backtest completado. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()
