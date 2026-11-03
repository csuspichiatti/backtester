# Backtester MultiWhale v2.0 — Guía de uso honesta

Backtester del `multiwhale_v6.py` con **fidelidad limitada**. Leé el disclaimer
antes de usarlo.

---

## DISCLAIMER (importante leer)

### Validación realizada

Comparación contra trades reales del bot del 21-27 abril (472 trades, -$226):

| Métrica | Real | BT v2 | Brecha |
|---|---|---|---|
| Trades totales | 472 | 924 | +96% |
| WR | 49% | 47% | OK |
| PnL semana | -$226 | -$831 | -3.7x |
| TP alcanzados | 8 (1.7%) | 13 (1.4%) | OK |
| SL/EXT_SL del total | 42% | 98% | +56pp |

### Limitación principal

**El bot real cierra el 42% de los trades por SL. El backtest cierra el 98%.**

Esto es porque el bot real tiene mecanismos que el backtest **NO reproduce**:

1. **`ext_manual` (21% de los trades reales)**: cierres externos por vos o por
   liberación de margen. Imposible de simular sin un modelo de tu comportamiento.

2. **`1m_rev` (8.7% reales vs 0.4% en BT)**: cierre prematuro por reversión 1m.
   Mi detección es muy estricta vs la del bot real.

3. **Re-evaluación cada 30s**: el bot real cambia su evaluación cada 30 segundos
   y puede cerrar trades **antes** de que toquen el SL. El backtest deja correr
   los SL hasta el final.

### Qué significa esto para vos

**El BT NO es confiable para**:
- Predecir el PnL absoluto (¿gana o pierde plata?)
- Decidir si activar/desactivar el bot completo
- Saber el equity final esperado

**El BT SÍ es útil para**:
- Comparar escenarios relativamente ("¿qué cambio mejora más?")
- Identificar símbolos sistemáticamente problemáticos
- Ver el impacto direccional de cambios de parámetros

**Regla de uso**: si el BT dice "el cambio X mejora $50", esperá entre $10-100
de mejora real. **Nunca confíes en el número absoluto, sí en la dirección y
magnitud aproximada**.

### Validación recomendada

Para cualquier conclusión del BT, validá con A/B testing en vivo:
1. Aplicá el cambio en el bot real durante 2-3 días
2. Mide el impacto real
3. Si confirma, dejá el cambio
4. Si no, revertí

---

## Setup (Windows)

```powershell
pip install pandas numpy requests matplotlib

# Carpeta del backtester
mkdir "C:\Users\csusp\OneDrive\Desktop\Code todos\Backtester"
cd "C:\Users\csusp\OneDrive\Desktop\Code todos\Backtester"

# Copiar archivos
# - backtester_multiwhale.py (este archivo)
# - config_multi.json (desde tu carpeta del multi)
```

---

## Uso típico

### Caso 1: ¿Qué pasa si saco símbolos perdedores?

```powershell
# Baseline: todo el universo
python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-28 `
    --output-dir results/baseline

# Solo símbolos rentables (de tu análisis previo)
python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-28 `
    --symbols SKYAI,HYPER,LDO,RAVE,CHIP,DOGE,SHIB,ORCA,GRIFFAIN,TRUMP `
    --output-dir results/solo_rentables
```

Comparar `pct_cambio` en los 2 `resumen.json`.
Si el "solo rentables" es mejor por **>5%**, vale la pena aplicarlo.

### Caso 2: ¿1M_REV ayuda o lastima?

```powershell
# Con 1M_REV
python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-28 `
    --output-dir results/con_1mrev

# Sin 1M_REV
python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-28 `
    --no-1m-rev --output-dir results/sin_1mrev
```

OJO: en mi BT 1M_REV apenas dispara, así que esta comparación va a ser poco
informativa. Mejor ver con A/B real.

### Caso 3: ¿Más leverage gana más?

```powershell
python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-28 `
    --leverage 5 --output-dir results/lev5

python backtester_multiwhale.py --start 2026-04-21 --end 2026-04-28 `
    --leverage 3 --output-dir results/lev3
```

Comparar % cambio. Más leverage típicamente amplifica tanto ganancias como pérdidas.

---

## Argumentos disponibles

```
--start YYYY-MM-DD          Fecha inicio
--end YYYY-MM-DD            Fecha fin (exclusivo)
--symbols S1,S2,S3          Lista de simbolos (default: del config)
--config path               Path al config_multi.json
--capital N                 Capital inicial USD (default: 3000)
--leverage N                Leverage (default: 5)
--max-pos N                 Max posiciones simultaneas (default: 5)
--fee-rate F                Fee por lado (default: 0.0006)
--no-1m-rev                 Desactivar cierre por reversion 1m
--no-btc                    No descargar BTC para contexto
--output-dir path           Directorio de output
```

---

## Output

Cada corrida genera `results/bt_<timestamp>/`:

- `trades.csv` — todos los trades
- `equity_curve.csv` — equity por vela
- `equity_curve.png` — grafico de equity
- `resumen.json` — resumen estructurado

---

## Lo que el BT v2 hace BIEN

- ✅ Reproducir indicadores (RSI Wilder, EMA, ATR, BB, MACD) idénticos al multi v6
- ✅ Sistema de scoring sl/ss del evaluar() exacto
- ✅ Umbral dinámico con los 9 ajustes
- ✅ Filtros estadísticos (Hurst, Z-score, GARCH)
- ✅ Trailing stop v4.5 (BE+colchón → trail 0.6%)
- ✅ SL/TP estructural ATR-based
- ✅ Sizing dinámico por ATR ratio y conf
- ✅ Direccionalidad correcta (longs en uptrend, shorts en downtrend)

## Lo que el BT v2 NO hace

- ❌ Re-evaluación de señal cada 30s (sólo cuando abre, no durante el trade)
- ❌ Cierres manuales tuyos (`ext_manual`, `manual`) — son ~25% de los trades reales
- ❌ Detección de reversión 1m al nivel del bot real (4 trades vs 41 reales)
- ❌ Funding rate histórico (asume 0)
- ❌ GT (game theory) completo (p_victoria aproximada)
- ❌ Macro F&G y Farside ETF
- ❌ Slippage real de orderbook (estimación fija 0.05%)

---

## Decisión sobre las 3 mejoras propuestas

Dado que el backtester tiene 50% de fidelidad para PnL absoluto pero ~80% para
comparaciones relativas, mi recomendación es:

### 1. ✅ Sacar símbolos no rentables (riesgo: BAJO)

**Acción**: editar `config_multi.json` y eliminar de `all_symbols` los símbolos
que perdieron consistentemente:

```
1INCH, APT, AXS, BASED, BIO, BSB, CAKE, FLOKI, GUN, KAT, M, MANA, MASK,
MOVR, PENGU, PEPE, PIEVERSE, PIPPIN, SAND, SPK, SUI, SUSHI, UAI, UB, WIF, ZBT
```

**Impacto esperado** (basado en análisis estadístico real, no BT):
- Eliminar pérdidas estructurales (~$50/semana evitadas)
- Riesgo: perderse algún símbolo que repunta en el futuro
- Mitigación: que el scout los re-detecte si vuelven a ser rentables

### 2. 🟡 Eliminar 1M_REV (riesgo: MEDIO)

**Acción**: parametrizar el bot para desactivar `cerrar_por_reversion_1m`.

**Impacto esperado** (datos reales):
- En la última semana, 1M_REV cerró 41 trades con +$21 (positivo)
- Si los hubiéramos dejado correr al TP completo: +$118 estimado
- **PERO** algunos hubieran ido al SL → mejora real probable: +$50

**Riesgo**: si los precios revierten realmente, perdés ganancias chicas.

**Validación recomendada**: A/B en vivo durante 1 semana antes de dejarlo permanente.

### 3. 🔴 Modificar trailing para reducir EXT_SL (riesgo: ALTO)

**Acción**: ajustar el trailing en `multiwhale_v6.py`:
- BE más conservador: BE_TRIGGER 0.5% → 0.8%
- Trailing más amplio: TRAIL_DIST 0.6% → 1.5%

**Impacto esperado**: reducir el daño de EXT_SL (-$186/semana actual).

**Riesgo**: con trailing más amplio, los pocos ganadores pueden ceder más
ganancia antes de cerrar. Necesita prueba en vivo.

---

## Resumen ejecutivo

Tras todo este análisis, **lo más fuerte que aprendimos**:

1. **El multi pierde -$226/semana estructuralmente** por R:R inverso
2. **Las fees no son la causa principal** (eras vos quien tenía razón)
3. **El asesino es EXT_SL** (-$186/semana, 82% de la pérdida)
4. **Los TPs configurados son irreales** (1.7% de trades los alcanzan)
5. **Hay 14 símbolos rentables** y 28 que pierden plata
6. **Reproducir el bot al 100% no es viable** sin re-construirlo

**Camino más sensato**: dejar el multi apagado por ahora, dejar al SwingWhale
correr para acumular data propia, y volver al multi si queremos mejorarlo en el
futuro con una versión limpia desde cero (no v7) que evite los problemas
estructurales identificados.
