# MarketFusion V1.0A — XAUUSD asset class

## Purpose and safety boundary

Gold is a separate `PRECIOUS_METAL` asset, not an alias for a Forex pair. The
V1.0A runtime is observational only: automatic execution is disabled, manual
confirmation is required, and no code path calls `order_send` or modifies an
MT5 order or position.

Selecting XAUUSD pauses active EURUSD collection only. It does not delete,
move, retrain, overwrite, or invalidate the existing EURUSD data, features,
models, registry, reports, research, or V0.8 observations. The staged migration
keeps all legacy EURUSD paths readable while new assets use symbol-aware paths.

## Broker discovery and contract truth

`src.assets.contracts.discover_broker_symbol` searches the connected MT5 symbol
catalogue. It accepts gold only when the broker reports XAU as the base currency
and USD as the profit currency; this prevents an equity or ETF called `GOLD`
from being mistaken for spot gold.

Live discovery on 2026-08-31 returned:

| Field | Broker value |
|---|---:|
| Symbol | XAUUSD |
| Description | Gold vs US Dollar |
| Digits | 2 |
| Point | 0.01 |
| Tick size | 0.01 |
| Tick value | 0.10 USD |
| Contract size | 100 oz |
| Volume min / max / step | 0.01 / 100 / 0.01 |
| Trade mode | 4 |
| Base / profit / margin | XAU / USD / USD |

The broker spread is dynamic; it must be read from each quote or completed bar,
not frozen from this discovery snapshot.

## Price, point, tick, spread, and volatility units

`BrokerSymbolSpec` converts raw price moves to broker points or ticks using the
actual MT5 `point` and `trade_tick_size`. It also supports relative-price and
ATR-normalized movement. The gold UI says `points`, never Forex `pips`.

Gold spread diagnostics retain:

- absolute price spread;
- broker points;
- spread divided by mid-price;
- spread divided by causal ATR;
- historical M5 spread percentiles.

No EURUSD spread or volatility threshold is reused. Gold risk remains an
observational profile based on spread percentile, ATR percentile, gap/outlier
state, and session. Event risk remains unavailable until verified external
intelligence is added.

## Symbol-aware paths and compatibility

New paths are:

```text
data/market/XAUUSD/bars/
data/market/XAUUSD/quotes/
data/features/XAUUSD/
data/evaluation/XAUUSD/
data/runtime/XAUUSD/
models/XAUUSD/
reports/XAUUSD/
```

EURUSD continues to read its legacy V0.5–V0.9 locations. No destructive data
migration is performed. The API and dashboard accept `symbol=EURUSD|XAUUSD`.

## Completed bars and data quality

The V1.0A collector is read-only and admits a candle only after its scheduled
close plus the stabilization guard. M1, M5, M15, H1, H4, and D1 contracts are
defined. The live broker returned M1 through H1; H4 and D1 requests blocked in
this terminal session and were reported unavailable rather than fabricated.
Partial M1 candles exist only on the live dashboard hot path and are explicitly
marked `DASHBOARD_ONLY_NOT_CAUSAL`.

All timestamps are normalized to UTC. Audits report first/last timestamp, row
count, duplicates, scheduled missing periods, weekend gaps, and broker breaks.

## Targets and research

The targets use exact future completed M5 closes at 15, 60, and 240 minutes.
If the exact timestamp is absent across a broker break, the target remains null.
The neutral band is the maximum of:

```text
1.5 × observed spread cost / decision close
0.10 × causal ATR(14) / decision close
```

Two narrower/wider candidates are reported before selection. The selected band
is justified by transaction-cost realism, volatility meaning, class balance,
and cross-horizon stability—not by model accuracy. Majority, class-prior, and
previous-direction baselines are recorded before any ML research.

## Model, evaluation, and observation isolation

`models/XAUUSD/registry.json` starts empty. XAUUSD 15m, 60m, and 240m therefore
start as `NO_APPROVED_MODEL`, with null probabilities. Asset registry validation
rejects a record whose `asset_id` or `symbol` does not match the selected asset.
Model IDs include family, asset, horizon, timestamp, and digest.

V0.8 observation identity already includes symbol, decision timestamp, horizon,
and model ID. V1.0A routes XAUUSD predictions/outcomes to its own evaluation
directory. Pattern and engine evidence is loaded from the selected asset’s bar
and feature roots and is never pooled with EURUSD.

No XAUUSD model is promoted in V1.0A. Future candidates must use strict
chronological walk-forward train/validation/test folds, train-only preprocessing,
calibration and threshold tuning, and the normal unchanged promotion standards.

## Startup and dashboard

Start the local gold runtime with:

```powershell
.\scripts\start_marketfusion_full.ps1 -Symbol XAUUSD -NoBrowser
```

The launcher runs the read-only gold audit, starts a bounded incremental
completed-bar collector, starts the XAUUSD quote hot path and fail-closed state
worker, and launches the asset-isolated V0.8 recorder, localhost API, and
dashboard. It does not start the EURUSD collector or XAUUSD training.

The selector offers EUR/USD and XAU/USD. Every state, quote, candle, engine,
research, model, and shadow request carries the selected asset. XAUUSD cards
show `NO APPROVED MODEL` and `N/A`; they cannot fall back to the EURUSD champion.

Quote freshness and completed-feature freshness are separate. A live quote does
not make an incomplete M5 feature row causal, and a waiting feature row does not
label a flowing quote feed stale.
