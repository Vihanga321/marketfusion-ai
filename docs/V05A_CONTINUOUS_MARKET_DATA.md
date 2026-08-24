# MarketFusion AI V0.5A — Continuous Market Data + Historical Update Engine

## Purpose

V0.5A turns the local MT5 terminal into a continuously growing, read-only EURUSD market-history source. It captures current market activity, stores only completed bars, preserves first-observed history, builds causal multi-timeframe features, and creates future-return labels only after their prediction horizons have actually elapsed.

V0.5A is **not** a model-retraining stage and is **not** a trading-execution stage.

## Live inputs

- EURUSD current bid/ask quote sampled by the local Python service.
- Completed MT5 bars for M1, M5, M15, and H1.
- Per-bar tick volume, real volume, and MT5 spread points.
- UTC timestamps from MT5 bar data and local UTC capture time.

The service uses the already logged-in MT5 desktop terminal through the local `MetaTrader5` Python package. No broker password is stored in MarketFusion.

## Storage

Runtime data is local and ignored by Git:

- `data/mt5/continuous/EURUSD_M1.parquet`
- `data/mt5/continuous/EURUSD_M5.parquet`
- `data/mt5/continuous/EURUSD_M15.parquet`
- `data/mt5/continuous/EURUSD_H1.parquet`
- `data/mt5/continuous/quotes/EURUSD_YYYY-MM-DD.parquet`
- `data/mt5/continuous/conflicts/*.csv`
- `data/mt5/continuous/runtime_status.json`
- `data/processed/v05a_eurusd_continuous_features.parquet`

Each completed bar stores `bar_open_utc`, `bar_close_utc`, OHLC, tick volume, spread points, real volume, `first_observed_utc`, and source.

## Completed-bar rule

An MT5 rate timestamp is treated as the bar-open timestamp. A bar is not admitted until:

`bar_close_utc <= capture_time_utc - stabilization_delay`

The default stabilization delay is 5 seconds. Current/incomplete candles are excluded.

## Immutable history rule

Once a completed bar has been observed and stored, a later pull is not allowed to silently rewrite that bar. The collector compares overlapping pulls with stored data. If a previously observed bar changes, the original row remains untouched, the disagreement is written to the conflict quarantine directory, and V0.5A stops fail-closed for review.

This makes "what MarketFusion knew at the time" reproducible.

## Multi-timeframe causal feature engine

The decision grid is M5. Every decision timestamp equals the close time of a completed M5 bar. M1, M15, and H1 features are joined only from bars whose `available_from_utc` is less than or equal to the M5 decision timestamp.

Current V0.5A feature families:

- M1: 1m/5m returns, 15m realized volatility, spread, tick volume.
- M5: 5m/15m/60m/240m returns, 15m/60m/240m realized volatility, spread, 60m spread median/max, tick volume, 60m mean tick volume.
- M15: 15m/60m/240m returns, 60m/240m realized volatility.
- H1: 60m/240m returns, 240m/24h realized volatility.
- Time/session context: UTC hour, weekday, Asia session, London session, New York session, London/New York overlap. London and New York session flags use timezone-aware DST conversion.

Outcome columns are physically excluded from the model feature registry.

## Label maturation

The three target horizons are 15 minutes, 60 minutes, and 240 minutes.

For a decision at time `T`, a label is created only when there is an exact future M5 close at `T + horizon`. No interpolation, forward fill, nearest-neighbor timestamp, or incomplete future bar is permitted.

Example:

- Decision at 10:00 UTC.
- 15m outcome becomes available only when the exact 10:15 UTC M5 close exists.
- 60m outcome becomes available only when the exact 11:00 UTC close exists.
- 240m outcome becomes available only when the exact 14:00 UTC close exists.

Until then the corresponding outcome fields remain null.

## Continuous-history behavior

While the V0.5A process is running:

1. It samples the latest EURUSD quote.
2. It re-reads an overlap window for each timeframe.
3. It appends newly completed bars only.
4. It checks for immutable-history conflicts.
5. When a new M5 bar closes, it rebuilds the causal feature/outcome dataset.
6. It runs the quality/leakage gate.
7. It writes `runtime_status.json`.
8. It repeats.

This means today's completed activity automatically becomes part of tomorrow's historical MarketFusion dataset without manual exporting.

## Fail-closed quality checks

V0.5A blocks on:

- duplicate bar timestamps;
- invalid/non-UTC times;
- wrong bar duration;
- bars observed before close;
- non-positive prices;
- OHLC enclosure violations;
- negative spread;
- immutable-bar conflicts;
- multi-timeframe feature availability after the decision timestamp;
- future/outcome fields in the feature registry;
- incorrect label horizon timestamps or label maturity times.

Large time gaps and stale bars are reported as warnings because weekends, holidays, and market closures can be legitimate.

## Local commands

Run the tests first:

```powershell
.\scripts\run_v05a_tests.ps1
```

With MT5 open and logged in, run one complete cycle:

```powershell
.\scripts\run_v05a_once.ps1
```

Then start continuous collection:

```powershell
.\scripts\start_v05a_continuous.ps1
```

The default poll interval is 15 seconds. Stop safely with `Ctrl+C`.

View the current state from another PowerShell terminal:

```powershell
.\scripts\show_v05a_status.ps1
```

## Training policy

V0.5A intentionally **does not retrain a model on every tick or bar**. New data is collected continuously and labels mature continuously, but model training/promotion belongs to a later controlled stage with purged walk-forward testing and champion/challenger promotion gates.

That separation prevents a live model from learning from incomplete outcomes, future information, or one abnormal trading day and immediately changing production behavior.

## Safety boundary

V0.5A is read-only. It reads symbol information, current quotes, and historical/current rates from the local MT5 terminal. It does not place, close, modify, or manage any order or position.
