# MarketFusion AI

MarketFusion AI is an experimental EUR/USD research pipeline. V0.1 is retained as a baseline because it exposed suspicious Yahoo Finance daily-candle geometry. The current primary market-data path is MetaTrader 5 (MT5); the earlier OANDA adapter is preserved as an optional/unused adapter for auditability.

No result in this repository should be interpreted as a live-trading recommendation or a demonstrated predictive edge.

## Versions

- **V0.1:** Yahoo Finance daily baseline. Preserved because validation exposed a candle-data artifact and a misleading high apparent accuracy.
- **V0.2:** Clean MT5 historical H1/H4 price pipeline plus a read-only live EUR/USD tick collector. OANDA files are retained but are not the active data path.
- **V0.3:** Point-in-time macroeconomic intelligence using official FRED/ALFRED, ECB, and quarantined BLS data where historical release timestamps are unavailable.

## Read-only MetaTrader 5 tick layer

The MT5 scripts connect only to the account already logged into the desktop terminal. They do not contain login details and do not place or modify trades.

```powershell
.\venv\Scripts\python.exe src\test_mt5_connection.py
.\venv\Scripts\python.exe src\live_mt5_feed.py
.\venv\Scripts\python.exe src\validate_mt5_ticks.py
```

The collector uses MT5 tick-history ranges to recover ticks between polling cycles and after reconnects. It writes UTC daily files under `data/mt5/ticks/`, flushes bounded buffers periodically, and stops cleanly with Ctrl+C. For a bounded diagnostic run, use `--duration-seconds 30`.

The live status timestamp comes from each tick's `time_msc`, not from the local computer clock. During a normal FX weekend closure, the collector prints `no fresh tick` and does not manufacture or save the stale `symbol_info_tick` snapshot.

### Historical MT5 bars

The historical pipeline reads H1 and H4 bars from the already logged-in terminal, removes the current incomplete bar, validates each timeframe, and gates all downstream work on the H1 result.

```powershell
.\venv\Scripts\python.exe src\download_mt5_history.py
.\venv\Scripts\python.exe src\validate_mt5_history.py
.\venv\Scripts\python.exe src\build_features_mt5.py
.\venv\Scripts\python.exe src\walk_forward_validation_mt5.py
```

Historical Parquet files are ignored by Git. Missing market-session bars are never synthesized, and forecast targets crossing a missing or weekend gap are excluded.

## Setup

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Keep real secrets only in `.env`; never put them in `.env.example`, screenshots, chat, or Git history.

For V0.3, edit `.env` and set:

```text
FRED_API_KEY=your_private_fred_key
BLS_API_KEY=
```

The macro helpers load the project-local `.env` directly, so the scripts work from VS Code or PowerShell even when terminal environment-file injection is disabled.

The optional OANDA adapter still understands:

```text
OANDA_API_TOKEN=...
OANDA_ACCOUNT_ID=...
OANDA_ENV=practice
```

## V0.2 MT5 workflow

```powershell
.\venv\Scripts\python.exe src\download_mt5_history.py
.\venv\Scripts\python.exe src\validate_mt5_history.py
```

Only after H1 validation passes:

```powershell
.\venv\Scripts\python.exe src\build_features_mt5.py
.\venv\Scripts\python.exe src\walk_forward_validation_mt5.py --horizon 1
.\venv\Scripts\python.exe src\walk_forward_validation_mt5.py --horizon 4
```

The clean MT5 price-only tests are retained as baselines. They did not establish a reliable predictive edge.

## Optional OANDA adapter

The OANDA files are preserved for auditability but are not required for the active MT5 pipeline:

```powershell
.\venv\Scripts\python.exe src\download_oanda.py
.\venv\Scripts\python.exe src\validate_oanda_data.py
```

## V0.3 macroeconomic intelligence

V0.3 treats publication time as part of every observation. It never joins a macro value before `available_from_utc`, and retains vintage metadata for audit. ALFRED date-only vintages are conservatively usable from the next UTC day. ECB revised series use SDMX `VALID_FROM`. BLS current values are quarantined because the time-series API does not provide their historical release timestamps.

Run the official adapters:

```powershell
.\venv\Scripts\python.exe src\macro\download_fred.py
.\venv\Scripts\python.exe src\macro\download_bls.py
.\venv\Scripts\python.exe src\macro\download_ecb.py
.\venv\Scripts\python.exe src\macro\validate_macro_data.py
```

Only after macro validation passes:

```powershell
.\venv\Scripts\python.exe src\build_features_mt5_macro.py
.\venv\Scripts\python.exe src\walk_forward_validation_mt5_macro.py --horizon 1
.\venv\Scripts\python.exe src\walk_forward_validation_mt5_macro.py --horizon 4
```

`download_fred.py` stops rather than silently falling back when `FRED_API_KEY` is absent. Only `data/macro/macro_safe.parquet` is eligible for feature joins. The matched walk-forward comparison uses the same yearly test boundaries and embargo as the clean MT5 price-only evaluation.

## Security and safety

- `.env`, Parquet datasets, MT5 tick files, and MT5 history are ignored by Git.
- `.env.example` must contain placeholders only.
- No `order_send`, trade execution, position modification, or embedded MT5 login/password code is permitted in this stage.
- Exposed API keys or demo credentials must be rotated/revoked; removing them from the latest file is not enough if they remain valid.

## Forecast timing and validation

H1/H4 targets use exact UTC elapsed horizons. Rows crossing weekend or missing-data gaps are excluded. Walk-forward folds are chronological, use expanding training windows, purge labels that overlap the test boundary, and add an embargo. Models are compared against random, majority, previous-direction, and logistic-regression baselines. No predictive edge is claimed unless improvement is consistent out of sample.
