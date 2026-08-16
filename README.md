# MarketFusion AI

MarketFusion AI is an experimental EUR/USD research pipeline. V0.1 is retained as a baseline because it exposed suspicious Yahoo Finance daily-candle geometry. V0.2 builds a separate hourly data pipeline using official OANDA v20 midpoint, bid, and ask candles.

No result in this repository should be interpreted as a live-trading recommendation or a demonstrated predictive edge.

## Versions

- **V0.1:** Existing Yahoo Finance daily files, feature pipeline, saved model, and validation reports. These files are preserved for auditability.
- **V0.2:** OANDA `EUR_USD` H1 data, strict quality validation, causal features, and purged walk-forward testing with observed spread costs.

V0.2 does not include news, macro data, deep learning, automated execution, or live trading.

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

Edit `.env` and supply:

- `OANDA_API_TOKEN`
- `OANDA_ACCOUNT_ID`
- `OANDA_ENV=practice` or `OANDA_ENV=live`

The token is sent as an HTTP bearer token. OANDA documents the practice and live REST URLs at <https://developer.oanda.com/rest-live-v20/development-guide/> and token authentication at <https://developer.oanda.com/rest-live-v20/authentication/>.

## V0.2 workflow

Download complete H1 candles. The default range starts on 2016-01-01 and ends at the current UTC hour; it can be changed with command-line arguments.

```powershell
.\venv\Scripts\python.exe src\download_oanda.py
.\venv\Scripts\python.exe src\download_oanda.py --start 2020-01-01 --end 2026-01-01
```

Validate before creating any features:

```powershell
.\venv\Scripts\python.exe src\validate_oanda_data.py
```

Only after the report says `VALIDATION_STATUS: PASS`:

```powershell
.\venv\Scripts\python.exe src\build_features_v02.py
.\venv\Scripts\python.exe src\walk_forward_validation_v02.py --horizon 1
.\venv\Scripts\python.exe src\walk_forward_validation_v02.py --horizon 4
```

Both downstream scripts independently rerun the OANDA validation gate. They stop without training when the raw dataset is absent or fails validation.

## V0.2 outputs

- Raw data: `data/oanda/eurusd_h1.parquet`
- Features: `data/processed/eurusd_h1_features_v02.parquet`
- Quality report: `reports/oanda_data_quality_report.txt`
- Model comparison: `reports/v02_validation_summary_{1h|4h}.csv`
- Yearly folds: `reports/v02_yearly_results_{1h|4h}.csv`
- Calibration bins: `reports/v02_calibration_{1h|4h}.csv`
- XGBoost importance: `reports/v02_feature_importance_{1h|4h}.csv`

Parquet datasets and `.env` are intentionally ignored by Git. `.env.example` is safe to commit because it contains placeholders only.

## V0.3 macroeconomic intelligence

V0.3 treats publication time as part of every observation. It never joins a
macro value before `available_from_utc`, and retains vintage metadata for audit.
ALFRED date-only vintages are conservatively usable from the next UTC day. ECB
revised series use SDMX `VALID_FROM`. BLS current values are quarantined because
the time-series API does not provide their historical release timestamps.

```powershell
.\venv\Scripts\python.exe src\macro\download_fred.py
.\venv\Scripts\python.exe src\macro\download_bls.py
.\venv\Scripts\python.exe src\macro\download_ecb.py
.\venv\Scripts\python.exe src\macro\validate_macro_data.py
.\venv\Scripts\python.exe src\build_features_mt5_macro.py
.\venv\Scripts\python.exe src\walk_forward_validation_mt5_macro.py
```

`download_fred.py` stops rather than silently falling back when `FRED_API_KEY`
is absent. Only `data/macro/macro_safe.parquet` is eligible for feature joins.
The matched walk-forward comparison uses the same yearly test boundaries and
embargo as the clean MT5 price-only evaluation.

## Forecast timing and costs

OANDA H1 candle timestamps identify the start of the candle. V0.2 records `decision_timestamp = timestamp + 1 hour`, because a complete candle can only be used after it closes. One-hour and four-hour targets require exact UTC horizon continuity, so rows crossing weekend or other data gaps are excluded. Walk-forward folds operate on decision/label availability times, use expanding training windows, purge labels that overlap the test boundary, and add a configurable embargo.

Cost-aware results assume an entry at the current close quote and an exit at the future close quote: longs pay current ask and exit at future bid; shorts receive current bid and cover at future ask. Per-signal results overlap at the four-hour horizon and are not a portfolio equity curve. This is a research approximation and does not include slippage, financing, latency, or order rejection.
