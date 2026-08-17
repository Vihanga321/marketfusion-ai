# MarketFusion AI

MarketFusion AI is an experimental EUR/USD research pipeline. V0.1 is retained as a baseline because it exposed suspicious Yahoo Finance daily-candle geometry. The current primary long-history/live market-data path is MetaTrader 5 (MT5); Dukascopy/JForex is used for small historical event windows where MT5 M1 coverage is insufficient. The earlier OANDA adapter is preserved as an optional/unused adapter for auditability.

No result in this repository should be interpreted as a live-trading recommendation or a demonstrated predictive edge.

## Versions

- **V0.1:** Yahoo Finance daily baseline. Preserved because validation exposed a candle-data artifact and a misleading high apparent accuracy.
- **V0.2:** Clean MT5 historical H1/H4 price pipeline plus a read-only live EUR/USD tick collector. OANDA files are retained but are not the active data path.
- **V0.3:** Point-in-time macroeconomic intelligence using official FRED/ALFRED and ECB data with explicit vintage/availability semantics. BLS values without historical release timestamps remain quarantined from the macro-level model.
- **V0.4A (in progress):** economic-release timestamp/provenance validation plus read-only historical EUR/USD event-window coverage. BLS direct archived-calendar access is preferred; when it is unavailable, the current fallback is explicitly labeled reconstruction from ALFRED first-vintage timing plus the official BLS 08:30 Eastern release convention. Consensus forecasts are still excluded until a trustworthy historical provider is established.

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

Dukascopy/JForex credentials are **not** stored in `.env` by the Java collector. Load them only into the current PowerShell process when required:

```powershell
$env:DUKASCOPY_USER = Read-Host "Dukascopy demo login"
$secret = Read-Host "Dukascopy demo password" -AsSecureString
$cred = New-Object System.Management.Automation.PSCredential("x", $secret)
$env:DUKASCOPY_PASSWORD = $cred.GetNetworkCredential().Password
Remove-Variable secret,cred
```

Verify presence without printing either value:

```powershell
[bool]$env:DUKASCOPY_USER
[bool]$env:DUKASCOPY_PASSWORD
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

## V0.4A economic event timestamps

V0.4A begins the Economic Surprise Engine by solving event time and provenance before adding forecasts or reaction models. `America/New_York` is used for historical Eastern-time conversion, so EST/EDT is not treated as a fixed UTC offset.

The event adapter first attempts official BLS release-calendar/archive sources. If those sources are blocked or unavailable, `reconstruct_bls_from_alfred.py` uses a conservative fallback: first non-bootstrap ALFRED vintage dates are cross-checked across related official series and paired with the official BLS 08:30 Eastern release convention. Such rows are labeled as reconstructed rather than pretending the timestamp was directly scraped. Ambiguous or failed cross-checks are quarantined.

```powershell
.\venv\Scripts\python.exe src\events\download_bls_release_calendar.py --start-year 2015
.\venv\Scripts\python.exe src\events\reconstruct_bls_from_alfred.py
.\venv\Scripts\python.exe src\events\validate_event_table.py
```

The initial event schema includes `actual`, `forecast`, `previous`, and `revised_previous`, but V0.4A intentionally leaves consensus forecasts empty until a trustworthy historical forecast provider is established. The validator fails if forecast values appear without that provenance. Later V0.4 stages will add BEA/FOMC/ECB event sources, actual/revision fields, reaction features, and historical analogue retrieval.

### Dukascopy event-window validation

MT5 M1 history is treated as an availability-limited source for old event windows. The JForex diagnostic path therefore retrieves small EUR/USD historical windows from Dukascopy. Tick reconstruction is validated against Dukascopy native BID M1 OHLC. Missing provider chunks are reported as `INCOMPLETE`; they are never interpolated or silently filled.

MT5 may return a cached bar outside an unavailable historical `copy_rates_range` window while reporting API success. The event coverage audit now quarantines every out-of-range response before testing exact reaction timestamps. Macro regime flags follow the same point-in-time rule as macro values: a release-day flag becomes true only at or after that day's first public release, never during pre-release hours.

The production validation contract is `v0.4a-native-bid-tick-rebuild-v3`:

- Window: UTC M1 bars from `T-10m` through `T+250m`, inclusive (261 minutes).
- BID and ASK reconstruction: open=first tick, high=maximum, low=minimum, close=last tick after stable timestamp ordering and exact-duplicate removal.
- Non-finite/non-positive quotes and `ask < bid` are invalid; ASK is never manufactured from an assumed spread.
- Native Dukascopy BID M1 is the independent reference. Complete ticks without native BID remain `NATIVE_REFERENCE_UNAVAILABLE`, not PASS.
- The numerical BID OHLC tolerance remains `1e-8`.

For a minute-aligned event at `T`, pre-event means the close of bar `T-1m`; +1m is the close of bar `T`; +5m is the close of `T+4m`; +15m, +60m, and +240m use `T+14m`, `T+59m`, and `T+239m` respectively.

The validation sample deliberately covers **each calendar year and each event type**:

```powershell
.\venv\Scripts\python.exe .\src\events\export_events_for_jforex.py `
  --sample-per-year 1 `
  --output .\data\dukascopy\tick_validation_sample.tsv
```

Compile the Java 8 JForex project and run the robust sample validator:

```powershell
cd .\jforex-event-exporter
mvn clean compile
mvn exec:java `
  "-Dexec.mainClass=ai.marketfusion.jforex.TickValidationBatchRobust" `
  "-Dexec.args=..\data\dukascopy\tick_validation_sample.tsv ..\data\dukascopy\tick_validation_robust"
```

For the complete model-eligible event set, use the batched runner. It exports all events, splits them into small deterministic batches, runs every batch, then verifies that every expected event appears exactly once in the aggregate result:

```powershell
cd C:\Users\vihan\marketfusion-ai
.\scripts\run_jforex_full_validation.ps1
```

The final human-readable report is written to:

```text
reports/dukascopy_full_validation_summary.txt
reports/dukascopy_incomplete_events.tsv
```

Interpretation is strict:

- `PASS`: complete M1 window and reconstructed BID exactly matches native BID within tolerance.
- `INCOMPLETE`: one or more historical provider chunks were unavailable; no fabricated fill is permitted.
- `MISMATCH`: retrieved tick reconstruction disagrees with native BID or contains an invalid spread.
- `ERROR`: validator/process failure for the event.

`PROVIDER_UNAVAILABLE`, `NETWORK_TIMEOUT`, `HISTORY_EMPTY`, `NATIVE_REFERENCE_UNAVAILABLE`, and `TICK_HISTORY_INCOMPLETE` retain the cause of an incomplete result. The quarantine TSV records the exact event, source, missing chunk, retry count, and provider diagnostic.

The runner is resumable but does not trust files by name alone. Each canonical result stores the event-input SHA-256, validator-source SHA-256, contract version, and configuration. Compatible PASS rows are preserved and only INCOMPLETE rows are retried. A source, input, or contract change invalidates reuse and causes a clean revalidation into a temporary attempt directory before atomic merge. MISMATCH and ERROR rows are preserved for the final failure report.

The completed 276-event run's four MISMATCH events have a separate, read-only forensic path. It first proves that the production aggregate's mismatch set is still exactly the approved four IDs, runs all non-live checks, and then retrieves only those four windows. It does not alter the production validator, reconstructor, tolerance, or aggregate:

```powershell
.\scripts\run_dukascopy_mismatch_diagnostic.ps1
```

`DUKASCOPY_USER` and `DUKASCOPY_PASSWORD` must already exist in that PowerShell process. The command writes minute-level differences, first/last tick context, and an evidence-based classification to `reports/dukascopy_mismatch_minutes.tsv`, `reports/dukascopy_mismatch_tick_context.tsv`, and `reports/dukascopy_mismatch_diagnostic.txt`. Missing tick minutes are never filled or interpolated.

The frozen V0.4A adjudication preserves the original result—242 PASS, 30 INCOMPLETE, four MISMATCH, and zero ERROR. The four mismatches are quarantined as provider native/tick disagreements; their forensic evidence also records five tick-history-hole minutes. There are zero demonstrated reconstruction bugs, invalid-tick-filtering causes, or minute-boundary issues. Together with the 30 incomplete events, 34 events are quarantined and only the 242 original strict PASS rows are eligible for reaction extraction.

Build the adjudication, strict manifest, and read-only reaction pipeline with:

```powershell
.\scripts\run_v04a_event_reaction_pipeline.ps1
```

The runner verifies SHA-256 fingerprints for all six frozen production/forensic artifacts before and after the new stage. It never reads `.env`; credentialed collection requires `DUKASCOPY_USER` and `DUKASCOPY_PASSWORD` in the current process. Without them it completes the non-live work and records `REACTION_EXTRACTION_NOT_RUN` rather than fabricating market data. Generated provider windows live under ignored `data/dukascopy/v04a_reaction/`, are written atomically, and are reused only after their event identity, UTC bounds, contract version, source, 261-minute coverage, and SHA-256 fingerprint pass validation.

Reaction timing is defined by M1 bar starts: pre-event is `T-1`; +1m is `T`; +5m is `T+4`; +15m is `T+14`; +60m is `T+59`; and +240m is `T+239`. EUR/USD uses one pip equal to `0.0001`. BID, ASK, MID, and spread are reconstructed from the same valid, ordered, exact-deduplicated ticks. MID high/low comes from actual per-tick midpoints, never averaged BID/ASK OHLC; spread geometry comes from each actual `ask-bid` observation.

Spread expansion is the maximum observed post-release tick spread through the selected horizon minus the pre-event bar's closing spread. Maximum upward excursion is positive relative to the pre-event midpoint; maximum downward excursion is signed and non-positive. Continuation requires two non-FLAT cumulative directions with the same sign, reversal requires opposite non-FLAT signs, and exact zero is the deterministic FLAT case.

The canonical wide schema prefixes pre-release values with `context_` and every post-release label with `outcome_`. A leakage gate rejects `outcome_` columns from pre-event feature matrices. Missing market minutes invalidate extraction: there is no interpolation, price forward fill, synthetic ASK, or assumed spread. The outputs are:

- `reports/dukascopy_v04a_adjudication.tsv` and `.txt`
- `data/processed/v04a_eligible_event_manifest.parquet`
- `reports/v04a_eligible_event_manifest.csv` and `_summary.csv`
- `data/processed/v04a_event_reactions.parquet` and `_long.parquet`
- `reports/v04a_event_reaction_sample.csv`
- `reports/v04a_event_reaction_coverage.csv`
- `reports/v04a_event_reaction_quality.txt`
- `reports/v04a_historical_event_memory_schema.csv`

Only rows with `model_eligible_market_reaction=true` may enter a future event model. The aggregator sets that flag only for strict PASS results; INCOMPLETE, MISMATCH, and ERROR are rejected by the training gate.

## Project preflight and CI

A local preflight checks Python syntax, unit tests, repository secret hygiene, the read-only trading boundary, all PowerShell syntax, event-table validation when local data exist, Java 8 compilation, and synthetic tick-reconstruction tests:

```powershell
.\scripts\preflight.ps1
```

To include a credentialed Dukascopy sample run:

```powershell
.\scripts\preflight.ps1 -RunJForexSample
```

GitHub Actions also contains non-credentialed Python/safety tests and a Java 8 Maven compile job. Live Dukascopy validation intentionally remains local because credentials are never committed to GitHub.

## Security and safety

- `.env`, Parquet datasets, Dukascopy outputs, MT5 tick files, and MT5 history are ignored by Git.
- `.env.example` must contain placeholders only.
- No `order_send`, JForex order execution, position modification, or embedded broker login/password code is permitted in this stage.
- Exposed API keys or demo credentials must be rotated/revoked; removing them from the latest file is not enough if they remain valid.
- `scripts/repo_safety_check.py` is a repository-level preflight guard, not a substitute for a dedicated secret scanner.

## Forecast timing and validation

H1/H4 targets use exact UTC elapsed horizons. Rows crossing weekend or missing-data gaps are excluded. Walk-forward folds are chronological, use expanding training windows, purge labels that overlap the test boundary, and add an embargo. Models are compared against random, majority, previous-direction, and logistic-regression baselines. No predictive edge is claimed unless improvement is consistent out of sample.
