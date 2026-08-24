# MarketFusion V0.4D — Free MT5 Economic Calendar Intelligence

This phase keeps MarketFusion at **zero recurring data-subscription cost**. It uses the MetaTrader 5 Economic Calendar for historical research candidates and creates MarketFusion's own pre-release forecast snapshots going forward.

## Scientific status

Historical MT5 rows may contain `forecast_value`, but a historical row alone does not prove exactly when the forecast first became available before release T. Therefore historical forecasts are intentionally labeled:

`HISTORICAL_FORECAST_PRESENT_UNVERIFIED_SNAPSHOT_TIME`

They are **not** model-eligible consensus.

The live snapshot EA records `captured_at_gmt` at the moment MarketFusion observes a forecast. That proves the capture time, but the row is still not model-eligible until the MT5 event identity/measure/reference period is independently matched to the frozen MarketFusion event table and the capture time is strictly before T.

## Files

- `mql5/MarketFusionCalendarExporter.mq5` — one-time historical US/USD calendar export.
- `mql5/MarketFusionCalendarSnapshotEA.mq5` — read-only EA that periodically snapshots upcoming US/USD calendar rows.
- `src/events/mt5_calendar_importer.py` — fail-closed Python audit/import layer.
- `tests/test_v04d_mt5_calendar.py` — chronology and eligibility gates.
- `scripts/run_v04d_mt5_calendar_audit.ps1` — local audit runner.

Neither MQL5 program sends orders, modifies positions, or requires an API key.

## 1. Compile the historical exporter

In MT5 choose **File → Open Data Folder**. Copy:

`mql5/MarketFusionCalendarExporter.mq5`

to:

`MQL5/Scripts/`

Open MetaEditor, compile the script, then run it on any chart. Default range starts at 2015-01-01 and filters to US/USD calendar rows.

The export is written to the terminal's `MQL5/Files/` directory as:

`marketfusion_mt5_calendar_history.tsv`

Important: MT5 Economic Calendar functions use trade-server time. The historical exporter deliberately preserves that time and does **not** pretend it is UTC.

## 2. Audit the historical export

From the repository root:

```powershell
.\scripts\run_v04d_mt5_calendar_audit.ps1 `
  -HistoryTsv "C:\path\to\MQL5\Files\marketfusion_mt5_calendar_history.tsv"
```

Generated local audit data is written under `data/mt5/calendar/` and ignored by Git. The readable quality report is:

`reports/v04d_mt5_calendar_quality.txt`

Expected status at this stage:

`V04D_FREE_MT5_CALENDAR_STATUS: PASS_FAIL_CLOSED`

and:

`historical_model_eligible_consensus_rows: 0`

That zero is intentional until point-in-time provenance is proved.

## 3. Start building verified forecast snapshots for future releases

Copy:

`mql5/MarketFusionCalendarSnapshotEA.mq5`

to:

`MQL5/Experts/`

Compile it in MetaEditor and attach it to one chart. By default it snapshots upcoming US/USD economic-calendar rows every 300 seconds over a 48-hour look-ahead window.

It writes:

`marketfusion_mt5_calendar_snapshots.tsv`

under `MQL5/Files/`.

Each snapshot stores both the broker/server representation of the event and a `captured_at_gmt` timestamp representing when MarketFusion observed that calendar state.

To audit snapshots:

```powershell
.\scripts\run_v04d_mt5_calendar_audit.ps1 `
  -SnapshotsTsv "C:\path\to\MQL5\Files\marketfusion_mt5_calendar_snapshots.tsv"
```

Or audit history and snapshots together:

```powershell
.\scripts\run_v04d_mt5_calendar_audit.ps1 `
  -HistoryTsv "C:\path\to\marketfusion_mt5_calendar_history.tsv" `
  -SnapshotsTsv "C:\path\to\marketfusion_mt5_calendar_snapshots.tsv"
```

## 4. What comes next

After the first real exports exist, the next V0.4D sub-stage is an **identity and reference-period adjudication** for exactly four measures:

- headline CPI YoY
- core CPI YoY
- nonfarm payroll change
- unemployment rate

The audit must establish stable MT5 `event_id` / `event_code` mappings and compare each reference period to frozen V0.4C event values. Only then can a future snapshot be accepted when:

`captured_at_gmt < event_timestamp_utc`

A snapshot at T or after T is rejected for PRE_RELEASE use.

## 5. Surprise activation rule

Do not calculate a model-eligible surprise until all of the following are true:

1. Event/measure identity is audited.
2. Reference period and units match the frozen MarketFusion event value.
3. Forecast is present.
4. Forecast capture timestamp is strictly before release T.
5. Actual is the already-audited release payload.

Then:

`raw_surprise = actual - forecast`

Historical MT5 forecasts without a proven pre-release snapshot time remain research-only and must not silently enter the final model.

## Cost

This V0.4D path requires no paid consensus API. MT5, Python, the local database, and the existing MarketFusion data pipeline can continue without a recurring market-data subscription for this stage.
