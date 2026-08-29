# MarketFusion V1.0A — Audited MT5 Historical Backfill + MARKET_CORE Expansion

## Purpose

V1.0A expands the existing V0.5A EUR/USD MARKET_CORE research history from the
same MT5 broker/terminal while preserving the current causal feature and outcome
contracts. It is a data-foundation stage, not a model-promotion stage.

The V0.9 read-only depth probe showed materially more broker history than the
continuous collector had accumulated. V1.0A therefore stages older completed
M1/M5/M15/H1 bars, audits every overlap, rebuilds the existing V0.5A feature
dataset with the existing causal builder, and only applies the expansion when the
integrity gates pass.

## Scientific boundary

Historical backfill is labeled `MT5_HISTORICAL_BACKFILL`. It is retrospective
research market history and **must not** be labeled `TRUE_FORWARD_SHADOW`.

The original V0.8 forward-shadow ledger is not changed, rewritten, or backfilled.

No V1.0A command trains a model, promotes a V0.5C challenger, or changes V0.6's
live advisory. With no current champion, V0.6 remains `WAIT`.

## Why M1 is the common-history limiter

The current MARKET_CORE contract contains M1-derived features as well as M5,
M15, and H1 features. A row is therefore eligible for V0.5C only when **all**
MARKET_CORE features are complete at the decision timestamp.

V1.0A deliberately hardens V0.5C so:
- `feature_complete == True` (or exact non-null completeness for legacy test
  frames) is required for training;
- all selected-horizon outcomes must be matured before use;
- MARKET_CORE eligibility counts only fully-featured, fully-matured rows.

This prevents older M5 rows from being treated as equivalent training rows when
the required M1 history is unavailable.

The V0.5C feature-contract hash now includes this availability rule. Existing
research artifacts made under the older availability contract remain historical
research evidence; they are not silently treated as current-contract champions.

## MT5 fetch policy

V1.0A uses only read-only MT5 rate retrieval. There is no order or position API.

Request ceilings:
- M1: 200,000 bars
- M5: 100,000 bars
- M15: 60,000 bars
- H1: 30,000 bars

The M1 ceiling is intentionally above the 100,000-row V0.9 probe. If the terminal
and broker permit deeper history, V1.0A can discover it. If MT5 returns fewer
bars because of broker availability or the terminal's `Max bars in chart`
setting, V1.0A reports the returned depth rather than inventing history.

## Two-phase workflow

### 1. Read-only audit

```powershell
.\scripts\run_v10a_tests.ps1
.\scripts\run_v10a_audit.ps1
.\scripts\show_v10a_status.ps1
```

The audit:
1. fetches completed historical bars;
2. stages them under `data/backfill/v10a/`;
3. compares all overlapping immutable OHLC/volume/spread fields;
4. constructs an in-memory merged bar history;
5. rebuilds V0.5A features/outcomes with the existing builder;
6. verifies point-in-time feature availability and exact future outcome timing;
7. compares previously materialized non-null feature/outcome values;
8. reports the fully-featured MARKET_CORE history.

Previously-null warm-up features may become populated after older history is
added. Previously non-null feature/outcome values may not mutate.

Expected successful dry status:

`PASS_AUDIT_READY_TO_APPLY`

Any immutable overlap conflict or causal dataset failure blocks apply.

### 2. Apply

Stop the managed MarketFusion collectors before applying so V0.5A cannot write
the same parquet stores concurrently. Keep MetaTrader 5 itself open.

```powershell
.\scripts\stop_marketfusion.ps1
.\scripts\apply_v10a_backfill.ps1
.\scripts\show_v10a_status.ps1
```

Apply:
- creates a timestamped local backup;
- atomically replaces the V0.5A bar stores and feature dataset;
- rereads the result;
- reruns V0.5A quality checks;
- automatically restores the backup if post-apply validation fails.

It does not delete V0.8 shadow history.

## History gate semantics

V0.5C's formal gate remains `MIN_PROMOTION_HISTORY_DAYS = 90`. Current V0.5C
code counts unique eligible decision dates, effectively trading/data days, not
the raw calendar span. V1.0A reports both:
- `eligible_trading_days`
- `calendar_span_days`

A 95-day calendar span does **not** automatically mean the 90-day formal gate is
met. The status is determined from the same eligible decision-day semantics used
by V0.5C.

If MT5's M1 history remains capped near 100,000 rows, the expansion may be
scientifically valid but still report `promotion_history_gate: NOT_MET`. In that
case, increasing MT5's history/max-bars availability can be considered and the
same audit rerun; the gate itself must not be weakened.

## Valid V1.0A statuses

- `PASS_AUDIT_READY_TO_APPLY`
- `PASS_APPLIED_MARKET_CORE_EXPANDED`
- `PASS_APPLIED_HISTORY_GATE_MET`
- `INSUFFICIENT_COMMON_HISTORY`
- `FAIL_OVERLAP_CONFLICT`
- `FAIL_DATASET_CAUSALITY`
- `FAIL`

`PASS_APPLIED_HISTORY_GATE_MET` means only that the history-length gate is met.
It does **not** mean any model has predictive edge or should be promoted.

## After a successful apply

Run the V0.5C tests, then one controlled training cycle:

```powershell
.\scripts\run_v05c_tests.ps1
.\scripts\run_v05c_daily_training.ps1
.\scripts\show_v05c_status.ps1
```

Do not reuse the rejected V0.9 candidate as a champion. V0.5C must retrain and
evaluate challengers under the current data/feature contract.

## Safety

- read-only MT5 data retrieval only;
- no trading execution;
- no account/balance/position exposure;
- no automatic V0.5C training;
- no automatic promotion;
- historical backfill is separate from true forward shadow evidence;
- runtime parquet, staging, backups, conflicts, and status are git-ignored.
