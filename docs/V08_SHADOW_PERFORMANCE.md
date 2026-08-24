# MarketFusion V0.8 Shadow Performance and Model Research

V0.8 is a read-only scientific monitoring layer. It records what the live V0.6 runtime actually said, waits for exact V0.5A horizons to mature, and then attaches outcomes in a separate ledger. It never sends a trading command, modifies the V0.5C registry, or presents descriptive returns as PnL.

## Forward shadow versus backtests

`TRUE_FORWARD_SHADOW` means V0.8 observed and hashed the V0.6 state before the earliest outcome could be known. A state first seen more than ten minutes after its completed M5 decision, or after the 15-minute horizon matures, is rejected as `RETROACTIVE_SHADOW_REJECTED`. V0.8 never runs today’s model over old rows to manufacture historical predictions.

Historical experiments use the distinct `BACKTEST_RESEARCH` scope and appear only in the research scorecard. Forward-shadow and research rows are never aggregated together.

## Immutable ledger and outcomes

The ignored runtime directory `data/evaluation/v08/` contains:

- `shadow_predictions.parquet`: one immutable record per observed V0.6 decision timestamp. It contains source-time market context, exact model IDs/probabilities, advisory, blockers, contracts, and a canonical SHA-256—but no outcomes.
- `conflicts/`: quarantined copies when a decision timestamp reappears with a different canonical payload. The original record is never overwritten.
- `shadow_outcomes.parquet`: outcomes bound to both prediction ID and prediction-payload SHA.
- `shadow_window_outcomes.parquet`: future-only MFE/MAE diagnostics for advisory windows actually emitted live.
- `provider_uptime.parquet`: append-only provider-cycle observations.

An outcome is eligible only at `decision_timestamp_utc + 15/60/240 minutes`, with an exact matching V0.5A timestamp and a matured-at timestamp no later than the evaluation time. Missing bars are not interpolated. The cost band uses the decision-time spread contract; future spread and unknown commission are never invented.

## Performance and WAIT

Reports include sample counts and explicit minimum-sample gates. Directional accuracy is calculated only for BUY_BIAS/SELL_BIAS observations. Probability scores are calculated only where approved-model probabilities were present in the immutable record. Wilson intervals and deterministic bootstrap intervals are descriptive confidence intervals, not evidence of future performance.

WAIT is an abstention policy, not automatically an error. V0.8 measures WAIT frequency, neutral-band outcomes, large subsequent moves, and the observed blocker category (no model, event, spread, or volatility). With no approved champion, WAIT/no-model and `NO_APPROVED_MODEL_DATA` calibration are the correct states.

V0.5B performance grouping is descriptive and uses only the intelligence-health value captured in the immutable V0.6 decision. Provider/topic activity that was not part of that causal decision payload is reported as unavailable; it is never reconstructed later from newer news or macro state.

## Calibration and drift

For actual champion probability rows, V0.8 reports multiclass Brier score, log loss, ECE, MCE, and confidence bins. Until those rows exist, calibration is `NO_APPROVED_MODEL_DATA`.

Feature drift uses PSI, standardized mean shift, and missing-rate shift. When there is no champion training reference, it is labeled `MARKET_REGIME_MONITOR`, not model drift. Champion-model drift remains `INSUFFICIENT_DATA` until an immutable matching training reference and adequate forward samples exist.

## Controlled improvement research

Heavy research is manual. It uses only MARKET_CORE until V0.5B passes its independent history gate. Configurations are predefined: elastic-net logistic regression, an isolated TRADEABLE/NO_TRADE then UP/DOWN architecture, fixed histogram gradient boosting, a fixed random-forest baseline, five deterministic feature-group ablations, and cost multipliers 1.0/1.25/1.5/2.0.

Each horizon uses chronological train, validation, purge, and untouched final holdout partitions. Configuration selection sees development validation only; the selected configuration is then evaluated once on the holdout. A configuration cannot be marked `PROMISING_FOR_V05C_CHALLENGER` when its untouched holdout cost-aware metric is non-positive. Selected linear configurations also record development-training-only coefficient rankings and an expanding-fit top-feature stability check. Results may be `PROMISING_FOR_V05C_CHALLENGER`, `REJECT`, or `INSUFFICIENT_DATA`. V0.8 cannot write `CHAMPION` and never modifies models or the V0.5C registry. Formal promotion still requires all V0.5C purged walk-forward, history, calibration, stability, and cost-aware gates.

## Commands

```powershell
# Deterministic synthetic validation
.\scripts\run_v08_tests.ps1

# One lightweight observation/outcome/analytics cycle
.\scripts\run_v08_once.ps1
.\scripts\show_v08_status.ps1

# Foreground continuous monitor (five-minute default)
.\scripts\start_v08_monitor.ps1

# Manual heavy historical research; never started by the normal runtime
.\scripts\run_v08_model_research.ps1
```

`start_marketfusion_full.ps1` starts the five-minute lightweight V0.8 monitor but explicitly does not start V0.5C training or V0.8 model research. The dashboard reads `/api/v08/status` and shows unavailable metrics as dashes or insufficient data.
