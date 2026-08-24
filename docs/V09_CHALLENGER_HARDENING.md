# MarketFusion V0.9 — Challenger Hardening Research

## Purpose

V0.9 is the final research-hardening layer before a promising configuration may be handed back to V0.5C as a challenger. It does **not** promote models and it does not change the live V0.6 advisory engine.

The primary source experiment is the V0.8 result `H60_ABLATE_VOLUME`: a 60-minute Elastic-Net Logistic Regression using MARKET_CORE with the VOLUME feature group removed. V0.8 marked it `PROMISING_FOR_V05C_CHALLENGER`; that label is research-only.

## Scientific flow

```text
V0.8 promising experiment
        ↓
exact-prefix reproducibility audit
        ↓
final temporal holdout sealed
        ↓
nested purged chronological development folds
        ↓
fixed candidate / feature-ablation families
        ↓
predefined target-cost multipliers
        ↓
predefined abstention probability + margin thresholds
        ↓
temporal calibration
        ↓
outer-fold stability + baseline comparison
        ↓
untouched final holdout opened once
        ↓
READY_FOR_V05C_CHALLENGER or REJECT
        ↓
formal promotion remains V0.5C only
```

There is no random K-fold and no shuffled time-series split. Each train segment must have its matured labels strictly before the next validation segment begins. The 60-minute research horizon therefore uses a minimum 60-minute purge.

## Fixed research choices

V0.9 deliberately keeps the search small and reviewable. Candidate families are Elastic-Net Logistic Regression, standard Logistic Regression, HistGradientBoosting, Random Forest baseline, and the two-stage `TRADEABLE/NO_TRADE → UP/DOWN` research architecture. XGBoost is not required by V0.9 CI.

Feature comparisons are predefined: all MARKET_CORE, no VOLUME, and Elastic-Net ablations of SESSION_TIME, SPREAD_LIQUIDITY, VOLATILITY and PRICE_RETURN. V0.5B news/macro features remain disabled for training until their own causal-history gate is satisfied.

Target cost multipliers are fixed at `1.00, 1.25, 1.50, 2.00`. Abstention thresholds are fixed at top probability `0.50, 0.55, 0.60` and top-vs-second probability margin `0.05, 0.08, 0.10`. V0.9 must not expand these grids after viewing the final holdout.

## Reproducibility

The current V0.5A training dataset grows over time, so V0.9 reconstructs the exact historical prefix size recorded by V0.8 for `H60_ABLATE_VOLUME`. It reruns the original V0.8 split and candidate implementation and compares development/holdout metrics to the tracked V0.8 scorecard within a small deterministic numerical tolerance.

If that result cannot be reproduced, V0.9 returns `RESEARCH_RESULT_NOT_REPRODUCIBLE` and does not create a ready challenger handoff.

## Promotion boundary

V0.9 can emit `READY_FOR_V05C_CHALLENGER`. That means only that a configuration survived V0.9 research hardening. It does **not** mean CHAMPION.

The formal V0.5C promotion-history gate remains at least 90 calendar days and is imported from the V0.5C contract. V0.9 cannot lower it. A configuration can therefore be research-ready while reporting `promotion_history_gate: NOT_MET`.

Live state remains unchanged until V0.5C legitimately promotes a model:

```text
15m champion: NONE
60m champion: NONE
240m champion: NONE
V0.6 advisory: WAIT
trading execution: DISABLED
```

## MT5 history-depth audit

The V0.9 research run performs a read-only MT5 depth probe for EURUSD M1/M5/M15/H1 using `copy_rates_from_pos`. It reports only returned row count and earliest/latest timestamps. The result is subject to broker/terminal history and the MT5 `Max bars` setting. It does not modify V0.5A, backfill production history, or expose account data.

## Reports

V0.9 writes tracked research evidence to:

- `reports/v09_reproducibility_audit.txt`
- `reports/v09_candidate_scorecard.csv`
- `reports/v09_nested_temporal_audit.csv`
- `reports/v09_feature_importance.csv`
- `reports/v09_feature_stability.csv`
- `reports/v09_abstention_research.csv`
- `reports/v09_target_robustness.csv`
- `reports/v09_regime_robustness.csv`
- `reports/v09_calibration.csv`
- `reports/v09_mt5_history_depth_audit.txt`
- `reports/v09_v05c_challenger_candidates.json`
- `reports/v09_validation.txt`

The latest local status JSON is generated under `data/research/v09/` and is git-ignored.

## Commands

Focused deterministic tests plus repository safety:

```powershell
.\scripts\run_v09_tests.ps1
```

Run the heavy research manually:

```powershell
.\scripts\run_v09_research.ps1
```

Show the latest local V0.9 status:

```powershell
.\scripts\show_v09_status.ps1
```

V0.9 research is intentionally not placed in the 15-second/5-minute runtime loops and it never performs trade execution.

## Valid outcomes

A normal current result may be `PASS_RESEARCH_READY_HISTORY_INSUFFICIENT` if the hardened 60-minute configuration survives but the 90-day V0.5C promotion-history requirement is not yet satisfied. `PASS_RESEARCH_REJECTED` is equally valid if the V0.8 candidate does not survive stronger testing.

Neither status is a profitability or predictive-edge claim.
