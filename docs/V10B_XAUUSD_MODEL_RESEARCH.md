# MarketFusion V1.0B — XAUUSD Controlled Model Research

## Purpose

V1.0B evaluates **XAUUSD only** and may register an approved champion only when all conservative promotion gates pass. EURUSD models, registries, shadow observations, and data remain isolated and unchanged.

This phase is research/shadow-only:

- Automatic execution: **DISABLED**
- Runtime: **SHADOW_ADVISORY_ONLY**
- Manual confirmation: **REQUIRED**
- No order placement, position management, or Algo Trading activation is implemented.

## Data contract

The runner consumes the local V1.0A stores:

- `data/features/XAUUSD/features.parquet`
- `data/features/XAUUSD/target_research.parquet`

These files are intentionally Git-ignored. GitHub/CI therefore validates code and causal contracts with deterministic fixtures; it does **not** fabricate XAUUSD historical metrics or a champion.

Targets remain the V1.0A cost-aware contract using the selected neutral band `max(1.5 × spread cost, 0.10 × causal ATR)` and exact future completed-M5 timestamps for 15m, 60m, and 240m.

## Feature research

All model features are available at the completed decision bar or are computed strictly from current/past completed M5 bars. Target/future columns are rejected from the model feature registry.

Ablation groups:

- MARKET_CORE
- TECHNICAL
- PRICE_ACTION
- STRUCTURE
- LIQUIDITY
- SUPPORT_RESISTANCE
- VOLATILITY
- SESSION
- REGIME
- PATTERNS

`PATTERNS` in V1.0B is a deterministic causal candle-geometry group (doji, engulfing, pin-bar, inside/outside bar). It is **not** a claim that every V0.9A.1 multi-bar lifecycle pattern has already been historically validated on XAUUSD.

## Model families

The bounded candidate set is:

- Logistic Regression
- Histogram Gradient Boosting
- Random Forest
- XGBoost when installed/compatible

The runner does not add LSTMs, Transformers, or large neural networks.

## Validation design

For each horizon:

1. Build the causal XAUUSD dataset.
2. Reserve the final 15% chronological holdout, with an additional horizon purge before holdout start.
3. Keep the final holdout untouched during ablation and model-family selection.
4. Screen feature groups with purged chronological walk-forward Logistic Regression.
5. Evaluate supported model families only on a bounded set of the strongest feature sets plus BASELINE and ALL_V10B.
6. Fit temporal calibration using only earlier training data and a later calibration partition.
7. Evaluate the selected candidate once on the final holdout.
8. Promote only when every required gate passes.

No shuffled train/test split is used.

## Promotion gates

Promotion requires all of the following:

- at least 90 historical decision days;
- five chronological walk-forward folds;
- walk-forward stability PASS;
- temporal calibration available;
- final-holdout balanced accuracy at least 0.02 above majority baseline;
- final-holdout macro F1 at least 0.02 above majority baseline;
- final-holdout log loss at least 0.01 better than the class-prior baseline;
- final-holdout cost-aware metric not worse than the previous-direction baseline;
- finite walk-forward and holdout probability metrics.

A valid result is `NO_APPROVED_MODEL`. Promotion thresholds must not be lowered merely to create a champion.

## Artifacts and registry

Research reports are written under `reports/XAUUSD/` and are Git-ignored.

Model artifacts are written under `models/XAUUSD/artifacts/` and should remain Git-ignored. Registry records include `asset_id=XAUUSD`, horizon, model family, feature contract hash, calibration status, holdout metrics, artifact SHA-256, and safety flags.

The runtime loader rejects:

- non-XAUUSD artifacts;
- non-approved records;
- missing artifacts;
- SHA mismatches;
- feature-contract mismatches.

## Run locally

With the V1.0A XAUUSD data present:

```powershell
cd C:\Users\vihan\marketfusion-ai
.\scripts\run_v10b_xauusd_research.ps1
```

For a dry research run that writes no models/registry/reports:

```powershell
.\scripts\run_v10b_xauusd_research.ps1 -NoPersist
```

The full persisted run can be CPU-intensive because it performs chronological ablation and multiple model-family evaluations. It is deliberately a controlled manual research job, not a continuously running service.

## Expected outcome

The only valid horizon states after a real local run are:

- `APPROVED_CHAMPION`, when all gates pass; or
- `NO_APPROVED_MODEL`, when any gate fails.

No `BEST_AVAILABLE`, forced promotion, or synthetic probability state is permitted.
