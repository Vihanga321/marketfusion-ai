# MarketFusion V1.0B.2 — XAUUSD Feature Evidence Research

## Why this phase exists

V1.0B.1 tested 20 combinations of XAUUSD horizons and cost/ATR-aware WAIT zones. The local result was `NO_TARGET_CANDIDATE`.

The strongest observed targets were still near coin-flip on the previously exposed final holdout:

- 30m + `SELECTIVE_MEDIUM`: holdout balanced accuracy about 0.5083;
- 15m + `V10B_REFERENCE`: walk-forward balanced accuracy about 0.5150 but holdout balanced accuracy about 0.5047.

Across all 20 target experiments, every target failed holdout balanced accuracy and holdout log loss, 19/20 failed walk-forward balanced accuracy, and 18/20 failed the cost-aware gate.

The correct next question is therefore not "which WAIT threshold should we tweak next?". It is:

> Do any of the existing causal price/structure/volatility/session/pattern feature families contain stable XAUUSD directional evidence at all?

## Holdout quarantine

V1.0B.2 does **not** evaluate the already-seen V1.0B.1 final tail again.

For each target specification it recreates the chronological directional sample, quarantines the final 15% (minimum 750 rows), and purges every earlier decision whose future outcome would overlap the quarantine boundary.

The quarantined tail is recorded only as metadata:

- `quarantine_start`
- `quarantine_rows`
- `quarantine_evaluated = false`
- `purge_ok`

This prevents repeated tuning against the same final period.

## Predeclared target specifications

Only the two strongest V1.0B.1 areas are screened:

1. 15m + `V10B_REFERENCE` (1.5x spread, 0.10x ATR)
2. 30m + `SELECTIVE_MEDIUM` (2.0x spread, 0.20x ATR)

No new target threshold is optimized in this phase.

## Feature families

Every screen starts from `MARKET_CORE` and compares controlled additions:

- Technical
- Price Action
- Structure
- Liquidity
- Support / Resistance
- Volatility
- Session
- Regime
- Patterns

Three predeclared combinations are also checked:

- Structure + Volatility
- Structure + Liquidity
- Technical + Regime

Finally, `ALL_EXISTING_PRICE_CONTEXT` tests all existing causal feature groups together.

No target, outcome, or future field may enter the feature matrix.

## Model screens

Two bounded model families are used only as evidence probes:

- class-balanced Logistic Regression;
- class-balanced Histogram Gradient Boosting.

The goal is not to find a production model. Logistic Regression tests roughly linear evidence, while Histogram Gradient Boosting checks whether meaningful non-linear evidence exists without launching a large model search.

## Evaluation

Only the pre-quarantine research segment is used.

Evaluation uses:

- purged chronological walk-forward;
- five folds;
- class-prior baseline;
- previous-direction baseline;
- balanced accuracy;
- macro F1;
- log loss;
- Brier score;
- cost-aware return-minus-WAIT-band diagnostic;
- recent-fold stability;
- feature-group delta versus `MARKET_CORE`.

## Feature evidence gate

A screen can become `FEATURE_EVIDENCE_CANDIDATE` only if it satisfies all research gates:

- mean balanced accuracy >= 0.515;
- recent fold balanced accuracy >= 0.51;
- at least 3/5 folds beat the class-prior baseline;
- mean log loss beats class prior by at least 0.002;
- mean cost-aware metric is not worse than previous-direction;
- added feature groups improve balanced accuracy over `MARKET_CORE` by at least 0.002.

This is still **not** `APPROVED_CHAMPION`.

## Decision

Possible phase outcomes:

### `FEATURE_EVIDENCE_FOUND`

At least one existing feature set shows enough stable research evidence to justify a later controlled model phase.

Next:

`V1.0B.3_CONTROLLED_MODEL_RESEARCH`

### `PRICE_ONLY_FEATURE_EVIDENCE_WEAK`

Existing price-derived context remains too weak.

Next:

`ADD_VERIFIED_GOLD_MACRO_INTERMARKET_CONTEXT`

That means MarketFusion should add verified causal gold drivers such as dollar/yield/macro context before training larger models.

## Outputs

Generated locally:

- `reports/XAUUSD/v10b2_feature_evidence.json`
- `reports/XAUUSD/v10b2_feature_leaderboard.csv`
- `reports/XAUUSD/v10b2_feature_diagnostics.csv`

These files do not change the production model registry.

## Run

```powershell
.\scripts\run_v10b2_xauusd_feature_evidence.ps1
```

No-persist validation:

```powershell
.\scripts\run_v10b2_xauusd_feature_evidence.ps1 -NoPersist
```

## Safety

- Automatic execution: `DISABLED`
- Runtime: `SHADOW_ADVISORY_ONLY`
- Manual confirmation: `REQUIRED`
- Model promotion: `NONE`
- Production integration: `FALSE`
