# MarketFusion V1.0C — Frozen XAUUSD Forward Point-in-Time Validation

## Why V1.0C exists

V1.0B through V1.0B.7 repeatedly inspected the same historical XAUUSD sample. Those phases were useful for rejecting weak ideas, but continuing to tune that history would increase overfitting risk.

V1.0B.7 therefore ended historical tuning. V1.0C starts a new evidence regime: contracts are frozen first and are judged only on observations whose completed-M5 decision timestamp occurs after an immutable forward-start marker created locally at initialization.

V1.0C is still research/shadow only. It cannot promote a model or place a trade.

## Frozen hypotheses

Only two raw-context hypotheses are carried forward.

### XAU15_SILVER_LR_V1

- target: XAUUSD only
- horizon: 15 minutes
- target contract: `V10B_REFERENCE`
- price baseline: `PRICE_CORE_NO_CLOCK`
- read-only context: XAGUSD / Silver
- model family: Logistic Regression
- paired comparison: frozen price-only LR vs frozen price+Silver LR

Reason: Silver was the most persistent 15-minute directional helper in B5/B7. Later Platt calibration and explicit XAU/XAG relationship features did not improve it.

### XAU30_FX_HGB_V1

- target: XAUUSD only
- horizon: 30 minutes
- target contract: `SELECTIVE_MEDIUM`
- price baseline: `PRICE_STRUCTURE_VOLATILITY_NO_CLOCK`
- read-only context: USDJPY + EURUSD
- model family: Histogram Gradient Boosting
- paired comparison: frozen price-only HGB vs frozen price+FX HGB

Reason: in B5 this contract beat its paired price-only baseline in 5/5 folds and passed the directional/paired gates, missing only the class-prior probability gate. Later calibration degraded the directional edge.

## Initialization is one-way

Run initialization exactly once:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1 -Initialize
```

Initialization:

1. runs the hardened V1.0B.4 MT5 broker-clock and instrument-identity audit;
2. requires the frozen context families to be available under the same verified identities;
3. collects completed-M5 historical context for fitting only;
4. fits one frozen price-only and one frozen price+context model for each contract;
5. writes model SHA256 values and historical training-data hashes;
6. writes `forward_start_utc` and the corresponding broker-epoch cutoff;
7. freezes the evidence gates below.

If a manifest already exists, initialization fails. V1.0C does not provide a force-reset switch because resetting the marker after seeing forward results would contaminate the experiment.

## Forward observations

After initialization, run continuously while MT5 and the PC/server are available:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1 -Continuous
```

Each cycle:

- reruns the broker-clock/instrument identity checks;
- requires the normalized broker-clock offset to match the initialization contract;
- uses completed M5 bars only;
- joins context by the exact raw MT5 broker-epoch decision key;
- records only decisions after the immutable forward-start marker;
- refuses decisions more than 10 minutes old, preventing retroactive backfill after downtime;
- records the paired baseline and context probabilities before the outcome matures;
- never retrains or recalibrates the frozen models.

The prediction ledger contains no future/target outcome field.

## Outcome maturation

After 15 or 30 minutes, V1.0C looks for the exact future completed XAUUSD M5 close. No as-of or forward fill is permitted.

The decision-time neutral/wait band is frozen from causal information available at prediction time:

```text
max(spread_multiplier × spread_relative_to_price,
    atr_multiplier × ATR14 / decision_close)
```

An outcome is directional only when the absolute future return exceeds that band. WAIT/neutral outcomes are retained in the maturity count but excluded from binary DOWN/UP directional scoring.

This mirrors the earlier directional-screen research and is intentionally not yet a deployable WAIT selector: future directional eligibility is known only after maturation.

## Predeclared minimum forward sample

The following values are frozen before collection begins and must not be lowered after results are observed:

- at least 20 active UTC trading dates per contract;
- at least 500 matured observations per contract;
- at least 400 matured directional outcomes per contract;
- four chronological directional stability blocks;
- at least 80 directional rows per block.

Until both contracts reach the minimum sample, the global decision remains:

```text
COLLECTING_FORWARD_DATA
```

## Forward evidence gates

Once the minimum sample is reached, a contract passes only if all of the following hold:

- context balanced accuracy >= 0.515;
- recent chronological block balanced accuracy >= 0.510;
- context beats its paired frozen price-only baseline in at least 3/4 blocks;
- context balanced-accuracy delta vs paired baseline >= 0.003;
- context log loss improves on paired baseline by >= 0.002;
- context log loss beats the class-prior probabilities frozen from historical training;
- context cost-aware metric is not worse than the paired baseline.

Possible global decisions after both contracts are sample-ready:

- `FORWARD_EVIDENCE_FOUND` — at least one frozen contract passes every gate;
- `FORWARD_EVIDENCE_WEAK` — neither frozen contract passes every gate.

A forward pass still does not create a production champion. It only justifies a later, separately controlled production-eligible research phase.

## Local files

Local generated evidence is stored under:

```text
data/evaluation/XAUUSD/v10c_forward_validation/
    manifest.json
    predictions.parquet
    outcomes.parquet
    latest_status.json

models/XAUUSD/forward_validation/v10c/
    XAU15_SILVER_LR_V1_baseline.joblib
    XAU15_SILVER_LR_V1_context.joblib
    XAU30_FX_HGB_V1_baseline.joblib
    XAU30_FX_HGB_V1_context.joblib
```

These are local research artifacts and are not registry champions.

## Useful commands

Initialize once:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1 -Initialize
```

Record/mature one cycle:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1
```

Run continuously:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1 -Continuous -IntervalSeconds 60
```

Read status without MT5 work:

```powershell
.\scripts\run_v10c_xauusd_forward_validation.ps1 -Status
```

## Safety

- prediction target: XAUUSD only;
- XAGUSD, USDJPY and EURUSD: read-only context only;
- forming M5 candle: excluded;
- exact broker-epoch context join only;
- no retroactive prediction backfill;
- models frozen at initialization;
- no retraining during forward collection;
- no model promotion;
- production integration: false;
- automatic execution: disabled;
- runtime: `SHADOW_ADVISORY_ONLY`;
- manual confirmation: required.
