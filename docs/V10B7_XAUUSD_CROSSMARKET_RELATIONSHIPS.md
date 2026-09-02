# MarketFusion V1.0B.7 — XAUUSD Causal Cross-Market Relationship Research

## Purpose

V1.0B.5 showed that raw intraday intermarket context can improve XAUUSD direction on paired rows, but no contract passed every evidence gate. V1.0B.6 then tested leakage-safe Platt calibration on four near-pass contracts. Calibration improved probability loss in places but weakened the directional edge and returned `CALIBRATED_INTRADAY_EVIDENCE_WEAK`.

V1.0B.7 therefore tests whether **explicit causal relationships** between XAUUSD and the strongest historical context families add stable evidence beyond the same raw-context model.

This is the **last historical exploratory phase** on the currently inspected sample. Because the contracts were selected after inspecting V1.0B.5/V1.0B.6, B7 is promotion-ineligible even if evidence is found.

## Frozen contracts

Only two contracts are evaluated:

1. **15m Silver relationship**
   - target: `15m:V10B_REFERENCE`
   - price baseline: `PRICE_CORE_NO_CLOCK`
   - raw context: `SILVER` / XAGUSD
   - model: Logistic Regression

2. **30m FX/USD relationship**
   - target: `30m:SELECTIVE_MEDIUM`
   - price baseline: `PRICE_STRUCTURE_VOLATILITY_NO_CLOCK`
   - raw context: `FX_USD` / USDJPY + EURUSD
   - model: Histogram Gradient Boosting

No new model families and no target/gate changes are introduced.

## Relationship features

### XAU/XAG

All values are calculated from current/past completed M5 information at the decision timestamp:

- XAU minus XAG return divergence at 5m, 15m and 60m;
- short-vs-long divergence momentum spread;
- XAU/XAG direction agreement at 5m and 15m;
- XAU/XAG 60m volatility ratio.

### XAU/USD composite

A simple USD impulse is constructed causally as:

```text
USD impulse = 0.5 × (USDJPY return - EURUSD return)
```

for 5m, 15m, 30m and 60m completed returns.

Additional features:

- USDJPY / inverse-EURUSD direction agreement;
- FX-leg dispersion;
- XAU + USD-impulse inverse-relationship residuals;
- XAU vs inverse-USD direction alignment;
- XAU/FX 60m volatility ratio;
- short-term USD impulse acceleration.

These are deterministic combinations of already available completed-bar inputs. They do not use future returns, target outcomes, forward fill, or forming candles.

## Paired evaluation

Each B7 experiment compares:

```text
RAW MODEL
price baseline + raw context

vs

RELATIONSHIP MODEL
same price baseline + same raw context + relationship features
```

Both models use the exact same aligned rows and the same model family. This isolates whether the explicit relationship representation adds evidence beyond the raw context itself.

## Validation

- exact completed-M5 broker-epoch joins only;
- final exposed XAUUSD tail remains quarantined;
- 5-fold purged chronological walk-forward;
- no probability calibration in this phase;
- no threshold lowering.

## Evidence gates

A `CROSSMARKET_RELATIONSHIP_EVIDENCE_CANDIDATE` must satisfy all of:

- mean balanced accuracy >= 0.515;
- recent-fold balanced accuracy >= 0.510;
- relationship model beats the raw-context model in at least 3/5 folds;
- balanced-accuracy improvement over raw context >= 0.003;
- log loss improves over raw context by at least 0.002;
- log loss beats the class-prior baseline;
- cost-aware performance is not worse than raw context.

## Decisions

- `CROSSMARKET_RELATIONSHIP_EVIDENCE_FOUND`
  - at least one frozen contract passes every gate;
  - next: freeze the relationship contract and validate on genuinely new forward point-in-time data.

- `CROSSMARKET_RELATIONSHIP_EVIDENCE_WEAK`
  - no frozen relationship contract passes;
  - next: stop historical tuning on this sample and collect genuinely new forward point-in-time data.

- `INSUFFICIENT_RELATIONSHIP_RESEARCH_HISTORY`
  - insufficient exact pre-quarantine overlap.

## Local run

```powershell
.\scripts\run_v10b7_xauusd_crossmarket_relationships.ps1
```

Outputs:

```text
reports/XAUUSD/v10b7_crossmarket_relationship_evidence.json
reports/XAUUSD/v10b7_crossmarket_relationship_leaderboard.csv
```

## Safety

- prediction target: XAUUSD only;
- context instruments: read-only;
- forming M5 candle: excluded;
- historical joins: exact broker-epoch keys only;
- final exposed tail: quarantined;
- historical sample reused: yes;
- promotion eligible: false;
- model promotion: none;
- production integration: false;
- automatic execution: disabled;
- runtime: `SHADOW_ADVISORY_ONLY`;
- manual confirmation: required.
