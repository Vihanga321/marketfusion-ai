# MarketFusion V1.0B.6 — Calibrated XAUUSD Intraday Intermarket Research

## Why this phase exists

The real V1.0B.5 leaderboard showed that intermarket context can improve XAUUSD direction relative to paired price-only baselines, but probability quality remained the dominant blocker.

The strongest V1.0B.5 near-pass was 30m `SELECTIVE_MEDIUM` with `PRICE_STRUCTURE_VOLATILITY_NO_CLOCK + FX_USD` using Histogram Gradient Boosting. It beat the paired baseline in 5/5 folds, had mean balanced accuracy above 0.515, recent-fold balanced accuracy above 0.510, and met the paired balanced-accuracy/log-loss/cost improvements. It failed only because its log loss remained worse than the class-prior baseline.

Several Silver and equity-risk Logistic Regression runs also passed the directional gates and failed mainly the paired log-loss margin.

V1.0B.6 therefore tests calibration before adding model complexity or more indicators.

## Frozen exploratory candidates

Only four contracts selected from the real V1.0B.5 leaderboard are re-tested:

1. 30m `SELECTIVE_MEDIUM` + `PRICE_STRUCTURE_VOLATILITY_NO_CLOCK` + `FX_USD` + Histogram Gradient Boosting.
2. 30m `SELECTIVE_MEDIUM` + `PRICE_CORE_NO_CLOCK` + `SILVER` + Logistic Regression.
3. 15m `V10B_REFERENCE` + `PRICE_CORE_NO_CLOCK` + `SILVER` + Logistic Regression.
4. 15m `V10B_REFERENCE` + `PRICE_CORE_NO_CLOCK` + `EQUITY_RISK` + Logistic Regression.

Because these were selected after inspecting V1.0B.5, V1.0B.6 is explicitly **exploratory** and cannot promote a model.

## Leakage-safe temporal calibration

For every outer purged walk-forward fold:

1. the outer training fold is kept chronological;
2. the trailing 20% (minimum 400 rows) becomes the calibration slice;
3. base-training rows whose future target outcome overlaps the calibration start are purged;
4. the base model is fitted only on the earlier base-training rows;
5. a Platt logistic calibrator is fitted only on raw probabilities from the trailing calibration slice;
6. the untouched outer validation fold is evaluated once;
7. paired price-only and price+context models use the same rows and the same calibration procedure.

The previously exposed final XAUUSD tail remains quarantined.

## Gates

V1.0B.6 keeps the V1.0B.5 evidence gates unchanged:

- mean balanced accuracy >= 0.515;
- recent-fold balanced accuracy >= 0.510;
- context beats paired baseline in at least 3/5 folds;
- balanced-accuracy delta >= 0.003;
- calibrated context log loss improves over calibrated paired baseline by at least 0.002;
- calibrated context log loss is better than class-prior log loss;
- cost-aware performance is not worse than the paired baseline.

It also requires calibration itself not to worsen context mean log loss versus the same uncalibrated base model.

No threshold is lowered to manufacture a candidate.

## Decisions

- `CALIBRATED_INTRADAY_EVIDENCE_FOUND`
  - at least one frozen exploratory candidate passes every gate;
  - next: freeze that contract and validate it on genuinely forward point-in-time data before any promotion research.

- `CALIBRATED_INTRADAY_EVIDENCE_WEAK`
  - calibration did not solve the remaining evidence problems;
  - next: add predeclared causal cross-market relationship features or collect new forward context, not larger models blindly.

- `INSUFFICIENT_CALIBRATED_INTERMARKET_HISTORY`
  - insufficient exact pre-quarantine context overlap for the nested temporal design.

## Local run

```powershell
.\scripts\run_v10b6_xauusd_calibrated_intermarket.ps1
```

Outputs:

```text
reports/XAUUSD/v10b6_calibrated_intermarket_evidence.json
reports/XAUUSD/v10b6_calibrated_intermarket_leaderboard.csv
```

## Safety

- prediction target: XAUUSD only;
- context symbols: read-only;
- forming M5 candle: excluded;
- exact broker-epoch historical joins only;
- no forward/as-of fill;
- final exposed XAUUSD tail: quarantined;
- model promotion: none;
- production integration: false;
- automatic execution: disabled;
- runtime: `SHADOW_ADVISORY_ONLY`;
- manual confirmation: required.
