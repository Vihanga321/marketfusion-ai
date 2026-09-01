# MarketFusion V1.0B.5 — XAUUSD Intraday Intermarket Evidence

## Purpose

V1.0B.4 verified that the connected MetaQuotes-Demo terminal exposes enough fast-moving context for controlled research. The verified READY set in the local audit was:

- XAGUSD — silver / U.S. dollar
- USDJPY — U.S. dollar / Japanese yen
- EURUSD — euro / U.S. dollar
- US500 — U.S. 500 equity-risk proxy
- USTEC — U.S. 100 technology-risk proxy

XAUUSD remains the **only prediction target**. Every other instrument is a read-only context sensor.

## Why the historical join uses broker epoch keys

V1.0B.4 measured the live MetaQuotes-Demo clock at approximately +03:00 relative to wall-clock UTC. Older XAUUSD research stores were created before this broker-clock issue was discovered and therefore contain the broker epoch timestamp in fields historically labelled UTC.

V1.0B.5 does **not** rewrite those immutable stores and does **not** invent historical DST corrections. Instead it:

1. runs a fresh V1.0B.4 audit and requires broker-clock calibration PASS;
2. collects completed M5 context history with `copy_rates_from_pos(..., 1, ...)`, excluding the forming candle;
3. derives a completed-bar decision key as broker bar open + 5 minutes;
4. joins context to XAUUSD on the exact broker decision timestamp only;
5. uses no forward fill, no as-of fill, and no future context;
6. excludes `utc_hour`, `day_of_week`, and session-clock features from the V1.0B.5 price baseline.

The raw broker key is used only to preserve exact chronology and alignment. It is explicitly reported as `MT5_BROKER_EPOCH_KEY_NOT_WALL_CLOCK_UTC`.

## Context features

For each verified context family V1.0B.5 derives only current/past completed-M5 information:

- 5-minute return
- 15-minute return
- 30-minute return
- 60-minute return
- rolling 60-minute return volatility
- current completed-bar range as a fraction of close

No future returns or target fields are included as model inputs.

## Predeclared context groups

- `SILVER`
- `FX_USD` = USDJPY + EURUSD
- `EQUITY_RISK` = US500 + US100/USTEC
- `SILVER_PLUS_FX`
- `ALL_VERIFIED_INTRADAY`

If a required family is not READY on the fresh audit or the deeper history collection, that group is recorded as unavailable rather than substituted with another ticker.

## Paired baseline design

Context is compared against price-only models on the **same aligned rows** so a context result cannot look better merely because it selects a different time period.

Two causal baselines are predeclared:

- `PRICE_CORE_NO_CLOCK`
- `PRICE_STRUCTURE_VOLATILITY_NO_CLOCK`

For each target, context group, baseline, and model family the research trains:

1. the paired price baseline;
2. the same baseline plus the context features.

Both use the exact same train/validation rows.

## Targets

Only the previously predeclared research targets are used:

- 15m `V10B_REFERENCE`
- 30m `SELECTIVE_MEDIUM`

The previously exposed final XAUUSD tail remains quarantined and is never evaluated in this phase.

## Models

Evidence probes only:

- Logistic Regression
- Histogram Gradient Boosting

This phase cannot create or promote a production champion.

## Evidence gates

An `INTRADAY_CONTEXT_EVIDENCE_CANDIDATE` must satisfy all of the following:

- mean walk-forward balanced accuracy >= 0.515;
- most recent fold balanced accuracy >= 0.510;
- context beats the paired baseline in at least 3 of 5 folds;
- mean balanced-accuracy improvement over the paired baseline >= 0.003;
- mean log loss improves over the paired baseline by at least 0.002;
- mean log loss is better than the class-prior baseline;
- cost-aware performance is not worse than the paired price baseline.

These are research gates, not profitability guarantees and not model-promotion gates.

## Decisions

Possible top-level decisions:

- `INTRADAY_CONTEXT_EVIDENCE_FOUND`
  - at least one predeclared context experiment passed all evidence gates;
  - next phase: controlled XAUUSD model research using only the supported context contract.

- `INTRADAY_CONTEXT_EVIDENCE_WEAK`
  - experiments ran but no context contract passed all gates;
  - next phase: expand/wait for additional verified point-in-time intraday context rather than increasing model complexity blindly.

- `INSUFFICIENT_PREQUARANTINE_INTERMARKET_HISTORY`
  - the broker did not provide enough exact pre-quarantine context overlap for a fair experiment;
  - next phase: collect more point-in-time context before research.

## Local run

```powershell
.\scripts\run_v10b5_xauusd_intraday_intermarket_evidence.ps1
```

Optional deeper/bounded history request:

```powershell
.\scripts\run_v10b5_xauusd_intraday_intermarket_evidence.ps1 -ContextRows 50000
```

Outputs:

```text
reports/XAUUSD/v10b5_intraday_intermarket_evidence.json
reports/XAUUSD/v10b5_intraday_intermarket_leaderboard.csv
```

Local raw context features are written under:

```text
data/market/XAUUSD/intermarket/v10b5/
```

They remain local research data and are not production model artifacts.

## Safety

- Prediction target: XAUUSD only
- Context instruments: read-only
- Forming candle: excluded
- Historical context fill: exact join only
- Previously exposed tail: quarantined
- Model promotion: none
- Production integration: false
- Automatic execution: disabled
- Runtime: `SHADOW_ADVISORY_ONLY`
- Manual confirmation: required
