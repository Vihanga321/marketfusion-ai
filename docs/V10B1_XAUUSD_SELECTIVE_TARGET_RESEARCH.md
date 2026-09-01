# MarketFusion V1.0B.1 — XAUUSD Selective Target Research

## Purpose

V1.0B completed successfully as a research phase, but no XAUUSD 15m/60m/240m candidate passed the full promotion gate. The strongest candidates also showed a repeated failure mode: the original three-class target produced a very small NEUTRAL class at longer horizons and many models effectively behaved like DOWN/UP classifiers while ignoring NEUTRAL.

V1.0B.1 therefore does **not** add a bigger model. It first tests whether the prediction problem itself should be reformulated.

This phase is research-only. It cannot promote a champion and cannot alter execution state.

- Automatic execution: `DISABLED`
- Runtime: `SHADOW_ADVISORY_ONLY`
- Manual confirmation: `REQUIRED`
- Production integration: `FALSE`

## Research question

For XAUUSD, which combination of horizon and economically meaningful WAIT zone creates a directional subset that is both:

1. large enough to study honestly; and
2. materially more learnable than trivial baselines on chronological out-of-sample data?

## Horizons

The controlled horizon set is:

- 15 minutes
- 30 minutes
- 60 minutes
- 120 minutes
- 240 minutes

30m and 120m are added only for research. Their outcomes are created by an **exact future completed-M5 timestamp join**. Missing bars are never replaced with row shifts or fabricated candles.

## WAIT-zone contracts

All target bands are constructed from information available at the decision timestamp:

`WAIT band = max(spread multiplier × spread/price, ATR multiplier × ATR/price)`

Contracts:

| Contract | Spread | ATR | Intent |
|---|---:|---:|---|
| V10B_REFERENCE | 1.5x | 0.10x | reproduce the original V1.0B semantics |
| SELECTIVE_MEDIUM | 2.0x | 0.20x | remove more small/noisy moves |
| SELECTIVE_WIDE | 3.0x | 0.35x | higher-conviction directional subset |
| SELECTIVE_STRICT | 4.0x | 0.50x | strict selective subset |

Each matured row is labeled:

- `DOWN` when future return < `-WAIT band`
- `WAIT` when the future return remains inside the band
- `UP` when future return > `+WAIT band`

The directional screening model only sees DOWN/UP rows. WAIT rows are excluded from that **screen**, but their fraction remains a core part of the target contract evaluation.

## Causal directional screen

The screen deliberately uses one bounded, interpretable model family: class-balanced Logistic Regression.

Features are limited to the existing causal V1.0B groups:

- MARKET_CORE
- STRUCTURE
- VOLATILITY

No `target_*`, `outcome_*`, or future field can enter the feature matrix.

Evaluation uses:

- purged chronological walk-forward folds;
- a separate chronological final holdout;
- train-only preprocessing;
- class-prior baseline;
- previous-direction baseline;
- balanced accuracy;
- macro F1;
- log loss;
- Brier score;
- cost-aware return-minus-band diagnostic;
- directional coverage.

This is a **target screen**, not a champion search. It intentionally does not calibrate or persist an approved model.

## Research-candidate gate

A target/horizon can be labeled `RESEARCH_CANDIDATE` only if it passes all screening gates, including:

- directional coverage between 10% and 90%;
- sufficient directional history and final-holdout sample;
- walk-forward balanced accuracy >= 0.515;
- final holdout balanced accuracy >= 0.52;
- sufficient fold-level baseline stability;
- holdout log loss beating the class-prior baseline by at least 0.01;
- cost-aware holdout diagnostic not worse than the previous-direction baseline.

`RESEARCH_CANDIDATE` **does not mean APPROVED_CHAMPION**. It only means the target contract deserves full V1.0B.2 model research.

## Outputs

Local generated reports:

- `reports/XAUUSD/v10b1_target_research.json`
- `reports/XAUUSD/v10b1_target_leaderboard.csv`

These remain local research artifacts and are not production model state.

## Run

From the repository root on the MarketFusion Windows machine:

```powershell
.\scripts\run_v10b1_xauusd_target_research.ps1
```

For a no-persist validation run:

```powershell
.\scripts\run_v10b1_xauusd_target_research.ps1 -NoPersist
```

## Decision after the run

If at least one target passes:

`TARGET_CANDIDATE_FOUND -> V1.0B.2 controlled XAUUSD model research`

If none pass:

`NO_TARGET_CANDIDATE -> revise XAUUSD target/feature design before adding model complexity`

V1.0C fusion remains blocked until at least one XAUUSD horizon earns a legitimately approved champion in a later controlled promotion phase.
