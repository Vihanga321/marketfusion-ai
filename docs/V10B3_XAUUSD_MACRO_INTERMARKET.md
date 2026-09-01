# MarketFusion V1.0B.3 — XAUUSD Verified Macro / Intermarket Evidence

## Why this phase exists

V1.0B.2 found `PRICE_ONLY_FEATURE_EVIDENCE_WEAK`. The existing XAUUSD chart,
technical, structure, liquidity, volatility, session, regime, and pattern
features did not show enough stable directional evidence to justify larger
models.

V1.0B.3 therefore adds verified external market context before any additional
model complexity.

This phase is **research only**.

- Automatic execution: `DISABLED`
- Runtime: `SHADOW_ADVISORY_ONLY`
- Manual confirmation: `REQUIRED`
- Model promotion: `NONE`
- Production integration: `FALSE`

## Verified context series

The first controlled provider is FRED / Federal Reserve market data:

| Key | FRED series | Meaning |
|---|---|---|
| `usd_broad` | `DTWEXBGS` | Nominal Broad U.S. Dollar Index |
| `yield_2y` | `DGS2` | 2-Year U.S. Treasury constant maturity yield |
| `yield_10y` | `DGS10` | 10-Year U.S. Treasury constant maturity yield |
| `real_yield_10y` | `DFII10` | 10-Year inflation-indexed Treasury yield |

Official series pages:

- https://fred.stlouisfed.org/series/DTWEXBGS
- https://fred.stlouisfed.org/series/DGS2
- https://fred.stlouisfed.org/series/DGS10
- https://fred.stlouisfed.org/series/DFII10

These are chosen because U.S. dollar strength and nominal/real Treasury yields
are directly relevant context for gold research. No synthetic news, sentiment,
or fabricated macro values are used.

## Important vintage limitation

The public `fredgraph.csv` download represents the provider's current historical
view and does not carry point-in-time ALFRED vintages/release timestamps.
Therefore V1.0B.3 marks every FRED row:

- `vintage_safe = false`
- `research_only = true`
- `production_model_eligible = false`

A positive V1.0B.3 result is **not sufficient for champion promotion**.
A later production model would require a point-in-time/vintage-safe provider or
non-revising intraday market source.

## Conservative availability contract

To reduce release-time leakage, a FRED observation is not visible to XAUUSD
research until:

`observation date + 4 calendar days + 23:59 UTC`

The as-of join then requires:

`context_available_at_utc <= XAUUSD decision_timestamp_utc`

Future context is a hard failure.

This lag is intentionally conservative. It sacrifices freshness rather than
pretending the public CSV contains historical publication timestamps.

## Derived context

For each series the research layer attaches:

- latest causally available level;
- 1-observation change;
- 5-observation change;
- context age in hours.

It also derives:

- `2s10s = DGS10 - DGS2`;
- `10y breakeven proxy = DGS10 - DFII10`.

## Targets

No target fishing is performed. The phase reuses only the two predeclared
research targets carried forward from V1.0B.2:

- 15m `V10B_REFERENCE`;
- 30m `SELECTIVE_MEDIUM`.

The previously exposed final XAUUSD tail remains quarantined and is never
scored in this phase.

## Evidence comparisons

For Logistic Regression and Histogram Gradient Boosting, compare:

1. `PRICE_CORE`
2. `PRICE_CORE_PLUS_MACRO`
3. `ALL_PRICE`
4. `ALL_PRICE_PLUS_MACRO`

Evaluation remains purged chronological walk-forward.

Macro context must improve over the price-only baseline and pass stability,
balanced-accuracy, and log-loss gates before it can be called a research
evidence candidate.

## Run

From the repository root:

```powershell
.\scripts\run_v10b3_xauusd_macro_intermarket.ps1
```

The normal run refreshes the four verified FRED series first.

After a successful refresh, an offline repeat can use the locally persisted
verified context store:

```powershell
.\scripts\run_v10b3_xauusd_macro_intermarket.ps1 -Offline
```

Outputs:

- `data/context/XAUUSD/fred/gold_macro_context.parquet`
- `data/context/XAUUSD/fred/manifest.json`
- `reports/XAUUSD/v10b3_macro_intermarket_evidence.json`
- `reports/XAUUSD/v10b3_macro_intermarket_leaderboard.csv`

Generated provider data and research reports are local runtime/research
artifacts and must not be treated as production model state.

## Decision

Possible research outcomes:

### `MACRO_CONTEXT_EVIDENCE_FOUND`

Verified daily dollar/yield context improved the quarantined walk-forward
research evidence. The next step is to obtain vintage-safe or intraday
intermarket sources before controlled production-eligible model research.

### `MACRO_CONTEXT_EVIDENCE_WEAK`

The conservative daily Federal Reserve context is still insufficient. The next
step should be richer **verified** context, especially read-only intraday
intermarket data if the broker/provider supports it (for example dollar index,
silver, and Treasury/rates proxies), not more indicator stacking.
