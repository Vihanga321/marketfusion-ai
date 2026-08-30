# MarketFusion V0.9A.1 Advanced Market Intelligence

V0.9A.1 is a deterministic, observational research layer. It does not alter V0.6 inference, fusion, risk gates, advisories, execution state, retraining, or promotion.

## Causal inputs

The layer reads immutable completed M5, M15, and H1 bars. A swing pivot becomes available only after its configured right-side bars close; downstream patterns, structure labels, levels, and liquidity events use the swing's confirmation timestamp. Incomplete H1 bars are excluded. H4 is intentionally not exposed because the continuous store has no independently verified H4 completed-bar contract.

## Analysis contracts

- Numeric candle geometry covers engulfing, hammer/inverted hammer, shooting star/hanging man with context, doji, morning/evening star, inside/outside bars, three soldiers/crows, and pin bars.
- Confirmed swings drive HH/HL/LH/LL, close-confirmed BOS/CHOCH, trend/range/transition, clustered support/resistance, and Fibonacci observations.
- Chart patterns cover double/triple tops and bottoms, head-and-shoulders and inverse, wedges, flags, pennants, four triangle types, and rectangle ranges. Pattern confidence is structural geometry fit, not a trading probability.
- Liquidity uses ATR/spread-scaled equal-extrema tolerances, close-confirmed sweeps, and mechanical three-candle fair-value gaps with causal partial/full fill timestamps.
- Vague order blocks are deliberately `NOT_IMPLEMENTED`; no subjective proxy is published.

`GET /api/engines/status` returns the multi-timeframe snapshot and compact event history. `GET /api/engines/patterns/recent?limit=20` returns bounded recent pattern records. Both endpoints are localhost-only, read-only, observational, and have no trading route.

## Research separation

`scripts/run_v09a1_pattern_research.py` creates sample-gated descriptive 15/60/240-minute pattern summaries by timeframe and session. Metrics remain null until the predefined sample threshold is met. The report explicitly disables production retraining and model promotion. Ablation groups are prepared for price action, structure, liquidity, patterns, and support/resistance but are not executed as production learning.
