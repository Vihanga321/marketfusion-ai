# MarketFusion V0.9A Multi-Engine Intelligence

V0.9A is an observational analysis layer. It produces structured analytical signals from completed V0.5A market rows; scores are not trading probabilities and do not alter V0.6 decisions or V0.8 recording.

## Engines

The first release exposes technical, price action, structure, liquidity, volatility, session, and regime engines. They use the existing causal V0.5A M5 return, volatility, spread, and timestamp fields. No indicator is silently reconstructed as a second copy under a different name.

Fundamental, news, sentiment, and intermarket outputs are contract-only and return `UNAVAILABLE` with `NO_VERIFIED_PROVIDER` until verified providers and point-in-time inputs exist.

## Output contract

Each output includes `engine_name`, `contract_version`, `decision_timestamp_utc`, `status`, `direction_score`, `confidence`, `regime`, `feature_count`, `input_freshness`, `reason_codes`, and raw `components`. Status is one of `AVAILABLE`, `PARTIAL`, `UNAVAILABLE`, or `INVALID`. Missing values remain missing; they are never converted to neutral scores.

## Causality

The API evaluates only rows with `decision_timestamp_utc <= now` and reads a bounded recent window. Completed V0.5A rows are the only input. A future row cannot change an earlier engine snapshot. Session state reuses the canonical V0.9 market calendar, including `WEEKEND` closure and DST-aware session boundaries.

## API and research status

`GET /api/engines/status` is localhost-only and read-only. V0.9A does not create a meta-model, retrain models, promote models, or add execution routes. Ablation and leaderboard work should consume these outputs later with existing V0.5C walk-forward controls; no improvement is claimed by this release.
