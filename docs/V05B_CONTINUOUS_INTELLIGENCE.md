# MarketFusion V0.5B — Continuous News + Macro Intelligence

## Purpose

V0.5B continuously captures free EUR/USD-relevant public information and turns each capture into causal, point-in-time context for later MarketFusion models. It does **not** place trades and does **not** retrain a production model.

The key rule is conservative availability: a headline or macro observation returned by a current endpoint is not treated as historically known merely because its provider timestamp is older. `available_from_utc` is MarketFusion's own `first_observed_utc`. This means the archive becomes stronger as it runs forward in time without inventing historical knowledge.

## Free sources

Official RSS feeds:

- Federal Reserve monetary-policy press releases — `https://www.federalreserve.gov/feeds/press_monetary.xml`
- Federal Reserve speeches — `https://www.federalreserve.gov/feeds/speeches.xml`
- ECB press releases/speeches/interviews — `https://www.ecb.europa.eu/rss/press.html`
- ECB statistical press releases — `https://www.ecb.europa.eu/rss/statpress.html`
- BLS Employment Situation — `https://www.bls.gov/feed/empsit.rss`
- BLS CPI — `https://www.bls.gov/feed/cpi.rss`

Broad discovery metadata:

- GDELT DOC 2.0 ArticleList JSON. Only headline/article metadata is retained; MarketFusion does not copy article bodies.

No-key macro snapshots:

- FRED public graph CSV: effective federal funds rate, U.S. 2-year yield, U.S. 10-year yield, and 10Y-2Y curve.
- ECB SDMX: main refinancing operations rate.

These current endpoints are used only from the moment MarketFusion observes them. They do not replace the older audited ALFRED/ECB historical-vintage work already present in V0.3/V0.4.

## Storage

Generated runtime data are local and git-ignored:

```text
data/intelligence/
├── news/news_items.parquet
├── macro/macro_observations.parquet
├── context/v05b_context_history.parquet
└── runtime_status.json
```

News is append-only by canonical `news_id`. The first captured version is preserved. Macro observations are append-only by `(series_id, observation_period, value)` so a later revised value becomes a new point-in-time observation rather than overwriting the prior one.

## Causal context features

Every V0.5B cycle writes one context row at its UTC decision timestamp. It aggregates only source rows with `first_observed_utc <= decision_timestamp_utc`.

News windows: 15 minutes, 60 minutes, 240 minutes, and 1440 minutes. Each window contains counts for official/GDELT sources, USD relevance, EUR relevance, high-impact terms, central-bank topics, rates, inflation, labor, growth, and risk/geopolitical topics.

Macro context contains the latest causally observed value for each configured series, its observation date, first-observed timestamp, and age. The derived policy-rate spread is `U.S. effective federal funds rate - ECB main refinancing rate`; it is descriptive context, not a signal.

No `outcome_*` field is allowed in V0.5B model context.

## Quality and failure policy

A current context row is not appended unless:

- there is at least one canonical news row and one macro row in the accumulated store;
- news and macro availability equals MarketFusion first-observed time;
- no first-observed timestamp is in the future;
- news IDs are unique;
- macro values are numeric;
- the context decision timestamp equals the capture timestamp;
- no future macro availability enters context;
- no outcome column exists in the model-visible context.

Individual provider failures are warnings once an accumulated valid store exists. A complete absence of usable news or macro data fails closed. Provider errors remain visible in runtime status.

## Commands

Run deterministic/offline tests:

```powershell
.\scripts\run_v05b_tests.ps1
```

Run one live free-source cycle:

```powershell
.\scripts\run_v05b_once.ps1
```

Start the continuous collector (default 5 minutes):

```powershell
.\scripts\start_v05b_continuous.ps1
```

Custom interval, minimum enforced by the Python runner is 60 seconds:

```powershell
.\scripts\start_v05b_continuous.ps1 -IntervalSeconds 300
```

Check current status from another terminal:

```powershell
.\scripts\show_v05b_status.ps1
```

## Interpretation

V0.5B creates a forward point-in-time intelligence memory. It does not claim that keyword-topic counts predict EUR/USD, that GDELT headlines are authoritative, or that current no-key macro endpoints are historical vintages. Official sources and aggregator metadata remain separately identified.

The next stage, V0.5C, should consume matured V0.5A market outcomes plus V0.5B causal context under purged walk-forward validation and a champion/challenger model registry. No model should be promoted merely because one recent day improves in-sample accuracy.
