# MarketFusion V0.4D — Historical Consensus Audit

V0.4D adds provider-independent historical consensus infrastructure without changing the frozen V0.4C event-value or historical-memory datasets. It remains research-only and does not create trading signals.

## Why this phase exists

V0.4C has audited actual, previous, and release-time revision values for four measures, but it has no historical economist consensus. MarketFusion therefore cannot yet calculate a defensible `actual - consensus` surprise.

A historical provider row is **not** sufficient by itself. For pre-release research MarketFusion requires evidence that the consensus was available strictly before the official event timestamp `T`.

## Provider policy

### Trading Economics

The economic-calendar schema distinguishes:

- `Forecast` / `ForecastValue`: survey consensus from a representative group of economists.
- `TEForecast` / `TEForecastValue`: Trading Economics' own projection.

`TEForecast` is never accepted as MarketFusion consensus.

Trading Economics documents an Economic Calendar Point-in-Time endpoint for historical/backtesting use, but the public calendar schema does not document a dedicated timestamp identifying when each `Forecast` first became available. `LastUpdate` is the most recent event-record update and may reflect actual/revision updates after release. V0.4D therefore **does not** substitute `LastUpdate` for `consensus_available_from_utc`.

Documentation:

- https://docs.tradingeconomics.com/economic_calendar/point-in-time/
- https://docs.tradingeconomics.com/economic_calendar/schema/

The provider starts as `CONDITIONALLY_APPROVED` and becomes `APPROVED_POINT_IN_TIME` only if an audit can prove strict pre-release availability.

### LSEG Reuters Polls Consensus

LSEG documents Reuters Polls consensus history from 1999 and says consensus estimates are traceable to specific polling points in time. MarketFusion records this as a strong candidate, but it is not activated until entitlement and field-level payloads are audited.

Documentation:

- https://www.lseg.com/en/data-catalogue/economics/economic-macro-forecasts/reuters-polls-consensus

## Frozen-source rule

The local V0.4C Parquet outputs are ignored by Git. Before a provider audit, initialize a fingerprint manifest exactly once from the already-audited local V0.4C outputs:

```powershell
.\scripts\run_v04d_consensus_audit.ps1 -InitializeFingerprint
```

Later runs omit that flag. Any source-byte mutation then hard-fails.

## Credentials

Trading Economics credentials are process-scoped only:

```powershell
$secret = Read-Host "Trading Economics API key" -AsSecureString
$cred = New-Object System.Management.Automation.PSCredential("x", $secret)
$env:TRADING_ECONOMICS_API_KEY = $cred.GetNetworkCredential().Password
Remove-Variable secret,cred
[bool]$env:TRADING_ECONOMICS_API_KEY
```

Never print or commit the value.

Without a credential the pipeline still runs the non-live framework and reports `TRADING_ECONOMICS_LIVE_AUDIT_NOT_RUN`.

## Stratified audit before full collection

The first live provider audit samples CPI and Employment events across 2015, 2018, 2020, 2022, 2024, and 2026. Full 242-event collection is blocked unless that sample reaches `APPROVED_POINT_IN_TIME`.

```powershell
.\scripts\run_v04d_consensus_audit.ps1
```

Only after provider approval:

```powershell
.\scripts\run_v04d_consensus_audit.ps1 -FullCollection
```

## Mapping rules

V0.4D starts with exactly the four audited V0.4C measures:

- headline CPI YoY (seasonally adjusted transformed measure)
- core CPI YoY (seasonally adjusted transformed measure)
- nonfarm payroll change
- unemployment rate

A provider row must pass country, event family, measure identity, release timestamp, reference month, and unit checks. Candidate Trading Economics names are not considered approved provider identifiers until a live mapping audit locks stable ticker/symbol metadata.

## Surprise activation

Raw surprise is calculated only when all of the following hold:

1. MarketFusion actual value is audited.
2. Provider consensus is present.
3. Units match.
4. `point_in_time_verified == true`.
5. `consensus_available_from_utc < event_timestamp_utc`.

Then:

```text
raw_surprise = actual_value - consensus_value
```

Raw arithmetic remains separate from interpretation. For example, lower-than-consensus unemployment can be stronger labor data, but MarketFusion never hard-codes that as an EUR/USD direction.

Normalized surprise uses prior same-measure releases only. Future events and the query event itself cannot enter the normalizer.

## Outputs

The runner writes readable reports under `reports/` and provider audit payloads under ignored `data/provider_audit/`. Generated Parquet files remain ignored.

Important outputs include:

- `reports/v04c_frozen_source_fingerprint.json`
- `reports/v04c_frozen_source_fingerprint.txt`
- `reports/v04d_consensus_source_registry.csv`
- `reports/v04d_consensus_mapping_audit.csv`
- `reports/v04d_consensus_candidate_coverage.csv`
- `reports/v04d_consensus_sample_audit.csv`
- `reports/v04d_consensus_source_audit.txt`
- `reports/v04d_consensus_quality.txt`
- `reports/v04d_surprise_quality.txt`

V0.4D does not claim predictive edge, profitability, or live-trading readiness.
