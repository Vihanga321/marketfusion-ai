# MarketFusion V1.0B.4 — XAUUSD Intraday Intermarket Audit

## Purpose

V1.0B.3 found that delayed daily Federal Reserve context did not produce material evidence for the short-horizon XAUUSD research problem. V1.0B.4 therefore begins by auditing faster MT5 intermarket context before adding model complexity.

XAUUSD remains the **only prediction target**. Every other symbol is read-only context research only.

## Context families audited

- Silver / U.S. Dollar
- U.S. Dollar Index, when the broker exposes one
- USD/JPY
- EUR/USD
- U.S. 500 equity-risk proxy
- U.S. 100 technology-risk proxy
- WTI crude-oil proxy
- U.S. 10-year Treasury market proxy, when exposed by the broker

Broker names are discovered from the connected MT5 symbol catalog. The audit does not assume that a specific alias exists.

## Audit evidence

For each family, MarketFusion checks:

1. broker symbol candidate identity;
2. `symbol_select` availability;
3. raw MT5 tick timestamp and non-clamped tick age;
4. bounded completed M5 history using position 1, never the forming bar;
5. bounded completed M15 history using position 1;
6. duplicate/future timestamp diagnostics.

A future tick/bar timestamp fails closed for that context family.

## Decisions

- `INTRADAY_CONTEXT_READY_FOR_RESEARCH`: at least two CORE context families have sufficient bounded M5/M15 history and no future-timestamp violation is present.
- `LIMITED_INTRADAY_CONTEXT_AVAILABLE`: some verified read-only context is available but the core set is too thin for the planned research design.
- `NO_VERIFIED_INTRADAY_CONTEXT`: no usable MT5 context family is available; a separate verified external intraday provider would be required.

## Safety

- Prediction target: `XAUUSD` only
- Context instruments: read-only
- Model training: none in this audit
- Model promotion: none
- Production integration: false
- Automatic execution: disabled
- Runtime: `SHADOW_ADVISORY_ONLY`
- Manual confirmation: required

## Local run

```powershell
.\scripts\run_v10b4_xauusd_intraday_context_audit.ps1
```

Report:

```text
reports/XAUUSD/v10b4_intraday_context_audit.json
```

A positive audit result only authorizes the next controlled research phase: collection and causal evaluation of the verified context symbols. It is not evidence of profitability and does not approve a trading model.
