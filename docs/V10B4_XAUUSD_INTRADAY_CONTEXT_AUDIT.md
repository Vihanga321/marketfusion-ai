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

## Broker clock normalization

The audit does not blindly treat MT5 Unix-like timestamps as already normalized UTC. It first measures the connected broker/server clock offset from the live XAUUSD tick.

The measured offset must be:
- within a bounded timezone-like range;
- close to a 15-minute timezone quantum;
- accompanied by only a small residual relative to the capture clock.

If calibration passes, the raw timestamps are preserved and normalized UTC timestamps are derived using only that measured offset. If calibration cannot be established, the audit fails closed with `RESOLVE_MT5_BROKER_CLOCK_BEFORE_CONTEXT_RESEARCH`.

This allows a coherent broker offset such as +03:00 to be handled without silently hard-coding a timezone correction.

## Symbol identity guards

Ticker text alone is not enough. MT5 catalogs can contain stocks or ETFs whose ticker collides with an intermarket alias.

The hardened audit rejects cases such as:
- an ETF ticker `USDX` when the contract is not a real U.S. Dollar Index;
- W&T Offshore Inc under ticker `WTI`, which is not WTI crude oil;
- Treasury ETFs such as IEF being mislabeled as a direct US10Y contract.

FX and precious-metal pairs require the expected base/profit currencies. Index, oil, and Treasury families require semantic contract evidence and reject unrelated stock/ETF collisions.

## Audit evidence

For each verified family, MarketFusion checks:

1. broker symbol candidate identity;
2. `symbol_select` availability;
3. raw MT5 tick timestamp and normalized non-clamped tick age;
4. bounded completed M5 history using position 1, never the forming bar;
5. bounded completed M15 history using position 1;
6. duplicate/future timestamp diagnostics after measured broker-clock normalization.

A genuinely future tick/bar timestamp still fails closed for that context family.

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
