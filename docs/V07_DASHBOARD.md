# MarketFusion V0.7 Local Dashboard

V0.7 is a localhost-only, read-only view over the canonical V0.6C runtime state. It is an advisory research interface. It has no trading routes, account access, order controls, or model-training trigger. Execution remains manual in MT5.

## Start and stop

From the repository root:

```powershell
# Collectors + V0.6 runtime + V0.7 API and web UI
.\scripts\start_marketfusion_full.ps1

# Existing collectors plus V0.6 runtime and V0.7 only
.\scripts\start_marketfusion_dashboard.ps1

# Stop only processes recorded by the V0.7 launchers
.\scripts\stop_marketfusion.ps1
```

Open `http://127.0.0.1:4173`. The API binds to `127.0.0.1:8765`. Port 4173 is the default; `-DashboardPort <port>` overrides it consistently for the full/dashboard launcher and API CORS. Direct Vite/API launchers use the same `MARKETFUSION_DASHBOARD_PORT` environment override. The dashboard launcher avoids duplicate tracked workers, creates PID records under ignored runtime storage, and does not stop MT5 or unrelated Python/Node processes.

## API contract

All routes are GET-only and return JSON. By default, CORS accepts only `http://127.0.0.1:4173` and `http://localhost:4173`; an explicit dashboard-port override updates those two local origins without enabling wildcard CORS.

| Route | Response |
|---|---|
| `/api/health` | API/state validity, freshness, V0.6 status, and disabled-trading flag. |
| `/api/state` | Canonical V0.6C JSON without value mutation. Adds only freshness/contract HTTP headers. Invalid or missing input produces an explicit degraded WAIT state. |
| `/api/operator/status` | DST-aware Forex-week/session state, source timestamp ages, next runtime reassessment, advisory-window state, and separate system-health/trading-state classifications. |
| `/api/market/candles?timeframe=M5&limit=300` | Completed local V0.5A OHLCV candles. Timeframe is exactly M1, M5, M15, or H1; limit is 1–1000. |
| `/api/market/summary` | Latest local bid, ask, mid, and spread when present. |
| `/api/runtime/history?limit=100` | Bounded V0.6 decision history, up to 1000 records. |
| `/api/events` | Audited V0.4D event identifiers, time, and verified consensus fields only. |
| `/api/intelligence` | V0.5B activity, macro values, 24-hour topic counts, and separate provider health. Missing values remain null. |
| `/api/research` | Research-only V0.5C candidate and eligibility metrics; never labels candidates as approved. |
| `/api/system/version` | API/state contract and localhost read-only mode. |

Core state sections are `contract_version`, `system`, `market`, `predictions`, `decision`, `trade_window`, `health`, and `reasons`. The service rejects a state containing account, balance, equity, position, order, credential, token, secret, or API-key fields. Null model probabilities remain null.

## Polling and safety behavior

The UI polls state every three seconds, candles every twenty seconds, and supplementary read-only data (including operator status) every fifteen seconds. Visible clocks, data ages, and countdowns tick locally once per second between server refreshes. Requests are aborted on replacement or unmount, so overlapping state requests are not retained. Browser polling never runs training or queries MT5 directly.

Fresh canonical state displays the backend advisory. A stale or disconnected state forces the visible advisory to WAIT/VERY LOW and suppresses any prior trade window. UTC timestamps are the source of truth and Sri Lanka time is rendered with `Intl.DateTimeFormat` and `Asia/Colombo`.

Run focused validation with:

```powershell
.\scripts\run_v07_tests.ps1
```

The dashboard intentionally renders valid fail-closed decisions such as WAIT, stale market data, weekend closure, and missing horizon models without presenting them as application failure. An approved horizon is displayed only when the runtime contract says `APPROVED_CHAMPION`; unavailable probabilities remain N/A. It never fills events, news, macro values, or trade windows with invented data.
