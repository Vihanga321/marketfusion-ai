# MarketFusion V0.6 — Complete Real-Time Shadow Prediction

V0.6 turns the existing causal data and controlled-learning pipeline into a single, inspectable shadow advisory state. It remains research-only: no component can place, modify, or close a trade, and every non-WAIT bias requires manual confirmation.

## Runtime flow

```text
V0.5A completed market features ─┐
V0.5B causal intelligence ──────┤
V0.5D/V0.4D exact event state ─┤
V0.4B historical memory ────────┼─> V0.6B risk fusion ─> V0.6C dashboard state
V0.5C CHAMPION registry ─> V0.6A┘
```

V0.6A rereads the V0.5C registry on every cycle. Only a `CHAMPION` is eligible, and its feature contract, feature order, artifact path, and SHA-256 must verify before inference. CHALLENGER and REJECTED artifacts are never operational fallbacks.

V0.6B classifies causal market regime, session, spread, exact-target event risk, and source freshness. It fuses 15m/60m/240m approved probabilities with weights 0.40/0.40/0.20, normalized across valid horizons. A direction needs at least two approved horizons, probability 0.55, margin 0.08, agreement, and every blocking risk gate to pass. V0.5B and Historical Event Memory are context only; neither can manufacture or override direction.

V0.6C publishes a stable JSON object with `system`, `market`, `predictions`, `decision`, `trade_window`, `event`, `intelligence`, `health`, and `reasons`. UTC remains canonical. Asia/Colombo timestamps are rendered with `zoneinfo`. History records only the first state for a unique decision timestamp and contains no outcome fields.

## Current expected result

V0.5C has no approved champions. The correct result is therefore:

```text
decision: WAIT
confidence: VERY_LOW
gate: WAIT_NO_MODEL
status: PASS_FAIL_CLOSED_NO_CHAMPION
trading_enabled: false
manual_confirmation_required: true
```

This is a safety result, not evidence of profitability, an edge, or model performance.

## Commands

```powershell
.\scripts\run_v06a_tests.ps1
.\scripts\run_v06b_tests.ps1
.\scripts\run_v06a_once.ps1
.\scripts\run_v06b_once.ps1
.\scripts\run_v06c_once.ps1
.\scripts\show_v06_status.ps1
```

Run only the unified foreground loop:

```powershell
.\scripts\start_v06_runtime.ps1
```

Start the local V0.5A/V0.5B collectors only when their validated PID files are not active, then run the unified loop:

```powershell
.\scripts\start_marketfusion_runtime.ps1
```

Generated runtime files are ignored under `data/runtime/v06/`. The optional evaluator writes only explicitly labelled `SHADOW PERFORMANCE` diagnostics after V0.5A outcomes mature.

## Next stage

V0.7 should build a read-only dashboard over the stable V0.6C state contract, with reason drill-down, freshness/health display, horizon probability charts, and shadow-performance views. It must not introduce execution or account access.
