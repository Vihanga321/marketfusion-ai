# MarketFusion V0.6A — Real-Time Shadow Inference

## Purpose

V0.6A is the first live inference runtime. It consumes the latest completed, causal V0.5A EUR/USD feature row and will load only V0.5C models that are explicitly registered as `CHAMPION`.

At the current validated project state there are no champions. Therefore the correct live behavior is fail-closed: every horizon reports `NO_APPROVED_MODEL` and `WAIT`. This is a successful safety behavior, not a model failure.

V0.6A is **shadow-only**. It cannot place, modify, or close trades.

## Data path

```text
V0.5A completed M5 decision features
        ↓
V0.5C runtime registry
        ↓
latest approved CHAMPION per horizon
        ↓
artifact SHA-256 + feature-contract verification
        ↓
15m / 60m / 240m calibrated probabilities
        ↓
staleness gate
        ↓
audited V0.4D target-event guard
        ↓
shadow direction or WAIT
```

The inference feature order must exactly equal the V0.5C training feature registry. Partial live candles are not reconstructed for inference because the candidates were trained on completed M5 decisions.

## Fail-closed gates

V0.6A returns `WAIT` when any of these apply:

- no approved V0.5C champion exists;
- the champion feature contract differs from the current V0.5C contract;
- the model artifact is missing or its SHA-256 differs from the registry;
- the loaded model's feature order differs from the training contract;
- the latest V0.5A feature row is stale or future-dated;
- an exact audited V0.4D target release is inside the configured event window;
- audited target-event data are unavailable/invalid;
- the probability vector is malformed;
- the top class is NEUTRAL;
- directional probability/margin do not pass the explicit research-only threshold.

V0.6A's target-event guard currently covers only the exact audited V0.4D CPI/Core CPI/NFP/Unemployment identities. It must not be interpreted as a complete news/event-risk engine. Broader event fusion belongs in V0.6B.

## Runtime files

Generated local files are git-ignored:

```text
data/inference/v06a/
├── latest_shadow_prediction.json
└── shadow_prediction_history.parquet
```

The history records at most one first shadow result per completed decision timestamp. The latest JSON is refreshed each cycle.

## Commands

Focused deterministic tests plus repository safety:

```powershell
.\scripts\run_v06a_tests.ps1
```

One shadow cycle:

```powershell
.\scripts\run_v06a_once.ps1
```

Continuous shadow runtime:

```powershell
.\scripts\start_v06a_shadow.ps1
```

Status from another terminal:

```powershell
.\scripts\show_v06a_status.ps1
```

## Expected current result

Until V0.5C promotes a model, the expected result is:

```text
V06A_STATUS: PASS_FAIL_CLOSED_NO_CHAMPION
15m: WAIT_NO_APPROVED_MODEL
60m: WAIT_NO_APPROVED_MODEL
240m: WAIT_NO_APPROVED_MODEL
trading: DISABLED / SHADOW ONLY
```

When a future daily V0.5C run legitimately promotes a champion, V0.6A will verify its registry metadata and artifact SHA before generating shadow probabilities. A promoted model does not automatically authorize trading.

## Next stage

V0.6B should combine approved model probabilities with broader event risk, V0.5B intelligence availability, Historical Event Memory, no-trade/risk policy, and suggested observation/reassessment windows. It should remain advisory/manual-trading until a separate execution-safety stage is intentionally designed and validated.
