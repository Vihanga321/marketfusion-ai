"""Attach matured V0.5A outcomes to immutable V0.6 shadow decisions.

Outputs are explicitly SHADOW PERFORMANCE diagnostics, never live performance or
profitability evidence.
"""
from __future__ import annotations

import json
import pandas as pd

from src.marketdata.v05a_contract import FEATURE_FILE
from src.runtime.v06c_contract import SHADOW_PERFORMANCE_FILE, STATE_HISTORY_FILE


def evaluate_matured_shadow() -> dict[str, object]:
    if not STATE_HISTORY_FILE.exists() or not FEATURE_FILE.exists():
        return {"status": "NO_MATURED_SHADOW_ROWS", "rows_added": 0, "label": "SHADOW PERFORMANCE"}
    history = pd.read_parquet(STATE_HISTORY_FILE)
    market = pd.read_parquet(FEATURE_FILE)
    market["decision_timestamp_utc"] = pd.to_datetime(market["decision_timestamp_utc"], utc=True, errors="coerce")
    existing = pd.read_parquet(SHADOW_PERFORMANCE_FILE) if SHADOW_PERFORMANCE_FILE.exists() else pd.DataFrame()
    keys = {(str(decision), int(horizon)) for decision, horizon in zip(existing.get("decision_timestamp_utc", []), existing.get("horizon_minutes", []))}
    rows: list[dict[str, object]] = []
    for _, item in history.iterrows():
        decision = pd.Timestamp(item["decision_timestamp_utc"])
        decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
        matched = market.loc[market["decision_timestamp_utc"] == decision]
        if matched.empty:
            continue
        state = json.loads(item["state_json"])
        probabilities = ((state.get("predictions") or {}).get("horizons") or {})
        source = matched.iloc[-1]
        for horizon in (15, 60, 240):
            outcome = source.get(f"outcome_direction_{horizon}m")
            matured = source.get(f"outcome_matured_at_utc_{horizon}m")
            key = (str(item["decision_timestamp_utc"]), horizon)
            if pd.isna(outcome) or pd.isna(matured) or key in keys:
                continue
            prediction = probabilities.get(str(horizon), {})
            rows.append({
                "decision_timestamp_utc": str(item["decision_timestamp_utc"]), "horizon_minutes": horizon,
                "model_id": prediction.get("model_id"), "shadow_direction": prediction.get("shadow_direction", "WAIT"),
                "outcome_direction": outcome, "outcome_matured_at_utc": matured,
                "evaluation_label": "SHADOW PERFORMANCE", "live_performance_claim": False,
            })
            keys.add(key)
    if rows:
        result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
        SHADOW_PERFORMANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SHADOW_PERFORMANCE_FILE.with_suffix(".parquet.tmp")
        result.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(SHADOW_PERFORMANCE_FILE)
    return {"status": "PASS", "rows_added": len(rows), "label": "SHADOW PERFORMANCE"}


if __name__ == "__main__":
    print(json.dumps(evaluate_matured_shadow(), indent=2, default=str))
