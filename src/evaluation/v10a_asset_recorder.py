"""Asset-isolated no-execution forward shadow recorder for XAUUSD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import pandas as pd

from src.assets.contracts import asset_paths, normalize_asset_id
from src.assets.registry import approved_champions
from src.evaluation.v08_ledger import ingest_state, recording_eligibility
from src.evaluation.v08_observations import observations_frame, shadow_summary
from src.evaluation.v10b_xauusd_outcomes import attach_matured_outcomes


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _commit() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _performance(summary: dict[str, Any], champion_count: int) -> dict[str, object]:
    horizons: dict[str, object] = {}
    for horizon in (15, 60, 240):
        item = (summary.get("horizons") or {}).get(str(horizon)) or {}
        total = int(item.get("total_recorded") or 0)
        evaluated = int(item.get("EVALUATED") or 0)
        predicted_neutral = int(item.get("predicted_neutral_count") or 0)
        directional_calls = max(0, total - predicted_neutral)
        horizons[str(horizon)] = {
            "matured_count": evaluated,
            "directional_calls": directional_calls,
            "wait_rate": None,
            "directional_accuracy": item.get("accuracy"),
            "balanced_accuracy": item.get("balanced_accuracy"),
            "macro_f1": item.get("macro_f1"),
            "log_loss": item.get("mean_log_loss"),
            "brier": item.get("mean_brier_score"),
            "sample_status": item.get("sample_status", "INSUFFICIENT_DATA"),
        }
    return {
        "recorded_predictions": int(summary.get("total_observations") or 0) // 3,
        "performance_observations": int(summary.get("performance_observations") or 0),
        "matured_outcomes": int((summary.get("counts") or {}).get("EVALUATED") or 0),
        "horizons": horizons,
        "wait": summary.get("wait") or {"wait_rate": None},
        "calibration_status": "INSUFFICIENT_SAMPLE" if champion_count else "NO_APPROVED_MODEL_DATA",
        "market_drift_status": "INSUFFICIENT_DATA",
        "model_drift_status": "INSUFFICIENT_DATA",
    }


def run_once(asset_id: str = "XAUUSD", now_utc: object | None = None) -> dict[str, object]:
    asset = normalize_asset_id(asset_id)
    if asset != "XAUUSD":
        raise RuntimeError("V1.0B asset recorder is restricted to XAUUSD")
    paths = asset_paths(asset)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    champions = approved_champions(asset)
    state = _json(paths.runtime / "latest_marketfusion_state.json")

    if not state:
        cycle: dict[str, object] = {
            "status": "STATE_UNAVAILABLE", "rows_added": 0, "reason": "ASSET_STATE_MISSING",
        }
    elif str((state.get("system") or {}).get("symbol", "")).upper() != asset:
        cycle = {
            "status": "STATE_CONTRACT_REJECTED", "rows_added": 0,
            "reason": "CROSS_SYMBOL_STATE_REJECTED",
        }
    else:
        eligibility = recording_eligibility(state)
        if not eligibility["eligible"]:
            cycle = {
                "status": eligibility["status"], "rows_added": 0,
                "reason": eligibility["reason"],
            }
        else:
            cycle = ingest_state(
                state, now, paths.shadow_predictions,
                paths.evaluation / "conflicts", _commit(),
            )
            cycle["reason"] = eligibility["reason"]

    outcome_cycle = attach_matured_outcomes(
        predictions_path=paths.shadow_predictions,
        target_path=paths.features / "target_research.parquet",
        outcomes_path=paths.shadow_outcomes,
        now_utc=now,
    )
    frame = observations_frame(
        predictions_path=paths.shadow_predictions,
        outcomes_path=paths.shadow_outcomes,
        now_utc=now,
    )
    summary = shadow_summary(frame, now)
    conflicts = list((paths.evaluation / "conflicts").glob("*.parquet")) if (paths.evaluation / "conflicts").exists() else []
    fatal = (
        bool(conflicts)
        or cycle.get("status") == "STATE_CONTRACT_REJECTED"
        or int(outcome_cycle.get("contract_mismatch", 0)) > 0
        or int(outcome_cycle.get("early_violations", 0)) > 0
    )
    if fatal:
        status = "FAIL"
    elif champions:
        status = "PASS_SHADOW_EVALUATION"
    else:
        status = "PASS_MONITORING_NO_CHAMPION"

    payload: dict[str, object] = {
        "contract_version": "v0.8-forward-shadow-monitor-v1",
        "symbol": asset,
        "updated_at_utc": now.isoformat(),
        "status": status,
        "ledger_cycle": cycle,
        "outcome_cycle": outcome_cycle,
        "live_shadow": summary,
        "performance": _performance(summary, len(champions)),
        "champions": {
            str(horizon): (champions.get(horizon) or {}).get("model_id", "NONE")
            for horizon in (15, 60, 240)
        },
        "provider_health": {
            "status": "NOT_CONFIGURED", "providers": {}, "uptime_percentage": {},
        },
        "research": {
            "status": "AVAILABLE_NOT_RUN", "experiments": 0,
            "best_experiment": None, "decision": "INSUFFICIENT_DATA",
        },
        "manual_execution_only": True,
        "trading_enabled": False,
    }
    _atomic_json(payload, paths.evaluation / "latest_status.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD", choices=("XAUUSD",))
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    while True:
        print(json.dumps(run_once(args.symbol), indent=2, default=str))
        if not args.continuous:
            return 0
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
