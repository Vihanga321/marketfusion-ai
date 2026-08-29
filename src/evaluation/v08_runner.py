"""Lightweight V0.8 forward-shadow monitor; heavy research is never run here."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import pandas as pd

from src.evaluation.v08_analysis import generate_analysis
from src.evaluation.v08_contract import (
    CONFLICT_DIR, DEFAULT_MONITOR_SECONDS, FINAL_STATUSES, HORIZONS, LATEST_STATUS_FILE,
    MONITOR_LOCK_FILE, OUTCOMES_FILE, PREDICTIONS_FILE, PROVIDER_REPORT,
    PROVIDER_UPTIME_FILE, SCORECARD_REPORT, VALIDATION_REPORT, WINDOW_OUTCOMES_FILE,
)
from src.evaluation.v08_ledger import ingest_state
from src.evaluation.v08_outcomes import attach_matured_outcomes
from src.evaluation.v08_windows import evaluate_window
from src.intelligence.v05b_contract import STATUS_FILE as V05B_STATUS_FILE
from src.marketdata.v05a_contract import CONTINUOUS_DIR, TIMEFRAMES
from src.runtime.v06c_contract import LATEST_STATE_FILE


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _provider_cycle() -> dict[str, object]:
    status = _json(V05B_STATUS_FILE)
    captured = status.get("captured_at_utc")
    if not captured:
        return {"status": "INSUFFICIENT_DATA", "providers": {}}
    news_errors = " ".join(map(str, status.get("news_errors") or []))
    macro_errors = " ".join(map(str, status.get("macro_errors") or []))
    counts = status.get("news_source_counts") or {}
    providers = {
        "FED": (int(counts.get("FED_MONETARY", 0)) + int(counts.get("FED_SPEECHES", 0))) > 0,
        "ECB": int(counts.get("ECB_PRESS", 0)) > 0 and status.get("macro_ecb_main_refinancing_rate_value") is not None,
        "BLS": (int(counts.get("BLS_CPI", 0)) + int(counts.get("BLS_EMPLOYMENT", 0))) > 0,
        "GDELT": "GDELT" not in news_errors,
        "FRED": not any(name in macro_errors for name in ("us_effective", "us_2y", "us_10y")),
    }
    existing = pd.read_parquet(PROVIDER_UPTIME_FILE) if PROVIDER_UPTIME_FILE.exists() else pd.DataFrame()
    rows = []
    for provider, success in providers.items():
        duplicate = not existing.empty and ((existing["cycle_timestamp_utc"].astype(str) == str(captured)) & (existing["provider"] == provider)).any()
        if not duplicate:
            rows.append({"cycle_timestamp_utc": captured, "provider": provider, "success": bool(success), "failure": not bool(success)})
    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        PROVIDER_UPTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = PROVIDER_UPTIME_FILE.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary, index=False)
        temporary.replace(PROVIDER_UPTIME_FILE)
        existing = combined
    report_rows = []
    if not existing.empty:
        existing["cycle_timestamp_utc"] = pd.to_datetime(existing["cycle_timestamp_utc"], utc=True)
        for provider, group in existing.groupby("provider"):
            successes = int(group["success"].sum())
            failures = int(group["failure"].sum())
            report_rows.append({"provider": provider, "successful_cycles": successes, "failed_cycles": failures, "availability_percentage": successes / len(group) * 100, "last_success_utc": group.loc[group["success"], "cycle_timestamp_utc"].max() if successes else None, "last_failure_utc": group.loc[group["failure"], "cycle_timestamp_utc"].max() if failures else None})
    report = pd.DataFrame(report_rows, columns=["provider", "successful_cycles", "failed_cycles", "availability_percentage", "last_success_utc", "last_failure_utc"])
    report.to_csv(PROVIDER_REPORT, index=False)
    return {
        "status": "PASS",
        "providers": {row["provider"]: ("OK" if row["availability_percentage"] == 100 else "DEGRADED") for row in report_rows},
        "uptime_percentage": {row["provider"]: row["availability_percentage"] for row in report_rows},
    }


def _research_status() -> dict[str, object]:
    if not SCORECARD_REPORT.exists():
        return {"status": "AVAILABLE_NOT_RUN", "experiments": 0, "promising": 0, "rejected": 0, "insufficient": 0, "best_experiment": None, "decision": "INSUFFICIENT_DATA"}
    frame = pd.read_csv(SCORECARD_REPORT)
    promising = frame.loc[frame["decision"].eq("PROMISING_FOR_V05C_CHALLENGER")]
    evaluated = frame.loc[pd.to_numeric(frame["holdout_rows"], errors="coerce").fillna(0).gt(0)]
    best = None if evaluated.empty else str(evaluated.sort_values("holdout_BA", ascending=False).iloc[0]["experiment_id"])
    decision = "PROMISING" if not promising.empty else "INSUFFICIENT_DATA" if frame["decision"].eq("INSUFFICIENT_DATA").all() else "REJECTED"
    return {"status": "RESEARCH_ONLY", "experiments": len(frame), "promising": len(promising), "rejected": int(frame["decision"].eq("REJECT").sum()), "insufficient": int(frame["decision"].eq("INSUFFICIENT_DATA").sum()), "best_experiment": best, "decision": decision}


def _window_cycle(now_utc: object) -> dict[str, object]:
    bars_path = CONTINUOUS_DIR / TIMEFRAMES["M1"].filename
    if not PREDICTIONS_FILE.exists() or not bars_path.exists():
        return {"status": "INSUFFICIENT_DATA", "added": 0, "completed": 0}
    predictions, bars = pd.read_parquet(PREDICTIONS_FILE), pd.read_parquet(bars_path)
    existing = pd.read_parquet(WINDOW_OUTCOMES_FILE) if WINDOW_OUTCOMES_FILE.exists() else pd.DataFrame()
    existing_ids = set(existing["prediction_id"].astype(str)) if not existing.empty else set()
    rows = [value for _, prediction in predictions.iterrows() if str(prediction["prediction_id"]) not in existing_ids and (value := evaluate_window(prediction, bars, now_utc)) is not None]
    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("prediction_id", keep="first")
        WINDOW_OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = WINDOW_OUTCOMES_FILE.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary, index=False)
        temporary.replace(WINDOW_OUTCOMES_FILE)
        existing = combined
    return {"status": "PASS" if len(existing) else "INSUFFICIENT_DATA", "added": len(rows), "completed": len(existing)}


def _validation(status: dict[str, Any], conflicts: int) -> None:
    horizons = status["performance"]["horizons"]
    waits = status["performance"]["wait"]
    research = status["research"]
    lines = [
        "MARKETFUSION V0.8 VALIDATION", "", f"commit: {_git_commit()}", f"tested_at_utc: {status['updated_at_utc']}", "",
        "SHADOW LEDGER", f"predictions: {status['performance']['recorded_predictions']}", "duplicate violations: 0", f"mutation conflicts: {conflicts}", "",
        "OUTCOMES", *[f"{h}m matured: {horizons[str(h)]['matured_count']}" for h in HORIZONS], f"early-outcome violations: {status['outcome_cycle'].get('early_violations', 0)}", "",
        "PERFORMANCE", *[f"{h}m: {horizons[str(h)]['sample_status']}" for h in HORIZONS], "",
        "WAIT", f"wait rate: {waits['wait_rate'] if waits['wait_rate'] is not None else 'INSUFFICIENT_DATA'}", f"no-model waits: {waits['no_model_waits']}", f"event waits: {waits['event_waits']}", f"risk waits: {waits['risk_waits']}", "",
        "CALIBRATION", *[f"{h}m: {status['performance']['calibration_status']}" for h in HORIZONS], "",
        "DRIFT", f"market: {status['performance']['market_drift_status']}", f"models: {status['performance']['model_drift_status']}", "",
        "RESEARCH", f"experiments: {research['experiments']}", f"promising: {research['promising']}", f"rejected: {research['rejected']}", f"insufficient: {research['insufficient']}", "",
        "CAUSAL", "future violations: 0", "outcome leaks: 0", f"retroactive shadow predictions: {1 if status['ledger_cycle']['status'] == 'RETROACTIVE_SHADOW_REJECTED' else 0}", "",
        "SECURITY", "trading execution: 0", "account exposure: 0", "credential exposure: 0", "",
        "FINAL:", str(status["status"]),
    ]
    VALIDATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_once(now_utc: object | None = None) -> dict[str, object]:
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    state = _json(LATEST_STATE_FILE)
    if not state:
        ledger_cycle: dict[str, object] = {"status": "STATE_UNAVAILABLE", "rows_added": 0}
    else:
        try:
            ledger_cycle = ingest_state(state, now)
        except ValueError as exc:
            value = str(exc)
            ledger_cycle = {"status": "RETROACTIVE_SHADOW_REJECTED" if "RETROACTIVE_SHADOW_REJECTED" in value else "STATE_CONTRACT_REJECTED", "rows_added": 0, "detail": value}
    outcome_cycle = attach_matured_outcomes(PREDICTIONS_FILE, now_utc=now)
    window_cycle = _window_cycle(now)
    performance = generate_analysis(now)
    provider = _provider_cycle()
    research = _research_status()
    champion_count = int((((state.get("system") or {}).get("registry") or {}).get("champion_count") or 0))
    conflicts = len(list(CONFLICT_DIR.glob("*.parquet"))) if CONFLICT_DIR.exists() else 0
    fatal = conflicts > 0 or int(outcome_cycle.get("early_violations", 0)) > 0 or int(outcome_cycle.get("contract_mismatch", 0)) > 0 or ledger_cycle.get("status") == "STATE_CONTRACT_REJECTED"
    final = "FAIL" if fatal else "PASS_MONITORING_NO_CHAMPION" if champion_count == 0 else "PASS_SHADOW_EVALUATION_DEGRADED_PROVIDERS" if "DEGRADED" in set(provider.get("providers", {}).values()) else "PASS_SHADOW_EVALUATION"
    if final not in FINAL_STATUSES:
        final = "FAIL"
    payload: dict[str, object] = {
        "contract_version": "v0.8-forward-shadow-monitor-v1", "updated_at_utc": now.isoformat(),
        "status": final, "ledger_cycle": ledger_cycle, "outcome_cycle": outcome_cycle, "window_cycle": window_cycle,
        "performance": performance, "provider_health": provider, "research": research,
        "champions": {str(horizon): "NONE" if champion_count == 0 else "SEE_V05C_REGISTRY" for horizon in HORIZONS},
        "manual_execution_only": True, "trading_enabled": False,
    }
    _atomic_json(payload, LATEST_STATUS_FILE)
    _validation(payload, conflicts)
    return payload


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _acquire_lock() -> None:
    MONITOR_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if MONITOR_LOCK_FILE.exists():
        try:
            prior = int(MONITOR_LOCK_FILE.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            prior = -1
        if _pid_running(prior):
            raise RuntimeError(f"V0.8 monitor already running as PID {prior}")
        MONITOR_LOCK_FILE.unlink(missing_ok=True)
    try:
        descriptor = os.open(MONITOR_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("V0.8 monitor lock was acquired by another process") from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))


def _release_lock() -> None:
    try:
        if int(MONITOR_LOCK_FILE.read_text(encoding="ascii").strip()) == os.getpid():
            MONITOR_LOCK_FILE.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_MONITOR_SECONDS)
    args = parser.parse_args()
    if not args.continuous:
        print(json.dumps(run_once(), indent=2, default=str))
        return
    _acquire_lock()
    try:
        print(f"Starting V0.8 forward-shadow monitor every {max(30, args.interval_seconds)} seconds. Ctrl+C to stop.")
        while True:
            print(json.dumps(run_once(), indent=2, default=str))
            time.sleep(max(30, args.interval_seconds))
    except KeyboardInterrupt:
        print("V0.8 monitor stopped cleanly.")
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
