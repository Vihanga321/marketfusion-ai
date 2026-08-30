"""Read-only per-horizon observation views over the immutable V0.8 ledgers."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.v08_contract import (
    APPROVED_MODEL_STATUSES, HORIZONS, MIN_FORWARD_SHADOW_ROWS,
    OBSERVATION_CONTRACT_VERSION, OUTCOMES_FILE, PREDICTIONS_FILE,
)
from src.evaluation.v08_ledger import utc
from src.runtime.v06c_contract import LOCAL_TIMEZONE

CLASS_LABELS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


def observation_id(symbol: str, decision_timestamp_utc: object, horizon: int, model_id: object) -> str:
    identity = f"{symbol}|{utc(decision_timestamp_utc).isoformat()}|{horizon}|{model_id or 'NO_APPROVED_MODEL'}"
    return sha256(identity.encode("utf-8")).hexdigest()


def _value(row: pd.Series, name: str, default: object = None) -> object:
    value = row.get(name, default)
    try:
        return default if value is None or bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    try:
        parsed = json.loads(str(value))
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _finite_probabilities(prediction: pd.Series, horizon: int) -> list[float] | None:
    names = [f"h{horizon}_p_down", f"h{horizon}_p_neutral", f"h{horizon}_p_up"]
    values = [_value(prediction, name) for name in names]
    if any(value is None for value in values):
        return None
    probabilities = [float(value) for value in values]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities):
        return None
    return probabilities if abs(sum(probabilities) - 1.0) <= 1e-6 else None


def _legacy_source_fresh(prediction: pd.Series) -> bool:
    explicit = _value(prediction, "source_fresh")
    if explicit is not None:
        return bool(explicit)
    if str(_value(prediction, "session", "")).upper() == "WEEKEND":
        return False
    try:
        delay = utc(_value(prediction, "recorded_at_utc")) - utc(_value(prediction, "decision_timestamp_utc"))
        return pd.Timedelta(0) <= delay <= pd.Timedelta(minutes=10)
    except Exception:
        return False


def observations_frame(
    predictions: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    now_utc: object | None = None,
    predictions_path: Path = PREDICTIONS_FILE,
    outcomes_path: Path = OUTCOMES_FILE,
) -> pd.DataFrame:
    """Expand one immutable decision record into three horizon observation views."""
    if predictions is None:
        predictions = pd.read_parquet(predictions_path) if predictions_path.exists() else pd.DataFrame()
    if outcomes is None:
        outcomes = pd.read_parquet(outcomes_path) if outcomes_path.exists() else pd.DataFrame()
    if predictions.empty:
        return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)
    outcome_map: dict[tuple[str, int], pd.Series] = {}
    if not outcomes.empty:
        for _, item in outcomes.iterrows():
            outcome_map[(str(item.get("prediction_id")), int(item.get("horizon_minutes")))] = item
    rows: list[dict[str, object]] = []
    for _, prediction in predictions.iterrows():
        decision = utc(_value(prediction, "decision_timestamp_utc"))
        symbol = str(_value(prediction, "symbol", "EURUSD"))
        source_fresh = _legacy_source_fresh(prediction)
        feature_complete = bool(_value(prediction, "feature_complete", True))
        blockers = _json_list(_value(prediction, "blockers", "[]"))
        for horizon in HORIZONS:
            model_id = _value(prediction, f"h{horizon}_model_id")
            model_status = str(_value(prediction, f"h{horizon}_status", "NO_APPROVED_MODEL"))
            approved = model_status in APPROVED_MODEL_STATUSES and model_id is not None
            probabilities = _finite_probabilities(prediction, horizon) if approved else None
            valid = approved and probabilities is not None and source_fresh and feature_complete
            predicted_index = int(np.argmax(probabilities)) if probabilities is not None else None
            due = decision + pd.Timedelta(minutes=horizon) if approved else None
            outcome = outcome_map.get((str(_value(prediction, "prediction_id")), horizon))
            # Legacy V0.8 created target rows for NO_APPROVED_MODEL horizons.  Keep
            # those immutable ledger rows as evidence, but never present or score
            # them as live model evaluations.
            if not approved:
                outcome = None
            outcome_status = str(_value(outcome, "outcome_status", "")) if outcome is not None else ""
            if not approved:
                evaluation_status = "NO_APPROVED_MODEL"
            elif outcome_status == "INVALID":
                evaluation_status = "INVALID"
            elif outcome_status == "MATURED":
                evaluation_status = "EVALUATED"
            elif due is not None and now >= due:
                evaluation_status = "PENDING_DATA"
            else:
                evaluation_status = "PENDING"
            actual_index_value = _value(outcome, "target_class") if outcome is not None else None
            actual_index = int(actual_index_value) if actual_index_value is not None else None
            direction_correct = bool(predicted_index == actual_index) if predicted_index is not None and actual_index is not None else None
            brier = None
            log_loss = None
            if probabilities is not None and actual_index is not None:
                expected = np.eye(3)[actual_index]
                brier = float(np.sum((np.asarray(probabilities) - expected) ** 2))
                log_loss = float(-math.log(max(probabilities[actual_index], 1e-15)))
            advisory = str(_value(prediction, "advisory", "WAIT"))
            rows.append({
                "observation_id": observation_id(symbol, decision, horizon, model_id),
                "prediction_id": str(_value(prediction, "prediction_id")),
                "contract_version": str(_value(prediction, "observation_contract_version", OBSERVATION_CONTRACT_VERSION)),
                "captured_at_utc": _value(prediction, "recorded_at_utc"),
                "decision_timestamp_utc": decision.isoformat(),
                "decision_timestamp_lkt": decision.tz_convert(LOCAL_TIMEZONE).isoformat(),
                "evaluation_due_utc": None if due is None else due.isoformat(),
                "evaluated_at_utc": _value(outcome, "evaluated_at_utc", _value(outcome, "matured_at_utc")) if outcome is not None else None,
                "symbol": symbol, "bid": _value(prediction, "market_bid"), "ask": _value(prediction, "market_ask"),
                "mid": _value(prediction, "market_mid"), "spread_points": _value(prediction, "spread_points"),
                "market_status": _value(prediction, "market_status", "CLOSED_WEEKEND" if str(_value(prediction, "session", "")).upper() == "WEEKEND" else "OPEN"),
                "session": _value(prediction, "session"), "active_sessions": _json_list(_value(prediction, "active_sessions")),
                "data_freshness": _value(prediction, "data_freshness", "FRESH" if source_fresh else "STALE"),
                "market_age_seconds": _value(prediction, "market_age_seconds"),
                "latest_tick_utc": _value(prediction, "latest_tick_utc"), "latest_m5_utc": _value(prediction, "latest_m5_utc", decision.isoformat()),
                "latest_m15_utc": _value(prediction, "latest_m15_utc"), "feature_timestamp_utc": _value(prediction, "feature_timestamp_utc", decision.isoformat()),
                "horizon_minutes": horizon, "model_id": model_id, "model_status": model_status,
                "prob_down": None if probabilities is None else probabilities[0],
                "prob_neutral": None if probabilities is None else probabilities[1],
                "prob_up": None if probabilities is None else probabilities[2],
                "predicted_class": None if predicted_index is None else CLASS_LABELS[predicted_index],
                "shadow_direction": _value(prediction, f"h{horizon}_shadow_direction"),
                "model_confidence": None if probabilities is None else max(probabilities),
                "v06a_gate": _value(prediction, f"h{horizon}_gate"),
                "v06b_action": advisory, "v06b_confidence": _value(prediction, "confidence"),
                "v06b_gate": _value(prediction, "v06b_gate", _value(prediction, "v06c_gate")),
                "blocking_reasons": blockers, "event_risk": _value(prediction, "event_risk"),
                "v06c_status": _value(prediction, "v06c_status"), "v06c_decision": advisory,
                "v06c_gate": _value(prediction, "v06c_gate"), "trading_window_status": _value(prediction, "trade_window_status"),
                "trading_enabled": False, "manual_confirmation_required": True,
                "evaluation_status": evaluation_status,
                "entry_reference_price": _value(outcome, "entry_mid_or_reference_price") if outcome is not None else _value(prediction, "market_mid"),
                "exit_reference_price": _value(outcome, "future_price") if outcome is not None else None,
                "entry_reference_type": _value(outcome, "entry_reference_type", _value(prediction, "entry_reference_type", "COMPLETED_M5_CLOSE")) if outcome is not None else _value(prediction, "entry_reference_type", "COMPLETED_M5_CLOSE"),
                "exit_reference_type": _value(outcome, "exit_reference_type", _value(prediction, "exit_reference_type", "EXACT_FUTURE_COMPLETED_M5_CLOSE")) if outcome is not None else _value(prediction, "exit_reference_type", "EXACT_FUTURE_COMPLETED_M5_CLOSE"),
                "actual_return": _value(outcome, "raw_return") if outcome is not None else None,
                "actual_class": None if actual_index is None else CLASS_LABELS[actual_index],
                "direction_correct": direction_correct, "probability_score": brier,
                "log_loss_contribution": log_loss,
                "evaluation_reason": _value(outcome, "evaluation_reason", "WAITING_FOR_EXACT_V05A_TARGET_EVENT") if outcome is not None else ("NO_APPROVED_MODEL" if not approved else "WAITING_FOR_EXACT_V05A_TARGET_EVENT"),
                "target_event_guard": _value(prediction, "target_event_guard"),
                "feature_complete": feature_complete, "source_fresh": source_fresh,
                "valid_for_performance_evaluation": valid,
                "invalid_reason": _value(outcome, "evaluation_reason") if outcome_status == "INVALID" else None,
                "advisory_wait": advisory == "WAIT",
                "underlying_direction_correct": direction_correct,
                "wait_protected": (not direction_correct) if advisory == "WAIT" and direction_correct is not None else None,
                "volatility_regime": _value(prediction, "volatility_regime"),
                "trend_regime": _value(prediction, "trend_regime"),
            })
    return pd.DataFrame(rows).sort_values(["decision_timestamp_utc", "horizon_minutes"], ascending=[False, True]).reset_index(drop=True)


def _macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(np.unique(actual)) < 2:
        return None
    scores = []
    for label in np.unique(np.concatenate([actual, predicted])):
        tp = int(((actual == label) & (predicted == label)).sum())
        fp = int(((actual != label) & (predicted == label)).sum())
        fn = int(((actual == label) & (predicted != label)).sum())
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else None


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    labels = np.unique(actual)
    return None if len(labels) < 2 else float(np.mean([(predicted[actual == label] == label).mean() for label in labels]))


def _performance(frame: pd.DataFrame, minimum: int = MIN_FORWARD_SHADOW_ROWS) -> dict[str, object]:
    evaluated = frame.loc[frame["evaluation_status"].eq("EVALUATED") & frame["valid_for_performance_evaluation"]].copy()
    count = len(evaluated)
    reliable = count >= minimum
    correct = int(evaluated["direction_correct"].fillna(False).sum()) if count else 0
    actual = evaluated["actual_class"].map({"DOWN": 0, "NEUTRAL": 1, "UP": 2}).to_numpy(dtype=int) if count else np.asarray([], dtype=int)
    predicted = evaluated["predicted_class"].map({"DOWN": 0, "NEUTRAL": 1, "UP": 2}).to_numpy(dtype=int) if count else np.asarray([], dtype=int)
    result: dict[str, object] = {
        "sample_count": count, "sample_status": "AVAILABLE" if reliable else "INSUFFICIENT_SAMPLE" if count else "INSUFFICIENT_DATA",
        "correct_count": correct, "incorrect_count": count - correct,
        "accuracy": float(correct / count) if reliable else None,
        "balanced_accuracy": _balanced_accuracy(actual, predicted) if reliable else None,
        "macro_f1": _macro_f1(actual, predicted) if reliable else None,
        "mean_log_loss": float(evaluated["log_loss_contribution"].mean()) if reliable else None,
        "mean_brier_score": float(evaluated["probability_score"].mean()) if reliable else None,
        "average_confidence": float(evaluated["model_confidence"].mean()) if reliable else None,
    }
    return result


def shadow_summary(frame: pd.DataFrame | None = None, now_utc: object | None = None) -> dict[str, object]:
    frame = observations_frame(now_utc=now_utc) if frame is None else frame.copy()
    empty_counts = {state: 0 for state in ("PENDING", "PENDING_DATA", "EVALUATED", "INVALID", "NO_APPROVED_MODEL")}
    if frame.empty:
        return {
            "contract_version": OBSERVATION_CONTRACT_VERSION, "status": "NO_LIVE_SHADOW_OBSERVATIONS_YET",
            "generated_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)).isoformat(),
            "recorder": {"status": "RUNNING", "trigger": "NEW_COMPLETED_CAUSAL_DECISION_ROW", "last_cycle_reason": "NO_OBSERVATIONS_RECORDED"},
            "total_observations": 0, "performance_observations": 0, "counts": empty_counts,
            "horizons": {str(horizon): {"horizon_minutes": horizon, "total_recorded": 0, **empty_counts, **_performance(pd.DataFrame(columns=["evaluation_status", "valid_for_performance_evaluation"]))} for horizon in HORIZONS},
            "sessions": [], "gates": [], "wait": {"decisions": 0, "evaluated": 0, "protected": 0, "underlying_correct": 0},
            "latest_observation": None, "trading_enabled": False, "manual_execution_only": True,
        }
    counts = {state: int(frame["evaluation_status"].eq(state).sum()) for state in empty_counts}
    horizons: dict[str, object] = {}
    for horizon in HORIZONS:
        subset = frame.loc[frame["horizon_minutes"].eq(horizon)]
        status_counts = {state: int(subset["evaluation_status"].eq(state).sum()) for state in empty_counts}
        values: dict[str, object] = {"horizon_minutes": horizon, "total_recorded": len(subset), **status_counts, **_performance(subset)}
        for label in ("DOWN", "NEUTRAL", "UP"):
            values[f"predicted_{label.lower()}_count"] = int(subset["predicted_class"].eq(label).sum())
            values[f"actual_{label.lower()}_count"] = int(subset["actual_class"].eq(label).sum())
        horizons[str(horizon)] = values
    sessions = []
    for session, subset in frame.loc[frame["model_status"].isin(APPROVED_MODEL_STATUSES)].groupby("session", dropna=False):
        sessions.append({"session": "UNKNOWN" if pd.isna(session) else str(session), **_performance(subset, 30)})
    gate_counts: dict[str, int] = {}
    decision_rows = frame.drop_duplicates("prediction_id")
    for _, item in decision_rows.iterrows():
        reasons = item["blocking_reasons"] if isinstance(item["blocking_reasons"], list) else []
        gate_value = _value(item, "v06c_gate", "UNAVAILABLE")
        codes = reasons or [str(gate_value)]
        for code in set(codes):
            gate_counts[str(code)] = gate_counts.get(str(code), 0) + 1
    gates = [{"gate": gate, "count": count} for gate, count in sorted(gate_counts.items(), key=lambda item: (-item[1], item[0]))]
    waits = frame.loc[frame["advisory_wait"] & frame["model_status"].isin(APPROVED_MODEL_STATUSES)]
    wait_evaluated = waits.loc[waits["evaluation_status"].eq("EVALUATED")]
    latest_candidates = frame.loc[frame["model_status"].isin(APPROVED_MODEL_STATUSES)]
    latest = None if latest_candidates.empty else latest_candidates.iloc[0].to_dict()
    return {
        "contract_version": OBSERVATION_CONTRACT_VERSION,
        "status": "AVAILABLE" if not latest_candidates.empty else "NO_LIVE_SHADOW_OBSERVATIONS_YET",
        "generated_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)).isoformat(),
        "recorder": {"status": "RUNNING", "trigger": "NEW_COMPLETED_CAUSAL_DECISION_ROW", "last_cycle_reason": "SEE_V08_STATUS"},
        "total_observations": len(frame),
        "performance_observations": int(frame["model_status"].isin(APPROVED_MODEL_STATUSES).sum()),
        "counts": counts, "horizons": horizons, "sessions": sessions, "gates": gates,
        "wait": {"decisions": len(waits), "evaluated": len(wait_evaluated),
                 "protected": int(wait_evaluated["wait_protected"].fillna(False).sum()),
                 "underlying_correct": int(wait_evaluated["underlying_direction_correct"].fillna(False).sum())},
        "latest_observation": latest, "trading_enabled": False, "manual_execution_only": True,
    }


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        clean: dict[str, object] = {}
        for key, value in record.items():
            try:
                clean[key] = None if value is None or bool(pd.isna(value)) else value.item() if hasattr(value, "item") else value
            except (TypeError, ValueError):
                clean[key] = value
        records.append(clean)
    return records
