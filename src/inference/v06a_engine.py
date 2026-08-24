"""Fail-closed V0.6A shadow inference engine for approved V0.5C champions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd

from src.inference.v06a_contract import (
    CLASS_NAMES,
    FEATURE_SOURCE,
    HIGH_CONFIDENCE_MARGIN,
    HIGH_CONFIDENCE_PROBABILITY,
    HISTORY_FILE,
    LATEST_STATUS_FILE,
    MAX_FEATURE_AGE_MINUTES,
    MEDIUM_CONFIDENCE_MARGIN,
    MEDIUM_CONFIDENCE_PROBABILITY,
    MIN_DIRECTION_MARGIN,
    MIN_DIRECTION_PROBABILITY,
    REQUIRED_FEATURES,
    ROOT,
    SYMBOL,
    TARGET_EVENT_BLOCK_AFTER_MINUTES,
    TARGET_EVENT_BLOCK_BEFORE_MINUTES,
    VERIFIED_CONSENSUS_FILE,
    V05C_MODEL_ROOT,
    V05C_REGISTRY,
)
from src.learning.v05c_dataset import feature_contract_hash, validate_availability, validate_feature_registry
from src.learning.v05c_registry import file_sha256, load_registry, validate_registry


@dataclass(frozen=True)
class EventGuard:
    status: str
    blocked: bool
    nearest_event_name: str | None = None
    nearest_event_timestamp_utc: str | None = None
    minutes_to_event: float | None = None


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def latest_market_snapshot(frame: pd.DataFrame, now_utc: object) -> tuple[pd.Series, float]:
    if frame.empty:
        raise RuntimeError("V0.5A feature store is empty")
    validate_feature_registry(REQUIRED_FEATURES)
    validate_availability(frame)
    missing = sorted({"decision_timestamp_utc", *REQUIRED_FEATURES} - set(frame.columns))
    if missing:
        raise RuntimeError("V0.5A feature store missing columns: " + ", ".join(missing))
    work = frame.copy()
    work["decision_timestamp_utc"] = pd.to_datetime(work["decision_timestamp_utc"], utc=True, errors="raise")
    work = work.sort_values("decision_timestamp_utc")
    row = work.iloc[-1]
    now = _utc(now_utc)
    decision = _utc(row["decision_timestamp_utc"])
    age_minutes = float((now - decision).total_seconds() / 60.0)
    if age_minutes < -0.05:
        raise RuntimeError("Latest V0.5A decision timestamp is in the future")
    return row, age_minutes


def select_latest_champion(records: list[dict[str, object]], horizon_minutes: int) -> dict[str, object] | None:
    candidates = [
        record for record in records
        if int(record.get("horizon_minutes", -1)) == horizon_minutes
        and str(record.get("promotion_status", "")) == "CHAMPION"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: _utc(item["created_at_utc"]))


def _artifact_path(record: dict[str, object]) -> Path:
    candidate = (ROOT / str(record["artifact_path"])).resolve()
    root = V05C_MODEL_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Champion artifact path escapes models/v05c") from exc
    return candidate


def load_verified_champion(record: dict[str, object], horizon_minutes: int) -> Any:
    if int(record["horizon_minutes"]) != horizon_minutes:
        raise RuntimeError("Champion horizon mismatch")
    expected_hash = feature_contract_hash(tuple(REQUIRED_FEATURES), horizon_minutes)
    if str(record["feature_contract_hash"]) != expected_hash:
        raise RuntimeError("Champion feature contract hash mismatch")
    if int(record["feature_count"]) != len(REQUIRED_FEATURES):
        raise RuntimeError("Champion feature count mismatch")
    path = _artifact_path(record)
    if not path.exists():
        raise RuntimeError(f"Champion artifact missing: {path}")
    if file_sha256(path) != str(record["artifact_sha256"]):
        raise RuntimeError("Champion artifact SHA256 mismatch")
    model = joblib.load(path)
    names = tuple(getattr(model, "feature_names", ()))
    if names != tuple(REQUIRED_FEATURES):
        raise RuntimeError("Loaded champion feature ordering mismatch")
    return model


def target_event_guard_from_frame(frame: pd.DataFrame, now_utc: object) -> EventGuard:
    now = _utc(now_utc)
    if frame.empty:
        return EventGuard("NO_VERIFIED_TARGET_EVENTS_IN_FILE", False)
    if "event_timestamp_utc" not in frame.columns:
        raise RuntimeError("Verified consensus file missing event_timestamp_utc")
    work = frame.copy()
    work["event_timestamp_utc"] = pd.to_datetime(work["event_timestamp_utc"], utc=True, errors="raise")
    deltas = (work["event_timestamp_utc"] - now).dt.total_seconds() / 60.0
    risk = deltas.between(-TARGET_EVENT_BLOCK_AFTER_MINUTES, TARGET_EVENT_BLOCK_BEFORE_MINUTES, inclusive="both")
    if risk.any():
        subset = work.loc[risk].copy()
        subset["abs_delta"] = deltas.loc[risk].abs()
        nearest = subset.sort_values("abs_delta").iloc[0]
        minutes = float((nearest["event_timestamp_utc"] - now).total_seconds() / 60.0)
        return EventGuard(
            "AUDITED_TARGET_EVENT_RISK_WINDOW",
            True,
            str(nearest.get("event_name") or "TARGET_EVENT"),
            _utc(nearest["event_timestamp_utc"]).isoformat(),
            minutes,
        )
    future = work.loc[work["event_timestamp_utc"] > now].sort_values("event_timestamp_utc")
    if future.empty:
        return EventGuard("NO_FUTURE_VERIFIED_TARGET_EVENT", False)
    nearest = future.iloc[0]
    minutes = float((nearest["event_timestamp_utc"] - now).total_seconds() / 60.0)
    return EventGuard(
        "VERIFIED_TARGET_EVENT_NOT_IN_BLOCK_WINDOW",
        False,
        str(nearest.get("event_name") or "TARGET_EVENT"),
        _utc(nearest["event_timestamp_utc"]).isoformat(),
        minutes,
    )


def load_target_event_guard(now_utc: object) -> EventGuard:
    if not VERIFIED_CONSENSUS_FILE.exists():
        return EventGuard("EVENT_DATA_UNAVAILABLE", True)
    try:
        return target_event_guard_from_frame(pd.read_csv(VERIFIED_CONSENSUS_FILE), now_utc)
    except Exception:
        return EventGuard("EVENT_DATA_INVALID", True)


def _confidence_band(max_probability: float, margin: float) -> str:
    if max_probability >= HIGH_CONFIDENCE_PROBABILITY and margin >= HIGH_CONFIDENCE_MARGIN:
        return "HIGH"
    if max_probability >= MEDIUM_CONFIDENCE_PROBABILITY and margin >= MEDIUM_CONFIDENCE_MARGIN:
        return "MEDIUM"
    return "LOW"


def probability_decision(probabilities: np.ndarray) -> dict[str, object]:
    vector = np.asarray(probabilities, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all() or (vector < 0).any():
        raise RuntimeError("Invalid three-class probability vector")
    total = float(vector.sum())
    if total <= 0:
        raise RuntimeError("Probability vector has zero mass")
    vector = vector / total
    order = np.argsort(vector)[::-1]
    top, second = int(order[0]), int(order[1])
    maximum = float(vector[top])
    margin = float(vector[top] - vector[second])
    result = {
        "prob_down": float(vector[0]),
        "prob_neutral": float(vector[1]),
        "prob_up": float(vector[2]),
        "top_class": CLASS_NAMES[top],
        "top_probability": maximum,
        "probability_margin": margin,
        "probability_confidence_band": _confidence_band(maximum, margin),
        "shadow_direction": "WAIT",
        "decision_gate": "WAIT_NEUTRAL" if top == 1 else "WAIT_LOW_CONFIDENCE",
    }
    if top != 1 and maximum >= MIN_DIRECTION_PROBABILITY and margin >= MIN_DIRECTION_MARGIN:
        result["shadow_direction"] = CLASS_NAMES[top]
        result["decision_gate"] = "SHADOW_DIRECTION_ELIGIBLE"
    return result


def infer_horizon(model: Any, feature_row: pd.Series) -> dict[str, object]:
    x = pd.DataFrame([{name: feature_row[name] for name in REQUIRED_FEATURES}], columns=list(REQUIRED_FEATURES))
    raw = np.asarray(model.predict_proba(x), dtype=float)
    if raw.shape != (1, 3):
        raise RuntimeError(f"Champion returned unexpected probability shape: {raw.shape}")
    return probability_decision(raw[0])


def _history_row(payload: dict[str, object]) -> pd.DataFrame:
    row: dict[str, object] = {
        "captured_at_utc": payload.get("captured_at_utc"),
        "decision_timestamp_utc": payload.get("decision_timestamp_utc"),
        "market_age_minutes": payload.get("market_age_minutes"),
        "market_fresh": payload.get("market_fresh"),
        "event_guard_status": (payload.get("event_guard") or {}).get("status"),
        "overall_status": payload.get("status"),
    }
    for horizon, result in (payload.get("horizons") or {}).items():
        prefix = f"h{horizon}m_"
        for key in (
            "model_id", "model_status", "prob_down", "prob_neutral", "prob_up",
            "top_class", "top_probability", "probability_margin",
            "probability_confidence_band", "shadow_direction", "decision_gate",
        ):
            row[prefix + key] = result.get(key)
    return pd.DataFrame([row])


def _write_runtime(payload: dict[str, object]) -> None:
    LATEST_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = LATEST_STATUS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(LATEST_STATUS_FILE)
    if not payload.get("decision_timestamp_utc"):
        return

    incoming = _history_row(payload)
    if HISTORY_FILE.exists():
        history = pd.read_parquet(HISTORY_FILE, engine="pyarrow")
        if not history.empty and str(history.iloc[-1]["decision_timestamp_utc"]) == str(incoming.iloc[0]["decision_timestamp_utc"]):
            return
        history = pd.concat([history, incoming], ignore_index=True)
    else:
        history = incoming
    temporary_history = HISTORY_FILE.with_suffix(".parquet.tmp")
    history.to_parquet(temporary_history, index=False, engine="pyarrow")
    temporary_history.replace(HISTORY_FILE)


def run_shadow_cycle(now_utc: object | None = None, persist: bool = True) -> dict[str, object]:
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc)
    payload: dict[str, object] = {
        "contract": "v0.6a-realtime-shadow-inference-v1",
        "symbol": SYMBOL,
        "captured_at_utc": now.isoformat(),
        "trading": "DISABLED",
        "shadow_only": True,
        "horizons": {},
        "errors": [],
    }
    if not FEATURE_SOURCE.exists():
        payload.update({"status": "FAIL_CLOSED", "reason": "V0.5A feature store missing"})
        if persist:
            _write_runtime(payload)
        return payload

    try:
        source = pd.read_parquet(FEATURE_SOURCE, engine="pyarrow")
        feature_row, age = latest_market_snapshot(source, now)
    except Exception as exc:
        payload.update({"status": "FAIL_CLOSED", "reason": f"market data gate: {type(exc).__name__}: {exc}"})
        if persist:
            _write_runtime(payload)
        return payload

    market_fresh = 0 <= age <= MAX_FEATURE_AGE_MINUTES
    payload["decision_timestamp_utc"] = _utc(feature_row["decision_timestamp_utc"]).isoformat()
    payload["market_age_minutes"] = age
    payload["market_fresh"] = market_fresh
    event_guard = load_target_event_guard(now)
    payload["event_guard"] = event_guard.__dict__

    try:
        records = load_registry(V05C_REGISTRY)
        validate_registry(records, verify_artifacts=False)
    except Exception as exc:
        payload.update({"status": "FAIL_CLOSED", "reason": f"registry gate: {type(exc).__name__}: {exc}"})
        if persist:
            _write_runtime(payload)
        return payload

    champion_count = 0
    inference_failures = 0
    for horizon in (15, 60, 240):
        record = select_latest_champion(records, horizon)
        if record is None:
            payload["horizons"][str(horizon)] = {
                "model_id": None,
                "model_status": "NO_APPROVED_MODEL",
                "shadow_direction": "WAIT",
                "decision_gate": "WAIT_NO_APPROVED_MODEL",
            }
            continue
        champion_count += 1
        result: dict[str, object] = {
            "model_id": str(record["model_id"]), "model_status": "APPROVED_CHAMPION",
            "model_created_at_utc": str(record["created_at_utc"]),
            "artifact_sha256": str(record["artifact_sha256"]),
            "feature_contract_hash": str(record["feature_contract_hash"]),
        }
        try:
            model = load_verified_champion(record, horizon)
            result.update(infer_horizon(model, feature_row))
            if not market_fresh:
                result["shadow_direction"] = "WAIT"
                result["decision_gate"] = "WAIT_STALE_MARKET_FEATURES"
            elif event_guard.blocked:
                result["shadow_direction"] = "WAIT"
                result["decision_gate"] = f"WAIT_{event_guard.status}"
        except Exception as exc:
            inference_failures += 1
            result.update({
                "model_status": "CHAMPION_REJECTED_AT_RUNTIME",
                "shadow_direction": "WAIT",
                "decision_gate": "WAIT_MODEL_INTEGRITY_FAILURE",
                "error": f"{type(exc).__name__}: {exc}",
            })
        payload["horizons"][str(horizon)] = result

    if inference_failures:
        payload["status"] = "FAIL_CLOSED"
        payload["reason"] = f"{inference_failures} approved champion(s) failed runtime integrity/inference"
    elif champion_count == 0:
        payload["status"] = "PASS_FAIL_CLOSED_NO_CHAMPION"
        payload["reason"] = "V0.5C has no approved champion; shadow runtime correctly returns WAIT"
    else:
        payload["status"] = "PASS_SHADOW_RUNNING"
        payload["reason"] = "approved champion inference available in research-only shadow mode"

    if persist:
        _write_runtime(payload)
    return payload
