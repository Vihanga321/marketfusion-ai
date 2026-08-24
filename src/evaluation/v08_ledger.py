"""Immutable, append-only ledger of genuinely observed V0.6 decisions."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from src.evaluation.v08_contract import (
    CONFLICT_DIR, CONTRACT_VERSION, FORBIDDEN_PREDICTION_FRAGMENTS, HORIZONS,
    MAX_FORWARD_RECORD_DELAY_MINUTES, PREDICTION_COLUMNS, PREDICTIONS_FILE, SOURCE_LABEL,
)


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _canonical(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return utc(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical(child) for key, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    return value.item() if hasattr(value, "item") else value


def prediction_sha(record: dict[str, object]) -> str:
    excluded = {"prediction_payload_sha256", "recorded_at_utc"}
    payload = {key: _canonical(record.get(key)) for key in PREDICTION_COLUMNS if key not in excluded}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def validate_prediction_columns(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = [name for name in columns if any(fragment in name.lower() for fragment in FORBIDDEN_PREDICTION_FRAGMENTS)]
    if forbidden:
        raise ValueError("Outcome-derived fields are forbidden in immutable predictions: " + ", ".join(forbidden))


def build_prediction_record(
    state: dict[str, Any], recorded_at_utc: object | None = None, git_commit: str | None = None,
    enforce_forward_deadline: bool = True,
) -> dict[str, object]:
    recorded = pd.Timestamp.now(tz="UTC") if recorded_at_utc is None else utc(recorded_at_utc)
    decision = ((state.get("decision") or {}).get("decision_time") or {}).get("utc")
    generated = ((state.get("system") or {}).get("generated_time") or {}).get("utc")
    if not decision or not generated:
        raise ValueError("V0.6 state lacks decision/source-generation timestamps")
    decision_time, generated_time = utc(decision), utc(generated)
    if generated_time < decision_time or generated_time > recorded + pd.Timedelta(seconds=5):
        raise ValueError("V0.6 source generation violates forward-observation chronology")
    if enforce_forward_deadline and recorded > decision_time + pd.Timedelta(minutes=MAX_FORWARD_RECORD_DELAY_MINUTES):
        raise ValueError("RETROACTIVE_SHADOW_REJECTED: decision was not observed before the forward-recording deadline")
    if enforce_forward_deadline and recorded >= decision_time + pd.Timedelta(minutes=min(HORIZONS)):
        raise ValueError("RETROACTIVE_SHADOW_REJECTED: earliest outcome horizon already matured")

    market = state.get("market") or {}
    regime = market.get("regime") or {}
    spread = market.get("spread") or {}
    predictions = state.get("predictions") or {}
    horizons = predictions.get("horizons") or {}
    fusion = predictions.get("fusion") or {}
    probabilities = fusion.get("probabilities") or {}
    decision_payload = state.get("decision") or {}
    event = state.get("event") or {}
    trade_window = state.get("trade_window") or {}
    health = state.get("health") or {}
    sources = health.get("sources") or {}
    reasons = state.get("reasons") or []
    blockers = sorted(str(item.get("code")) for item in reasons if item.get("blocking"))
    risk = "BLOCKING" if blockers else "CAUTION" if reasons else "NORMAL"
    model_hashes = sorted({str(item.get("feature_contract_hash")) for item in horizons.values() if item.get("feature_contract_hash")})
    prediction_id = sha256(f"{(state.get('system') or {}).get('symbol', 'EURUSD')}|{decision_time.isoformat()}".encode()).hexdigest()
    record: dict[str, object] = {
        "prediction_id": prediction_id, "prediction_payload_sha256": None,
        "source_label": SOURCE_LABEL, "decision_timestamp_utc": decision_time.isoformat(),
        "recorded_at_utc": recorded.isoformat(), "source_generated_at_utc": generated_time.isoformat(),
        "symbol": (state.get("system") or {}).get("symbol", "EURUSD"),
        "market_mid": market.get("close"), "spread_points": spread.get("current_points"),
        "session": market.get("session"), "trend_regime": regime.get("trend_regime"),
        "volatility_regime": regime.get("volatility_regime"), "spread_regime": spread.get("status"),
        "fusion_agreement": fusion.get("status"), "weighted_down": probabilities.get("down"),
        "weighted_neutral": probabilities.get("neutral"), "weighted_up": probabilities.get("up"),
        "advisory": decision_payload.get("action", "WAIT"), "confidence": decision_payload.get("confidence", "VERY_LOW"),
        "risk": risk, "blockers": json.dumps(blockers, separators=(",", ":")),
        "event_risk": event.get("status"), "minutes_to_event": event.get("minutes_to_event"),
        "trade_window_status": trade_window.get("status"), "suggested_start_utc": trade_window.get("start_utc"),
        "suggested_end_utc": trade_window.get("end_utc"),
        "next_reassessment_utc": (decision_payload.get("next_reassessment") or {}).get("utc"),
        "v05b_health": sources.get("v05b_intelligence"), "event_health": sources.get("v05d_event"),
        "model_health": sources.get("v06a_inference"), "git_commit": git_commit or _git_commit(),
        "v06_contract": state.get("contract_version"), "feature_contract_hash": ",".join(model_hashes) or None,
        "manual_execution_only": True,
    }
    for horizon in HORIZONS:
        item = horizons.get(str(horizon)) or {}
        for target, source in (
            ("model_id", "model_id"), ("status", "model_status"), ("p_down", "prob_down"),
            ("p_neutral", "prob_neutral"), ("p_up", "prob_up"), ("shadow_direction", "shadow_direction"),
        ):
            record[f"h{horizon}_{target}"] = item.get(source)
    record["prediction_payload_sha256"] = prediction_sha(record)
    validate_prediction_columns(tuple(record))
    return {key: record.get(key) for key in PREDICTION_COLUMNS}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def append_prediction(
    record: dict[str, object], path: Path = PREDICTIONS_FILE, conflict_dir: Path = CONFLICT_DIR,
) -> dict[str, object]:
    validate_prediction_columns(tuple(record))
    incoming_hash = prediction_sha(record)
    if incoming_hash != record.get("prediction_payload_sha256"):
        raise ValueError("Prediction payload SHA mismatch")
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=PREDICTION_COLUMNS)
    matching = existing.loc[existing.get("decision_timestamp_utc", pd.Series(dtype=str)).astype(str).eq(str(record["decision_timestamp_utc"]))]
    if not matching.empty:
        prior = str(matching.iloc[0]["prediction_payload_sha256"])
        if prior == incoming_hash:
            return {"status": "DEDUPLICATED", "rows_added": 0, "prediction_id": record["prediction_id"]}
        conflict_dir.mkdir(parents=True, exist_ok=True)
        conflict = pd.DataFrame([{**record, "original_payload_sha256": prior, "conflict_status": "PREDICTION_MUTATION_CONFLICT"}])
        conflict_path = conflict_dir / f"{str(record['prediction_id'])[:16]}-{incoming_hash[:12]}.parquet"
        _atomic_parquet(conflict, conflict_path)
        return {"status": "PREDICTION_MUTATION_CONFLICT", "rows_added": 0, "prediction_id": record["prediction_id"]}
    combined = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    _atomic_parquet(combined[list(PREDICTION_COLUMNS)], path)
    return {"status": "APPENDED", "rows_added": 1, "prediction_id": record["prediction_id"]}


def ingest_state(
    state: dict[str, Any], recorded_at_utc: object | None = None, path: Path = PREDICTIONS_FILE,
    conflict_dir: Path = CONFLICT_DIR, git_commit: str | None = None,
) -> dict[str, object]:
    decision = (((state.get("decision") or {}).get("decision_time") or {}).get("utc"))
    already_recorded = False
    if decision and path.exists():
        existing = pd.read_parquet(path, columns=["decision_timestamp_utc"])
        already_recorded = existing["decision_timestamp_utc"].astype(str).eq(utc(decision).isoformat()).any()
    record = build_prediction_record(
        state, recorded_at_utc, git_commit, enforce_forward_deadline=not already_recorded,
    )
    return append_prediction(record, path, conflict_dir)
