"""Immutable, append-only ledger of genuinely observed V0.6 decisions."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from src.evaluation.v08_contract import (
    APPROVED_MODEL_STATUSES, CONFLICT_DIR, CONTRACT_VERSION, FORBIDDEN_PREDICTION_FRAGMENTS,
    HORIZONS, LEGACY_PREDICTION_COLUMNS, LIVE_DATA_FRESHNESS,
    MAX_FORWARD_RECORD_DELAY_MINUTES, OBSERVATION_CONTRACT_VERSION, PREDICTION_COLUMNS,
    PREDICTIONS_FILE, SOURCE_LABEL,
    V1_PREDICTION_COLUMNS,
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
    if not isinstance(value, (dict, list, tuple)):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return utc(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical(child) for key, child in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    return value.item() if hasattr(value, "item") else value


def prediction_sha(record: dict[str, object]) -> str:
    excluded = {"prediction_payload_sha256", "recorded_at_utc"}
    version = record.get("observation_contract_version")
    try:
        has_version = version is not None and not bool(pd.isna(version))
    except (TypeError, ValueError):
        has_version = bool(version)
    if version == OBSERVATION_CONTRACT_VERSION:
        columns = PREDICTION_COLUMNS
    elif has_version:
        columns = V1_PREDICTION_COLUMNS
    else:
        columns = LEGACY_PREDICTION_COLUMNS
    payload = {key: _canonical(record.get(key)) for key in columns if key not in excluded}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def validate_probabilities(item: dict[str, Any], horizon: int) -> None:
    """Reject malformed approved-model output; missing models must stay null."""
    status = str(item.get("model_status") or "NO_APPROVED_MODEL")
    values = [item.get("prob_down"), item.get("prob_neutral"), item.get("prob_up")]
    if status not in APPROVED_MODEL_STATUSES:
        if any(value is not None and not pd.isna(value) for value in values):
            raise ValueError(f"Non-approved {horizon}m model published probabilities")
        return
    if any(value is None or pd.isna(value) for value in values):
        raise ValueError(f"Approved {horizon}m model lacks complete probabilities")
    probabilities = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError(f"Approved {horizon}m probabilities are outside [0,1]")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError(f"Approved {horizon}m probabilities do not sum to one")


def recording_eligibility(state: dict[str, Any]) -> dict[str, object]:
    """Gate persistence to a newly observed, completed, causal feature decision."""
    decision = (((state.get("decision") or {}).get("decision_time") or {}).get("utc"))
    generated = (((state.get("system") or {}).get("generated_time") or {}).get("utc"))
    market = state.get("market") or {}
    freshness = str((market.get("freshness") or {}).get("status") or "UNAVAILABLE").upper()
    session = str(market.get("session") or "UNKNOWN").upper()
    if not decision or not generated:
        return {"eligible": False, "status": "STATE_CONTRACT_REJECTED", "reason": "MISSING_DECISION_TIMESTAMP"}
    if session == "WEEKEND":
        return {"eligible": False, "status": "NO_NEW_FRESH_DECISION_EVENT", "reason": "MARKET_CLOSED_WEEKEND"}
    if freshness not in LIVE_DATA_FRESHNESS:
        return {"eligible": False, "status": "NO_NEW_FRESH_DECISION_EVENT", "reason": f"MARKET_DATA_{freshness}"}
    observed = (market.get("freshness") or {}).get("observed_at_utc")
    if observed and utc(observed) != utc(decision):
        return {"eligible": False, "status": "STATE_CONTRACT_REJECTED", "reason": "FEATURE_DECISION_TIMESTAMP_MISMATCH"}
    horizons = ((state.get("predictions") or {}).get("horizons") or {})
    blocker_codes = {str(item.get("code")) for item in state.get("reasons") or [] if item.get("blocking")}
    if "INSUFFICIENT_FEATURES" in blocker_codes:
        return {"eligible": False, "status": "NO_NEW_VALID_DECISION_EVENT", "reason": "FEATURE_ROW_INCOMPLETE"}
    try:
        for horizon in HORIZONS:
            validate_probabilities(horizons.get(str(horizon)) or {}, horizon)
    except ValueError as exc:
        return {"eligible": False, "status": "STATE_CONTRACT_REJECTED", "reason": str(exc)}
    return {"eligible": True, "status": "ELIGIBLE_NEW_DECISION_EVENT", "reason": "LIVE_FRESH_COMPLETED_CAUSAL_FEATURE_ROW"}


def validate_prediction_columns(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = [name for name in columns if any(fragment in name.lower() for fragment in FORBIDDEN_PREDICTION_FRAGMENTS)]
    if forbidden:
        raise ValueError("Outcome-derived fields are forbidden in immutable predictions: " + ", ".join(forbidden))


def build_prediction_record(
    state: dict[str, Any], recorded_at_utc: object | None = None, git_commit: str | None = None,
    enforce_forward_deadline: bool = True, engine_observed_at_utc: object | None = None,
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
    try:
        from src.engines.engine_layer import decision_engine_snapshot
        engine_snapshot = decision_engine_snapshot(decision_time, engine_observed_at_utc or recorded)
    except Exception as exc:
        engine_snapshot = {
            "contract_version": None, "decision_timestamp_utc": decision_time.isoformat(),
            "engines": {}, "chart_pattern": None, "structure_event": None,
            "liquidity_event": None, "observational_only": True,
            "status": "UNAVAILABLE", "reason": type(exc).__name__,
        }
    chart_pattern = engine_snapshot.get("chart_pattern") or {}
    structure_event = engine_snapshot.get("structure_event") or {}
    liquidity_event = engine_snapshot.get("liquidity_event") or {}
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
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "market_status": "CLOSED_WEEKEND" if str(market.get("session", "")).upper() == "WEEKEND" else "OPEN",
        "active_sessions": json.dumps([value.strip().replace(" ", "_") for value in str(market.get("session") or "UNKNOWN").split("+")], separators=(",", ":")),
        "market_bid": market.get("bid"), "market_ask": market.get("ask"),
        "data_freshness": (market.get("freshness") or {}).get("status"),
        "market_age_seconds": None if (market.get("freshness") or {}).get("age_minutes") is None else float((market.get("freshness") or {}).get("age_minutes")) * 60.0,
        "latest_tick_utc": market.get("latest_tick_utc"), "latest_m5_utc": market.get("latest_m5_utc") or decision_time.isoformat(),
        "latest_m15_utc": market.get("latest_m15_utc"), "feature_timestamp_utc": (market.get("freshness") or {}).get("observed_at_utc") or decision_time.isoformat(),
        "feature_complete": "INSUFFICIENT_FEATURES" not in blockers, "source_fresh": str((market.get("freshness") or {}).get("status") or "").upper() in LIVE_DATA_FRESHNESS,
        "target_event_guard": event.get("status"), "v06b_gate": decision_payload.get("gate"),
        "v06c_status": (state.get("system") or {}).get("status"), "v06c_gate": decision_payload.get("gate"),
        "trading_enabled": False, "manual_confirmation_required": True,
        "entry_reference_type": "COMPLETED_M5_CLOSE", "exit_reference_type": "EXACT_FUTURE_COMPLETED_M5_CLOSE",
        "engine_snapshot_contract": engine_snapshot.get("contract_version"),
        "engine_snapshot_json": json.dumps(_canonical(engine_snapshot), sort_keys=True, separators=(",", ":")),
        "chart_pattern_type": chart_pattern.get("pattern_type"), "chart_pattern_score": chart_pattern.get("score"),
        "chart_pattern_timeframe": chart_pattern.get("timeframe"),
        "nearest_support_distance_price": engine_snapshot.get("nearest_support_distance_price"),
        "nearest_resistance_distance_price": engine_snapshot.get("nearest_resistance_distance_price"),
        "structure_event_type": structure_event.get("event_type"), "structure_event_time_utc": structure_event.get("detected_at_utc"),
        "liquidity_event_type": liquidity_event.get("event_type"), "liquidity_event_time_utc": liquidity_event.get("detected_at_utc"),
        "engine_observational_only": True,
    }
    for horizon in HORIZONS:
        item = horizons.get(str(horizon)) or {}
        for target, source in (
            ("model_id", "model_id"), ("status", "model_status"), ("p_down", "prob_down"),
            ("p_neutral", "prob_neutral"), ("p_up", "prob_up"), ("shadow_direction", "shadow_direction"),
        ):
            record[f"h{horizon}_{target}"] = item.get(source)
        record[f"h{horizon}_gate"] = item.get("decision_gate")
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
    if not matching.empty and "symbol" in matching:
        matching = matching.loc[matching["symbol"].astype(str).eq(str(record.get("symbol")))]
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
    original_observed_at = None
    if decision and path.exists():
        existing = pd.read_parquet(path, columns=["decision_timestamp_utc", "recorded_at_utc"])
        matching = existing.loc[existing["decision_timestamp_utc"].astype(str).eq(utc(decision).isoformat())]
        already_recorded = not matching.empty
        if already_recorded:
            original_observed_at = matching.iloc[0]["recorded_at_utc"]
    record = build_prediction_record(
        state, recorded_at_utc, git_commit, enforce_forward_deadline=not already_recorded,
        engine_observed_at_utc=original_observed_at,
    )
    return append_prediction(record, path, conflict_dir)
