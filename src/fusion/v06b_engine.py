"""Deterministic V0.6B fusion engine. Advisory only; no execution path exists."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fusion.v06b_contract import (
    CONTRACT_VERSION, HORIZONS, HORIZON_WEIGHTS, LATEST_ADVISORY_FILE,
    MANUAL_CONFIRMATION_REQUIRED, MAX_EVENT_CAPTURE_AGE_HOURS,
    MAX_INFERENCE_AGE_MINUTES, MAX_INTELLIGENCE_AGE_MINUTES,
    MAX_MARKET_AGE_MINUTES, MAX_MODEL_ARTIFACT_AGE_DAYS, MIN_APPROVED_HORIZONS, MIN_DIRECTION_MARGIN,
    MIN_DIRECTION_PROBABILITY, MIN_TRADE_WINDOW_MINUTES, REASON_PRIORITIES,
    SYMBOL, TRADING_ENABLED,
)
from src.fusion.v06b_risk import classify_event_risk, classify_market_regime, classify_session, classify_spread, freshness, utc
from src.inference.v06a_contract import VERIFIED_CONSENSUS_FILE
from src.inference.v06a_engine import run_shadow_cycle
from src.intelligence.v05b_contract import CONTEXT_HISTORY_FILE, STATUS_FILE as INTEL_STATUS_FILE
from src.marketdata.v05a_contract import FEATURE_FILE
from src.memory.v04b_contract import REACTION_PATH


def reason(code: str, message: str, source: str, blocking: bool, evidence: object = None) -> dict[str, object]:
    return {"code": code, "priority": REASON_PRIORITIES[code], "blocking": blocking, "source": source, "message": message, "evidence": evidence}


def _probability_vector(item: dict[str, Any]) -> np.ndarray | None:
    values = np.asarray([item.get("prob_down"), item.get("prob_neutral"), item.get("prob_up")], dtype=object)
    try:
        vector = values.astype(float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all() or (vector < 0).any() or float(vector.sum()) <= 0:
        return None
    return vector / vector.sum()


def fuse_probabilities(horizons: dict[str, dict[str, Any]]) -> dict[str, object]:
    valid: list[tuple[int, np.ndarray, str]] = []
    malformed: list[int] = []
    for horizon in HORIZONS:
        item = horizons.get(str(horizon), {})
        if item.get("model_status") != "APPROVED_CHAMPION":
            continue
        vector = _probability_vector(item)
        if vector is None:
            malformed.append(horizon)
            continue
        top = int(np.argmax(vector))
        order = np.sort(vector)[::-1]
        direction = ("DOWN", "NEUTRAL", "UP")[top]
        eligible = direction != "NEUTRAL" and float(order[0]) >= MIN_DIRECTION_PROBABILITY and float(order[0] - order[1]) >= MIN_DIRECTION_MARGIN
        valid.append((horizon, vector, direction if eligible else "WAIT"))
    if not valid:
        return {"valid_horizons": [], "malformed_horizons": malformed, "probabilities": None, "direction": "WAIT", "confidence": "VERY_LOW", "status": "NO_APPROVED_MODEL"}
    total_weight = sum(HORIZON_WEIGHTS[h] for h, _, _ in valid)
    combined = sum(HORIZON_WEIGHTS[h] * vector for h, vector, _ in valid) / total_weight
    directional = [direction for _, _, direction in valid if direction != "WAIT"]
    disagreement = len(set(directional)) > 1
    order = np.argsort(combined)[::-1]
    top, second = int(order[0]), int(order[1])
    maximum, margin = float(combined[top]), float(combined[top] - combined[second])
    fused_direction = ("DOWN", "NEUTRAL", "UP")[top]
    eligible = (
        len(valid) >= MIN_APPROVED_HORIZONS and not malformed and not disagreement
        and fused_direction != "NEUTRAL" and maximum >= MIN_DIRECTION_PROBABILITY
        and margin >= MIN_DIRECTION_MARGIN and bool(directional)
    )
    confidence = "HIGH" if eligible and len(valid) == 3 and maximum >= 0.70 and margin >= 0.20 else "MEDIUM" if eligible else "LOW"
    status = "DIRECTIONAL" if eligible else "HORIZON_DISAGREEMENT" if disagreement else "INSUFFICIENT_APPROVED_HORIZONS" if len(valid) < MIN_APPROVED_HORIZONS else "LOW_CONFIDENCE"
    return {
        "valid_horizons": [h for h, _, _ in valid], "malformed_horizons": malformed,
        "normalized_weights": {str(h): HORIZON_WEIGHTS[h] / total_weight for h, _, _ in valid},
        "probabilities": {"down": float(combined[0]), "neutral": float(combined[1]), "up": float(combined[2])},
        "top_probability": maximum, "margin": margin,
        "direction": fused_direction if eligible else "WAIT", "confidence": confidence,
        "status": status, "horizon_directions": {str(h): d for h, _, d in valid},
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_events() -> pd.DataFrame | None:
    try:
        return pd.read_csv(VERIFIED_CONSENSUS_FILE) if VERIFIED_CONSENSUS_FILE.exists() else None
    except Exception:
        return None


def _latest_context(decision: pd.Timestamp) -> tuple[dict[str, Any], dict[str, Any]]:
    status = _load_json(INTEL_STATUS_FILE)
    if not CONTEXT_HISTORY_FILE.exists():
        return {}, status
    try:
        frame = pd.read_parquet(CONTEXT_HISTORY_FILE)
        frame["decision_timestamp_utc"] = pd.to_datetime(frame["decision_timestamp_utc"], utc=True, errors="coerce")
        eligible = frame.loc[frame["decision_timestamp_utc"] <= decision].sort_values("decision_timestamp_utc")
        return ({} if eligible.empty else eligible.iloc[-1].to_dict()), status
    except Exception:
        return {}, status


def _memory_context() -> dict[str, object]:
    if not REACTION_PATH.exists():
        return {"status": "UNAVAILABLE", "rows": 0, "directional_override": False, "outcomes_used_for_direction": False}
    try:
        frame = pd.read_parquet(REACTION_PATH, columns=["event_type"])
        return {
            "status": "AVAILABLE_CONTEXT_ONLY", "rows": len(frame),
            "event_types": sorted(frame["event_type"].dropna().astype(str).unique().tolist()),
            "directional_override": False, "outcomes_used_for_direction": False,
        }
    except Exception:
        return {"status": "DEGRADED", "rows": None, "directional_override": False, "outcomes_used_for_direction": False}


def _next_m5(now: pd.Timestamp) -> pd.Timestamp:
    floor = now.floor("5min")
    return floor + pd.Timedelta(minutes=5)


def trade_window(now: pd.Timestamp, direction: str, valid_horizons: list[int], event: dict[str, Any]) -> dict[str, object]:
    if direction == "WAIT":
        return {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None, "horizon_minutes": None}
    horizon = min(valid_horizons)
    start = _next_m5(now)
    end = start + pd.Timedelta(minutes=horizon)
    nearest = event.get("nearest_event") or {}
    if nearest.get("event_timestamp_utc"):
        block_start = utc(nearest["event_timestamp_utc"]) - pd.Timedelta(minutes=15)
        if start < block_start < end:
            end = block_start
    duration = float((end - start).total_seconds() / 60)
    if duration < MIN_TRADE_WINDOW_MINUTES:
        return {"status": "TOO_SHORT_WAIT", "start_utc": None, "end_utc": None, "horizon_minutes": horizon}
    return {"status": "ADVISORY_WINDOW", "start_utc": start.isoformat(), "end_utc": end.isoformat(), "horizon_minutes": horizon}


def build_advisory(
    market: pd.DataFrame, shadow: dict[str, Any], events: pd.DataFrame | None,
    context: dict[str, Any], intel_status: dict[str, Any], now_utc: object,
) -> dict[str, object]:
    now = utc(now_utc)
    rows = market.copy()
    rows["decision_timestamp_utc"] = pd.to_datetime(rows["decision_timestamp_utc"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["decision_timestamp_utc"]).sort_values("decision_timestamp_utc")
    if rows.empty:
        raise RuntimeError("V0.5A feature store has no valid decision rows")
    row = rows.iloc[-1]
    decision = utc(row["decision_timestamp_utc"])
    market_fresh = freshness(decision, now, MAX_MARKET_AGE_MINUTES)
    inference_fresh = freshness(shadow.get("captured_at_utc"), now, MAX_INFERENCE_AGE_MINUTES)
    intel_degraded = bool(intel_status.get("news_errors") or intel_status.get("macro_errors") or str(intel_status.get("status", "")).endswith("DEGRADED"))
    intel_observed = context.get("decision_timestamp_utc") or intel_status.get("captured_at_utc")
    intel_fresh = freshness(intel_observed, now, MAX_INTELLIGENCE_AGE_MINUTES, intel_degraded)
    event_risk = classify_event_risk(events, now)
    event_capture = None
    if events is not None and not events.empty and "captured_at_gmt" in events:
        event_capture = pd.to_datetime(events["captured_at_gmt"], utc=True, errors="coerce").max()
    event_fresh = freshness(event_capture, now, MAX_EVENT_CAPTURE_AGE_HOURS * 60, event_risk["data_status"] != "FRESH")
    regime = classify_market_regime(rows, row)
    spread = classify_spread(row)
    fusion = fuse_probabilities(shadow.get("horizons") or {})
    reasons: list[dict[str, object]] = []
    if fusion["status"] == "NO_APPROVED_MODEL":
        reasons.append(reason("NO_APPROVED_MODEL", "No V0.5C CHAMPION is available; no probabilities were fabricated.", "V0.5C/V0.6A", True))
    if fusion.get("malformed_horizons"):
        reasons.append(reason("MALFORMED_PROBABILITIES", "Approved-model probabilities are malformed.", "V0.6A", True, fusion["malformed_horizons"]))
    if shadow.get("status") == "FAIL_CLOSED":
        reasons.append(reason("MODEL_INTEGRITY_FAILURE", "V0.6A failed its registry, artifact, contract, or inference gate.", "V0.6A", True, shadow.get("reason")))
    if market_fresh["status"] != "FRESH":
        reasons.append(reason("STALE_MARKET_DATA", "Latest completed market feature row is unavailable or stale.", "V0.5A", True, market_fresh))
    if inference_fresh["status"] != "FRESH":
        reasons.append(reason("STALE_MODEL_DATA", "V0.6A inference output is unavailable or stale.", "V0.6A", True, inference_fresh))
    stale_artifacts = []
    for horizon, item in (shadow.get("horizons") or {}).items():
        if item.get("model_status") == "APPROVED_CHAMPION" and item.get("model_created_at_utc"):
            if (now - utc(item["model_created_at_utc"])) > pd.Timedelta(days=MAX_MODEL_ARTIFACT_AGE_DAYS):
                stale_artifacts.append(horizon)
    if stale_artifacts:
        reasons.append(reason("STALE_MODEL_DATA", "Approved model artifact age exceeds the explicit 45-day policy.", "V0.5C", True, stale_artifacts))
    if event_risk["data_status"] != "FRESH" or event_fresh["status"] in {"STALE", "UNAVAILABLE"}:
        reasons.append(reason("EVENT_DATA_INCOMPLETE", "Exact-target event coverage is unavailable, incomplete, or stale.", "V0.4D", True, event_fresh))
    if event_risk["blocks_direction"] and event_risk["data_status"] == "FRESH":
        reasons.append(reason("EVENT_BLOCK_WINDOW", f"Exact target event risk is {event_risk['status']}.", "V0.4D", True, event_risk.get("nearest_event")))
    if spread["status"] in {"EXTREME", "UNAVAILABLE"}:
        reasons.append(reason("EXTREME_SPREAD", "Spread is extreme or invalid relative to its causal recent median.", "V0.5A", True, spread))
    elif spread["status"] == "WIDE":
        reasons.append(reason("EXTREME_SPREAD", "Spread is wide relative to its causal recent median.", "V0.5A", True, spread))
    if regime["volatility_regime"] == "EXTREME":
        reasons.append(reason("EXTREME_VOLATILITY", "Current volatility exceeds the prior-history 95th percentile.", "V0.5A", True, regime.get("thresholds")))
    if regime["volatility_regime"] == "UNAVAILABLE" or not bool(row.get("feature_complete", True)):
        reasons.append(reason("INSUFFICIENT_FEATURES", "The market regime or full feature row is incomplete.", "V0.5A", True))
    if fusion["status"] == "INSUFFICIENT_APPROVED_HORIZONS":
        reasons.append(reason("INSUFFICIENT_APPROVED_HORIZONS", "Fewer than two approved horizons passed validation.", "V0.6A", True, fusion.get("valid_horizons")))
    elif fusion["status"] == "HORIZON_DISAGREEMENT":
        reasons.append(reason("HORIZON_DISAGREEMENT", "Approved horizons disagree directionally.", "V0.6A", True, fusion.get("horizon_directions")))
    elif fusion["status"] == "LOW_CONFIDENCE":
        reasons.append(reason("LOW_DIRECTIONAL_CONFIDENCE", "Fused probability or margin does not pass the conservative threshold.", "V0.6B", True))
    if intel_fresh["status"] == "STALE":
        reasons.append(reason("INTELLIGENCE_STALE", "V0.5B context is stale; it is not used as a directional override.", "V0.5B", True, intel_fresh))
    elif intel_fresh["status"] in {"DEGRADED", "UNAVAILABLE"}:
        reasons.append(reason("INTELLIGENCE_DEGRADED", "V0.5B context is degraded; it is context only.", "V0.5B", False, intel_fresh))
    if not REACTION_PATH.exists():
        reasons.append(reason("MEMORY_UNAVAILABLE", "Historical Event Memory is unavailable; no memory-based direction was inferred.", "V0.4B", False))
    blockers = sorted((item for item in reasons if item["blocking"]), key=lambda item: (-int(item["priority"]), str(item["code"])))
    direction = "WAIT" if blockers else str(fusion["direction"])
    if direction != "WAIT":
        reasons.append(reason("DIRECTIONAL_CONSENSUS", "Approved horizons passed fusion and all risk gates.", "V0.6B", False, fusion.get("probabilities")))
    reasons.sort(key=lambda item: (-int(item["priority"]), str(item["code"])))
    action = "BUY_BIAS" if direction == "UP" else "SELL_BIAS" if direction == "DOWN" else "WAIT"
    window = trade_window(now, direction, list(fusion.get("valid_horizons") or []), event_risk)
    if window["status"] == "TOO_SHORT_WAIT":
        action, direction = "WAIT", "WAIT"
        reasons = [item for item in reasons if item["code"] != "DIRECTIONAL_CONSENSUS"]
        short_reason = reason("TRADE_WINDOW_TOO_SHORT", "The remaining advisory window is shorter than 10 minutes.", "V0.6B", True, window)
        reasons.append(short_reason)
        blockers.append(short_reason)
        reasons.sort(key=lambda item: (-int(item["priority"]), str(item["code"])))
        blockers.sort(key=lambda item: (-int(item["priority"]), str(item["code"])))
    next_boundary = _next_m5(now)
    reassess_candidates = [next_boundary]
    nearest = event_risk.get("nearest_event") or {}
    if nearest.get("event_timestamp_utc"):
        event_time = utc(nearest["event_timestamp_utc"])
        for offset in (-240, -60, -15, 15, 60):
            transition = event_time + pd.Timedelta(minutes=offset)
            if transition > now:
                reassess_candidates.append(transition)
    confidence = "VERY_LOW" if fusion["status"] == "NO_APPROVED_MODEL" else "LOW" if blockers else fusion["confidence"]
    main_reason = blockers[0]["code"] if blockers else "DIRECTIONAL_CONSENSUS"
    wait_gate = "WAIT_NO_MODEL" if main_reason == "NO_APPROVED_MODEL" else f"WAIT_{main_reason}" if action == "WAIT" else "DIRECTIONAL_ADVISORY"
    action_meaning = {
        "WAIT": "No directional advisory; reassess at the published time.",
        "BUY_BIAS": "Research-only upward bias; manual confirmation is required.",
        "SELL_BIAS": "Research-only downward bias; manual confirmation is required.",
    }[action]
    confidence_meaning = {
        "VERY_LOW": "No approved model probability support.",
        "LOW": "A blocker or weak/partial model support prevents a directional advisory.",
        "MEDIUM": "Conservative directional gates passed with approved model support.",
        "HIGH": "All three approved horizons passed stronger probability and margin gates.",
    }[confidence]
    context_summary = {
        "status": intel_fresh["status"], "directional_override": False,
        "news_15m": context.get("news_15m_count"), "news_60m": context.get("news_60m_count"),
        "news_240m": context.get("news_240m_count"), "high_impact_240m": context.get("news_240m_high_impact_flag_count"),
        "provider_errors": len(intel_status.get("news_errors") or []) + len(intel_status.get("macro_errors") or []),
    }
    return {
        "contract_version": CONTRACT_VERSION, "symbol": SYMBOL,
        "generated_at_utc": now.isoformat(), "decision_timestamp_utc": decision.isoformat(),
        "action": action, "action_meaning": action_meaning, "direction": direction,
        "confidence": confidence, "confidence_meaning": confidence_meaning, "decision_gate": wait_gate,
        "manual_confirmation_required": MANUAL_CONFIRMATION_REQUIRED, "trading_enabled": TRADING_ENABLED,
        "market": {"freshness": market_fresh, "session": classify_session(row, now), "regime": regime, "spread": spread, "close": row.get("m5_close")},
        "model": {"source_status": shadow.get("status"), "freshness": inference_fresh, "horizons": shadow.get("horizons") or {}, "fusion": fusion},
        "event": {**event_risk, "freshness": event_fresh}, "intelligence": context_summary,
        "memory": _memory_context(),
        "trade_window": window, "next_reassessment_utc": min(reassess_candidates).isoformat(),
        "reasons": reasons, "blocking_reason_codes": [item["code"] for item in blockers],
    }


def _write(payload: dict[str, object]) -> None:
    LATEST_ADVISORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = LATEST_ADVISORY_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(LATEST_ADVISORY_FILE)


def run_fusion_cycle(now_utc: object | None = None, persist: bool = True, shadow_payload: dict[str, Any] | None = None) -> dict[str, object]:
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)
    try:
        if not FEATURE_FILE.exists():
            raise RuntimeError("V0.5A feature store missing")
        market = pd.read_parquet(FEATURE_FILE)
        shadow = shadow_payload if shadow_payload is not None else run_shadow_cycle(now, persist=True)
        decision = utc(market["decision_timestamp_utc"].iloc[-1])
        context, intel_status = _latest_context(decision)
        payload = build_advisory(market, shadow, _load_events(), context, intel_status, now)
        payload["status"] = "PASS_FAIL_CLOSED_NO_CHAMPION" if payload["decision_gate"] == "WAIT_NO_MODEL" else "PASS_ADVISORY" if payload["action"] != "WAIT" else "PASS_WAIT"
    except Exception as exc:
        payload = {
            "contract_version": CONTRACT_VERSION, "symbol": SYMBOL, "generated_at_utc": now.isoformat(),
            "status": "FAIL_CLOSED", "action": "WAIT", "direction": "WAIT", "confidence": "VERY_LOW",
            "decision_gate": "WAIT_RUNTIME_ERROR", "manual_confirmation_required": True, "trading_enabled": False,
            "reasons": [reason("RUNTIME_ERROR", f"{type(exc).__name__}: {exc}", "V0.6B", True)],
            "blocking_reason_codes": ["RUNTIME_ERROR"], "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None},
            "next_reassessment_utc": _next_m5(now).isoformat(),
        }
    if persist:
        _write(payload)
    return payload
