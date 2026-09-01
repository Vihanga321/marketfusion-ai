"""Fail-closed asset-aware state for XAUUSD shadow advisory runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.assets.contracts import asset_paths, get_asset, normalize_asset_id
from src.assets.registry import approved_champions
from src.marketdata.market_calendar import forex_session_state, market_calendar
from src.marketdata.realtime_quote import RUNTIME_DIR


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _age_seconds(value: object, now: pd.Timestamp) -> float | None:
    try:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        delta = float((now - stamp).total_seconds())
        # A future causal market timestamp is not silently clamped to fresh.
        return delta if delta >= 0 else None
    except Exception:
        return None


def _same_utc(left: object, right: object) -> bool:
    try:
        a = pd.Timestamp(left)
        b = pd.Timestamp(right)
        a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
        b = b.tz_localize("UTC") if b.tzinfo is None else b.tz_convert("UTC")
        return a == b
    except Exception:
        return False


def _empty_horizons(champions: dict[int, dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for horizon in (15, 60, 240):
        champion = champions.get(horizon)
        result[str(horizon)] = {
            "model_status": "APPROVED_CHAMPION" if champion else "NO_APPROVED_MODEL",
            "model_id": None if champion is None else champion.get("model_id"),
            "prob_down": None, "prob_neutral": None, "prob_up": None,
            "shadow_direction": "WAIT",
            "decision_gate": "WAIT_INFERENCE_NOT_RUN" if champion else "WAIT_NO_APPROVED_MODEL",
        }
    return result


def build_asset_state(asset_id: str = "XAUUSD", now_utc: object | None = None) -> dict[str, object]:
    asset_id = normalize_asset_id(asset_id)
    asset = get_asset(asset_id)
    paths = asset_paths(asset_id)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    quote = _json(RUNTIME_DIR / f"{asset_id}_quote.json")
    data_status = _json(paths.runtime / "asset_status.json")
    feature_path = paths.features / "features.parquet"
    feature_time = None
    decision_close = None
    spread_atr = None
    if feature_path.exists():
        try:
            latest = pd.read_parquet(feature_path).iloc[-1]
            feature_time = pd.Timestamp(latest["decision_timestamp_utc"]).isoformat()
            decision_close = float(latest["close"])
            spread_atr = None if pd.isna(latest.get("spread_relative_to_atr")) else float(latest["spread_relative_to_atr"])
        except Exception:
            feature_time = None
            decision_close = None

    quote_time = quote.get("normalized_tick_utc") or quote.get("received_at_utc")
    quote_age = _age_seconds(quote_time, now)
    generic_feature_age = _age_seconds(feature_time, now)
    quote_freshness = "LIVE" if quote_age is not None and quote_age <= 3 else "STALE" if quote_age is not None else "UNAVAILABLE"
    generic_feature_freshness = "FRESH" if generic_feature_age is not None and generic_feature_age <= 900 else "STALE" if generic_feature_age is not None else "WAITING_FOR_COMPLETED_M5"

    champions = approved_champions(asset_id)
    horizons = _empty_horizons(champions)
    inference_error: str | None = None
    inference_time = feature_time
    if asset_id == "XAUUSD" and champions:
        try:
            from src.research.v10b_xauusd_models import infer_latest_from_disk
            inference = infer_latest_from_disk()
            if str(inference.get("asset_id")) != "XAUUSD":
                raise RuntimeError("CROSS_SYMBOL_INFERENCE_REJECTED")
            inference_horizons = inference.get("horizons")
            if isinstance(inference_horizons, dict):
                horizons = inference_horizons
            inference_time = inference.get("decision_timestamp_utc") or feature_time
            # The immutable forward ledger uses market.close as the completed-M5
            # decision reference. Never substitute the faster live quote mid.
            if feature_time is None or inference_time is None or not _same_utc(inference_time, feature_time):
                raise RuntimeError("XAUUSD_INFERENCE_FEATURE_REFERENCE_MISMATCH")
            if decision_close is None:
                raise RuntimeError("XAUUSD_COMPLETED_DECISION_CLOSE_UNAVAILABLE")
        except Exception as exc:
            inference_error = f"{type(exc).__name__}: {exc}"
            horizons = _empty_horizons(champions)

    decision_feature_age = _age_seconds(inference_time, now)
    decision_feature_freshness = (
        "FRESH" if decision_feature_age is not None and decision_feature_age <= 900
        else "STALE" if decision_feature_age is not None
        else "WAITING_FOR_COMPLETED_M5"
    )
    valid_horizons = [
        int(horizon) for horizon, item in horizons.items()
        if isinstance(item, dict)
        and item.get("model_status") == "APPROVED_CHAMPION"
        and all(item.get(name) is not None for name in ("prob_down", "prob_neutral", "prob_up"))
    ]

    if not champions:
        runtime_status = "PASS_FAIL_CLOSED_NO_CHAMPION"
        v06a_status = "PASS_FAIL_CLOSED_NO_CHAMPION"
        fusion_status = "NO_APPROVED_MODEL"
        gate = "WAIT_NO_MODEL"
        action_meaning = "No approved asset-specific model is available."
        reason = {
            "code": "NO_APPROVED_MODEL", "priority": 100, "blocking": True,
            "source": "V1.0A_ASSET_REGISTRY",
            "message": f"{asset_id} has no approved model; EURUSD artifacts are not eligible.",
            "evidence": None,
        }
    elif inference_error:
        runtime_status = "DEGRADED_FAIL_CLOSED_INFERENCE"
        v06a_status = "FAIL_CLOSED_INFERENCE_UNAVAILABLE"
        fusion_status = "INFERENCE_UNAVAILABLE"
        gate = "WAIT_INFERENCE_UNAVAILABLE"
        action_meaning = "An approved XAUUSD model exists, but safe shadow inference is unavailable."
        reason = {
            "code": "XAUUSD_INFERENCE_UNAVAILABLE", "priority": 100, "blocking": True,
            "source": "V1.0B_XAUUSD_INFERENCE", "message": inference_error, "evidence": None,
        }
    else:
        runtime_status = "PASS_SHADOW_MODEL_AVAILABLE_WAIT_FUSION"
        v06a_status = "PASS_SHADOW_RUNNING"
        if len(valid_horizons) < 2:
            fusion_status = "INSUFFICIENT_APPROVED_HORIZONS"
            gate = "WAIT_INSUFFICIENT_APPROVED_HORIZONS"
            action_meaning = "Approved XAUUSD model output is available, but multi-horizon fusion remains blocked."
            reason = {
                "code": "INSUFFICIENT_APPROVED_HORIZONS", "priority": 90, "blocking": True,
                "source": "V1.0B_XAUUSD_INFERENCE",
                "message": "Fewer than two approved XAUUSD horizons have valid current probabilities.",
                "evidence": {"valid_horizons": valid_horizons},
            }
        else:
            fusion_status = "XAUUSD_RISK_FUSION_NOT_VALIDATED"
            gate = "WAIT_XAUUSD_FUSION_NOT_VALIDATED"
            action_meaning = "XAUUSD model outputs are available; production risk fusion has not been validated for gold."
            reason = {
                "code": "XAUUSD_RISK_FUSION_NOT_VALIDATED", "priority": 90, "blocking": True,
                "source": "V1.0B_XAUUSD_INFERENCE",
                "message": "Gold-specific multi-horizon fusion remains research-only.",
                "evidence": {"valid_horizons": valid_horizons},
            }

    calendar = market_calendar(now)
    session = forex_session_state(now)
    generated = now.isoformat()
    return {
        "contract_version": "v0.6c-unified-runtime-v1",
        "system": {
            "status": runtime_status,
            "generated_time": {"utc": generated, "asia_colombo": now.tz_convert("Asia/Colombo").isoformat()},
            "mode": "SHADOW_ADVISORY_ONLY", "symbol": asset_id, "asset_id": asset_id,
            "asset_class": asset.asset_class, "display_name": asset.display_name,
            "registry": {
                "status": "AVAILABLE_EMPTY" if not champions else "AVAILABLE",
                "champion_count": len(champions), "asset_id": asset_id,
            },
        },
        "market": {
            # Completed causal M5 reference used by shadow evaluation.
            "close": decision_close,
            # Fast hot-path quote remains separately visible to the dashboard.
            "live_mid": quote.get("mid"), "bid": quote.get("bid"), "ask": quote.get("ask"),
            "quote_freshness": {"status": quote_freshness, "observed_at_utc": quote_time, "age_seconds": quote_age},
            "feature_freshness": {"status": generic_feature_freshness, "observed_at_utc": feature_time, "age_seconds": generic_feature_age},
            "freshness": {
                "status": decision_feature_freshness, "observed_at_utc": inference_time,
                "age_minutes": None if decision_feature_age is None else decision_feature_age / 60,
            },
            "session": session["current_session"], "market_status": calendar["status"],
            "spread": {
                "status": "ASSET_SPECIFIC", "current_points": quote.get("spread_points"),
                "current_price": quote.get("spread"),
                "relative_to_price": None if quote.get("spread") is None or quote.get("mid") in (None, 0) else float(quote["spread"]) / float(quote["mid"]),
                "relative_to_atr": spread_atr,
            },
        },
        "predictions": {
            "v06a_status": v06a_status,
            "horizons": horizons,
            "fusion": {
                "status": fusion_status, "direction": "WAIT", "confidence": "VERY_LOW",
                "probabilities": None, "valid_horizons": valid_horizons,
            },
        },
        "decision": {
            "action": "WAIT", "direction": "WAIT", "action_meaning": action_meaning,
            "confidence": "VERY_LOW", "gate": gate, "manual_confirmation_required": True,
            "trading_enabled": False,
            "decision_time": {"utc": inference_time, "asia_colombo": None},
            "next_reassessment": {"utc": None, "asia_colombo": None},
        },
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None, "horizon_minutes": None},
        "event": {"status": "EXTERNAL_EVENT_INTELLIGENCE_NOT_CONFIGURED", "nearest_event": None},
        "intelligence": {"status": "ASSET_SPECIFIC_RESEARCH_PENDING"},
        "health": {
            "sources": {
                "asset_market": data_status.get("status", "UNAVAILABLE"), "quote": quote_freshness,
                "features": generic_feature_freshness, "asset_registry": "PASS",
                "v10b_inference": "NO_CHAMPION" if not champions else "FAIL" if inference_error else "PASS",
            },
            "failed_sources": ["v10b_inference"] if inference_error else [],
            "degraded_sources": [] if quote_freshness == "LIVE" else ["quote"],
        },
        "reasons": [reason],
        "trading_enabled": False,
    }


def write_asset_state(asset_id: str = "XAUUSD") -> Path:
    paths = asset_paths(asset_id)
    path = paths.runtime / "latest_marketfusion_state.json"
    _atomic_json(build_asset_state(asset_id), path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD", choices=("EURUSD", "XAUUSD"))
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=15)
    args = parser.parse_args()
    while True:
        path = write_asset_state(args.symbol)
        print(f"V10A_ASSET_STATE: PASS {path}")
        if not args.continuous:
            return 0
        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
