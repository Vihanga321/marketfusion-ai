"""Fail-closed asset-aware state for XAUUSD shadow advisory runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
        return max(0.0, float((now - stamp).total_seconds()))
    except Exception:
        return None


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
    close = quote.get("mid")
    spread_atr = None
    if feature_path.exists():
        try:
            latest = pd.read_parquet(feature_path).iloc[-1]
            feature_time = pd.Timestamp(latest["decision_timestamp_utc"]).isoformat()
            close = quote.get("mid") if quote.get("mid") is not None else float(latest["close"])
            spread_atr = None if pd.isna(latest.get("spread_relative_to_atr")) else float(latest["spread_relative_to_atr"])
        except Exception:
            feature_time = None
    quote_time = quote.get("normalized_tick_utc") or quote.get("received_at_utc")
    quote_age = _age_seconds(quote_time, now)
    feature_age = _age_seconds(feature_time, now)
    quote_freshness = "LIVE" if quote_age is not None and quote_age <= 3 else "STALE" if quote_age is not None else "UNAVAILABLE"
    feature_freshness = "FRESH" if feature_age is not None and feature_age <= 900 else "STALE" if feature_age is not None else "WAITING_FOR_COMPLETED_M5"
    champions = approved_champions(asset_id)
    horizons: dict[str, object] = {}
    for horizon in (15, 60, 240):
        champion = champions.get(horizon)
        horizons[str(horizon)] = {
            "model_status": "APPROVED_CHAMPION" if champion else "NO_APPROVED_MODEL",
            "model_id": None if champion is None else champion.get("model_id"),
            "prob_down": None, "prob_neutral": None, "prob_up": None,
            "shadow_direction": "WAIT", "decision_gate": "WAIT_NO_APPROVED_MODEL" if champion is None else "WAIT_INFERENCE_NOT_RUN",
        }
    calendar = market_calendar(now)
    session = forex_session_state(now)
    generated = now.isoformat()
    return {
        "contract_version": "v0.6c-unified-runtime-v1",
        "system": {
            "status": "PASS_FAIL_CLOSED_NO_CHAMPION" if not champions else "PASS_SHADOW_READY",
            "generated_time": {"utc": generated, "asia_colombo": now.tz_convert("Asia/Colombo").isoformat()},
            "mode": "SHADOW_ADVISORY_ONLY", "symbol": asset_id, "asset_id": asset_id,
            "asset_class": asset.asset_class, "display_name": asset.display_name,
            "registry": {"status": "AVAILABLE_EMPTY" if not champions else "AVAILABLE", "champion_count": len(champions), "asset_id": asset_id},
        },
        "market": {
            "close": close, "bid": quote.get("bid"), "ask": quote.get("ask"),
            "quote_freshness": {"status": quote_freshness, "observed_at_utc": quote_time, "age_seconds": quote_age},
            "feature_freshness": {"status": feature_freshness, "observed_at_utc": feature_time, "age_seconds": feature_age},
            "freshness": {"status": feature_freshness, "observed_at_utc": feature_time, "age_minutes": None if feature_age is None else feature_age / 60},
            "session": session["current_session"], "market_status": calendar["status"],
            "spread": {
                "status": "ASSET_SPECIFIC", "current_points": quote.get("spread_points"),
                "current_price": quote.get("spread"), "relative_to_price": None if quote.get("spread") is None or quote.get("mid") in (None, 0) else float(quote["spread"]) / float(quote["mid"]),
                "relative_to_atr": spread_atr,
            },
        },
        "predictions": {
            "v06a_status": "PASS_FAIL_CLOSED_NO_CHAMPION" if not champions else "WAIT_INFERENCE_NOT_RUN",
            "horizons": horizons,
            "fusion": {"status": "NO_APPROVED_MODEL", "direction": "WAIT", "confidence": "VERY_LOW", "probabilities": None, "valid_horizons": []},
        },
        "decision": {
            "action": "WAIT", "direction": "WAIT", "action_meaning": "No approved asset-specific model is available.",
            "confidence": "VERY_LOW", "gate": "WAIT_NO_MODEL", "manual_confirmation_required": True,
            "trading_enabled": False, "decision_time": {"utc": feature_time, "asia_colombo": None},
            "next_reassessment": {"utc": None, "asia_colombo": None},
        },
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None, "horizon_minutes": None},
        "event": {"status": "EXTERNAL_EVENT_INTELLIGENCE_NOT_CONFIGURED", "nearest_event": None},
        "intelligence": {"status": "ASSET_SPECIFIC_RESEARCH_PENDING"},
        "health": {
            "sources": {"asset_market": data_status.get("status", "UNAVAILABLE"), "quote": quote_freshness, "features": feature_freshness, "asset_registry": "PASS"},
            "failed_sources": [], "degraded_sources": [] if quote_freshness == "LIVE" else ["quote"],
        },
        "reasons": [{"code": "NO_APPROVED_MODEL", "priority": 100, "blocking": True, "source": "V1.0A_ASSET_REGISTRY", "message": f"{asset_id} has no approved model; EURUSD artifacts are not eligible.", "evidence": None}],
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
