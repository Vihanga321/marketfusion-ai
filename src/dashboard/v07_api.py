"""Localhost-only, read-only FastAPI surface for MarketFusion V0.7."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.assets.contracts import asset_paths, get_asset, normalize_asset_id
from src.dashboard.v07_contract import (
    ACTIVE_SYMBOL, CONTRACT_VERSION, DEFAULT_CANDLE_LIMIT, FORBIDDEN_RESPONSE_KEYS, HOST,
    MAX_CANDLE_LIMIT, MAX_HISTORY_LIMIT, PORT, REQUIRED_STATE_KEYS,
    STATE_STALE_SECONDS, TIMEFRAMES, TRADING_ROUTE_FRAGMENTS, V04D_EVENTS_FILE,
    V05A_STATUS_FILE, V05B_CONTEXT_FILE, V05B_STATUS_FILE, V05C_CANDIDATES_FILE,
    V05C_ELIGIBILITY_FILE, V05C_TRAINING_FILE, V06_HISTORY_FILE, V06_STATE_FILE,
    V08_STATUS_FILE, VITE_ORIGINS,
)
from src.dashboard.v07_market import load_candles
from src.dashboard.v07_operator import build_operator_status, latest_quote_timestamp
from src.evaluation.v08_contract import MAX_OBSERVATION_API_LIMIT, OBSERVATION_CONTRACT_VERSION
from src.evaluation.v08_observations import json_records, observations_frame, shadow_summary
from src.engines.engine_layer import engine_status, recent_patterns
from src.dashboard.realtime import STREAM_TELEMETRY, load_quote_snapshot, prepare_stream_payload


def _asset(value: str | None) -> str:
    return normalize_asset_id(ACTIVE_SYMBOL if value is None else value)


def _state_path(symbol: str) -> Path:
    return V06_STATE_FILE if symbol == "EURUSD" else asset_paths(symbol).runtime / "latest_marketfusion_state.json"


def _market_status_path(symbol: str) -> Path:
    return V05A_STATUS_FILE if symbol == "EURUSD" else asset_paths(symbol).runtime / "asset_status.json"


def _engine_inputs(symbol: str) -> tuple[Path, Path]:
    if symbol == "EURUSD":
        from src.marketdata.v05a_contract import CONTINUOUS_DIR, FEATURE_FILE
        return FEATURE_FILE, CONTINUOUS_DIR
    paths = asset_paths(symbol)
    return paths.features / "features.parquet", paths.bars


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _forbidden_keys(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_RESPONSE_KEYS:
                found.append(normalized)
            found.extend(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_forbidden_keys(child))
    return found


def validate_state(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "state is not a JSON object"
    missing = sorted(REQUIRED_STATE_KEYS.difference(payload))
    if missing:
        return False, "missing required sections: " + ", ".join(missing)
    if not isinstance(payload.get("contract_version"), str) or not str(payload["contract_version"]).startswith("v0.6c-"):
        return False, "unsupported V0.6C contract"
    if not isinstance(payload.get("decision"), dict) or payload["decision"].get("action") not in {"WAIT", "BUY_BIAS", "SELL_BIAS"}:
        return False, "invalid decision action"
    if payload["decision"].get("trading_enabled") is not False:
        return False, "trading must remain disabled"
    forbidden = sorted(set(_forbidden_keys(payload)))
    if forbidden:
        return False, "forbidden response fields: " + ", ".join(forbidden)
    return True, "PASS"


def degraded_state(reason: str, symbol: str = "EURUSD") -> dict[str, object]:
    asset = get_asset(symbol)
    return {
        "contract_version": "v0.6c-dashboard-degraded-v1",
        "system": {"status": "SYSTEM_DATA_INVALID", "mode": "SHADOW_ADVISORY_ONLY", "symbol": asset.asset_id, "asset_id": asset.asset_id, "asset_class": asset.asset_class, "display_name": asset.display_name, "generated_time": {"utc": None, "asia_colombo": None}},
        "market": {"freshness": {"status": "UNAVAILABLE", "observed_at_utc": None, "age_minutes": None}},
        "predictions": {"v06a_status": "DATA_UNAVAILABLE", "horizons": {str(h): {"model_id": None, "model_status": "NO_APPROVED_MODEL", "prob_down": None, "prob_neutral": None, "prob_up": None, "shadow_direction": "WAIT"} for h in (15, 60, 240)}, "fusion": {"status": "DATA_UNAVAILABLE", "direction": "WAIT", "confidence": "VERY_LOW", "probabilities": None, "valid_horizons": []}},
        "decision": {"action": "WAIT", "direction": "WAIT", "confidence": "VERY_LOW", "gate": "WAIT_SYSTEM_DATA_INVALID", "manual_confirmation_required": True, "trading_enabled": False, "decision_time": {"utc": None, "asia_colombo": None}, "next_reassessment": {"utc": None, "asia_colombo": None}},
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None, "horizon_minutes": None},
        "event": {"status": "EVENT_DATA_INCOMPLETE", "nearest_event": None},
        "intelligence": {"status": "UNAVAILABLE"},
        "health": {"sources": {}, "failed_sources": ["v06_state"], "degraded_sources": []},
        "reasons": [{"code": "SYSTEM_DATA_INVALID", "priority": 100, "blocking": True, "source": "V0.7", "message": reason, "evidence": None}],
    }


def load_state(path: Path | None = None, *, symbol: str | None = None) -> tuple[dict[str, Any], str, str]:
    asset = _asset(symbol)
    path = _state_path(asset) if path is None else path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return degraded_state("V0.6C state file is unavailable.", asset), "INVALID", "state file missing"
    except (OSError, json.JSONDecodeError) as exc:
        return degraded_state("V0.6C state file is malformed or unreadable.", asset), "INVALID", type(exc).__name__
    valid, detail = validate_state(raw)
    if not valid:
        return degraded_state(detail, asset), "INVALID", detail
    generated = ((raw.get("system") or {}).get("generated_time") or {}).get("utc")
    try:
        age_seconds = float((pd.Timestamp.now(tz="UTC") - _utc(generated)).total_seconds())
        freshness = "FRESH" if -1 <= age_seconds <= STATE_STALE_SECONDS else "STALE"
    except Exception:
        freshness = "STALE"
    return raw, freshness, "PASS"


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _json_scalar(value: object) -> object:
    """Convert pandas/numpy values to strict JSON primitives without inventing data."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value


app = FastAPI(
    title="MarketFusion V0.7 Local Dashboard API",
    version=CONTRACT_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VITE_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
)


@app.get("/api/health")
def health(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    state, freshness, detail = load_state(symbol=asset)
    return {"status": "PASS" if detail == "PASS" else "DEGRADED", "symbol": asset, "state_valid": detail == "PASS", "state_freshness": freshness, "v06_status": (state.get("system") or {}).get("status"), "trading_enabled": False}


@app.get("/api/state")
def state(response: Response, symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, Any]:
    payload, freshness, detail = load_state(symbol=_asset(symbol))
    response.headers["X-MarketFusion-State-Freshness"] = freshness
    response.headers["X-MarketFusion-State-Contract"] = detail
    response.headers["Cache-Control"] = "no-store"
    return payload


@app.get("/api/market/candles")
def candles(timeframe: Literal["M1", "M5", "M15", "H1"] = Query("M5"), limit: int = Query(DEFAULT_CANDLE_LIMIT, ge=1, le=MAX_CANDLE_LIMIT), symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    return load_candles(timeframe, limit, _asset(symbol))


@app.get("/api/market/summary")
def market_summary(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    live = load_quote_snapshot(symbol=asset)
    if live.get("bid") is not None:
        return {
            "status": live.get("connection", "UNAVAILABLE"), "symbol": asset, "captured_at_utc": live.get("received_at_utc"),
            "bid": live.get("bid"), "ask": live.get("ask"), "mid": live.get("mid"),
            "spread_points": live.get("spread_points"), "normalized_tick_utc": live.get("normalized_tick_utc"),
            "freshness": live.get("freshness"),
        }
    payload = _safe_json(_market_status_path(asset))
    fallback_quote = payload.get("quote") if isinstance(payload.get("quote"), dict) else {}
    return {
        "status": payload.get("status", "UNAVAILABLE"), "symbol": asset, "captured_at_utc": payload.get("captured_at_utc"),
        "bid": payload.get("latest_bid", fallback_quote.get("bid")), "ask": payload.get("latest_ask", fallback_quote.get("ask")),
        "mid": fallback_quote.get("mid"), "spread_points": payload.get("latest_spread_points", fallback_quote.get("spread_points")),
    }


@app.get("/api/market/quote")
def realtime_quote(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    return prepare_stream_payload(load_quote_snapshot(symbol=_asset(symbol)))


@app.get("/api/performance/realtime")
def realtime_performance(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    payload = load_quote_snapshot(symbol=_asset(symbol))
    return {
        "status": payload.get("connection", "UNAVAILABLE"),
        "collector": payload.get("performance") or {},
        "api_stream_delivery": STREAM_TELEMETRY.snapshot(),
        "poll_interval_ms": payload.get("poll_interval_ms"),
        "trading_enabled": False,
    }


@app.websocket("/ws/market/{symbol}")
async def market_stream(websocket: WebSocket, symbol: str) -> None:
    try:
        asset = _asset(symbol)
    except ValueError:
        await websocket.close(code=1008)
        return
    origin = websocket.headers.get("origin")
    allowed = {None, *VITE_ORIGINS}
    if origin not in allowed:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    last_sequence: object = None
    try:
        while True:
            snapshot = load_quote_snapshot(symbol=asset)
            sequence = snapshot.get("sequence")
            if sequence != last_sequence:
                payload = prepare_stream_payload(snapshot)
                STREAM_TELEMETRY.observe(payload.get("feed_delivery_ms"))
                await websocket.send_json(payload)
                last_sequence = sequence
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


@app.get("/api/operator/status")
def operator_status(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    payload, freshness, detail = load_state(symbol=asset)
    v05a = _safe_json(_market_status_path(asset))
    return build_operator_status(
        payload, v05a, quote_timestamp=latest_quote_timestamp(v05a),
        state_valid=detail == "PASS", state_freshness=freshness,
    )


@app.get("/api/runtime/history")
def runtime_history(limit: int = Query(100, ge=1, le=MAX_HISTORY_LIMIT)) -> dict[str, object]:
    if not V06_HISTORY_FILE.exists():
        return {"status": "DATA_UNAVAILABLE", "count": 0, "items": []}
    frame = pd.read_parquet(V06_HISTORY_FILE).tail(limit)
    items = [{key: _json_scalar(row.get(key)) for key in ("decision_timestamp_utc", "generated_at_utc", "status", "action", "confidence", "decision_gate")} for _, row in frame.iterrows()]
    return {"status": "PASS", "count": len(items), "items": items}


@app.get("/api/events")
def events(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    if _asset(symbol) != "EURUSD":
        return {"status": "EXTERNAL_EVENT_INTELLIGENCE_NOT_CONFIGURED", "symbol": _asset(symbol), "events": []}
    if not V04D_EVENTS_FILE.exists():
        return {"status": "DATA_UNAVAILABLE", "events": []}
    try:
        frame = pd.read_csv(V04D_EVENTS_FILE)
        allowed = ("event_id", "event_name", "event_code", "event_timestamp_utc", "forecast_value", "consensus_status")
        return {"status": "PASS", "events": [{key: _json_scalar(row.get(key)) for key in allowed} for _, row in frame.iterrows()]}
    except Exception:
        return {"status": "DATA_UNAVAILABLE", "events": []}


@app.get("/api/intelligence")
def intelligence(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    if _asset(symbol) != "EURUSD":
        return {"status": "ASSET_SPECIFIC_RESEARCH_PENDING", "symbol": _asset(symbol), "captured_at_utc": None, "news_counts": {}, "topic_counts_24h": {}, "macro": {}, "provider_health": {}}
    payload = _safe_json(V05B_STATUS_FILE)
    macro_fields = (
        "macro_us_effective_fed_funds_value", "macro_us_2y_yield_value",
        "macro_us_10y_yield_value", "macro_us_10y_minus_2y_value",
        "macro_ecb_main_refinancing_rate_value", "macro_policy_rate_spread_us_minus_ecb_pctpt",
    )
    news_counts = {window: payload.get(f"current_news_{window}m") for window in (15, 60, 240, 1440)}
    topics: dict[str, object] = {}
    if V05B_CONTEXT_FILE.exists():
        try:
            latest = pd.read_parquet(V05B_CONTEXT_FILE).iloc[-1]
            topics = {
                topic: _json_scalar(latest.get(f"news_1440m_topic_{topic}_count"))
                for topic in ("central_bank", "rates", "inflation", "labor", "growth", "risk")
            }
            topics["usd_relevant"] = _json_scalar(latest.get("news_1440m_usd_relevant_count"))
            topics["eur_relevant"] = _json_scalar(latest.get("news_1440m_eur_relevant_count"))
        except Exception:
            topics = {}
    provider_health = {
        "FED_RSS": "OK" if sum(int((payload.get("news_source_counts") or {}).get(name, 0)) for name in ("FED_MONETARY", "FED_SPEECHES")) else "UNAVAILABLE",
        "ECB": "OK" if int((payload.get("news_source_counts") or {}).get("ECB_PRESS", 0)) and payload.get("macro_ecb_main_refinancing_rate_value") is not None else "DEGRADED",
        "BLS": "OK" if sum(int((payload.get("news_source_counts") or {}).get(name, 0)) for name in ("BLS_CPI", "BLS_EMPLOYMENT")) else "UNAVAILABLE",
        "GDELT": "DEGRADED" if any("GDELT" in str(item) for item in payload.get("news_errors") or []) else "OK",
        "FRED": "DEGRADED" if any("us_" in str(item) for item in payload.get("macro_errors") or []) else "OK",
    }
    return {"status": payload.get("status", "UNAVAILABLE"), "captured_at_utc": payload.get("captured_at_utc"), "news_counts": news_counts, "topic_counts_24h": topics, "macro": {key: payload.get(key) for key in macro_fields}, "provider_health": provider_health}


@app.get("/api/engines/status")
def engines_status(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    feature_path, bar_root = _engine_inputs(asset)
    return engine_status(symbol=asset, path=feature_path, bar_root=bar_root)


@app.get("/api/engines/patterns/recent")
def engines_recent_patterns(limit: int = Query(20, ge=1, le=100), symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    feature_path, bar_root = _engine_inputs(asset)
    return recent_patterns(limit=limit, symbol=asset, path=feature_path, bar_root=bar_root)


@app.get("/api/research")
def research(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    if asset != "EURUSD":
        paths = asset_paths(asset)
        targets = _safe_json(paths.reports / "target_research.json")
        baselines = _safe_json(paths.reports / "baseline_research.json")
        return {"status": "RESEARCH_ONLY_NO_APPROVED_MODEL", "symbol": asset, "approved_champions": 0, "candidates": {}, "target_research": targets, "baselines": baselines, "last_training_utc": None, "market_core_rows": None, "history_days": None, "v05b_eligibility": "NOT_APPLICABLE", "live_surprise_samples": 0, "next_recommended_training": "WALK_FORWARD_RESEARCH_AFTER_DATA_AUDIT"}
    candidates: dict[str, object] = {}
    if V05C_CANDIDATES_FILE.exists():
        try:
            frame = pd.read_csv(V05C_CANDIDATES_FILE)
            models = frame.loc[frame["candidate_type"].eq("MODEL")]
            for horizon in (15, 60, 240):
                subset = models.loc[models["horizon_minutes"].eq(horizon)].sort_values("balanced_accuracy", ascending=False)
                if not subset.empty:
                    row = subset.iloc[0]
                    candidates[str(horizon)] = {"family": row["family"], "balanced_accuracy": float(row["balanced_accuracy"]), "promotion_status": row["promotion_status"]}
        except Exception:
            candidates = {}
    last_training = None
    market_core_rows = None
    if V05C_TRAINING_FILE.exists():
        for line in V05C_TRAINING_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("created_at_utc:"):
                last_training = line.split(":", 1)[1].strip()
            if line.startswith("training_rows:") and market_core_rows is None:
                market_core_rows = int(line.split(":", 1)[1].strip())
    history_days = None
    if V05C_ELIGIBILITY_FILE.exists():
        for line in V05C_ELIGIBILITY_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("MARKET_CORE|"):
                parts = line.split("|")
                if len(parts) >= 4:
                    market_core_rows, history_days = int(parts[2]), int(parts[3])
                break
    return {"status": "RESEARCH_ONLY", "approved_champions": 0, "candidates": candidates, "last_training_utc": last_training, "market_core_rows": market_core_rows, "history_days": history_days, "v05b_eligibility": "INSUFFICIENT_HISTORY", "live_surprise_samples": 0, "next_recommended_training": "DAILY_MANUAL_SCHEDULE"}


@app.get("/api/system/version")
def version() -> dict[str, object]:
    return {"api_contract": CONTRACT_VERSION, "state_contract": "v0.6c-unified-runtime-v1", "mode": "LOCALHOST_READ_ONLY", "trading_enabled": False}


@app.get("/api/v08/status")
def v08_status(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    status_path = V08_STATUS_FILE if asset == "EURUSD" else asset_paths(asset).evaluation / "latest_status.json"
    payload = _safe_json(status_path)
    if payload:
        return payload
    return {
        "contract_version": "v0.8-forward-shadow-monitor-v1", "status": "INSUFFICIENT_DATA", "symbol": asset,
        "performance": {
            "recorded_predictions": 0, "matured_outcomes": 0,
            "horizons": {str(h): {"matured_count": 0, "directional_calls": 0, "wait_rate": None, "directional_accuracy": None, "brier": None, "sample_status": "INSUFFICIENT_DATA"} for h in (15, 60, 240)},
            "wait": {"wait_rate": None}, "calibration_status": "NO_APPROVED_MODEL_DATA",
            "market_drift_status": "INSUFFICIENT_DATA", "model_drift_status": "INSUFFICIENT_DATA",
        },
        "champions": {"15": "NONE", "60": "NONE", "240": "NONE"},
        "provider_health": {"status": "INSUFFICIENT_DATA", "providers": {}, "uptime_percentage": {}},
        "research": {"status": "AVAILABLE_NOT_RUN", "experiments": 0, "best_experiment": None, "decision": "INSUFFICIENT_DATA"},
        "manual_execution_only": True, "trading_enabled": False,
    }


@app.get("/api/evaluation/shadow/summary")
def shadow_evaluation_summary(symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    if asset == "EURUSD":
        summary = shadow_summary()
        monitor = _safe_json(V08_STATUS_FILE)
    else:
        paths = asset_paths(asset)
        summary = shadow_summary(observations_frame(predictions_path=paths.shadow_predictions, outcomes_path=paths.shadow_outcomes))
        monitor = _safe_json(paths.evaluation / "latest_status.json")
    summary["symbol"] = asset
    cycle = monitor.get("ledger_cycle") if isinstance(monitor.get("ledger_cycle"), dict) else {}
    summary["recorder"] = {
        "status": "RUNNING" if monitor else "STATUS_UNAVAILABLE",
        "trigger": "NEW_COMPLETED_CAUSAL_DECISION_ROW",
        "last_cycle_status": cycle.get("status", "NOT_RUN"),
        "last_cycle_reason": cycle.get("reason") or cycle.get("detail") or "NOT_AVAILABLE",
        "updated_at_utc": monitor.get("updated_at_utc"),
    }
    return summary


@app.get("/api/evaluation/shadow/recent")
def recent_shadow_observations(limit: int = Query(50, ge=1, le=MAX_OBSERVATION_API_LIMIT), symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    asset = _asset(symbol)
    paths = asset_paths(asset)
    frame = observations_frame() if asset == "EURUSD" else observations_frame(predictions_path=paths.shadow_predictions, outcomes_path=paths.shadow_outcomes)
    items = json_records(frame.head(limit)) if not frame.empty else []
    return {
        "contract_version": OBSERVATION_CONTRACT_VERSION, "status": "PASS", "symbol": asset,
        "count": len(items), "limit": limit, "items": items,
        "trading_enabled": False,
    }


@app.get("/api/evaluation/shadow/observations/{observation_id}")
def shadow_observation_detail(observation_id: str, symbol: str = Query(ACTIVE_SYMBOL)) -> dict[str, object]:
    if len(observation_id) != 64 or any(value not in "0123456789abcdef" for value in observation_id.lower()):
        raise HTTPException(status_code=404, detail="Shadow observation not found")
    asset = _asset(symbol)
    paths = asset_paths(asset)
    frame = observations_frame() if asset == "EURUSD" else observations_frame(predictions_path=paths.shadow_predictions, outcomes_path=paths.shadow_outcomes)
    if frame.empty:
        raise HTTPException(status_code=404, detail="Shadow observation not found")
    matching = frame.loc[frame["observation_id"].eq(observation_id)]
    if matching.empty:
        raise HTTPException(status_code=404, detail="Shadow observation not found")
    return json_records(matching.head(1))[0]


def trading_route_count() -> int:
    return sum(1 for route in app.routes if any(fragment in route.path.lower() for fragment in TRADING_ROUTE_FRAGMENTS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("V0.7 API binds to localhost only; LAN mode is not enabled.")
    import uvicorn
    uvicorn.run("src.dashboard.v07_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
