"""Unified V0.6 runtime state assembler; read-only advisory operation only."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.fusion.v06b_engine import run_fusion_cycle
from src.inference.v06a_engine import run_shadow_cycle
from src.intelligence.v05b_contract import STATUS_FILE as V05B_STATUS_FILE
from src.learning.v05c_contract import RUNTIME_REGISTRY
from src.marketdata.v05a_contract import STATUS_FILE as V05A_STATUS_FILE
from src.runtime.v06c_contract import CONTRACT_VERSION, LATEST_STATE_FILE, LOCAL_TIMEZONE, STATE_HISTORY_FILE


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def dual_time(value: object | None) -> dict[str, str | None]:
    if value is None:
        return {"utc": None, "asia_colombo": None}
    stamp = utc(value)
    return {"utc": stamp.isoformat(), "asia_colombo": stamp.to_pydatetime().astimezone(ZoneInfo(LOCAL_TIMEZONE)).isoformat()}


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _registry_state() -> dict[str, object]:
    if not RUNTIME_REGISTRY.exists():
        return {"status": "AVAILABLE_EMPTY", "sha256": None, "champion_count": 0, "reloaded_this_cycle": True}
    try:
        raw = RUNTIME_REGISTRY.read_bytes()
        records = json.loads(raw)
        champions = [record for record in records if record.get("promotion_status") == "CHAMPION"]
        return {"status": "AVAILABLE" if champions else "AVAILABLE_EMPTY", "sha256": sha256(raw).hexdigest(), "champion_count": len(champions), "reloaded_this_cycle": True}
    except Exception as exc:
        return {"status": "FAIL", "sha256": None, "champion_count": 0, "reloaded_this_cycle": True, "error": f"{type(exc).__name__}: {exc}"}


def _health(v05a: dict[str, Any], v05b: dict[str, Any], advisory: dict[str, Any], registry: dict[str, Any]) -> dict[str, object]:
    market = advisory.get("market") or {}
    intelligence = advisory.get("intelligence") or {}
    event = advisory.get("event") or {}
    sources = {
        "v05a_market": "PASS" if v05a and (market.get("freshness") or {}).get("status") == "FRESH" else "FAIL",
        "v05b_intelligence": "PASS_DEGRADED" if intelligence.get("status") == "DEGRADED" else "PASS" if intelligence.get("status") == "FRESH" else "PASS_WITH_LIMITED_COVERAGE",
        "v05d_event": "PASS" if event.get("data_status") == "FRESH" else "PASS_WITH_LIMITED_COVERAGE" if event.get("data_status") == "DEGRADED" else "FAIL",
        "v05c_registry": "PASS" if registry.get("status") in {"AVAILABLE", "AVAILABLE_EMPTY"} else "FAIL",
        "v06a_inference": "PASS_FAIL_CLOSED_NO_CHAMPION" if advisory.get("decision_gate") == "WAIT_NO_MODEL" else "PASS" if advisory.get("status") != "FAIL_CLOSED" else "FAIL",
        "v06b_fusion": advisory.get("status", "FAIL"),
    }
    return {"sources": sources, "failed_sources": [name for name, value in sources.items() if value == "FAIL"], "degraded_sources": [name for name, value in sources.items() if "DEGRADED" in value or "LIMITED" in value]}


def assemble_state(advisory: dict[str, Any], shadow: dict[str, Any], v05a: dict[str, Any], v05b: dict[str, Any], registry: dict[str, Any]) -> dict[str, object]:
    generated = advisory.get("generated_at_utc") or pd.Timestamp.now(tz="UTC").isoformat()
    health = _health(v05a, v05b, advisory, registry)
    if advisory.get("status") == "FAIL_CLOSED":
        overall = "FAIL_CLOSED"
    elif advisory.get("decision_gate") == "WAIT_NO_MODEL":
        overall = "PASS_FAIL_CLOSED_NO_CHAMPION"
    elif health["failed_sources"]:
        overall = "FAIL_CLOSED"
    elif any("DEGRADED" in value for value in health["sources"].values()):
        overall = "PASS_DEGRADED"
    elif health["degraded_sources"]:
        overall = "PASS_WITH_LIMITED_COVERAGE"
    else:
        overall = "PASS"
    market = advisory.get("market") or {}
    decision = {
        "action": advisory.get("action", "WAIT"), "action_meaning": advisory.get("action_meaning"),
        "direction": advisory.get("direction", "WAIT"), "confidence": advisory.get("confidence", "VERY_LOW"),
        "confidence_meaning": advisory.get("confidence_meaning"), "gate": advisory.get("decision_gate", "WAIT_RUNTIME_ERROR"),
        "manual_confirmation_required": True, "trading_enabled": False,
        "decision_time": dual_time(advisory.get("decision_timestamp_utc")),
        "next_reassessment": dual_time(advisory.get("next_reassessment_utc")),
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "system": {"status": overall, "generated_time": dual_time(generated), "mode": "SHADOW_ADVISORY_ONLY", "symbol": advisory.get("symbol", "EURUSD"), "registry": registry},
        "market": market,
        "predictions": {"v06a_status": shadow.get("status"), "horizons": shadow.get("horizons") or {}, "fusion": (advisory.get("model") or {}).get("fusion")},
        "decision": decision,
        "trade_window": advisory.get("trade_window") or {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None},
        "event": advisory.get("event") or {}, "intelligence": advisory.get("intelligence") or {},
        "health": health, "reasons": advisory.get("reasons") or [],
    }


def _history_row(state: dict[str, Any]) -> pd.DataFrame:
    decision = state.get("decision") or {}
    decision_time = decision.get("decision_time") or {}
    generated = (state.get("system") or {}).get("generated_time") or {}
    return pd.DataFrame([{
        "decision_timestamp_utc": decision_time.get("utc"), "generated_at_utc": generated.get("utc"),
        "status": (state.get("system") or {}).get("status"), "action": decision.get("action"),
        "confidence": decision.get("confidence"), "decision_gate": decision.get("gate"),
        "state_json": json.dumps(state, sort_keys=True, default=str),
    }])


def persist_state(state: dict[str, Any]) -> bool:
    LATEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = LATEST_STATE_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(LATEST_STATE_FILE)
    incoming = _history_row(state)
    key = incoming.iloc[0]["decision_timestamp_utc"]
    if not key:
        return False
    if STATE_HISTORY_FILE.exists():
        history = pd.read_parquet(STATE_HISTORY_FILE, engine="pyarrow")
        if str(key) in set(history["decision_timestamp_utc"].astype(str)):
            return False
        history = pd.concat([history, incoming], ignore_index=True)
    else:
        history = incoming
    temp_history = STATE_HISTORY_FILE.with_suffix(".parquet.tmp")
    history.to_parquet(temp_history, index=False, engine="pyarrow")
    temp_history.replace(STATE_HISTORY_FILE)
    return True


def run_runtime_cycle(now_utc: object | None = None, persist: bool = True) -> dict[str, object]:
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)
    shadow = run_shadow_cycle(now, persist=persist)
    advisory = run_fusion_cycle(now, persist=persist, shadow_payload=shadow)
    state = assemble_state(advisory, shadow, _json(V05A_STATUS_FILE), _json(V05B_STATUS_FILE), _registry_state())
    if persist:
        state["system"]["history_appended"] = persist_state(state)
    return state
