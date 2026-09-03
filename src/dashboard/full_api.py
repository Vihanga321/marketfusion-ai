"""Extended localhost-only, read-only API for the complete MarketFusion console."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Query

from src.assets.contracts import ROOT, asset_paths, normalize_asset_id
from src.dashboard.v07_api import app

RELIABILITY_HEALTH_FILE = ROOT / "data" / "runtime" / "v10c1" / "collector_health.json"
RELIABILITY_OUTAGE_FILE = ROOT / "data" / "runtime" / "v10c1" / "provider_outages.jsonl"
RELIABILITY_HEARTBEAT_STALE_SECONDS = 180


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _forward_status_candidates(asset: str) -> list[Path]:
    paths = asset_paths(asset)
    return [
        paths.evaluation / "v10c1_forward_validation" / "latest_status.json",
        paths.evaluation / "v10c_forward_validation" / "latest_status.json",
    ]


def _recent_jsonl(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    items: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


def _heartbeat_age_seconds(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds())


@app.get("/api/evaluation/forward/status")
def forward_validation_status(symbol: str = Query("XAUUSD")) -> dict[str, Any]:
    """Return the newest explicit XAUUSD frozen-forward status without mutating it."""
    asset = normalize_asset_id(symbol)
    if asset != "XAUUSD":
        raise HTTPException(status_code=404, detail="Forward validation is not configured for this asset")

    for path in _forward_status_candidates(asset):
        if not path.exists():
            continue
        payload = _read_object(path)
        if payload is None:
            raise HTTPException(status_code=503, detail="Forward validation status is unreadable")
        contracts = payload.get("contracts")
        if not isinstance(payload.get("contract_version"), str) or not isinstance(payload.get("decision"), str) or not isinstance(contracts, dict):
            raise HTTPException(status_code=503, detail="Forward validation status contract is invalid")
        return payload

    raise HTTPException(status_code=404, detail="Forward validation status is not available")


@app.get("/api/system/reliability")
def reliability_status(symbol: str = Query("XAUUSD")) -> dict[str, Any]:
    """Read the V1.0C.1 supervisor heartbeat and recent outage ledger."""
    asset = normalize_asset_id(symbol)
    if asset != "XAUUSD":
        raise HTTPException(status_code=404, detail="Reliability supervisor is not configured for this asset")
    health = _read_object(RELIABILITY_HEALTH_FILE)
    if health is None:
        return {
            "contract_version": "v1.0c1-xauusd-reliability-supervisor-v1",
            "status": "NOT_STARTED",
            "collector": {"state": "NOT_STARTED"},
            "heartbeat_age_seconds": None,
            "recent_outages": _recent_jsonl(RELIABILITY_OUTAGE_FILE),
            "safety": {
                "automatic_execution": "DISABLED",
                "runtime": "SHADOW_ADVISORY_ONLY",
                "manual_confirmation": "REQUIRED",
                "backfill": "PROHIBITED",
            },
        }
    collector = health.get("collector") if isinstance(health.get("collector"), dict) else {}
    state = str(collector.get("state", "UNKNOWN"))
    heartbeat_age = _heartbeat_age_seconds(health.get("updated_at_utc"))
    stale = heartbeat_age is None or heartbeat_age > RELIABILITY_HEARTBEAT_STALE_SECONDS
    api_status = "STALE_HEARTBEAT" if state == "RUNNING" and stale else ("PASS" if state == "RUNNING" else state)
    return {
        **health,
        "status": api_status,
        "heartbeat_age_seconds": heartbeat_age,
        "recent_outages": _recent_jsonl(RELIABILITY_OUTAGE_FILE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("MarketFusion full API binds to localhost only; LAN mode is not enabled.")
    import uvicorn

    uvicorn.run("src.dashboard.full_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
