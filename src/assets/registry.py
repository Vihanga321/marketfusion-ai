"""Asset-isolated model registry helpers for V1.0A."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.assets.contracts import asset_paths, normalize_asset_id

HORIZONS = (15, 60, 240)


def empty_promotion_state(asset_id: str) -> dict[str, object]:
    asset = normalize_asset_id(asset_id)
    return {
        "asset_id": asset,
        "champions": {str(h): {"model_status": "NO_APPROVED_MODEL", "model_id": None} for h in HORIZONS},
        "trading_enabled": False,
    }


def model_id(asset_id: str, family: str, horizon: int, timestamp: str, digest: str) -> str:
    asset = normalize_asset_id(asset_id).lower()
    if int(horizon) not in HORIZONS:
        raise ValueError("Unsupported horizon")
    clean_family = "".join(char for char in family.lower() if char.isalnum() or char == "_").strip("_")
    clean_stamp = "".join(char for char in timestamp if char.isdigit())
    clean_digest = "".join(char for char in digest.lower() if char in "0123456789abcdef")[:12]
    if not clean_family or not clean_stamp or len(clean_digest) < 8:
        raise ValueError("Invalid model identity components")
    return f"{clean_family}_{asset}_{int(horizon)}m_{clean_stamp}_{clean_digest}"


def load_asset_registry(asset_id: str, path: Path | None = None) -> list[dict[str, Any]]:
    asset = normalize_asset_id(asset_id)
    registry = asset_paths(asset).model_registry if path is None else path
    if not registry.exists():
        return []
    payload = json.loads(registry.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Asset model registry must be a JSON list")
    for record in payload:
        if normalize_asset_id(str(record.get("asset_id") or record.get("symbol"))) != asset:
            raise RuntimeError(f"Cross-symbol model registry collision in {registry}")
        if int(record.get("horizon_minutes", -1)) not in HORIZONS:
            raise ValueError("Invalid registry horizon")
    return payload


def approved_champions(asset_id: str, path: Path | None = None) -> dict[int, dict[str, Any]]:
    records = load_asset_registry(asset_id, path)
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("promotion_status") == "APPROVED_CHAMPION":
            horizon = int(record["horizon_minutes"])
            if horizon in result:
                raise RuntimeError(f"Multiple approved champions for {asset_id} {horizon}m")
            result[horizon] = record
    return result
