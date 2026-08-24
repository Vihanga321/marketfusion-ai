"""Immutable V0.5C model artifact and registry operations."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.learning.v05c_contract import MODEL_REGISTRY_COLUMNS


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable_bytes(path: Path, payload: bytes) -> str:
    expected = sha256(payload).hexdigest()
    if path.exists():
        if file_sha256(path) != expected:
            raise RuntimeError(f"Immutable artifact mutation rejected: {path}")
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return expected


def dump_immutable_model(path: Path, model: Any) -> str:
    if path.exists():
        raise RuntimeError(f"Registered model artifact cannot be silently overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    digest = file_sha256(temporary)
    temporary.replace(path)
    return digest


def load_registry(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Model registry must be a JSON list")
    return payload


def validate_registry(records: list[dict[str, object]], verify_artifacts: bool = True) -> None:
    ids: set[str] = set()
    for record in records:
        missing = sorted(set(MODEL_REGISTRY_COLUMNS) - set(record))
        if missing:
            raise ValueError("Model registry record missing fields: " + ", ".join(missing))
        model_id = str(record["model_id"])
        if model_id in ids:
            raise ValueError(f"Duplicate model_id: {model_id}")
        ids.add(model_id)
        if int(record["horizon_minutes"]) not in (15, 60, 240):
            raise ValueError("Invalid model horizon")
        if verify_artifacts:
            path = Path(str(record["artifact_path"]))
            if not path.exists() or file_sha256(path) != str(record["artifact_sha256"]):
                raise RuntimeError(f"Artifact SHA mismatch: {model_id}")


def append_registry(path: Path, new_records: list[dict[str, object]]) -> list[dict[str, object]]:
    records = load_registry(path)
    existing = {str(record["model_id"]): record for record in records}
    for record in new_records:
        model_id = str(record["model_id"])
        if model_id in existing:
            if existing[model_id] != record:
                raise RuntimeError(f"Registered model mutation rejected: {model_id}")
            continue
        records.append(record)
        existing[model_id] = record
    validate_registry(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)
    return records


def registry_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=MODEL_REGISTRY_COLUMNS)
