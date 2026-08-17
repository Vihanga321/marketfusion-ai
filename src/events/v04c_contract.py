"""Frozen inputs and contracts for V0.4C event-value intelligence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V04B_MEMORY_PATH = ROOT / "data" / "processed" / "v04b_historical_event_memory.parquet"
V04B_SHA256 = "6A6439D2BA8D317BFCE3989296F4039C1A0EC02F705153EEDD6D76F34E98708C"
V04B_ROWS = 242
V04B_COLUMNS = 221
V04B_CONTRACT = "v0.4b-historical-event-memory-v1"
V04B_MIN_TIMESTAMP = pd.Timestamp("2015-01-09T13:30:00Z")
V04B_MAX_TIMESTAMP = pd.Timestamp("2026-08-12T12:30:00Z")
V04B_EVENT_TYPE_COUNTS = {"us_employment_situation": 124, "us_cpi_release": 118}

EVENT_VALUE_CONTRACT = "v0.4c-event-values-v1"
EVENT_VALUE_FEATURE_CONTRACT = "v0.4c-event-value-features-v1"
ENRICHED_MEMORY_CONTRACT = "v0.4c-historical-event-memory-v1"
EXPECTED_MEASURES_PER_EVENT = 2
EXPECTED_EVENT_VALUE_ROWS = V04B_ROWS * EXPECTED_MEASURES_PER_EVENT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_v04b_memory(path: Path = V04B_MEMORY_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"Missing frozen V0.4B memory: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != V04B_SHA256:
        raise RuntimeError(
            f"Frozen V0.4B memory SHA-256 changed; expected={V04B_SHA256}; actual={actual_hash}"
        )
    frame = pd.read_parquet(path, engine="pyarrow")
    timestamps = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="raise")
    errors: list[str] = []
    if frame.shape != (V04B_ROWS, V04B_COLUMNS):
        errors.append(f"shape={frame.shape}")
    if frame["event_id"].nunique() != V04B_ROWS:
        errors.append("event IDs are not unique")
    if frame["event_type"].value_counts().to_dict() != V04B_EVENT_TYPE_COUNTS:
        errors.append(f"event type counts={frame['event_type'].value_counts().to_dict()}")
    if timestamps.min() != V04B_MIN_TIMESTAMP or timestamps.max() != V04B_MAX_TIMESTAMP:
        errors.append(f"timestamp range={timestamps.min()}..{timestamps.max()}")
    if frame["memory_contract_version"].value_counts().to_dict() != {V04B_CONTRACT: V04B_ROWS}:
        errors.append("retrieval contract/version changed")
    if errors:
        raise RuntimeError("Frozen V0.4B metadata changed: " + "; ".join(errors))
    return frame, {
        "sha256": actual_hash,
        "rows": len(frame),
        "columns": len(frame.columns),
        "event_count": frame["event_id"].nunique(),
        "event_type_counts": frame["event_type"].value_counts().to_dict(),
        "min_timestamp": timestamps.min(),
        "max_timestamp": timestamps.max(),
        "retrieval_contract": V04B_CONTRACT,
    }
