"""Frozen contracts for the V0.4B Historical Event Memory."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REACTION_PATH = ROOT / "data" / "processed" / "v04a_event_reactions.parquet"
REACTION_SHA256 = "5914E7DA16680CA0F5CF6ADC49892A50F141118EF9FB167571B46F6DB9EAFD12"
REACTION_ROWS = 242
REACTION_COLUMNS = 74
REACTION_CONTRACT = "v0.4a-event-reaction-v1"
REACTION_MIN_TIMESTAMP = pd.Timestamp("2015-01-09T13:30:00Z")
REACTION_MAX_TIMESTAMP = pd.Timestamp("2026-08-12T12:30:00Z")
REACTION_EVENT_TYPE_COUNTS = {"us_employment_situation": 124, "us_cpi_release": 118}
MEMORY_CONTRACT = "v0.4b-historical-event-memory-v1"
PIP_SIZE = 0.0001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_reaction_dataset(path: Path = REACTION_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.is_file():
        raise RuntimeError(f"Missing frozen V0.4A reaction dataset: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != REACTION_SHA256:
        raise RuntimeError(
            f"Frozen V0.4A reaction SHA-256 changed; expected={REACTION_SHA256}; actual={actual_hash}"
        )
    frame = pd.read_parquet(path, engine="pyarrow")
    timestamps = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="raise")
    event_counts = frame["event_type"].value_counts().to_dict()
    contract_counts = frame["reaction_contract_version"].value_counts().to_dict()
    errors = []
    if frame.shape != (REACTION_ROWS, REACTION_COLUMNS):
        errors.append(f"shape={frame.shape}, expected={(REACTION_ROWS, REACTION_COLUMNS)}")
    if timestamps.min() != REACTION_MIN_TIMESTAMP or timestamps.max() != REACTION_MAX_TIMESTAMP:
        errors.append(f"timestamp_range={timestamps.min()}..{timestamps.max()}")
    if event_counts != REACTION_EVENT_TYPE_COUNTS:
        errors.append(f"event_type_counts={event_counts}")
    if contract_counts != {REACTION_CONTRACT: REACTION_ROWS}:
        errors.append(f"reaction_contract={contract_counts}")
    if errors:
        raise RuntimeError("Frozen V0.4A reaction metadata changed: " + "; ".join(errors))
    metadata = {
        "sha256": actual_hash,
        "rows": len(frame),
        "columns": len(frame.columns),
        "contract_version": REACTION_CONTRACT,
        "min_timestamp": timestamps.min(),
        "max_timestamp": timestamps.max(),
        "event_type_counts": event_counts,
    }
    return frame, metadata
