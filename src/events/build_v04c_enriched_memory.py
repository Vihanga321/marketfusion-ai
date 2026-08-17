"""Create V0.4C event-level features and enriched memory without overwriting V0.4B."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.events.build_v04c_event_values import OUTPUT as EVENT_VALUES_PATH, validate_event_values
from src.events.decision_mode_registry import DecisionMode, project_mode
from src.events.event_value_source_registry import SOURCES
from src.events.v04c_contract import (
    ENRICHED_MEMORY_CONTRACT,
    EVENT_VALUE_FEATURE_CONTRACT,
    ROOT,
    V04B_ROWS,
    verify_v04b_memory,
)
from src.events.v04c_retrieval_policy import policy_registry_frame


FEATURE_OUTPUT = ROOT / "data" / "processed" / "v04c_event_value_features.parquet"
MEMORY_OUTPUT = ROOT / "data" / "processed" / "v04c_historical_event_memory.parquet"
QUALITY_REPORT = ROOT / "reports" / "v04c_historical_event_memory_quality.txt"
PRE_REGISTRY_REPORT = ROOT / "reports" / "v04c_pre_release_feature_registry.csv"
POST_REGISTRY_REPORT = ROOT / "reports" / "v04c_post_release_feature_registry.csv"


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False, engine="pyarrow")
    else:
        frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_event_level_features(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected = {source.event_type: sorted(s.measure_id for s in SOURCES if s.event_type == source.event_type) for source in SOURCES}
    for event_id, group in values.groupby("event_id", sort=False):
        group = group.sort_values("measure_id")
        event_type = str(group["event_type"].iloc[0])
        if group["measure_id"].tolist() != expected[event_type]:
            raise RuntimeError(f"Unexpected release composition: {event_id}")
        row: dict[str, object] = {
            "event_id": event_id,
            "event_type": event_type,
            "event_timestamp_utc": group["event_timestamp_utc"].iloc[0],
            "release_component_count": len(group),
            "release_actual_component_count": int(group["actual_value"].notna().sum()),
            "release_consensus_component_count": int(group["consensus_value"].notna().sum()),
            "release_mixed_signal_flag": pd.NA,
            "event_value_feature_contract_version": EVENT_VALUE_FEATURE_CONTRACT,
        }
        for value in group.itertuples(index=False):
            measure = value.measure_id
            row[f"pre_release_context_event_{measure}_previous_value"] = value.previous_value_pre_release
            row[f"pre_release_context_event_{measure}_previous_available_from_utc"] = value.previous_available_from_utc
            row[f"pre_release_context_event_{measure}_consensus_value"] = value.consensus_value
            row[f"pre_release_context_event_{measure}_consensus_available_from_utc"] = value.consensus_available_from_utc
            row[f"release_payload_event_{measure}_actual_value"] = value.actual_value
            row[f"release_payload_event_{measure}_actual_available_from_utc"] = value.actual_available_from_utc
            row[f"release_payload_event_{measure}_revision_raw"] = value.event_revision_raw
            row[f"release_payload_event_{measure}_revision_available_from_utc"] = value.revision_available_from_utc
            row[f"release_payload_event_{measure}_prior_period_newly_released_at_release"] = value.prior_period_newly_released_at_release
            row[f"release_payload_event_{measure}_prior_period_new_release_available_from_utc"] = value.prior_period_new_release_available_from_utc
            row[f"release_payload_event_{measure}_surprise_signed_raw"] = value.surprise_signed_raw
            row[f"release_payload_event_{measure}_surprise_z_robust"] = value.surprise_z_robust
            row[f"release_payload_event_{measure}_unit"] = value.unit
            row[f"release_payload_event_{measure}_value_status"] = value.value_status
            row[f"release_payload_event_{measure}_consensus_status"] = value.consensus_status
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["event_timestamp_utc", "event_id"]).reset_index(drop=True)


def validate_event_level_features(features: pd.DataFrame) -> None:
    if len(features) != V04B_ROWS or features["event_id"].duplicated().any():
        raise RuntimeError("Event-level feature view must contain 242 unique events")
    if any(column.startswith("outcome_") for column in features):
        raise RuntimeError("Historical outcome entered event-value feature construction")
    event_times = pd.to_datetime(features["event_timestamp_utc"], utc=True)
    for column in (name for name in features if name.endswith("previous_available_from_utc")):
        available = pd.to_datetime(features[column], utc=True)
        present = available.notna()
        if (present & ~(available < event_times)).any():
            raise RuntimeError(f"Pre-release previous-value chronology failure: {column}")
    for column in (name for name in features if name.endswith("consensus_available_from_utc")):
        available = pd.to_datetime(features[column], utc=True)
        present = available.notna()
        if (present & ~(available < event_times)).any():
            raise RuntimeError(f"Post-event consensus: {column}")
    for column in (name for name in features if name.endswith("actual_available_from_utc") or name.endswith("revision_available_from_utc")):
        available = pd.to_datetime(features[column], utc=True)
        present = available.notna()
        if (present & ~available.eq(event_times)).any():
            raise RuntimeError(f"Release payload not available at T: {column}")
    pre = project_mode(features, DecisionMode.PRE_RELEASE)
    post = project_mode(features, DecisionMode.POST_RELEASE_IMMEDIATE)
    if any("actual_value" in column or "surprise" in column for column in pre):
        raise RuntimeError("PRE_RELEASE projection contains actual/surprise")
    if not any("actual_value" in column for column in post):
        raise RuntimeError("POST_RELEASE_IMMEDIATE projection lacks audited actual payload")


def build_enriched_memory(memory: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = memory.merge(
        features.drop(columns=["event_type", "event_timestamp_utc"]),
        on="event_id", how="left", validate="one_to_one",
    )
    if len(joined) != V04B_ROWS or joined["event_value_feature_contract_version"].isna().any():
        raise RuntimeError("V0.4C enriched memory join lost or duplicated events")
    if not joined["adjudication_class"].eq("STRICT_PASS").all():
        raise RuntimeError("Quarantined event entered V0.4C memory")
    joined["v04c_memory_contract_version"] = ENRICHED_MEMORY_CONTRACT
    return joined


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    memory, metadata = verify_v04b_memory()
    if not EVENT_VALUES_PATH.is_file():
        raise RuntimeError(f"Missing V0.4C normalized event values: {EVENT_VALUES_PATH}")
    values = pd.read_parquet(EVENT_VALUES_PATH, engine="pyarrow")
    validate_event_values(values)
    features = build_event_level_features(values)
    validate_event_level_features(features)
    enriched = build_enriched_memory(memory, features)
    _atomic_frame(features, FEATURE_OUTPUT)
    _atomic_frame(enriched, MEMORY_OUTPUT)
    _atomic_frame(policy_registry_frame(DecisionMode.PRE_RELEASE), PRE_REGISTRY_REPORT)
    _atomic_frame(policy_registry_frame(DecisionMode.POST_RELEASE_IMMEDIATE), POST_REGISTRY_REPORT)
    lines = [
        "MARKETFUSION V0.4C ENRICHED HISTORICAL MEMORY QUALITY",
        f"source_v04b_sha256: {metadata['sha256']}",
        f"records: {len(enriched)}",
        f"columns: {len(enriched.columns)}",
        "quarantined_events: 0",
        f"event_value_feature_columns: {len(features.columns)}",
        "logical_groups: pre_release_context | release_payload | historical_outcomes",
        "PRE_RELEASE_GATE: PASS",
        "POST_RELEASE_IMMEDIATE_GATE: PASS",
        "OUTCOME_ISOLATION: PASS",
        "ENRICHED_MEMORY_STATUS: PASS",
        "",
    ]
    _atomic_text("\n".join(lines), QUALITY_REPORT)
    print("V04C_ENRICHED_MEMORY_STATUS: PASS")
    print(f"Records: {len(enriched)}")
    print(f"Columns: {len(enriched.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
