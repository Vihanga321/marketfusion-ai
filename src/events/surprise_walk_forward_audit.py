"""Walk-forward surprise audit with an explicit insufficient-consensus sample gate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.events.build_v04c_event_values import OUTPUT as EVENT_VALUES_PATH, validate_event_values
from src.events.v04c_contract import ROOT, verify_v04b_memory


AUDIT_REPORT = ROOT / "reports" / "v04c_surprise_walk_forward_audit.csv"
EXAMPLES_REPORT = ROOT / "reports" / "v04c_surprise_analogue_examples.txt"
HORIZONS = ("1m", "5m", "15m", "60m", "240m")
MINIMUM_QUERY_SAMPLE = 20


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def eligible_surprise_events(values: pd.DataFrame) -> pd.DataFrame:
    audited = values[
        values["consensus_value"].notna()
        & values["surprise_signed_raw"].notna()
        & values["surprise_z_robust"].notna()
        & values["point_in_time_verified"].fillna(False).astype(bool)
    ]
    return audited[["event_id", "event_type", "event_timestamp_utc"]].drop_duplicates()


def insufficient_sample_audit(values: pd.DataFrame) -> pd.DataFrame:
    eligible = eligible_surprise_events(values)
    start = values["event_timestamp_utc"].min()
    end = values["event_timestamp_utc"].max()
    rows = []
    for horizon in HORIZONS:
        rows.append(
            {
                "horizon": horizon,
                "eligible_query_count": len(eligible),
                "event_type": "ALL",
                "time_span_start": start,
                "time_span_end": end,
                "surprise_analogue_directional_hit_rate": pd.NA,
                "v04b_context_only_directional_hit_rate": pd.NA,
                "same_event_type_unconditional_hit_rate": pd.NA,
                "recent_n_hit_rate": pd.NA,
                "audit_status": "INSUFFICIENT_SAMPLE",
                "reason": "No audited historical pre-release consensus values",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify_v04b_memory()
    if not EVENT_VALUES_PATH.is_file():
        raise RuntimeError(f"Missing normalized event values: {EVENT_VALUES_PATH}")
    values = pd.read_parquet(EVENT_VALUES_PATH, engine="pyarrow")
    validate_event_values(values)
    eligible = eligible_surprise_events(values)
    if len(eligible) >= MINIMUM_QUERY_SAMPLE:
        raise RuntimeError(
            "Audited consensus coverage now exists; implement and independently review the "
            "surprise-aware ranking policy before producing metrics"
        )
    audit = insufficient_sample_audit(values)
    _atomic_frame(audit, AUDIT_REPORT)
    lines = [
        "MARKETFUSION V0.4C SURPRISE ANALOGUE EXAMPLES",
        f"eligible_surprise_queries: {len(eligible)}",
        f"minimum_descriptive_sample: {MINIMUM_QUERY_SAMPLE}",
        "SURPRISE_ANALOGUE_STATUS: INSUFFICIENT_SAMPLE",
        "",
        "No analogue examples or directional statistics were generated because no audited",
        "historical consensus provider is present. Actual values were not substituted for",
        "consensus, market prices were not used to infer expectations, and future reactions",
        "were not used to construct surprise fields.",
        "",
        "The frozen V0.4B context-only audit remains the comparison baseline. A new surprise",
        "audit must be enabled only after sufficient strictly pre-release consensus coverage",
        "and an independently reviewed POST_RELEASE_IMMEDIATE retrieval policy exist.",
        "",
    ]
    _atomic_text("\n".join(lines), EXAMPLES_REPORT)
    print("V04C_SURPRISE_AUDIT_STATUS: INSUFFICIENT_SAMPLE")
    print(f"Eligible queries: {len(eligible)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
