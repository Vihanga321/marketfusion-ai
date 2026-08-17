"""Prepare the immutable four-event input for the Dukascopy forensic run.

This tool is intentionally separate from the production validator.  It reads
the canonical event export and the completed aggregate, verifies the exact
mismatch set, and emits only the four approved rows.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_EVENTS = ROOT / "data" / "dukascopy" / "events_for_jforex.tsv"
PRODUCTION_SUMMARY = ROOT / "data" / "dukascopy" / "full_validation_summary.tsv"
DEFAULT_OUTPUT = ROOT / "data" / "dukascopy" / "mismatch_diagnostic" / "events.tsv"

TARGET_EVENT_IDS = (
    "bls:us_employment_situation:20220902T1230Z",
    "bls:us_cpi_release:20230913T1230Z",
    "bls:us_employment_situation:20250404T1230Z",
    "bls:us_cpi_release:20260512T1230Z",
)


def _read_tsv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing required TSV: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")
    if frame["event_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["event_id"].duplicated(False), "event_id"].unique())
        raise RuntimeError(f"{path} contains duplicate event_id values: {duplicates}")
    return frame


def select_target_events(events_path: Path, summary_path: Path) -> pd.DataFrame:
    events = _read_tsv(
        events_path,
        {"event_id", "event_type", "reference_period", "event_timestamp_utc"},
    )
    summary = _read_tsv(summary_path, {"event_id", "status"})

    actual_mismatches = set(summary.loc[summary["status"] == "MISMATCH", "event_id"])
    expected = set(TARGET_EVENT_IDS)
    if actual_mismatches != expected:
        raise RuntimeError(
            "Completed production mismatch set does not equal the approved four-event set; "
            f"expected={sorted(expected)}, actual={sorted(actual_mismatches)}"
        )

    selected = events.loc[events["event_id"].isin(expected)].copy()
    selected = selected.set_index("event_id", drop=False).reindex(TARGET_EVENT_IDS).reset_index(drop=True)
    if selected["event_id"].isna().any() or set(selected["event_id"]) != expected:
        present = set(events["event_id"])
        raise RuntimeError(f"Canonical export lacks target event(s): {sorted(expected - present)}")

    summary_targets = summary.set_index("event_id").loc[list(TARGET_EVENT_IDS)]
    if not summary_targets["status"].eq("MISMATCH").all():
        raise RuntimeError("Every approved diagnostic event must remain MISMATCH in production")
    return selected


def atomic_tsv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, sep="\t", index=False, lineterminator="\n")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=CANONICAL_EVENTS)
    parser.add_argument("--summary", type=Path, default=PRODUCTION_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selected = select_target_events(args.events, args.summary)
    atomic_tsv(selected, args.output)
    print("DUKASCOPY_MISMATCH_DIAGNOSTIC_INPUT: READY")
    print(f"Events: {len(selected)} (exact approved set)")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
