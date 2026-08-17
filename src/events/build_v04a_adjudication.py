"""Freeze V0.4A validation adjudication and the strict eligible event universe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from v04a_contract import (
        CONTRACT_VERSION,
        EXPECTED_ELIGIBLE_EVENTS,
        EXPECTED_PRODUCTION_COUNTS,
        MISMATCH_EVENT_IDS,
        REACTION_CONTRACT_VERSION,
        ROOT,
        verify_frozen_evidence,
    )
except ImportError:  # pragma: no cover - package import path used by tests
    from src.events.v04a_contract import (
        CONTRACT_VERSION,
        EXPECTED_ELIGIBLE_EVENTS,
        EXPECTED_PRODUCTION_COUNTS,
        MISMATCH_EVENT_IDS,
        REACTION_CONTRACT_VERSION,
        ROOT,
        verify_frozen_evidence,
    )


SUMMARY = ROOT / "data" / "dukascopy" / "full_validation_summary.tsv"
EVENTS = ROOT / "data" / "dukascopy" / "events_for_jforex.tsv"
MISMATCH_MINUTES = ROOT / "reports" / "dukascopy_mismatch_minutes.tsv"
ADJUDICATION_TSV = ROOT / "reports" / "dukascopy_v04a_adjudication.tsv"
ADJUDICATION_TEXT = ROOT / "reports" / "dukascopy_v04a_adjudication.txt"
ELIGIBLE_PARQUET = ROOT / "data" / "processed" / "v04a_eligible_event_manifest.parquet"
ELIGIBLE_CSV = ROOT / "reports" / "v04a_eligible_event_manifest.csv"
ELIGIBLE_SUMMARY = ROOT / "reports" / "v04a_eligible_event_manifest_summary.csv"
COLLECTOR_INPUT = ROOT / "data" / "dukascopy" / "v04a_reaction" / "eligible_events.tsv"


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing required evidence: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path} lacks columns: {missing}")
    return frame


def _atomic_frame(frame: pd.DataFrame, path: Path, *, separator: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False, engine="pyarrow")
    else:
        frame.to_csv(temporary, sep=separator or ",", index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _diagnostic_evidence(minutes: pd.DataFrame) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for event_id, rows in minutes.groupby("event_id", sort=False):
        details = []
        for row in rows.sort_values("minute_utc").itertuples(index=False):
            maximum = f" max_diff={row.max_diff}" if row.max_diff else ""
            details.append(
                f"{row.minute_utc} {row.classification}{maximum}: {row.evidence}"
            )
        evidence[event_id] = " | ".join(details)
    return evidence


def build_adjudication(
    summary_path: Path = SUMMARY,
    events_path: Path = EVENTS,
    mismatch_path: Path = MISMATCH_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = _read(
        summary_path,
        {"event_id", "event_timestamp_utc", "status", "failure_reason", "validator_version"},
    )
    events = _read(
        events_path,
        {"event_id", "event_type", "reference_period", "event_timestamp_utc", "timestamp_precision", "timestamp_source"},
    )
    mismatch = _read(
        mismatch_path,
        {"event_id", "minute_utc", "classification", "max_diff", "evidence"},
    )

    if summary["event_id"].duplicated().any() or events["event_id"].duplicated().any():
        raise RuntimeError("event_id must be unique in production summary and canonical events")
    counts = summary["status"].value_counts().to_dict()
    observed = {status: int(counts.get(status, 0)) for status in EXPECTED_PRODUCTION_COUNTS}
    if observed != EXPECTED_PRODUCTION_COUNTS or len(summary) != sum(EXPECTED_PRODUCTION_COUNTS.values()):
        raise RuntimeError(f"Production counts changed: expected={EXPECTED_PRODUCTION_COUNTS}, actual={observed}")
    if set(summary.loc[summary["status"] == "MISMATCH", "event_id"]) != MISMATCH_EVENT_IDS:
        raise RuntimeError("Production mismatch event set changed")
    if set(mismatch["event_id"]) != MISMATCH_EVENT_IDS:
        raise RuntimeError("Forensic mismatch evidence does not cover exactly the four mismatch events")
    if not mismatch["classification"].isin(
        {"PROVIDER_NATIVE_TICK_DISAGREEMENT", "TICK_HISTORY_HOLE"}
    ).all():
        raise RuntimeError("Forensic evidence contains an unapproved or newly demonstrated root cause")
    if not summary["validator_version"].eq(CONTRACT_VERSION).all():
        raise RuntimeError("Production validator contract changed")

    event_lookup = events.set_index("event_id")
    canonical_types = summary["event_id"].map(event_lookup["event_type"])
    canonical_times = summary["event_id"].map(event_lookup["event_timestamp_utc"])
    if canonical_types.isna().any() or canonical_times.isna().any():
        raise RuntimeError("Production summary contains event IDs absent from the canonical export")
    if not summary["event_type"].eq(canonical_types).all() or not summary["event_timestamp_utc"].eq(canonical_times).all():
        raise RuntimeError("Production event type/timestamp differs from canonical event metadata")
    metadata = events.drop(columns=["event_timestamp_utc", "event_type"])
    adjudication = summary.merge(metadata, on="event_id", how="left", validate="one_to_one")
    if adjudication["event_type"].eq("").any():
        raise RuntimeError("Canonical event metadata missing after production join")
    forensic = _diagnostic_evidence(mismatch)

    class_map = {
        "PASS": "STRICT_PASS",
        "INCOMPLETE": "PROVIDER_HISTORY_INCOMPLETE",
        "MISMATCH": "PROVIDER_NATIVE_TICK_DISAGREEMENT",
        "ERROR": "VALIDATOR_ERROR",
    }
    adjudication["production_status"] = adjudication["status"]
    adjudication["production_failure_reason"] = adjudication["failure_reason"]
    adjudication["adjudication_class"] = adjudication["production_status"].map(class_map)
    adjudication["diagnostic_evidence"] = adjudication["event_id"].map(forensic).fillna("")
    adjudication["model_eligible_market_reaction"] = adjudication["production_status"].eq("PASS")
    adjudication["quarantine_reason"] = ""
    incomplete = adjudication["production_status"].eq("INCOMPLETE")
    mismatch_rows = adjudication["production_status"].eq("MISMATCH")
    errors = adjudication["production_status"].eq("ERROR")
    adjudication.loc[incomplete, "quarantine_reason"] = (
        "PROVIDER_HISTORY_INCOMPLETE:" + adjudication.loc[incomplete, "production_failure_reason"]
    )
    adjudication.loc[mismatch_rows, "quarantine_reason"] = "PROVIDER_NATIVE_TICK_DISAGREEMENT"
    adjudication.loc[errors, "quarantine_reason"] = "VALIDATOR_ERROR"

    columns = [
        "event_id", "event_type", "reference_period", "event_timestamp_utc",
        "timestamp_precision", "timestamp_source", "production_status",
        "production_failure_reason", "adjudication_class", "diagnostic_evidence",
        "model_eligible_market_reaction", "quarantine_reason", "validator_version",
    ]
    adjudication = adjudication[columns].sort_values(
        ["event_timestamp_utc", "event_type", "event_id"]
    ).reset_index(drop=True)

    eligible = adjudication.loc[
        adjudication["production_status"].eq("PASS")
        & adjudication["model_eligible_market_reaction"],
        [
            "event_id", "event_type", "reference_period", "event_timestamp_utc",
            "timestamp_precision", "timestamp_source", "production_status",
            "adjudication_class", "model_eligible_market_reaction", "validator_version",
        ],
    ].copy()
    eligible["reaction_contract_version"] = REACTION_CONTRACT_VERSION
    eligible["market_data_source"] = "Dukascopy JForex historical ticks"

    if len(eligible) != EXPECTED_ELIGIBLE_EVENTS:
        raise RuntimeError(f"Eligible count is {len(eligible)}, expected {EXPECTED_ELIGIBLE_EVENTS}")
    if eligible["event_id"].duplicated().any():
        raise RuntimeError("Eligible event IDs are not unique")
    if eligible.duplicated(["event_type", "event_timestamp_utc"]).any():
        raise RuntimeError("Eligible event type/release timestamp relationship is not unique")
    if not eligible["production_status"].eq("PASS").all():
        raise RuntimeError("Non-PASS event entered eligible universe")
    if set(eligible["event_id"]) & set(adjudication.loc[adjudication["quarantine_reason"].ne(""), "event_id"]):
        raise RuntimeError("Quarantined event entered eligible universe")

    dates = pd.to_datetime(eligible["event_timestamp_utc"], utc=True, errors="raise")
    eligible["event_timestamp_utc"] = dates
    eligible["reference_period"] = pd.to_datetime(eligible["reference_period"], utc=True, errors="raise")
    eligible["event_year"] = dates.dt.year
    eligible["event_month"] = dates.dt.strftime("%Y-%m")

    summary_rows = []
    for dimension, column in (
        ("event_type", "event_type"),
        ("year", "event_year"),
        ("month", "event_month"),
        ("timestamp_precision", "timestamp_precision"),
        ("timestamp_provenance", "timestamp_source"),
    ):
        for value, count in eligible.groupby(column, dropna=False).size().items():
            summary_rows.append({"dimension": dimension, "value": str(value), "eligible_events": int(count)})
    eligible_summary = pd.DataFrame(summary_rows)
    return adjudication, eligible, eligible_summary


def write_outputs(adjudication: pd.DataFrame, eligible: pd.DataFrame, eligible_summary: pd.DataFrame) -> None:
    _atomic_frame(adjudication, ADJUDICATION_TSV, separator="\t")
    counts = adjudication["adjudication_class"].value_counts().to_dict()
    text = "\n".join(
        [
            "MARKETFUSION V0.4A FINAL ADJUDICATION",
            f"Production contract: {CONTRACT_VERSION}",
            "Original production statuses are preserved without relabeling.",
            "",
            f"STRICT_PASS: {counts.get('STRICT_PASS', 0)}",
            f"PROVIDER_HISTORY_INCOMPLETE: {counts.get('PROVIDER_HISTORY_INCOMPLETE', 0)}",
            f"PROVIDER_NATIVE_TICK_DISAGREEMENT: {counts.get('PROVIDER_NATIVE_TICK_DISAGREEMENT', 0)}",
            "RECONSTRUCTION_BUG: 0",
            f"VALIDATOR_ERROR: {counts.get('VALIDATOR_ERROR', 0)}",
            f"TRAINING_ELIGIBLE: {int(adjudication['model_eligible_market_reaction'].sum())}",
            f"QUARANTINED: {int((~adjudication['model_eligible_market_reaction']).sum())}",
            "",
            "MISMATCH remains MISMATCH. INCOMPLETE remains INCOMPLETE.",
            "No demonstrated reconstruction bug, invalid filtering, or minute-boundary issue.",
            "ADJUDICATION_STATUS: PASS",
            "",
        ]
    )
    _atomic_text(text, ADJUDICATION_TEXT)
    csv_eligible = eligible.copy()
    csv_eligible["event_timestamp_utc"] = csv_eligible["event_timestamp_utc"].map(lambda x: x.isoformat().replace("+00:00", "Z"))
    csv_eligible["reference_period"] = csv_eligible["reference_period"].map(lambda x: x.isoformat().replace("+00:00", "Z"))
    _atomic_frame(eligible, ELIGIBLE_PARQUET)
    _atomic_frame(csv_eligible, ELIGIBLE_CSV)
    _atomic_frame(eligible_summary, ELIGIBLE_SUMMARY)
    collector_columns = [
        "event_id", "event_type", "reference_period", "event_timestamp_utc",
        "timestamp_precision", "timestamp_source", "production_status",
        "adjudication_class", "model_eligible_market_reaction", "validator_version",
        "reaction_contract_version", "market_data_source",
    ]
    _atomic_frame(csv_eligible[collector_columns], COLLECTOR_INPUT, separator="\t")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify_frozen_evidence()
    adjudication, eligible, eligible_summary = build_adjudication()
    write_outputs(adjudication, eligible, eligible_summary)
    counts = adjudication["adjudication_class"].value_counts().to_dict()
    print("V04A_ADJUDICATION_STATUS: PASS")
    print(f"Adjudicated: {len(adjudication)}")
    print(f"Strict eligible: {len(eligible)}")
    print(f"Quarantined: {len(adjudication) - len(eligible)}")
    print(f"Classes: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
