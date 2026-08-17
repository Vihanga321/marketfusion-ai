"""Export validated V0.4A economic events for the read-only JForex collector.

The Parquet source remains the canonical event table. This helper writes a small
TSV bridge for Java so the JForex collector does not need a Parquet dependency.
No market data is downloaded and no trading action is possible in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVENT_FILE = ROOT / "data" / "events" / "bls_release_events.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "dukascopy" / "events_for_jforex.tsv"

BASE_COLUMNS = [
    "event_id",
    "event_type",
    "reference_period",
    "event_timestamp_utc",
    "timestamp_precision",
    "timestamp_source",
]


def _iso_utc(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, utc=True, errors="coerce")
    if values.isna().any():
        raise RuntimeError("Cannot export rows with missing or invalid UTC timestamps")
    return values.map(lambda value: value.isoformat().replace("+00:00", "Z"))


def build_export(
    event_id: str | None = None,
    limit: int | None = None,
    sample_per_year: int | None = None,
) -> pd.DataFrame:
    if not EVENT_FILE.exists():
        raise RuntimeError(f"Missing validated event file: {EVENT_FILE}")

    frame = pd.read_parquet(EVENT_FILE, engine="pyarrow")
    required = set(BASE_COLUMNS + ["model_eligible"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Event file is missing columns: " + ", ".join(missing))

    eligible = frame[frame["model_eligible"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        raise RuntimeError("No model-eligible events are available")

    eligible["event_timestamp_utc"] = pd.to_datetime(
        eligible["event_timestamp_utc"], utc=True, errors="coerce"
    )
    eligible["reference_period"] = pd.to_datetime(
        eligible["reference_period"], utc=True, errors="coerce"
    )

    if eligible["event_timestamp_utc"].isna().any() or eligible["reference_period"].isna().any():
        raise RuntimeError("Eligible events contain invalid event/reference timestamps")
    if eligible["event_id"].duplicated().any():
        raise RuntimeError("Eligible event table contains duplicate event_id values")

    if event_id:
        eligible = eligible[eligible["event_id"].eq(event_id)].copy()
        if eligible.empty:
            raise RuntimeError(f"Requested event_id not found among model-eligible rows: {event_id}")

    eligible = eligible.sort_values("event_timestamp_utc").reset_index(drop=True)

    if sample_per_year is not None:
        if sample_per_year <= 0:
            raise RuntimeError("--sample-per-year must be greater than zero")
        if event_id is not None:
            raise RuntimeError("--sample-per-year cannot be combined with --event-id")
        eligible = eligible.assign(_sample_year=eligible["event_timestamp_utc"].dt.year)
        eligible = (
            eligible.groupby("_sample_year", sort=True, group_keys=False)
            .head(sample_per_year)
            .drop(columns=["_sample_year"])
            .reset_index(drop=True)
        )

    if limit is not None:
        if limit <= 0:
            raise RuntimeError("--limit must be greater than zero")
        eligible = eligible.head(limit).copy()

    exported = eligible[BASE_COLUMNS].copy()
    exported["event_timestamp_utc"] = _iso_utc(exported["event_timestamp_utc"])
    exported["reference_period"] = _iso_utc(exported["reference_period"])

    for column in BASE_COLUMNS:
        text = exported[column].astype(str)
        if text.str.contains("\t|\r|\n", regex=True).any():
            raise RuntimeError(f"Column {column!r} contains a tab or newline and is unsafe for TSV export")

    return exported


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export validated economic events to a JForex-friendly TSV file."
    )
    parser.add_argument("--event-id", help="Export only one exact event_id")
    parser.add_argument("--limit", type=int, help="Export only the earliest N eligible events")
    parser.add_argument(
        "--sample-per-year",
        type=int,
        help="Deterministically export the earliest N eligible events from each calendar year",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    exported = build_export(
        event_id=args.event_id,
        limit=args.limit,
        sample_per_year=args.sample_per_year,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output, sep="\t", index=False, encoding="utf-8")

    print(f"Exported events: {len(exported):,}")
    print(f"First event UTC: {exported['event_timestamp_utc'].iloc[0]}")
    print(f"Last event UTC:  {exported['event_timestamp_utc'].iloc[-1]}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
