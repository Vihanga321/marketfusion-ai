"""Split the full validated JForex event bridge into small deterministic batches.

Small batches keep each credentialed JForex run comfortably below the Java
validator timeout and make transient Dukascopy datafeed gaps retryable without
restarting the complete historical audit.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "dukascopy" / "events_for_jforex.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "dukascopy" / "full_validation_batches"
REQUIRED_COLUMNS = {
    "event_id",
    "event_type",
    "reference_period",
    "event_timestamp_utc",
    "timestamp_precision",
    "timestamp_source",
}


def _load_events(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing event TSV: {path}")

    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError("Event TSV is missing columns: " + ", ".join(missing))
    if frame.empty:
        raise RuntimeError("Event TSV contains no rows")
    if frame["event_id"].eq("").any():
        raise RuntimeError("Event TSV contains empty event_id values")
    if frame["event_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["event_id"].duplicated(False), "event_id"].unique())
        raise RuntimeError("Event TSV contains duplicate event_id values: " + ", ".join(duplicates[:10]))
    if frame["event_type"].str.strip().eq("").any():
        raise RuntimeError("Event TSV contains empty event_type values")

    timestamps = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise RuntimeError("Event TSV contains invalid event_timestamp_utc values")

    frame = frame.assign(_event_timestamp=timestamps)
    frame = frame.sort_values(["_event_timestamp", "event_type", "event_id"]).reset_index(drop=True)
    return frame


def prepare_batches(input_path: Path, output_dir: Path, batch_size: int) -> pd.DataFrame:
    if batch_size < 1:
        raise RuntimeError("--batch-size must be at least 1")

    frame = _load_events(input_path)

    # Start clean so a shorter rerun can never leave stale batch files that would
    # be accidentally included in a later aggregation.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, object]] = []
    exported_columns = [column for column in frame.columns if not column.startswith("_")]

    for batch_number, start in enumerate(range(0, len(frame), batch_size), start=1):
        batch = frame.iloc[start : start + batch_size].copy()
        batch_id = f"batch_{batch_number:03d}"
        batch_dir = output_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_file = batch_dir / "events.tsv"
        batch[exported_columns].to_csv(batch_file, sep="\t", index=False, encoding="utf-8")

        manifests.append(
            {
                "batch_id": batch_id,
                "event_count": len(batch),
                "first_event_utc": batch["_event_timestamp"].iloc[0].isoformat().replace("+00:00", "Z"),
                "last_event_utc": batch["_event_timestamp"].iloc[-1].isoformat().replace("+00:00", "Z"),
                "event_types": ",".join(sorted(batch["event_type"].unique())),
                "event_file": str(batch_file.resolve()),
            }
        )

    manifest = pd.DataFrame(manifests)
    manifest_file = output_dir / "batch_manifest.tsv"
    manifest.to_csv(manifest_file, sep="\t", index=False, encoding="utf-8")

    if int(manifest["event_count"].sum()) != len(frame):
        raise RuntimeError("Batch manifest row count does not equal source event count")

    print(f"Events prepared: {len(frame):,}")
    print(f"Batch size: {batch_size}")
    print(f"Batches: {len(manifest):,}")
    print(f"Manifest: {manifest_file.resolve()}")
    return manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    prepare_batches(args.input.resolve(), args.output_dir.resolve(), args.batch_size)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
