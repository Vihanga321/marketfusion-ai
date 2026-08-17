"""Aggregate batched robust JForex validation outputs with strict integrity checks.

The aggregator never fills missing market data. It verifies that every expected
validated event appears exactly once, separates provider-side INCOMPLETE windows
from true reconstruction MISMATCH/ERROR states, and writes one auditable report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPECTED = ROOT / "data" / "dukascopy" / "events_for_jforex.tsv"
DEFAULT_BATCH_ROOT = ROOT / "data" / "dukascopy" / "full_validation_batches"
DEFAULT_OUTPUT = ROOT / "data" / "dukascopy" / "full_validation_summary.tsv"
DEFAULT_TEXT_REPORT = ROOT / "reports" / "dukascopy_full_validation_summary.txt"

SUMMARY_FILENAME = "robust_tick_validation_summary.tsv"
CHUNK_FILENAME = "robust_tick_chunk_status.tsv"
VALID_STATUSES = {"PASS", "INCOMPLETE", "MISMATCH", "ERROR"}
REQUIRED_SUMMARY_COLUMNS = {
    "event_id",
    "event_timestamp_utc",
    "expected_minutes",
    "native_bid_bars",
    "historical_ticks",
    "rebuilt_minutes",
    "matched_minutes",
    "mismatched_minutes",
    "missing_minutes",
    "invalid_spread_ticks",
    "missing_chunks",
    "max_abs_ohlc_diff",
    "status",
    "error",
}


def _read_expected(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing expected-events TSV: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"event_id", "event_type", "event_timestamp_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Expected-events TSV missing columns: " + ", ".join(missing))
    if frame.empty:
        raise RuntimeError("Expected-events TSV contains no rows")
    if frame["event_id"].eq("").any() or frame["event_id"].duplicated().any():
        raise RuntimeError("Expected-events TSV must contain unique non-empty event_id values")
    return frame


def _summary_files(batch_root: Path) -> list[Path]:
    if not batch_root.is_dir():
        raise RuntimeError(f"Missing validation batch directory: {batch_root}")
    files = sorted(batch_root.glob(f"batch_*/result/{SUMMARY_FILENAME}"))
    if not files:
        raise RuntimeError(f"No {SUMMARY_FILENAME} files found under {batch_root}")
    return files


def aggregate(expected_path: Path, batch_root: Path, output_path: Path, text_report: Path) -> pd.DataFrame:
    expected = _read_expected(expected_path)
    files = _summary_files(batch_root)

    parts: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        missing = sorted(REQUIRED_SUMMARY_COLUMNS.difference(frame.columns))
        if missing:
            raise RuntimeError(f"{path} missing summary columns: {', '.join(missing)}")
        frame = frame.copy()
        frame["batch_id"] = path.parents[1].name
        parts.append(frame)

    result = pd.concat(parts, ignore_index=True)
    if result.empty:
        raise RuntimeError("Validation summaries contain no rows")
    if result["event_id"].eq("").any():
        raise RuntimeError("Validation summaries contain an empty event_id")
    if result["event_id"].duplicated().any():
        duplicates = sorted(result.loc[result["event_id"].duplicated(False), "event_id"].unique())
        raise RuntimeError("Validation summaries contain duplicate events: " + ", ".join(duplicates[:10]))

    expected_ids = set(expected["event_id"])
    result_ids = set(result["event_id"])
    missing_ids = sorted(expected_ids.difference(result_ids))
    extra_ids = sorted(result_ids.difference(expected_ids))
    if missing_ids or extra_ids:
        details = []
        if missing_ids:
            details.append(f"missing expected events={len(missing_ids)}")
        if extra_ids:
            details.append(f"unexpected events={len(extra_ids)}")
        raise RuntimeError("Full-validation event-set mismatch: " + ", ".join(details))

    unknown_statuses = sorted(set(result["status"]).difference(VALID_STATUSES))
    if unknown_statuses:
        raise RuntimeError("Unknown validation status values: " + ", ".join(unknown_statuses))

    numeric_columns = [
        "expected_minutes",
        "native_bid_bars",
        "historical_ticks",
        "rebuilt_minutes",
        "matched_minutes",
        "mismatched_minutes",
        "missing_minutes",
        "invalid_spread_ticks",
        "missing_chunks",
        "max_abs_ohlc_diff",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"Validation summaries contain non-numeric {column} values")
        result[column] = values

    # A PASS row is accepted only if it is actually complete. This catches a
    # future validator regression where the label says PASS but coverage fields do not.
    pass_rows = result[result["status"].eq("PASS")]
    bad_pass = pass_rows[
        (pass_rows["native_bid_bars"] != pass_rows["expected_minutes"])
        | (pass_rows["rebuilt_minutes"] != pass_rows["expected_minutes"])
        | (pass_rows["matched_minutes"] != pass_rows["expected_minutes"])
        | (pass_rows["mismatched_minutes"] != 0)
        | (pass_rows["missing_minutes"] != 0)
        | (pass_rows["invalid_spread_ticks"] != 0)
        | (pass_rows["missing_chunks"] != 0)
    ]
    if not bad_pass.empty:
        raise RuntimeError(f"{len(bad_pass)} PASS row(s) fail strict coverage invariants")

    event_metadata = expected[["event_id", "event_type"]]
    result = result.merge(event_metadata, on="event_id", how="left", validate="one_to_one")
    result = result.sort_values(["event_timestamp_utc", "event_type", "event_id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    counts = result["status"].value_counts().reindex(["PASS", "INCOMPLETE", "MISMATCH", "ERROR"], fill_value=0)
    by_type = (
        result.groupby(["event_type", "status"]).size().unstack(fill_value=0)
        .reindex(columns=["PASS", "INCOMPLETE", "MISMATCH", "ERROR"], fill_value=0)
    )

    if counts["MISMATCH"] > 0 or counts["ERROR"] > 0:
        overall = "FAIL"
    elif counts["INCOMPLETE"] > 0:
        overall = "INCOMPLETE"
    else:
        overall = "PASS"

    incomplete = result[result["status"].eq("INCOMPLETE")]
    lines = [
        "MARKETFUSION AI - DUKASCOPY FULL EVENT VALIDATION",
        "=" * 72,
        f"Expected events: {len(expected):,}",
        f"Validated events: {len(result):,}",
        f"Batch summaries: {len(files):,}",
        f"PASS: {int(counts['PASS']):,}",
        f"INCOMPLETE: {int(counts['INCOMPLETE']):,}",
        f"MISMATCH: {int(counts['MISMATCH']):,}",
        f"ERROR: {int(counts['ERROR']):,}",
        f"FULL_VALIDATION_STATUS: {overall}",
        "",
        "BY EVENT TYPE",
        by_type.to_string(),
        "",
        "INTERPRETATION",
        "- PASS means complete minute coverage and exact native-BID OHLC agreement within tolerance.",
        "- INCOMPLETE means provider data were unavailable; missing minutes are never interpolated or fabricated.",
        "- MISMATCH means retrieved tick reconstruction disagreed with native BID or contained invalid spreads.",
        "- ERROR means the validator itself could not complete that event.",
    ]
    if not incomplete.empty:
        lines.extend([
            "",
            "INCOMPLETE EVENTS",
            incomplete[["event_id", "event_type", "missing_minutes", "missing_chunks", "error"]].to_string(index=False),
        ])

    text_report.parent.mkdir(parents=True, exist_ok=True)
    text_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Expected events: {len(expected):,}")
    print(f"Validated events: {len(result):,}")
    print(f"PASS: {int(counts['PASS']):,}")
    print(f"INCOMPLETE: {int(counts['INCOMPLETE']):,}")
    print(f"MISMATCH: {int(counts['MISMATCH']):,}")
    print(f"ERROR: {int(counts['ERROR']):,}")
    print(f"FULL_VALIDATION_STATUS: {overall}")
    print(f"Summary TSV: {output_path.resolve()}")
    print(f"Report: {text_report.resolve()}")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-events", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_TEXT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    result = aggregate(
        args.expected_events.resolve(),
        args.batch_root.resolve(),
        args.output.resolve(),
        args.report.resolve(),
    )
    # Provider-side incompleteness is represented in the report but is not a
    # process failure. MISMATCH/ERROR is a hard failure for automation.
    hard_fail = result["status"].isin(["MISMATCH", "ERROR"]).any()
    return 1 if hard_fail else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
