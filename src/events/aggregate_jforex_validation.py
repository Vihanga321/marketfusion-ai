"""Aggregate batched robust JForex validation outputs with strict integrity checks.

The aggregator never fills missing market data. It verifies that every expected
validated event appears exactly once, separates provider-side INCOMPLETE windows
from true reconstruction MISMATCH/ERROR states, and writes one auditable report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .resume_jforex_validation import compatible_result
except ImportError:  # Support direct execution.
    from resume_jforex_validation import compatible_result


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPECTED = ROOT / "data" / "dukascopy" / "events_for_jforex.tsv"
DEFAULT_BATCH_ROOT = ROOT / "data" / "dukascopy" / "full_validation_batches"
DEFAULT_OUTPUT = ROOT / "data" / "dukascopy" / "full_validation_summary.tsv"
DEFAULT_TEXT_REPORT = ROOT / "reports" / "dukascopy_full_validation_summary.txt"
DEFAULT_INCOMPLETE_REPORT = ROOT / "reports" / "dukascopy_incomplete_events.tsv"

SUMMARY_FILENAME = "robust_tick_validation_summary.tsv"
CHUNK_FILENAME = "robust_tick_chunk_status.tsv"
VALID_STATUSES = {"PASS", "INCOMPLETE", "MISMATCH", "ERROR"}
PRICE_TOLERANCE = 1.0e-8
EXPECTED_MINUTES = 261
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


def _atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _failure_reason(row: pd.Series) -> str:
    reported = str(row.get("failure_reason", "")).strip()
    if reported:
        return reported
    if row["status"] == "PASS":
        return ""
    if row["status"] == "MISMATCH":
        return "INVALID_SPREAD" if row["invalid_spread_ticks"] > 0 else "DATA_MISMATCH"
    if row["status"] == "ERROR":
        return "CODE_ERROR"
    native_missing = row["native_bid_bars"] != row["expected_minutes"]
    tick_missing = row["rebuilt_minutes"] != row["expected_minutes"]
    if native_missing and tick_missing:
        return "NATIVE_AND_TICK_HISTORY_INCOMPLETE"
    if native_missing:
        return "NATIVE_REFERENCE_UNAVAILABLE"
    if tick_missing:
        return "TICK_HISTORY_INCOMPLETE"
    return "PROVIDER_UNAVAILABLE"


def _classify_provider_error(error: str) -> str:
    lowered = str(error).lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "NETWORK_TIMEOUT"
    if "503" in lowered or "service unavailable" in lowered or "datacache" in lowered:
        return "PROVIDER_UNAVAILABLE"
    if "no ticks" in lowered or "no native" in lowered or "no bars" in lowered:
        return "HISTORY_EMPTY"
    return "PROVIDER_UNAVAILABLE"


def _incomplete_chunks(
    files: list[Path], result: pd.DataFrame, destination: Path
) -> pd.DataFrame:
    parts = []
    for summary_path in files:
        chunk_path = summary_path.with_name(CHUNK_FILENAME)
        if not chunk_path.is_file():
            raise RuntimeError(f"Missing chunk diagnostics: {chunk_path}")
        chunks = pd.read_csv(chunk_path, sep="\t", dtype=str, keep_default_na=False)
        required = {"event_id", "data_kind", "chunk_from_utc", "chunk_to_utc", "attempts", "rows", "status", "error"}
        missing = sorted(required.difference(chunks.columns))
        if missing:
            raise RuntimeError(f"{chunk_path} missing chunk columns: {', '.join(missing)}")
        unknown = sorted(set(chunks["status"]).difference({"OK", "MISSING"}))
        if unknown:
            raise RuntimeError(f"{chunk_path} contains unknown chunk statuses: {', '.join(unknown)}")
        attempts = pd.to_numeric(chunks["attempts"], errors="coerce")
        rows = pd.to_numeric(chunks["rows"], errors="coerce")
        if attempts.isna().any() or rows.isna().any() or (attempts < 1).any() or (rows < 0).any():
            raise RuntimeError(f"{chunk_path} contains invalid numeric diagnostics")
        chunks["attempts"] = attempts.astype("int64")
        chunks["rows"] = rows.astype("int64")
        missing_chunks = chunks[chunks["status"].eq("MISSING")].copy()
        if not missing_chunks.empty:
            missing_chunks["batch_id"] = summary_path.parents[1].name
            parts.append(missing_chunks)

    columns = [
        "event_id", "event_type", "event_timestamp_utc", "batch_id", "missing_source",
        "missing_window_from_utc", "missing_window_to_utc", "reason", "retry_count", "provider_error",
    ]
    if parts:
        gaps = pd.concat(parts, ignore_index=True)
        incomplete_ids = set(result.loc[result["status"].eq("INCOMPLETE"), "event_id"])
        gaps = gaps[gaps["event_id"].isin(incomplete_ids)].copy()
        metadata = result.set_index("event_id")[["event_type", "event_timestamp_utc"]]
        gaps = gaps.join(metadata, on="event_id")
        gaps["missing_source"] = gaps["data_kind"].map(
            lambda value: "native_bid_m1" if str(value).startswith("native_bid") else "tick_history"
        )
        if "reason" in gaps.columns:
            gaps["reason"] = gaps["reason"].where(
                gaps["reason"].astype(str).str.strip().ne(""),
                gaps["error"].map(_classify_provider_error),
            )
        else:
            gaps["reason"] = gaps["error"].map(_classify_provider_error)
        output = pd.DataFrame({
            "event_id": gaps["event_id"],
            "event_type": gaps["event_type"],
            "event_timestamp_utc": gaps["event_timestamp_utc"],
            "batch_id": gaps["batch_id"],
            "missing_source": gaps["missing_source"],
            "missing_window_from_utc": gaps["chunk_from_utc"],
            "missing_window_to_utc": gaps["chunk_to_utc"],
            "reason": gaps["reason"],
            "retry_count": gaps["attempts"],
            "provider_error": gaps["error"],
        })
    else:
        output = pd.DataFrame(columns=columns)
    incomplete_rows = result[result["status"].eq("INCOMPLETE")]
    represented = set(output["event_id"]) if not output.empty else set()
    uncovered = incomplete_rows[~incomplete_rows["event_id"].isin(represented)]
    if not uncovered.empty:
        synthetic = pd.DataFrame({
            "event_id": uncovered["event_id"],
            "event_type": uncovered["event_type"],
            "event_timestamp_utc": uncovered["event_timestamp_utc"],
            "batch_id": uncovered["batch_id"],
            "missing_source": uncovered["failure_reason"].map(
                lambda reason: "native_bid_m1" if "NATIVE" in str(reason) else "tick_history"
            ),
            "missing_window_from_utc": "",
            "missing_window_to_utc": "",
            "reason": uncovered["failure_reason"],
            "retry_count": 0,
            "provider_error": uncovered["error"],
        })
        output = pd.concat([output, synthetic], ignore_index=True)
    _atomic_tsv(output, destination)
    return output


def aggregate(
    expected_path: Path,
    batch_root: Path,
    output_path: Path,
    text_report: Path,
    incomplete_report: Path | None = None,
    require_metadata: bool = True,
) -> pd.DataFrame:
    if incomplete_report is None:
        incomplete_report = text_report.with_name("dukascopy_incomplete_events.tsv")
    expected = _read_expected(expected_path)
    files = _summary_files(batch_root)

    parts: list[pd.DataFrame] = []
    for path in files:
        if require_metadata:
            batch_dir = path.parents[1]
            if not compatible_result(batch_dir / "events.tsv", path.parent):
                raise RuntimeError(f"Incompatible or missing result metadata for {batch_dir.name}")
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

    optional_integer_columns = [
        column for column in (
            "invalid_quote_ticks", "duplicate_ticks", "native_missing_chunks", "tick_missing_chunks"
        ) if column in result.columns
    ]
    for column in optional_integer_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise RuntimeError(f"Validation summaries contain invalid non-negative integer field {column}")
        result[column] = values.astype("int64")
    if "invalid_quote_ticks" not in result.columns:
        result["invalid_quote_ticks"] = result["invalid_spread_ticks"]

    integer_columns = [column for column in numeric_columns if column != "max_abs_ohlc_diff"]
    for column in integer_columns:
        values = result[column]
        if (values < 0).any() or (values % 1 != 0).any():
            raise RuntimeError(f"Validation summaries contain invalid non-negative integer field {column}")
        result[column] = values.astype("int64")
    if (result["max_abs_ohlc_diff"] < 0).any():
        raise RuntimeError("Validation summaries contain negative max_abs_ohlc_diff")
    if result["expected_minutes"].ne(EXPECTED_MINUTES).any():
        raise RuntimeError(f"Every validation row must expect exactly {EXPECTED_MINUTES} M1 minutes")

    expected_timestamps = pd.to_datetime(
        expected.set_index("event_id")["event_timestamp_utc"], utc=True, errors="coerce"
    )
    actual_timestamps = pd.to_datetime(result["event_timestamp_utc"], utc=True, errors="coerce")
    aligned_expected = pd.to_datetime(result["event_id"].map(expected_timestamps), utc=True, errors="coerce")
    if actual_timestamps.isna().any() or aligned_expected.isna().any() or (actual_timestamps != aligned_expected).any():
        raise RuntimeError("Validation event timestamps do not match the canonical event table")

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
        | (pass_rows["invalid_quote_ticks"] != 0)
        | (pass_rows["missing_chunks"] != 0)
        | (pass_rows["max_abs_ohlc_diff"] > PRICE_TOLERANCE)
    ]
    if not bad_pass.empty:
        raise RuntimeError(f"{len(bad_pass)} PASS row(s) fail strict coverage invariants")

    incomplete_rows = result[result["status"].eq("INCOMPLETE")]
    bad_incomplete = incomplete_rows[
        (incomplete_rows["mismatched_minutes"] != 0)
        | (incomplete_rows["invalid_spread_ticks"] != 0)
        | (incomplete_rows["invalid_quote_ticks"] != 0)
        | (
            (incomplete_rows["native_bid_bars"] == incomplete_rows["expected_minutes"])
            & (incomplete_rows["rebuilt_minutes"] == incomplete_rows["expected_minutes"])
            & (incomplete_rows["missing_minutes"] == 0)
            & (incomplete_rows["missing_chunks"] == 0)
        )
    ]
    if not bad_incomplete.empty:
        raise RuntimeError(f"{len(bad_incomplete)} INCOMPLETE row(s) have inconsistent diagnostics")
    mismatch_rows = result[result["status"].eq("MISMATCH")]
    bad_mismatch = mismatch_rows[
        (mismatch_rows["mismatched_minutes"] == 0)
        & (mismatch_rows["invalid_spread_ticks"] == 0)
        & (mismatch_rows["invalid_quote_ticks"] == 0)
        & (mismatch_rows["max_abs_ohlc_diff"] <= PRICE_TOLERANCE)
    ]
    if not bad_mismatch.empty:
        raise RuntimeError(f"{len(bad_mismatch)} MISMATCH row(s) have no mismatch evidence")
    error_rows = result[result["status"].eq("ERROR")]
    if error_rows["error"].astype(str).str.strip().eq("").any():
        raise RuntimeError("ERROR rows must contain a diagnostic message")

    event_metadata = expected[["event_id", "event_type"]]
    result = result.merge(event_metadata, on="event_id", how="left", validate="one_to_one")
    result["failure_reason"] = result.apply(_failure_reason, axis=1)
    result["model_eligible_market_reaction"] = result["status"].eq("PASS")
    result = result.sort_values(["event_timestamp_utc", "event_type", "event_id"]).reset_index(drop=True)

    _atomic_tsv(result, output_path)
    incomplete_chunks = _incomplete_chunks(files, result, incomplete_report)

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
    incomplete_causes = incomplete["failure_reason"].value_counts()
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
        "- model_eligible_market_reaction is true only for strict PASS rows.",
    ]
    if not incomplete.empty:
        lines.extend([
            "",
            "INCOMPLETE EVENTS",
            incomplete[["event_id", "event_type", "missing_minutes", "missing_chunks", "error"]].to_string(index=False),
        ])
        lines.extend(["", "INCOMPLETE CAUSES", incomplete_causes.to_string()])
        lines.append(f"Detailed provider-gap rows: {len(incomplete_chunks):,} ({incomplete_report})")

    _atomic_text(text_report, "\n".join(lines) + "\n")

    print(f"Expected events: {len(expected):,}")
    print(f"Validated events: {len(result):,}")
    print(f"PASS: {int(counts['PASS']):,}")
    print(f"INCOMPLETE: {int(counts['INCOMPLETE']):,}")
    print(f"MISMATCH: {int(counts['MISMATCH']):,}")
    print(f"ERROR: {int(counts['ERROR']):,}")
    print(f"FULL_VALIDATION_STATUS: {overall}")
    print(f"Summary TSV: {output_path.resolve()}")
    print(f"Report: {text_report.resolve()}")
    print(f"Incomplete-event quarantine: {incomplete_report.resolve()}")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-events", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_TEXT_REPORT)
    parser.add_argument("--incomplete-report", type=Path, default=DEFAULT_INCOMPLETE_REPORT)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    result = aggregate(
        args.expected_events.resolve(),
        args.batch_root.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        args.incomplete_report.resolve(),
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
