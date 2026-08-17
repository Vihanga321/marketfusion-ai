"""Validate V0.4A economic-event timestamps before any market reaction join."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from .common import DATA_DIRECTORY, REPORT_DIRECTORY, STANDARD_EVENT_COLUMNS
except ImportError:  # Support direct execution.
    from common import DATA_DIRECTORY, REPORT_DIRECTORY, STANDARD_EVENT_COLUMNS


BLS_FILE = DATA_DIRECTORY / "bls_release_events.parquet"
REPORT_FILE = REPORT_DIRECTORY / "event_data_quality_report.txt"
COVERAGE_FILE = REPORT_DIRECTORY / "event_coverage.csv"
EXPECTED_BLS_TYPES = {"us_cpi_release", "us_employment_situation"}
DIRECT_TIMESTAMP_SOURCE = "BLS annual release calendar exact Eastern Time"
RECONSTRUCTED_TIMESTAMP_SOURCE = "ALFRED first vintage date + official BLS 08:30 Eastern release time"
ALLOWED_TIMESTAMP_SOURCES = {DIRECT_TIMESTAMP_SOURCE, RECONSTRUCTED_TIMESTAMP_SOURCE}
NEW_YORK = ZoneInfo("America/New_York")
MIN_RELEASE_LAG_DAYS = 1
MAX_RELEASE_LAG_DAYS = 25


@dataclass
class ValidationResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    rows: int


def validate(write_report: bool = True) -> ValidationResult:
    failures: list[str] = []
    warnings: list[str] = []

    if not BLS_FILE.exists():
        failures.append(f"Missing BLS event file: {BLS_FILE}")
        frame = pd.DataFrame()
    else:
        frame = pd.read_parquet(BLS_FILE, engine="pyarrow")

    if not frame.empty:
        missing = sorted(set(STANDARD_EVENT_COLUMNS).difference(frame.columns))
        if missing:
            failures.append("Missing event columns: " + ", ".join(missing))
        else:
            frame["event_timestamp_utc"] = pd.to_datetime(
                frame["event_timestamp_utc"], utc=True, errors="coerce"
            )
            frame["reference_period"] = pd.to_datetime(
                frame["reference_period"], utc=True, errors="coerce"
            )
            frame["model_eligible"] = frame["model_eligible"].fillna(False).astype(bool)
            eligible = frame[frame["model_eligible"]].copy()

            if eligible.empty:
                failures.append("No model-eligible events")
            if eligible["event_timestamp_utc"].isna().any():
                failures.append("Eligible events contain missing/invalid UTC timestamps")
            if eligible["reference_period"].isna().any():
                failures.append("Eligible events contain missing/invalid reference periods")
            if eligible["event_id"].duplicated().any():
                failures.append("Duplicate eligible event_id values detected")
            missing_event_type = eligible["event_type"].isna() | eligible["event_type"].astype(str).str.strip().eq("")
            if missing_event_type.any():
                failures.append(f"{int(missing_event_type.sum())} eligible rows have missing event_type")

            not_minute_aligned = (
                eligible["event_timestamp_utc"].dt.second.ne(0)
                | eligible["event_timestamp_utc"].dt.microsecond.ne(0)
                | eligible["event_timestamp_utc"].dt.nanosecond.ne(0)
            )
            if not_minute_aligned.any():
                failures.append(f"{int(not_minute_aligned.sum())} eligible timestamps are not exact minute boundaries")

            bad_reference_start = (
                eligible["reference_period"].dt.day.ne(1)
                | eligible["reference_period"].dt.hour.ne(0)
                | eligible["reference_period"].dt.minute.ne(0)
                | eligible["reference_period"].dt.second.ne(0)
            )
            if bad_reference_start.any():
                failures.append(f"{int(bad_reference_start.sum())} reference periods are not UTC month starts")

            duplicate_reference = eligible.duplicated(
                ["event_type", "reference_period"], keep=False
            )
            if duplicate_reference.any():
                failures.append(
                    f"{int(duplicate_reference.sum())} eligible rows duplicate event_type/reference_period"
                )

            future = eligible["event_timestamp_utc"] > pd.Timestamp.now(tz="UTC")
            if future.any():
                failures.append(f"{int(future.sum())} eligible events have future timestamps")

            # Compare year-month ordinals directly. Using .dt.to_period("M") on
            # timezone-aware timestamps emits a warning because pandas drops the
            # timezone even though only the calendar month is needed here.
            release_month = (
                eligible["event_timestamp_utc"].dt.year * 12
                + eligible["event_timestamp_utc"].dt.month
            )
            reference_month = (
                eligible["reference_period"].dt.year * 12
                + eligible["reference_period"].dt.month
            )
            before_reference = release_month < reference_month
            if before_reference.any():
                failures.append(
                    f"{int(before_reference.sum())} releases occur before their reference month"
                )

            wrong_source = ~eligible["timestamp_source"].isin(ALLOWED_TIMESTAMP_SOURCES)
            if wrong_source.any():
                failures.append(
                    f"{int(wrong_source.sum())} eligible BLS rows use unsupported timestamp provenance"
                )

            reconstructed = eligible["timestamp_source"].eq(RECONSTRUCTED_TIMESTAMP_SOURCE)
            if reconstructed.any():
                warnings.append(
                    f"{int(reconstructed.sum())} eligible events use dual-official reconstruction: "
                    "ALFRED first-vintage release date plus BLS 08:30 Eastern time; direct BLS calendar access was blocked"
                )
                if "provenance_tier" not in eligible.columns:
                    failures.append("Reconstructed events lack provenance_tier metadata")
                else:
                    bad_tier = reconstructed & eligible["provenance_tier"].ne("dual_official_reconstruction")
                    if bad_tier.any():
                        failures.append(
                            f"{int(bad_tier.sum())} reconstructed events have invalid provenance_tier"
                        )
                if "alfred_crosscheck_series" not in eligible.columns:
                    failures.append("Reconstructed events lack ALFRED cross-check metadata")

            direct = eligible["timestamp_source"].eq(DIRECT_TIMESTAMP_SOURCE)
            if direct.any() and "provenance_tier" in eligible.columns:
                bad_direct = direct & eligible["provenance_tier"].ne("direct_official_calendar")
                if bad_direct.any():
                    failures.append(f"{int(bad_direct.sum())} direct-calendar events have invalid provenance tier")

            allowed_precision = eligible["timestamp_precision"].isin({"minute", "minute_reconstructed"})
            if (~allowed_precision).any():
                failures.append(
                    f"{int((~allowed_precision).sum())} eligible rows lack minute-level timestamp precision"
                )
            bad_reconstructed_precision = reconstructed & eligible["timestamp_precision"].ne("minute_reconstructed")
            if bad_reconstructed_precision.any():
                failures.append(
                    f"{int(bad_reconstructed_precision.sum())} reconstructed rows are not explicitly labeled minute_reconstructed"
                )

            wrong_timezone = eligible["local_timezone"].ne("America/New_York")
            if wrong_timezone.any():
                failures.append(
                    f"{int(wrong_timezone.sum())} eligible BLS rows use the wrong local timezone"
                )

            bad_local = ~eligible["stated_timezone_abbreviation"].isin({"EST", "EDT"})
            if bad_local.any():
                failures.append(
                    f"{int(bad_local.sum())} eligible rows lack a valid historical Eastern timezone abbreviation"
                )

            local_failures = 0
            lag_failures = 0
            id_failures = 0
            for row in eligible.itertuples():
                utc_timestamp = pd.Timestamp(row.event_timestamp_utc)
                expected_local = utc_timestamp.to_pydatetime().astimezone(NEW_YORK)
                try:
                    parsed_local = datetime.fromisoformat(str(row.event_timestamp_local).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    local_failures += 1
                    continue
                if (
                    parsed_local.tzinfo is None
                    or parsed_local.astimezone(ZoneInfo("UTC")) != utc_timestamp.to_pydatetime()
                    or parsed_local.utcoffset() != expected_local.utcoffset()
                    or (parsed_local.hour, parsed_local.minute, parsed_local.second, parsed_local.microsecond)
                    != (8, 30, 0, 0)
                    or str(row.stated_timezone_abbreviation) != expected_local.tzname()
                ):
                    local_failures += 1

                expected_suffix = utc_timestamp.strftime("%Y%m%dT%H%MZ")
                if not str(row.event_id).endswith(expected_suffix):
                    id_failures += 1

                reference_month_end = (pd.Timestamp(row.reference_period) + pd.offsets.MonthEnd(0)).date()
                lag_days = (expected_local.date() - reference_month_end).days
                if not (MIN_RELEASE_LAG_DAYS <= lag_days <= MAX_RELEASE_LAG_DAYS):
                    lag_failures += 1
            if local_failures:
                failures.append(f"{local_failures} eligible rows have inconsistent local/UTC/DST release timestamps")
            if id_failures:
                failures.append(f"{id_failures} eligible event IDs do not encode their UTC release timestamp")
            if lag_failures:
                failures.append(
                    f"{lag_failures} eligible releases fall outside the {MIN_RELEASE_LAG_DAYS}.."
                    f"{MAX_RELEASE_LAG_DAYS}-day post-reference-month window"
                )

            ordering_failures = 0
            for _, group in eligible.sort_values("reference_period").groupby("event_type"):
                if not group["event_timestamp_utc"].is_monotonic_increasing:
                    ordering_failures += 1
            if ordering_failures:
                failures.append(f"{ordering_failures} event types have non-monotonic release ordering")

            represented = set(eligible["event_type"].unique())
            missing_types = sorted(EXPECTED_BLS_TYPES.difference(represented))
            if missing_types:
                failures.append("Missing expected BLS event types: " + ", ".join(missing_types))

            if eligible["forecast"].notna().any():
                failures.append("Forecast values exist before a trusted consensus source is configured")

            coverage = (
                eligible.groupby("event_type")
                .agg(
                    event_count=("event_id", "count"),
                    reference_months=("reference_period", "nunique"),
                    first_release_utc=("event_timestamp_utc", "min"),
                    last_release_utc=("event_timestamp_utc", "max"),
                )
                .reset_index()
            )
            for row in coverage.itertuples():
                span_years = max(
                    1,
                    row.last_release_utc.year - row.first_release_utc.year + 1,
                )
                conservative_minimum = span_years * 8
                if row.reference_months < conservative_minimum:
                    failures.append(
                        f"{row.event_type} coverage is unexpectedly sparse: "
                        f"{row.reference_months} reference months across {span_years} years"
                    )
        if "eligible" not in locals():
            eligible = pd.DataFrame()
            coverage = pd.DataFrame()
    else:
        eligible = pd.DataFrame()
        coverage = pd.DataFrame()

    passed = not failures and not eligible.empty
    try:
        input_display = BLS_FILE.relative_to(DATA_DIRECTORY.parents[1])
    except ValueError:  # Unit-test/external inputs can legitimately live outside the repository.
        input_display = BLS_FILE
    lines = [
        "MARKETFUSION AI V0.4A - EVENT TIMESTAMP QUALITY REPORT",
        "=" * 72,
        "Validation contract: exact UTC minute + America/New_York DST + point-in-time provenance",
        f"Input: {input_display}",
        f"Eligible events: {len(eligible):,}",
        "",
        "COVERAGE",
    ]
    if not coverage.empty:
        for row in coverage.itertuples():
            lines.append(
                f"- {row.event_type}: events={row.event_count:,}, "
                f"reference_months={row.reference_months:,}, "
                f"UTC={row.first_release_utc} to {row.last_release_utc}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "FAILURES"])
    lines.extend([f"- {item}" for item in failures] or ["- None"])
    lines.extend(["", "WARNINGS"])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend(
        [
            "",
            "V0.4A SAFETY RULES",
            "- Preferred timestamps come directly from official annual BLS release calendars.",
            "- If BLS blocks calendar access, dates may be reconstructed from first non-bootstrap ALFRED vintages and cross-checked against a second official series.",
            "- Reconstructed rows use the official 08:30 Eastern CPI/Employment release time and are explicitly labeled minute_reconstructed.",
            "- America/New_York performs historical DST conversion; no fixed UTC offset is used.",
            "- Future scheduled releases are quarantined rather than treated as historical observations.",
            "- Consensus forecast values remain empty until a trustworthy historical provider exists.",
            "",
            f"VALIDATION_STATUS: {'PASS' if passed else 'FAIL'}",
        ]
    )

    if write_report:
        REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not coverage.empty:
            coverage.to_csv(COVERAGE_FILE, index=False)
        print(f"VALIDATION_STATUS: {'PASS' if passed else 'FAIL'}")
        print(f"Eligible events: {len(eligible):,}")
        print(f"Report: {REPORT_FILE}")
        if not coverage.empty:
            print(f"Coverage: {COVERAGE_FILE}")

    return ValidationResult(passed, failures, warnings, len(eligible))


if __name__ == "__main__":
    result = validate(write_report=True)
    if not result.passed:
        raise SystemExit("ERROR: Event validation failed: " + "; ".join(result.failures))
