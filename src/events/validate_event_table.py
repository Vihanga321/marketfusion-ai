"""Validate V0.4A economic-event timestamps before any market reaction join."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from .common import DATA_DIRECTORY, REPORT_DIRECTORY, STANDARD_EVENT_COLUMNS
except ImportError:  # Support direct execution.
    from common import DATA_DIRECTORY, REPORT_DIRECTORY, STANDARD_EVENT_COLUMNS


BLS_FILE = DATA_DIRECTORY / "bls_release_events.parquet"
REPORT_FILE = REPORT_DIRECTORY / "event_data_quality_report.txt"
COVERAGE_FILE = REPORT_DIRECTORY / "event_coverage.csv"
EXPECTED_BLS_TYPES = {"us_cpi_release", "us_employment_situation"}


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

            before_reference = (
                eligible["event_timestamp_utc"].dt.to_period("M").astype(str)
                < eligible["reference_period"].dt.to_period("M").astype(str)
            )
            if before_reference.any():
                failures.append(
                    f"{int(before_reference.sum())} releases occur before their reference month"
                )

            wrong_source = eligible["timestamp_source"].ne("BLS archived release embargo line")
            if wrong_source.any():
                failures.append(
                    f"{int(wrong_source.sum())} eligible BLS rows do not use archive embargo timestamps"
                )

            wrong_precision = eligible["timestamp_precision"].ne("minute")
            if wrong_precision.any():
                failures.append(
                    f"{int(wrong_precision.sum())} eligible rows lack minute timestamp precision"
                )

            wrong_timezone = eligible["local_timezone"].ne("America/New_York")
            if wrong_timezone.any():
                failures.append(
                    f"{int(wrong_timezone.sum())} eligible BLS rows use the wrong local timezone"
                )

            represented = set(eligible["event_type"].unique())
            missing_types = sorted(EXPECTED_BLS_TYPES.difference(represented))
            if missing_types:
                failures.append("Missing expected BLS event types: " + ", ".join(missing_types))

            # V0.4A deliberately has no consensus provider yet. A non-null forecast
            # would be a provenance error, not a feature.
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
    lines = [
        "MARKETFUSION AI V0.4A - EVENT TIMESTAMP QUALITY REPORT",
        "=" * 72,
        f"Generated UTC: {pd.Timestamp.now(tz='UTC').isoformat()}",
        f"Input: {BLS_FILE}",
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
            "- Exact model-eligible BLS timestamps come from archived release embargo lines.",
            "- America/New_York performs historical DST conversion; no fixed UTC offset is used.",
            "- Archive filename dates are audit checks, never the source of the event time.",
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
