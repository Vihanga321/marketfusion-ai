"""Validate macro provenance and publish only point-in-time-safe observations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import DATA_DIRECTORY, QUARANTINE_DIRECTORY, REPORT_DIRECTORY, STANDARD_COLUMNS, atomic_parquet
except ImportError:  # Support direct execution.
    from common import DATA_DIRECTORY, QUARANTINE_DIRECTORY, REPORT_DIRECTORY, STANDARD_COLUMNS, atomic_parquet


SAFE_OUTPUT = DATA_DIRECTORY / "macro_safe.parquet"
REPORT_FILE = REPORT_DIRECTORY / "macro_data_quality_report.txt"
ADAPTER_FILES = (DATA_DIRECTORY / "fred_macro.parquet", DATA_DIRECTORY / "ecb_macro.parquet")
REQUIRED_COMPLETE_FILES = {"fred_macro.parquet", "ecb_macro.parquet"}
EXPECTED_SERIES = {
    "fed_funds_rate", "us_2y_yield", "us_10y_yield", "us_cpi_yoy", "us_core_cpi_yoy",
    "us_pce", "us_core_pce", "us_unemployment", "us_payroll_growth", "us_gdp_growth",
    "us_retail_sales_yoy", "ecb_rate", "euro_hicp", "euro_core_hicp",
    "euro_unemployment", "euro_gdp_growth", "euro_2y_yield", "euro_10y_yield",
}

# Complete V0.3 is defined as a 2015-present US macro history. These minimums
# prevent a malformed adapter response from passing merely because every series
# name exists. They are deliberately conservative relative to the expected
# number of observations over the full period.
FRED_MIN_UNIQUE_OBSERVATIONS = {
    "fed_funds_rate": 1_000,
    "us_2y_yield": 1_000,
    "us_10y_yield": 1_000,
    "us_cpi_yoy": 100,
    "us_core_cpi_yoy": 100,
    "us_pce": 100,
    "us_core_pce": 100,
    "us_unemployment": 100,
    "us_payroll_growth": 100,
    "us_gdp_growth": 30,
    "us_retail_sales_yoy": 100,
}


@dataclass
class ValidationResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    safe_rows: int


def load_frames(paths: tuple[Path, ...]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    loaded = []
    for path in paths:
        if path.exists():
            frame = pd.read_parquet(path, engine="pyarrow")
            frame["input_file"] = path.name
            frames.append(frame)
            loaded.append(path.name)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), loaded


def validate(write_report: bool = True, require_complete: bool = False) -> ValidationResult:
    data, loaded = load_frames(ADAPTER_FILES)
    quarantine_paths = tuple(sorted(QUARANTINE_DIRECTORY.glob("*.parquet"))) if QUARANTINE_DIRECTORY.exists() else ()
    quarantine, quarantine_loaded = load_frames(quarantine_paths)
    failures: list[str] = []
    warnings: list[str] = []
    details: list[str] = []

    missing_required_files = sorted(REQUIRED_COMPLETE_FILES.difference(loaded))
    if require_complete and missing_required_files:
        failures.append("Complete macro validation requires adapter files: " + ", ".join(missing_required_files))

    if data.empty:
        failures.append("No model-eligible adapter files were found")
        safe = pd.DataFrame()
    else:
        missing_columns = sorted(set(STANDARD_COLUMNS).difference(data.columns))
        if missing_columns:
            failures.append("Missing standard columns: " + ", ".join(missing_columns))
            safe = pd.DataFrame()
        else:
            data["observation_period"] = pd.to_datetime(data["observation_period"], utc=True, errors="coerce")
            data["available_from_utc"] = pd.to_datetime(data["available_from_utc"], utc=True, errors="coerce")
            data["vintage_date"] = pd.to_datetime(data["vintage_date"], utc=True, errors="coerce")
            data["value"] = pd.to_numeric(data["value"], errors="coerce")
            eligible = data.get("model_eligible", pd.Series(True, index=data.index)).fillna(False).astype(bool)
            safe = data[eligible].copy()

            invalid_timestamps = safe[["observation_period", "available_from_utc"]].isna().any(axis=1)
            if invalid_timestamps.any():
                failures.append(f"{int(invalid_timestamps.sum())} eligible rows have invalid timestamps")
            nan_values = safe["value"].isna()
            if nan_values.any():
                failures.append(f"{int(nan_values.sum())} eligible rows have NaN values")
            finite = np.isfinite(safe["value"].to_numpy(dtype=float, na_value=np.nan))
            if (~finite).any():
                failures.append(f"{int((~finite).sum())} eligible rows have non-finite values")
            before_period = safe["available_from_utc"] < safe["observation_period"]
            if before_period.any():
                failures.append(f"{int(before_period.sum())} rows are available before their observation period")
            future = safe["available_from_utc"] > pd.Timestamp.now(tz="UTC")
            if future.any():
                failures.append(f"{int(future.sum())} rows have future availability timestamps")
            missing_vintage = safe["vintage_date"].isna()
            if missing_vintage.any():
                failures.append(f"{int(missing_vintage.sum())} eligible rows lack vintage metadata")

            duplicate_key = safe.duplicated(
                ["series_id", "observation_period", "available_from_utc", "value"], keep=False
            )
            if duplicate_key.any():
                failures.append(f"{int(duplicate_key.sum())} eligible rows participate in exact duplicates")
            ambiguous = (
                safe.groupby(["series_id", "observation_period", "available_from_utc"], dropna=False)["value"]
                .nunique(dropna=False).gt(1)
            )
            if ambiguous.any():
                failures.append(f"{int(ambiguous.sum())} release keys contain conflicting values")

            frequency_count = safe.groupby("series_id")["frequency"].nunique(dropna=False)
            inconsistent = frequency_count[frequency_count != 1]
            if not inconsistent.empty:
                failures.append("Inconsistent frequency metadata: " + ", ".join(inconsistent.index))

            revision_counts = safe.groupby(["series_id", "observation_period"])["value"].nunique()
            revisions = revision_counts[revision_counts > 1]
            details.append(f"Observation periods with multiple recorded values: {len(revisions):,}")
            if len(revisions):
                details.append("Revision counts by series: " + str(
                    revisions.reset_index().groupby("series_id").size().to_dict()
                ))

            available_series = set(safe["series_id"].unique())
            missing_series = sorted(EXPECTED_SERIES.difference(available_series))
            if missing_series:
                message = "Expected series unavailable for safe modeling: " + ", ".join(missing_series)
                if require_complete:
                    failures.append(message)
                else:
                    warnings.append(message)

            if "fred_macro.parquet" not in loaded:
                warnings.append("FRED/ALFRED file absent; U.S. point-in-time macro coverage is incomplete")
            if "ecb_macro.parquet" not in loaded:
                warnings.append("ECB file absent; Eurozone point-in-time macro coverage is incomplete")

            if require_complete and "fred_macro.parquet" in loaded:
                fred = safe[safe["input_file"] == "fred_macro.parquet"]
                freshness_floor = pd.Timestamp(
                    year=max(2015, pd.Timestamp.now(tz="UTC").year - 1), month=1, day=1, tz="UTC"
                )
                for series_id, minimum in FRED_MIN_UNIQUE_OBSERVATIONS.items():
                    group = fred[fred["series_id"] == series_id]
                    if group.empty:
                        failures.append(f"FRED complete coverage missing series: {series_id}")
                        continue
                    observations = int(group["observation_period"].nunique())
                    if observations < minimum:
                        failures.append(
                            f"FRED {series_id} has only {observations} unique observations; "
                            f"minimum for complete 2015-present coverage is {minimum}"
                        )
                    availability_years = int(group["available_from_utc"].dt.year.nunique())
                    if availability_years < 5:
                        failures.append(
                            f"FRED {series_id} spans only {availability_years} availability years; "
                            "historical point-in-time coverage is incomplete"
                        )
                    latest = group["available_from_utc"].max()
                    if pd.isna(latest) or latest < freshness_floor:
                        failures.append(
                            f"FRED {series_id} latest availability is {latest}; expected coverage into "
                            f"{freshness_floor.year} or later"
                        )

    if not quarantine.empty:
        warnings.append(f"{len(quarantine):,} non-model-eligible rows remain quarantined")

    passed = not failures and not safe.empty
    if passed:
        publish = safe.drop(columns=["input_file"], errors="ignore").sort_values(
            ["series_id", "available_from_utc", "observation_period"]
        )
        atomic_parquet(publish, SAFE_OUTPUT)

    lines = [
        "MARKETFUSION AI V0.3 - MACRO DATA QUALITY REPORT",
        "=" * 68,
        "Validation contract: point-in-time availability and vintage metadata",
        f"Validation mode: {'COMPLETE' if require_complete else 'SAFE_SUBSET'}",
        f"Adapter files: {', '.join(loaded) if loaded else 'none'}",
        f"Quarantine files: {', '.join(quarantine_loaded) if quarantine_loaded else 'none'}",
        f"Eligible input rows: {len(safe):,}",
        f"Series represented: {safe['series_id'].nunique() if not safe.empty else 0}",
        "",
        "SERIES COVERAGE",
    ]
    if not safe.empty:
        for series_id, group in safe.groupby("series_id"):
            lines.append(
                f"- {series_id}: rows={len(group):,}, observations={group['observation_period'].nunique():,}, "
                f"available={group['available_from_utc'].min()} to {group['available_from_utc'].max()}"
            )
    lines.extend(["", "REVISION CHECKS", *(details or ["No revision detail available"]), "", "FAILURES"])
    lines.extend([f"- {item}" for item in failures] or ["- None"])
    lines.extend(["", "WARNINGS"])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend([
        "",
        "LEAKAGE POLICY",
        "- Eligible values require available_from_utc, vintage_date, and available_from_utc >= observation_period.",
        "- ALFRED date-only vintages are delayed to the next UTC day.",
        "- ECB revised series use SDMX VALID_FROM; policy-rate effective dates are treated as non-revised.",
        "- BLS current values without release/vintage timestamps are quarantined.",
        "",
        f"VALIDATION_STATUS: {'PASS' if passed else 'FAIL'}",
    ])
    if write_report:
        REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"VALIDATION_STATUS: {'PASS' if passed else 'FAIL'}")
        print(f"Validation mode: {'COMPLETE' if require_complete else 'SAFE_SUBSET'}")
        print(f"Safe rows: {len(safe):,}")
        print(f"Report: {REPORT_FILE}")
        if passed:
            print(f"Published safe macro data: {SAFE_OUTPUT}")
    return ValidationResult(passed, failures, warnings, len(safe))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless both FRED/ALFRED and ECB model-eligible adapter files are present with complete US coverage.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    result = validate(write_report=True, require_complete=args.require_complete)
    if not result.passed:
        raise SystemExit("ERROR: Macro validation failed: " + "; ".join(result.failures))
