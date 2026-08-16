"""Reconstruct CPI and Employment Situation release timestamps from local ALFRED data.

Use this only when direct BLS annual-calendar access is blocked. The script keeps
only first non-bootstrap vintages that fall in a plausible next-month release
window, cross-checks each release date against a second official ALFRED series,
and quarantines ambiguous same-timestamp candidates instead of silently deduping.

No actual values or consensus forecasts are fabricated.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from .common import (
        DATA_DIRECTORY,
        QUARANTINE_DIRECTORY,
        atomic_parquet,
        normalize_event_schema,
    )
except ImportError:  # Support direct execution.
    from common import (
        DATA_DIRECTORY,
        QUARANTINE_DIRECTORY,
        atomic_parquet,
        normalize_event_schema,
    )


ROOT = Path(__file__).resolve().parents[2]
FRED_MACRO_FILE = ROOT / "data" / "macro" / "fred_macro.parquet"
OUTPUT_FILE = DATA_DIRECTORY / "bls_release_events.parquet"
QUARANTINE_FILE = QUARANTINE_DIRECTORY / "bls_release_events_unusable.parquet"
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
TIMESTAMP_SOURCE = "ALFRED first vintage date + official BLS 08:30 Eastern release time"

# CPI and Employment Situation are monthly releases. Their initial release should
# occur shortly after the reference month ends. A wide 1..25 day window rejects
# benchmark/backfill vintages while allowing holidays and unusual schedules.
MIN_RELEASE_LAG_DAYS = 1
MAX_RELEASE_LAG_DAYS = 25

RELEASES = {
    "us_cpi_release": {
        "event_name": "Consumer Price Index",
        "primary_series": "us_cpi_yoy",
        "crosscheck_series": "us_core_cpi_yoy",
        "source_series_id": "CPIAUCSL",
    },
    "us_employment_situation": {
        "event_name": "Employment Situation",
        "primary_series": "us_payroll_growth",
        "crosscheck_series": "us_unemployment",
        "source_series_id": "PAYEMS",
    },
}


def first_real_vintage(frame: pd.DataFrame, series_id: str) -> pd.DataFrame:
    source = frame[frame["series_id"].eq(series_id)].copy()
    if source.empty:
        return source
    if "availability_method" in source.columns:
        source = source[
            ~source["availability_method"].astype(str).str.contains(
                "bootstrap", case=False, na=False
            )
        ]
    source["observation_period"] = pd.to_datetime(
        source["observation_period"], utc=True, errors="coerce"
    )
    source["vintage_date"] = pd.to_datetime(
        source["vintage_date"], utc=True, errors="coerce"
    )
    source = source.dropna(subset=["observation_period", "vintage_date"])
    source = source.sort_values(["observation_period", "vintage_date"])
    return source.drop_duplicates("observation_period", keep="first")


def append_reason(current: str, reason: str) -> str:
    parts = [part for part in str(current).split(";") if part]
    if reason not in parts:
        parts.append(reason)
    return ";".join(parts)


def build_candidates(macro: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict] = []

    for event_type, spec in RELEASES.items():
        primary = first_real_vintage(macro, spec["primary_series"])
        crosscheck = first_real_vintage(macro, spec["crosscheck_series"])
        if primary.empty:
            raise RuntimeError(f"No ALFRED vintages available for {spec['primary_series']}")

        cross_map = dict(zip(crosscheck["observation_period"], crosscheck["vintage_date"]))

        for record in primary.itertuples():
            reference_period = pd.Timestamp(record.observation_period)
            vintage = pd.Timestamp(record.vintage_date)
            release_date = vintage.date()
            if not (start_year <= release_date.year <= end_year):
                continue

            reasons: list[str] = []
            cross_vintage = cross_map.get(reference_period)
            cross_release_date = None
            if cross_vintage is None or pd.isna(cross_vintage):
                reasons.append("official_series_release_date_crosscheck_missing")
            else:
                cross_release_date = pd.Timestamp(cross_vintage).date()
                if cross_release_date != release_date:
                    reasons.append("official_series_release_date_crosscheck_disagrees")

            reference_month_end = (reference_period + pd.offsets.MonthEnd(0)).date()
            release_lag_days = (release_date - reference_month_end).days
            if not (MIN_RELEASE_LAG_DAYS <= release_lag_days <= MAX_RELEASE_LAG_DAYS):
                reasons.append("first_vintage_outside_expected_next_month_release_window")

            local_dt = datetime(
                release_date.year,
                release_date.month,
                release_date.day,
                8,
                30,
                tzinfo=NEW_YORK,
            )
            utc_dt = local_dt.astimezone(UTC)
            if utc_dt > datetime.now(tz=UTC):
                reasons.append("future_scheduled_release")

            rows.append(
                {
                    "event_id": f"bls:{event_type}:{utc_dt.strftime('%Y%m%dT%H%MZ')}",
                    "event_timestamp_utc": utc_dt,
                    "event_timestamp_local": local_dt.isoformat(),
                    "local_timezone": "America/New_York",
                    "country": "US",
                    "currency": "USD",
                    "event_type": event_type,
                    "event_name": spec["event_name"],
                    "reference_period": reference_period,
                    "actual": np.nan,
                    "forecast": np.nan,
                    "previous": np.nan,
                    "revised_previous": np.nan,
                    "source": "Federal Reserve Bank of St. Louis ALFRED + U.S. Bureau of Labor Statistics",
                    "source_url": f"https://fred.stlouisfed.org/series/{spec['source_series_id']}",
                    "timestamp_source": TIMESTAMP_SOURCE,
                    "timestamp_precision": "minute_reconstructed",
                    "stated_timezone_abbreviation": local_dt.tzname(),
                    "model_eligible": not reasons,
                    "quarantine_reason": ";".join(reasons),
                    "alfred_primary_series": spec["primary_series"],
                    "alfred_crosscheck_series": spec["crosscheck_series"],
                    "alfred_first_vintage_date": release_date.isoformat(),
                    "alfred_crosscheck_vintage_date": (
                        cross_release_date.isoformat() if cross_release_date is not None else None
                    ),
                    "reference_month_end": reference_month_end.isoformat(),
                    "release_lag_days_after_reference_month_end": release_lag_days,
                    "provenance_tier": "dual_official_reconstruction",
                }
            )

    if not rows:
        raise RuntimeError("ALFRED reconstruction produced no candidate BLS events")

    frame = pd.DataFrame(rows)

    # Multiple observation periods can acquire a first recorded vintage on the same
    # date during benchmark revisions/backfills. Never hide those by changing the
    # event ID. Quarantine every still-plausible collision so only unambiguous
    # release timestamps are model-eligible.
    eligible_mask = frame["model_eligible"].fillna(False).astype(bool)
    duplicate_mask = frame.loc[eligible_mask].duplicated(
        ["event_type", "event_timestamp_utc"], keep=False
    )
    duplicate_indices = frame.loc[eligible_mask].index[duplicate_mask]
    if len(duplicate_indices):
        frame.loc[duplicate_indices, "model_eligible"] = False
        for index in duplicate_indices:
            frame.at[index, "quarantine_reason"] = append_reason(
                frame.at[index, "quarantine_reason"],
                "ambiguous_multiple_reference_periods_share_release_timestamp",
            )

    return frame


def reconstruct(start_year: int = 2015, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or datetime.now(tz=UTC).year
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year")
    if not FRED_MACRO_FILE.exists():
        raise RuntimeError(f"Missing local point-in-time FRED file: {FRED_MACRO_FILE}")

    macro = pd.read_parquet(FRED_MACRO_FILE, engine="pyarrow")
    required = {"series_id", "observation_period", "vintage_date"}
    missing = sorted(required.difference(macro.columns))
    if missing:
        raise RuntimeError("FRED macro file is missing columns: " + ", ".join(missing))

    frame = normalize_event_schema(build_candidates(macro, start_year, end_year))
    safe = frame[frame["model_eligible"]].copy()
    quarantine = frame[~frame["model_eligible"]].copy()

    if safe["event_id"].duplicated().any():
        raise RuntimeError("Duplicate eligible event IDs remain after conservative screening")
    if safe.duplicated(["event_type", "reference_period"]).any():
        raise RuntimeError("Duplicate eligible event/reference rows remain after screening")

    atomic_parquet(safe, OUTPUT_FILE)
    if not quarantine.empty:
        atomic_parquet(quarantine, QUARANTINE_FILE)

    counts = safe["event_type"].value_counts().to_dict()
    print("BLS direct web access is blocked; built conservative ALFRED/BLS reconstruction.")
    print(f"Model-eligible events: {len(safe):,} {counts}")
    print(f"Quarantined candidates: {len(quarantine):,}")
    if not quarantine.empty:
        print("Quarantine reasons:")
        print(quarantine["quarantine_reason"].value_counts().to_string())
    print(f"Saved: {OUTPUT_FILE}")
    if not quarantine.empty:
        print(f"Quarantine: {QUARANTINE_FILE}")
    return frame


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    try:
        reconstruct(args.start_year, args.end_year)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
