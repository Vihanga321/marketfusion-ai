"""Build historical BLS CPI and Employment Situation release timestamps.

Primary source: official annual BLS release calendars. If BLS blocks automated
calendar access with HTTP 403, V0.4A falls back to a dual-official-source
reconstruction using the first non-bootstrap ALFRED vintage date already stored
locally by V0.3 plus the documented 08:30 America/New_York release time used for
CPI and Employment Situation. The fallback is explicitly labeled in provenance
and cross-checks CPI against Core CPI and Payroll against Unemployment dates.

No browser impersonation, proxy bypass, fabricated actual values, or fabricated
consensus forecasts are used.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from .common import (
        DATA_DIRECTORY,
        QUARANTINE_DIRECTORY,
        atomic_parquet,
        http_session,
        normalize_event_schema,
    )
except ImportError:  # Support direct execution.
    from common import (
        DATA_DIRECTORY,
        QUARANTINE_DIRECTORY,
        atomic_parquet,
        http_session,
        normalize_event_schema,
    )


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = DATA_DIRECTORY / "bls_release_events.parquet"
QUARANTINE_FILE = QUARANTINE_DIRECTORY / "bls_release_events_unusable.parquet"
FRED_MACRO_FILE = ROOT / "data" / "macro" / "fred_macro.parquet"
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
SCHEDULE_URL = "https://www.bls.gov/schedule/{year}/"
CALENDAR_TIMESTAMP_SOURCE = "BLS annual release calendar exact Eastern Time"
ALFRED_TIMESTAMP_SOURCE = "ALFRED first vintage date + official BLS 08:30 Eastern release time"

RELEASES = {
    "us_cpi_release": {
        "release_prefix": "Consumer Price Index for ",
        "event_name": "Consumer Price Index",
        "alfred_series": "us_cpi_yoy",
        "crosscheck_series": "us_core_cpi_yoy",
        "source_series_id": "CPIAUCSL",
    },
    "us_employment_situation": {
        "release_prefix": "Employment Situation for ",
        "event_name": "Employment Situation",
        "alfred_series": "us_payroll_growth",
        "crosscheck_series": "us_unemployment",
        "source_series_id": "PAYEMS",
    },
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
DATE_PATTERN = re.compile(
    rf"(?:(?:{WEEKDAYS})\s*,?\s*)?(?P<month>{MONTHS})\s+"
    rf"(?P<day>\d{{1,2}}),\s*(?P<year>\d{{4}})",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>AM|PM)", re.IGNORECASE
)
REFERENCE_PATTERN = re.compile(
    rf"\bfor\s+(?P<month>{MONTHS})\s+(?P<year>\d{{4}})\b", re.IGNORECASE
)


class TableRowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            text = data.strip()
            if text:
                self.current_cell.append(text)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.in_row and tag in {"td", "th"} and self.in_cell:
            text = re.sub(r"\s+", " ", " ".join(self.current_cell)).strip()
            self.current_row.append(text)
            self.current_cell = []
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
            self.in_row = False
            self.in_cell = False


def parse_date(text: str) -> datetime | None:
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    try:
        month = datetime.strptime(match.group("month"), "%B").month
        return datetime(int(match.group("year")), month, int(match.group("day")))
    except ValueError:
        return None


def parse_time(text: str) -> tuple[int, int] | None:
    match = TIME_PATTERN.search(text)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    ampm = match.group("ampm").upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def parse_reference_period(release_text: str) -> pd.Timestamp | pd.NaT:
    match = REFERENCE_PATTERN.search(release_text)
    if not match:
        return pd.NaT
    try:
        parsed = datetime.strptime(
            f"{match.group('month')} {match.group('year')}", "%B %Y"
        )
    except ValueError:
        return pd.NaT
    return pd.Timestamp(parsed.date(), tz="UTC")


def classify_release(release_text: str) -> tuple[str, dict] | None:
    normalized = re.sub(r"\s+", " ", release_text).strip()
    for event_type, spec in RELEASES.items():
        if normalized.lower().startswith(spec["release_prefix"].lower()):
            return event_type, spec
    return None


def build_calendar_row(
    *, event_type: str, spec: dict, release_text: str,
    release_date: datetime | None, release_time: tuple[int, int] | None,
    source_url: str, calendar_year: int,
) -> dict:
    reasons: list[str] = []
    reference_period = parse_reference_period(release_text)
    if release_date is None:
        reasons.append("calendar_release_date_not_parsed")
    if release_time is None:
        reasons.append("calendar_release_time_not_parsed")
    if pd.isna(reference_period):
        reasons.append("reference_period_not_parsed")

    local_dt = None
    utc_dt = pd.NaT
    if release_date is not None and release_time is not None:
        local_dt = release_date.replace(
            hour=release_time[0], minute=release_time[1], second=0, microsecond=0,
            tzinfo=NEW_YORK,
        )
        utc_dt = local_dt.astimezone(UTC)
        if utc_dt > datetime.now(tz=UTC):
            reasons.append("future_scheduled_release")
    if release_date is not None and release_date.year != calendar_year:
        reasons.append("release_date_outside_calendar_year")

    model_eligible = not reasons
    if local_dt is not None:
        event_id = f"bls:{event_type}:{utc_dt.strftime('%Y%m%dT%H%MZ')}"
        local_iso = local_dt.isoformat()
        tz_abbreviation = local_dt.tzname()
        release_date_iso = release_date.date().isoformat()
    else:
        event_id = f"bls:{event_type}:unparsed:{calendar_year}:{release_text[:40]}"
        local_iso = None
        tz_abbreviation = None
        release_date_iso = release_date.date().isoformat() if release_date is not None else None

    return {
        "event_id": event_id,
        "event_timestamp_utc": utc_dt,
        "event_timestamp_local": local_iso,
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
        "source": "U.S. Bureau of Labor Statistics",
        "source_url": source_url,
        "timestamp_source": CALENDAR_TIMESTAMP_SOURCE,
        "timestamp_precision": "minute",
        "stated_timezone_abbreviation": tz_abbreviation,
        "model_eligible": model_eligible,
        "quarantine_reason": ";".join(reasons),
        "calendar_release_text": release_text,
        "calendar_release_date": release_date_iso,
        "calendar_year": calendar_year,
        "provenance_tier": "direct_official_calendar",
    }


def parse_schedule_page(content: str, source_url: str, calendar_year: int) -> list[dict]:
    parser = TableRowParser()
    parser.feed(content)
    rows: list[dict] = []
    last_date: datetime | None = None
    last_time: tuple[int, int] | None = None

    for cells in parser.rows:
        row_date = next((parse_date(cell) for cell in cells if parse_date(cell) is not None), None)
        if row_date is not None:
            last_date = row_date
            last_time = None
        row_time = next((parse_time(cell) for cell in cells if parse_time(cell) is not None), None)
        if row_time is not None:
            last_time = row_time

        release_cell = None
        classification = None
        for cell in cells:
            classification = classify_release(cell)
            if classification is not None:
                release_cell = cell
                break
        if classification is None or release_cell is None:
            continue
        event_type, spec = classification
        rows.append(build_calendar_row(
            event_type=event_type,
            spec=spec,
            release_text=release_cell,
            release_date=row_date or last_date,
            release_time=row_time or last_time,
            source_url=source_url,
            calendar_year=calendar_year,
        ))
    return rows


def first_real_vintage(frame: pd.DataFrame, series_id: str) -> pd.DataFrame:
    source = frame[frame["series_id"].eq(series_id)].copy()
    if source.empty:
        return source
    if "availability_method" in source.columns:
        source = source[~source["availability_method"].astype(str).str.contains("bootstrap", case=False, na=False)]
    source["observation_period"] = pd.to_datetime(source["observation_period"], utc=True, errors="coerce")
    source["vintage_date"] = pd.to_datetime(source["vintage_date"], utc=True, errors="coerce")
    source = source.dropna(subset=["observation_period", "vintage_date"])
    source = source.sort_values(["observation_period", "vintage_date"])
    return source.drop_duplicates("observation_period", keep="first")


def alfred_fallback(start_year: int, end_year: int) -> list[dict]:
    if not FRED_MACRO_FILE.exists():
        raise RuntimeError(
            "BLS returned HTTP 403 and the local FRED/ALFRED macro file is missing. "
            f"Expected: {FRED_MACRO_FILE}"
        )
    macro = pd.read_parquet(FRED_MACRO_FILE, engine="pyarrow")
    required = {"series_id", "observation_period", "vintage_date"}
    missing = sorted(required.difference(macro.columns))
    if missing:
        raise RuntimeError("FRED macro fallback is missing columns: " + ", ".join(missing))

    rows: list[dict] = []
    for event_type, spec in RELEASES.items():
        primary = first_real_vintage(macro, spec["alfred_series"])
        crosscheck = first_real_vintage(macro, spec["crosscheck_series"])
        if primary.empty:
            raise RuntimeError(f"No ALFRED vintages available for {spec['alfred_series']}")
        cross_map = dict(zip(crosscheck["observation_period"], crosscheck["vintage_date"]))

        for record in primary.itertuples():
            reference_period = record.observation_period
            vintage = record.vintage_date
            release_date = vintage.date()
            if not (start_year <= release_date.year <= end_year):
                continue

            reasons: list[str] = []
            cross_vintage = cross_map.get(reference_period)
            if cross_vintage is None:
                reasons.append("official_series_release_date_crosscheck_missing")
            elif pd.Timestamp(cross_vintage).date() != release_date:
                reasons.append("official_series_release_date_crosscheck_disagrees")

            local_dt = datetime(
                release_date.year, release_date.month, release_date.day,
                8, 30, tzinfo=NEW_YORK,
            )
            utc_dt = local_dt.astimezone(UTC)
            if utc_dt > datetime.now(tz=UTC):
                reasons.append("future_scheduled_release")

            event_id = f"bls:{event_type}:{utc_dt.strftime('%Y%m%dT%H%MZ')}"
            rows.append({
                "event_id": event_id,
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
                "timestamp_source": ALFRED_TIMESTAMP_SOURCE,
                "timestamp_precision": "minute_reconstructed",
                "stated_timezone_abbreviation": local_dt.tzname(),
                "model_eligible": not reasons,
                "quarantine_reason": ";".join(reasons),
                "alfred_primary_series": spec["alfred_series"],
                "alfred_crosscheck_series": spec["crosscheck_series"],
                "alfred_first_vintage_date": release_date.isoformat(),
                "provenance_tier": "dual_official_reconstruction",
            })

    if not rows:
        raise RuntimeError("ALFRED fallback produced no candidate BLS events")
    print(
        "BLS calendar access is blocked; using labeled dual-official ALFRED/BLS timestamp reconstruction.",
        flush=True,
    )
    return rows


def download(start_year: int = 2015, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or datetime.now(tz=UTC).year
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year")

    session = http_session()
    rows: list[dict] = []
    blocked = False
    for year in range(start_year, end_year + 1):
        url = SCHEDULE_URL.format(year=year)
        response = session.get(url, timeout=90)
        if response.status_code == 403:
            blocked = True
            break
        if not response.ok:
            raise RuntimeError(
                f"BLS annual schedule request failed for {year} (HTTP {response.status_code}): {url}"
            )
        year_rows = parse_schedule_page(response.text, url, year)
        if not year_rows:
            raise RuntimeError(f"No CPI/Employment Situation rows parsed from BLS schedule {year}")
        rows.extend(year_rows)
        counts = pd.Series([row["event_type"] for row in year_rows]).value_counts().to_dict()
        print(f"BLS schedule {year}: {len(year_rows):,} target events {counts}", flush=True)
        time.sleep(0.75)

    if blocked:
        rows = alfred_fallback(start_year, end_year)

    frame = normalize_event_schema(pd.DataFrame(rows))
    safe = frame[frame["model_eligible"]].copy()
    quarantine = frame[~frame["model_eligible"]].copy()

    duplicate_ids = safe["event_id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise RuntimeError(f"Duplicate eligible BLS event IDs: {int(duplicate_ids.sum())}")
    duplicate_reference = safe.duplicated(["event_type", "reference_period"], keep=False)
    if duplicate_reference.any():
        raise RuntimeError(
            f"Duplicate eligible BLS event/reference rows: {int(duplicate_reference.sum())}"
        )

    atomic_parquet(safe, OUTPUT_FILE)
    if not quarantine.empty:
        atomic_parquet(quarantine, QUARANTINE_FILE)

    print(f"BLS model-eligible events: {len(safe):,}")
    print(f"BLS quarantined events: {len(quarantine):,}")
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
        download(args.start_year, args.end_year)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
