"""Build exact historical BLS CPI and Employment Situation release timestamps.

The downloader uses only official BLS archive pages. Model-eligible timestamps must
come from the release's own embargo line; dates inferred from filenames are used
only as an audit cross-check. Historical Eastern time is converted with
America/New_York so daylight-saving transitions are handled by timezone rules.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urljoin
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
except ImportError:  # Support direct execution: python src/events/download_bls_release_calendar.py
    from common import (
        DATA_DIRECTORY,
        QUARANTINE_DIRECTORY,
        atomic_parquet,
        http_session,
        normalize_event_schema,
    )


BLS_BASE = "https://www.bls.gov"
OUTPUT_FILE = DATA_DIRECTORY / "bls_release_events.parquet"
QUARANTINE_FILE = QUARANTINE_DIRECTORY / "bls_release_events_unusable.parquet"
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

RELEASES = {
    "us_cpi_release": {
        "archive_index": "https://www.bls.gov/bls/news-release/cpi.htm",
        "archive_slug": "cpi",
        "event_name": "Consumer Price Index",
        "reference_label": "Consumer Price Index",
    },
    "us_employment_situation": {
        "archive_index": "https://www.bls.gov/bls/news-release/empsit.htm",
        "archive_slug": "empsit",
        "event_name": "Employment Situation",
        "reference_label": "Employment Situation",
    },
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"

# BLS archived releases historically use wording such as:
#   8:30 a.m. (EST) January 20, 2016
#   8:30 a.m. (EDT) Friday, July 8, 2016
# Newer pages may use ET. We still localize through America/New_York rather than
# applying a fixed UTC offset.
EMBARGO_PATTERN = re.compile(
    rf"(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})\s*"
    rf"(?P<ampm>a\.?\s*m\.?|p\.?\s*m\.?)\s*"
    rf"\((?P<tz>EST|EDT|ET)\)\s*"
    rf"(?:(?:{WEEKDAYS})\s*,?\s*)?"
    rf"(?P<month>{MONTHS})\s+(?P<day>\d{{1,2}}),\s*(?P<year>\d{{4}})",
    flags=re.IGNORECASE,
)


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._href: str | None = None
        self._text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        self._href = dict(attrs).get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            text = " ".join(part.strip() for part in self._text if part.strip())
            self.links.append((self._href, text))
            self._href = None
            self._text = []


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def html_text(content: str) -> str:
    parser = TextExtractor()
    parser.feed(content)
    return parser.text()


def discover_archive_links(session, spec: dict, start_year: int, end_year: int) -> list[dict]:
    response = session.get(spec["archive_index"], timeout=90)
    response.raise_for_status()
    parser = LinkCollector()
    parser.feed(response.text)

    slug = re.escape(spec["archive_slug"])
    pattern = re.compile(rf"/news\.release/archives/{slug}_(\d{{8}})\.htm$", re.IGNORECASE)
    discovered: dict[str, dict] = {}
    for href, link_text in parser.links:
        match = pattern.search(href)
        if not match:
            continue
        release_date = datetime.strptime(match.group(1), "%m%d%Y").date()
        if not (start_year <= release_date.year <= end_year):
            continue
        url = urljoin(BLS_BASE, href)
        discovered[url] = {
            "url": url,
            "link_text": link_text,
            "filename_release_date": release_date,
        }
    return sorted(discovered.values(), key=lambda item: item["filename_release_date"])


def parse_reference_period(link_text: str) -> pd.Timestamp | pd.NaT:
    match = re.search(rf"(?P<month>{MONTHS})\s+(?P<year>\d{{4}})", link_text, re.IGNORECASE)
    if not match:
        return pd.NaT
    try:
        parsed = datetime.strptime(f"{match.group('month')} {match.group('year')}", "%B %Y")
    except ValueError:
        return pd.NaT
    return pd.Timestamp(parsed.date(), tz="UTC")


def parse_embargo_timestamp(text: str) -> tuple[datetime | None, str | None]:
    marker = text.lower().find("embargoed until")
    search_text = text[marker : marker + 700] if marker >= 0 else text[:1200]
    match = EMBARGO_PATTERN.search(search_text)
    if not match:
        return None, None

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    ampm = re.sub(r"[^apm]", "", match.group("ampm").lower())
    if ampm.startswith("p") and hour != 12:
        hour += 12
    elif ampm.startswith("a") and hour == 12:
        hour = 0

    local_dt = datetime(
        int(match.group("year")),
        datetime.strptime(match.group("month"), "%B").month,
        int(match.group("day")),
        hour,
        minute,
        tzinfo=NEW_YORK,
    )
    return local_dt, match.group("tz").upper()


def build_row(event_type: str, spec: dict, item: dict, page_text: str) -> dict:
    local_dt, stated_tz = parse_embargo_timestamp(page_text)
    reference_period = parse_reference_period(item["link_text"])
    reasons: list[str] = []

    if local_dt is None:
        reasons.append("embargo_timestamp_not_parsed")
    else:
        if local_dt.date() != item["filename_release_date"]:
            reasons.append("embargo_date_disagrees_with_archive_filename")
        actual_tz = local_dt.tzname()
        if stated_tz in {"EST", "EDT"} and actual_tz != stated_tz:
            reasons.append(f"stated_timezone_{stated_tz}_disagrees_with_America_New_York_{actual_tz}")

    if pd.isna(reference_period):
        reasons.append("reference_period_not_parsed_from_archive_index")

    utc_dt = local_dt.astimezone(UTC) if local_dt is not None else pd.NaT
    local_iso = local_dt.isoformat() if local_dt is not None else None
    model_eligible = not reasons
    event_id = (
        f"bls:{event_type}:{utc_dt.strftime('%Y%m%dT%H%MZ')}"
        if local_dt is not None
        else f"bls:{event_type}:unparsed:{item['filename_release_date'].isoformat()}"
    )

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
        "source_url": item["url"],
        "timestamp_source": "BLS archived release embargo line",
        "timestamp_precision": "minute",
        "stated_timezone_abbreviation": stated_tz,
        "model_eligible": model_eligible,
        "quarantine_reason": ";".join(reasons),
        "archive_link_text": item["link_text"],
        "filename_release_date": item["filename_release_date"].isoformat(),
    }


def download(start_year: int = 2015, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or datetime.now(tz=UTC).year
    if start_year > end_year:
        raise ValueError("start_year must not be after end_year")

    session = http_session()
    rows: list[dict] = []
    for event_type, spec in RELEASES.items():
        links = discover_archive_links(session, spec, start_year, end_year)
        if not links:
            raise RuntimeError(f"No BLS archive links discovered for {event_type}")
        print(f"{event_type}: discovered {len(links):,} archived releases", flush=True)

        for index, item in enumerate(links, start=1):
            response = session.get(item["url"], timeout=90)
            response.raise_for_status()
            rows.append(build_row(event_type, spec, item, html_text(response.text)))
            if index % 25 == 0 or index == len(links):
                print(f"  parsed {index:,}/{len(links):,}", flush=True)

    frame = normalize_event_schema(pd.DataFrame(rows))
    safe = frame[frame["model_eligible"]].copy()
    quarantine = frame[~frame["model_eligible"]].copy()

    duplicate_ids = safe["event_id"].duplicated(keep=False)
    if duplicate_ids.any():
        raise RuntimeError(f"Duplicate eligible BLS event IDs: {int(duplicate_ids.sum())}")

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
