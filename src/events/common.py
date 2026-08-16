"""Shared schema, paths, HTTP, and persistence helpers for V0.4 event intelligence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = ROOT / "data" / "events"
QUARANTINE_DIRECTORY = DATA_DIRECTORY / "quarantine"
REPORT_DIRECTORY = ROOT / "reports"

STANDARD_EVENT_COLUMNS = [
    "event_id",
    "event_timestamp_utc",
    "event_timestamp_local",
    "local_timezone",
    "country",
    "currency",
    "event_type",
    "event_name",
    "reference_period",
    "actual",
    "forecast",
    "previous",
    "revised_previous",
    "source",
    "source_url",
    "timestamp_source",
    "timestamp_precision",
    "stated_timezone_abbreviation",
    "model_eligible",
    "quarantine_reason",
]


def http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    # BLS may block automated clients that do not identify an owner. Keep the
    # crawler transparent and low-volume rather than pretending to be a browser.
    session.headers.update(
        {
            "User-Agent": (
                "MarketFusion-AI/0.4 read-only research "
                "(+https://github.com/Vihanga321/marketfusion-ai)"
            ),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_event_schema(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(STANDARD_EVENT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing standard event columns: {missing}")

    result = frame.copy()
    result["event_timestamp_utc"] = pd.to_datetime(
        result["event_timestamp_utc"], utc=True, errors="coerce"
    )
    result["reference_period"] = pd.to_datetime(
        result["reference_period"], utc=True, errors="coerce"
    )
    for column in ("actual", "forecast", "previous", "revised_previous"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["model_eligible"] = result["model_eligible"].fillna(False).astype(bool)

    ordered = STANDARD_EVENT_COLUMNS + [
        column for column in result.columns if column not in STANDARD_EVENT_COLUMNS
    ]
    return result[ordered].sort_values(
        ["event_timestamp_utc", "event_type", "event_id"], na_position="last"
    ).reset_index(drop=True)


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(destination)
