"""Shared paths, schema rules, environment loading, and HTTP helpers for macro adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = ROOT / "data" / "macro"
QUARANTINE_DIRECTORY = DATA_DIRECTORY / "quarantine"
REPORT_DIRECTORY = ROOT / "reports"

# Load the project-local .env deterministically. This works when scripts are
# launched from VS Code, PowerShell, or another working directory. Existing
# process environment variables keep precedence over values in .env.
load_dotenv(ROOT / ".env", override=False)

STANDARD_COLUMNS = [
    "series_id",
    "source",
    "observation_period",
    "available_from_utc",
    "value",
    "vintage_date",
    "frequency",
]


def http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MarketFusion-AI/0.3 (read-only research)"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def normalize_schema(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(STANDARD_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing standard macro columns: {sorted(missing)}")
    result = frame.copy()
    result["observation_period"] = pd.to_datetime(result["observation_period"], utc=True, errors="coerce")
    result["available_from_utc"] = pd.to_datetime(result["available_from_utc"], utc=True, errors="coerce")
    result["vintage_date"] = pd.to_datetime(result["vintage_date"], utc=True, errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result["model_eligible"] = result.get("model_eligible", True).astype(bool)
    ordered = STANDARD_COLUMNS + [column for column in result.columns if column not in STANDARD_COLUMNS]
    return result[ordered].sort_values(
        ["series_id", "available_from_utc", "observation_period"], na_position="last"
    ).reset_index(drop=True)


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(destination)
