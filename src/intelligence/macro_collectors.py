"""Free no-key V0.5B macro snapshot collectors.

The collector intentionally records MarketFusion's capture time as the causal
availability boundary. Provider observation dates are descriptive only; this
stage does not backdate information merely because a current endpoint exposes an
older observation.
"""
from __future__ import annotations

import io

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .v05b_contract import MACRO_SOURCES

USER_AGENT = "MarketFusionAI/0.5B research collector"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2))
    return session


def parse_fred_graph_csv(text: str, series_name: str, captured_at: pd.Timestamp) -> pd.DataFrame:
    spec = MACRO_SOURCES[series_name]
    raw = pd.read_csv(io.StringIO(text))
    date_column = "DATE" if "DATE" in raw.columns else raw.columns[0]
    value_column = spec["series"] if spec["series"] in raw.columns else raw.columns[-1]
    values = pd.to_numeric(raw[value_column].replace(".", pd.NA), errors="coerce")
    dates = pd.to_datetime(raw[date_column], utc=True, errors="coerce")
    valid = values.notna() & dates.notna()
    if not valid.any():
        raise ValueError(f"No valid values returned for {series_name}")
    idx = valid[valid].index[-1]
    return pd.DataFrame([{
        "series_id": series_name,
        "provider": spec["provider"],
        "source_series_id": spec["series"],
        "observation_period": dates.loc[idx],
        "value": float(values.loc[idx]),
        "unit": spec["unit"],
        "source_url": spec["url"],
        "provider_valid_from_utc": pd.NaT,
        "first_observed_utc": captured_at,
        "available_from_utc": captured_at,
        "availability_method": "marketfusion_first_observed_capture_from_current_public_endpoint",
    }])


def fetch_fred_graph(series_name: str, captured_at: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    spec = MACRO_SOURCES[series_name]
    start = (captured_at - pd.Timedelta(days=45)).date().isoformat()
    end = captured_at.date().isoformat()
    response = session.get(
        spec["url"],
        params={"cosd": start, "coed": end},
        timeout=(10, 20),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv, text/plain;q=0.9, */*;q=0.2",
            # FRED occasionally resets long-lived keep-alive sockets on Windows.
            "Connection": "close",
        },
    )
    response.raise_for_status()
    return parse_fred_graph_csv(response.text, series_name, captured_at)


def parse_ecb_csv(text: str, series_name: str, captured_at: pd.Timestamp) -> pd.DataFrame:
    spec = MACRO_SOURCES[series_name]
    raw = pd.read_csv(io.StringIO(text), low_memory=False)
    if not {"TIME_PERIOD", "OBS_VALUE"}.issubset(raw.columns):
        raise ValueError("ECB response missing TIME_PERIOD/OBS_VALUE")
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    raw["TIME_PERIOD_PARSED"] = pd.to_datetime(raw["TIME_PERIOD"], utc=True, errors="coerce")
    valid = raw["OBS_VALUE"].notna() & raw["TIME_PERIOD_PARSED"].notna()
    if not valid.any():
        raise ValueError("ECB response has no valid observation")
    row = raw.loc[valid].sort_values("TIME_PERIOD_PARSED").iloc[-1]
    provider_valid = pd.to_datetime(row.get("VALID_FROM"), utc=True, errors="coerce")
    return pd.DataFrame([{
        "series_id": series_name,
        "provider": spec["provider"],
        "source_series_id": spec["series"],
        "observation_period": row["TIME_PERIOD_PARSED"],
        "value": float(row["OBS_VALUE"]),
        "unit": spec["unit"],
        "source_url": spec["url"],
        "provider_valid_from_utc": provider_valid,
        "first_observed_utc": captured_at,
        "available_from_utc": captured_at,
        "availability_method": "marketfusion_first_observed_capture_from_ecb_sdmx",
    }])


def fetch_ecb(series_name: str, captured_at: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    spec = MACRO_SOURCES[series_name]
    start = (captured_at - pd.Timedelta(days=60)).date().isoformat()
    response = session.get(
        spec["url"],
        params={"startPeriod": start, "format": "csvdata", "includeHistory": "true"},
        timeout=(10, 30),
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv, */*;q=0.2"},
    )
    response.raise_for_status()
    return parse_ecb_csv(response.text, series_name, captured_at)


def collect_macro(captured_at: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    capture = captured_at or utc_now()
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    counts: dict[str, int] = {}

    # Use a fresh short-lived session for each provider row. This costs almost
    # nothing at a five-minute cadence and avoids a reset socket poisoning the
    # remaining FRED requests on Windows.
    for series_name, spec in MACRO_SOURCES.items():
        try:
            with _session() as session:
                frame = (
                    fetch_ecb(series_name, capture, session)
                    if spec["provider"] == "ECB_SDMX"
                    else fetch_fred_graph(series_name, capture, session)
                )
            counts[series_name] = len(frame)
            frames.append(frame)
        except Exception as exc:
            counts[series_name] = 0
            errors.append(f"{series_name}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors, counts
    return pd.concat(frames, ignore_index=True), errors, counts
