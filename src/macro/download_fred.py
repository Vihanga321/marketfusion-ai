"""Download point-in-time US macro vintages from the official FRED/ALFRED API."""

from __future__ import annotations

import argparse
import os
import re
from datetime import date

import pandas as pd

try:
    from .common import DATA_DIRECTORY, atomic_parquet, http_session, normalize_schema
except ImportError:  # Support direct execution: python src/macro/download_fred.py
    from common import DATA_DIRECTORY, atomic_parquet, http_session, normalize_schema


ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
OUTPUT_FILE = DATA_DIRECTORY / "fred_macro.parquet"
FRED_KEY_PATTERN = re.compile(r"^[a-z0-9]{32}$")

# ALFRED output_type=2 retains revisions. Daily market/rate observations use
# output_type=4 (initial release) to avoid an impractically large vintage cube.
SERIES = {
    "fed_funds_rate": {"source_id": "DFF", "frequency": "D", "units": "lin", "output_type": 4},
    "us_2y_yield": {"source_id": "DGS2", "frequency": "D", "units": "lin", "output_type": 4},
    "us_10y_yield": {"source_id": "DGS10", "frequency": "D", "units": "lin", "output_type": 4},
    "us_cpi_yoy": {"source_id": "CPIAUCSL", "frequency": "M", "units": "pc1", "output_type": 2},
    "us_core_cpi_yoy": {"source_id": "CPILFESL", "frequency": "M", "units": "pc1", "output_type": 2},
    "us_pce": {"source_id": "PCEPI", "frequency": "M", "units": "pc1", "output_type": 2},
    "us_core_pce": {"source_id": "PCEPILFE", "frequency": "M", "units": "pc1", "output_type": 2},
    "us_unemployment": {"source_id": "UNRATE", "frequency": "M", "units": "lin", "output_type": 2},
    "us_payroll_growth": {"source_id": "PAYEMS", "frequency": "M", "units": "chg", "output_type": 2},
    "us_gdp_growth": {"source_id": "A191RL1Q225SBEA", "frequency": "Q", "units": "lin", "output_type": 2},
    "us_retail_sales_yoy": {"source_id": "RSAFS", "frequency": "M", "units": "pc1", "output_type": 2},
}


def fred_api_key() -> str:
    """Return a validated FRED key without ever printing or logging it."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Put the private key in the project .env file; "
            "no substitute or revised-latest data was used."
        )
    if api_key == "replace_with_your_fred_api_key" or not FRED_KEY_PATTERN.fullmatch(api_key):
        raise RuntimeError(
            "FRED_API_KEY is present but does not look like a registered FRED key "
            "(expected 32 lowercase alphanumeric characters)."
        )
    return api_key


def fetch_series(name: str, spec: dict, api_key: str, start: str, end: str) -> pd.DataFrame:
    session = http_session()
    rows: list[dict] = []
    offset = 0
    while True:
        parameters = {
            "api_key": api_key,
            "file_type": "json",
            "series_id": spec["source_id"],
            "observation_start": start,
            "observation_end": end,
            "realtime_start": start,
            "realtime_end": end,
            "units": spec["units"],
            "output_type": spec["output_type"],
            "sort_order": "asc",
            "limit": 100_000,
            "offset": offset,
        }
        response = session.get(ENDPOINT, params=parameters, timeout=90)
        if not response.ok:
            try:
                payload = response.json()
                message = payload.get("error_message") or payload.get("message") or "unknown API error"
            except ValueError:
                message = "non-JSON API error response"
            raise RuntimeError(
                f"FRED/ALFRED request failed for {name} (HTTP {response.status_code}): {message}"
            )
        payload = response.json()
        observations = payload.get("observations", [])
        for observation in observations:
            if observation.get("value") in {None, "."}:
                continue
            vintage = pd.to_datetime(observation.get("realtime_start"), utc=True, errors="coerce")
            # ALFRED vintage metadata is date-granular. Delaying use until the
            # next UTC day prevents same-day release-time look-ahead.
            available = vintage + pd.Timedelta(days=1) if pd.notna(vintage) else pd.NaT
            rows.append(
                {
                    "series_id": name,
                    "source": "FRED/ALFRED",
                    "observation_period": observation.get("date"),
                    "available_from_utc": available,
                    "value": observation.get("value"),
                    "vintage_date": vintage,
                    "frequency": spec["frequency"],
                    "source_series_id": spec["source_id"],
                    "units_transform": spec["units"],
                    "realtime_end": observation.get("realtime_end"),
                    "availability_method": "alfred_vintage_date_plus_1d",
                    "model_eligible": True,
                }
            )
        offset += len(observations)
        if not observations or offset >= int(payload.get("count", offset)):
            break
    return pd.DataFrame(rows)


def download(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    api_key = fred_api_key()
    end = end or date.today().isoformat()
    frames = []
    for name, spec in SERIES.items():
        frame = fetch_series(name, spec, api_key, start, end)
        frames.append(frame)
        print(f"FRED/ALFRED {name}: {len(frame):,} point-in-time rows")
    result = normalize_schema(pd.concat(frames, ignore_index=True))
    result = result.drop_duplicates(
        ["series_id", "observation_period", "available_from_utc", "value"], keep="last"
    )
    atomic_parquet(result, OUTPUT_FILE)
    print(f"Saved {len(result):,} rows: {OUTPUT_FILE}")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    try:
        download(args.start, args.end)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
