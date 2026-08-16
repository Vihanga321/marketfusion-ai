"""Download official BLS values and quarantine rows lacking release vintages."""

from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd

try:
    from .common import QUARANTINE_DIRECTORY, atomic_parquet, http_session, normalize_schema
except ImportError:  # Support direct execution.
    from common import QUARANTINE_DIRECTORY, atomic_parquet, http_session, normalize_schema


ENDPOINT = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
OUTPUT_FILE = QUARANTINE_DIRECTORY / "bls_current_values.parquet"
SERIES = {
    "us_cpi_level_bls": "CUSR0000SA0",
    "us_core_cpi_level_bls": "CUSR0000SA0L1E",
    "us_unemployment_bls": "LNS14000000",
    "us_nonfarm_payroll_bls": "CES0000000001",
}


def period_timestamp(year: str, period: str) -> pd.Timestamp | pd.NaT:
    if not period.startswith("M") or period == "M13":
        return pd.NaT
    return pd.Timestamp(year=int(year), month=int(period[1:]), day=1, tz="UTC")


def download(start_year: int = 2015, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year
    api_key = os.getenv("BLS_API_KEY", "").strip()
    session = http_session()
    source_to_name = {source_id: name for name, source_id in SERIES.items()}
    rows: list[dict] = []

    # The unregistered API has a ten-year window limit, so request bounded chunks.
    for chunk_start in range(start_year, end_year + 1, 10):
        chunk_end = min(chunk_start + 9, end_year)
        body: dict[str, object] = {
            "seriesid": list(SERIES.values()),
            "startyear": str(chunk_start),
            "endyear": str(chunk_end),
        }
        if api_key:
            body["registrationkey"] = api_key
        response = session.post(ENDPOINT, json=body, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError("BLS request failed: " + "; ".join(payload.get("message", [])))
        for series in payload.get("Results", {}).get("series", []):
            source_id = series["seriesID"]
            for observation in series.get("data", []):
                observation_period = period_timestamp(observation["year"], observation["period"])
                if pd.isna(observation_period):
                    continue
                rows.append(
                    {
                        "series_id": source_to_name[source_id],
                        "source": "BLS",
                        "observation_period": observation_period,
                        "available_from_utc": pd.NaT,
                        "value": observation.get("value"),
                        "vintage_date": pd.NaT,
                        "frequency": "M",
                        "source_series_id": source_id,
                        "availability_method": "not_provided_by_bls_timeseries_api",
                        "model_eligible": False,
                        "quarantine_reason": "Historical release/vintage timestamp unavailable",
                    }
                )
    result = normalize_schema(pd.DataFrame(rows))
    result = result.drop_duplicates(["series_id", "observation_period"], keep="first")
    atomic_parquet(result, OUTPUT_FILE)
    print(f"Downloaded {len(result):,} official BLS observations.")
    print("MODEL_ELIGIBLE: NO - BLS response does not contain historical release/vintage timestamps.")
    print(f"Quarantined: {OUTPUT_FILE}")
    return result


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
