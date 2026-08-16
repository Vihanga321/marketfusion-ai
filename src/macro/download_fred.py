"""Download point-in-time US macro vintages from the official FRED/ALFRED API."""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta

import pandas as pd

try:
    from .common import DATA_DIRECTORY, atomic_parquet, http_session, normalize_schema
except ImportError:  # Support direct execution: python src/macro/download_fred.py
    from common import DATA_DIRECTORY, atomic_parquet, http_session, normalize_schema


ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
OUTPUT_FILE = DATA_DIRECTORY / "fred_macro.parquet"
CHECKPOINT_DIRECTORY = DATA_DIRECTORY / "fred_parts"
FRED_KEY_PATTERN = re.compile(r"^[a-z0-9]{32}$")

# FRED limits JSON/XML observation requests to 2,000 vintage dates. Querying
# 2015-present in one real-time window exceeds that limit for daily series, so
# every series is downloaded in non-overlapping calendar-year real-time windows.
#
# output_type=3 keeps only new/revised observations for revision-prone macro
# series. That is enough to reconstruct point-in-time state without downloading
# the very large repeated vintage cube from output_type=2.
# output_type=4 returns initial releases only and is used for daily rates/yields.
SERIES = {
    "fed_funds_rate": {"source_id": "DFF", "frequency": "D", "units": "lin", "output_type": 4},
    "us_2y_yield": {"source_id": "DGS2", "frequency": "D", "units": "lin", "output_type": 4},
    "us_10y_yield": {"source_id": "DGS10", "frequency": "D", "units": "lin", "output_type": 4},
    "us_cpi_yoy": {"source_id": "CPIAUCSL", "frequency": "M", "units": "pc1", "output_type": 3},
    "us_core_cpi_yoy": {"source_id": "CPILFESL", "frequency": "M", "units": "pc1", "output_type": 3},
    "us_pce": {"source_id": "PCEPI", "frequency": "M", "units": "pc1", "output_type": 3},
    "us_core_pce": {"source_id": "PCEPILFE", "frequency": "M", "units": "pc1", "output_type": 3},
    "us_unemployment": {"source_id": "UNRATE", "frequency": "M", "units": "lin", "output_type": 3},
    "us_payroll_growth": {"source_id": "PAYEMS", "frequency": "M", "units": "chg", "output_type": 3},
    "us_gdp_growth": {"source_id": "A191RL1Q225SBEA", "frequency": "Q", "units": "lin", "output_type": 3},
    "us_retail_sales_yoy": {"source_id": "RSAFS", "frequency": "M", "units": "pc1", "output_type": 3},
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


def parse_fred_date(value: object) -> pd.Timestamp | pd.NaT:
    """Parse FRED's YYYY-MM-DD date fields without pandas' generic parser.

    FRED/ALFRED returns date-only ISO strings here. On the project's Python 3.14
    environment, repeatedly calling ``pd.to_datetime`` for thousands of scalar
    strings can spend excessive time inside pandas/typing internals. The stdlib
    ISO parser is deterministic and much cheaper for this known format.
    """
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError:
        return pd.NaT
    return pd.Timestamp(parsed, tz="UTC")


def realtime_windows(start: str, end: str):
    """Yield non-overlapping calendar-year real-time windows."""
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if first > last:
        raise ValueError("Start date must not be after end date")
    cursor = first
    while cursor <= last:
        window_end = min(date(cursor.year, 12, 31), last)
        yield cursor.isoformat(), window_end.isoformat()
        cursor = window_end + timedelta(days=1)


def checkpoint_path(name: str, spec: dict, start: str, end: str):
    return CHECKPOINT_DIRECTORY / (
        f"{name}_{spec['source_id']}_ot{spec['output_type']}_{start}_{end}.parquet"
    )


def fetch_window(
    session,
    name: str,
    spec: dict,
    api_key: str,
    observation_start: str,
    observation_end: str,
    realtime_start: str,
    realtime_end: str,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        parameters = {
            "api_key": api_key,
            "file_type": "json",
            "series_id": spec["source_id"],
            "observation_start": observation_start,
            "observation_end": observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
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
                f"FRED/ALFRED request failed for {name} "
                f"(real-time {realtime_start}..{realtime_end}, HTTP {response.status_code}): {message}"
            )
        payload = response.json()
        observations = payload.get("observations", [])
        for observation in observations:
            if observation.get("value") in {None, "."}:
                continue
            vintage = parse_fred_date(observation.get("realtime_start"))
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
                    "fred_output_type": spec["output_type"],
                    "model_eligible": True,
                }
            )
        offset += len(observations)
        if not observations or offset >= int(payload.get("count", offset)):
            break
    return rows


def fetch_series(name: str, spec: dict, api_key: str, start: str, end: str) -> pd.DataFrame:
    rows: list[dict] = []
    session = http_session()
    for realtime_start, realtime_end in realtime_windows(start, end):
        chunk = fetch_window(
            session,
            name,
            spec,
            api_key,
            observation_start=start,
            observation_end=end,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
        )
        rows.extend(chunk)
        print(
            f"  {name} {realtime_start}..{realtime_end}: "
            f"{len(chunk):,} new/revised rows",
            flush=True,
        )
    return pd.DataFrame(rows)


def download(start: str = "2015-01-01", end: str | None = None, resume: bool = True) -> pd.DataFrame:
    api_key = fred_api_key()
    end = end or date.today().isoformat()
    frames = []
    for name, spec in SERIES.items():
        checkpoint = checkpoint_path(name, spec, start, end)
        if resume and checkpoint.exists():
            frame = pd.read_parquet(checkpoint, engine="pyarrow")
            print(f"FRED/ALFRED {name}: resumed {len(frame):,} rows from checkpoint", flush=True)
        else:
            frame = fetch_series(name, spec, api_key, start, end)
            if frame.empty:
                raise RuntimeError(f"FRED/ALFRED returned no usable rows for {name}")
            CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            atomic_parquet(frame, checkpoint)
            print(f"FRED/ALFRED {name}: {len(frame):,} point-in-time rows", flush=True)
        frames.append(frame)

    result = normalize_schema(pd.concat(frames, ignore_index=True))
    result = result.drop_duplicates(
        ["series_id", "observation_period", "available_from_utc", "value"], keep="last"
    )
    atomic_parquet(result, OUTPUT_FILE)
    print(f"Saved {len(result):,} rows: {OUTPUT_FILE}", flush=True)
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore per-series checkpoints and redownload every series.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    try:
        download(args.start, args.end, resume=not args.no_resume)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
