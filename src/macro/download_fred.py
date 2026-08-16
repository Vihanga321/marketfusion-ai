"""Download point-in-time US macro vintages from the official FRED/ALFRED API."""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    from .common import DATA_DIRECTORY, atomic_parquet, http_session, normalize_schema
except ImportError:  # Support direct execution: python src/macro/download_fred.py
    from common import DATA_DIRECTORY, atomic_parquet, http_session, normalize_schema


ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
OUTPUT_FILE = DATA_DIRECTORY / "fred_macro.parquet"
CHECKPOINT_DIRECTORY = DATA_DIRECTORY / "fred_parts"
FRED_KEY_PATTERN = re.compile(r"^[a-z0-9]{32}$")

# FRED only permits transformed units with the all-vintages output format.
# Instead of parsing that cross-tab format, MarketFusion downloads raw levels
# with output_type=3 (new/revised observations) and reconstructs the requested
# transformations locally using the official FRED formulas. This keeps the
# event stream explicit and point-in-time auditable.
SERIES = {
    "fed_funds_rate": {
        "source_id": "DFF", "frequency": "D", "transform": "lin", "api_output_type": 4,
    },
    "us_2y_yield": {
        "source_id": "DGS2", "frequency": "D", "transform": "lin", "api_output_type": 4,
    },
    "us_10y_yield": {
        "source_id": "DGS10", "frequency": "D", "transform": "lin", "api_output_type": 4,
    },
    "us_cpi_yoy": {
        "source_id": "CPIAUCSL", "frequency": "M", "transform": "pc1",
        "api_output_type": 3, "lag_months": 12,
    },
    "us_core_cpi_yoy": {
        "source_id": "CPILFESL", "frequency": "M", "transform": "pc1",
        "api_output_type": 3, "lag_months": 12,
    },
    "us_pce": {
        "source_id": "PCEPI", "frequency": "M", "transform": "pc1",
        "api_output_type": 3, "lag_months": 12,
    },
    "us_core_pce": {
        "source_id": "PCEPILFE", "frequency": "M", "transform": "pc1",
        "api_output_type": 3, "lag_months": 12,
    },
    "us_unemployment": {
        "source_id": "UNRATE", "frequency": "M", "transform": "lin", "api_output_type": 3,
    },
    "us_payroll_growth": {
        "source_id": "PAYEMS", "frequency": "M", "transform": "chg",
        "api_output_type": 3, "lag_months": 1,
    },
    "us_gdp_growth": {
        "source_id": "A191RL1Q225SBEA", "frequency": "Q", "transform": "lin", "api_output_type": 3,
    },
    "us_retail_sales_yoy": {
        "source_id": "RSAFS", "frequency": "M", "transform": "pc1",
        "api_output_type": 3, "lag_months": 12,
    },
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
    """Parse FRED's YYYY-MM-DD date fields without pandas' generic scalar parser."""
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
    output_type = spec["api_output_type"]
    transform = spec["transform"]
    # Preserve compatibility with the already-created rate/yield checkpoints.
    if output_type == 4 and transform == "lin":
        filename = f"{name}_{spec['source_id']}_ot4_{start}_{end}.parquet"
    else:
        filename = f"{name}_{spec['source_id']}_ot{output_type}_{transform}_{start}_{end}.parquet"
    return CHECKPOINT_DIRECTORY / filename


def raw_window(
    session,
    *,
    name: str,
    source_id: str,
    frequency: str,
    api_key: str,
    observation_start: str,
    observation_end: str,
    realtime_start: str,
    realtime_end: str,
    output_type: int,
) -> pd.DataFrame:
    """Fetch one standard FRED observation window in untransformed levels."""
    rows: list[dict] = []
    offset = 0
    while True:
        parameters = {
            "api_key": api_key,
            "file_type": "json",
            "series_id": source_id,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "units": "lin",
            "output_type": output_type,
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
                f"(real-time {realtime_start}..{realtime_end}, output_type={output_type}, "
                f"HTTP {response.status_code}): {message}"
            )

        payload = response.json()
        observations = payload.get("observations", [])
        for observation in observations:
            value = observation.get("value")
            if value in {None, "."}:
                continue
            vintage = parse_fred_date(observation.get("realtime_start"))
            period = parse_fred_date(observation.get("date"))
            if pd.isna(vintage) or pd.isna(period):
                continue
            rows.append(
                {
                    "observation_period": period,
                    "vintage_date": vintage,
                    "raw_value": float(value),
                    "raw_realtime_end": observation.get("realtime_end"),
                    "frequency": frequency,
                }
            )

        offset += len(observations)
        if not observations or offset >= int(payload.get("count", offset)):
            break

    return pd.DataFrame(rows)


def fetch_yearly_events(
    session,
    *,
    name: str,
    spec: dict,
    api_key: str,
    observation_start: str,
    observation_end: str,
    realtime_start: str,
    realtime_end: str,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    if date.fromisoformat(realtime_start) > date.fromisoformat(realtime_end):
        return pd.DataFrame()
    for window_start, window_end in realtime_windows(realtime_start, realtime_end):
        chunk = raw_window(
            session,
            name=name,
            source_id=spec["source_id"],
            frequency=spec["frequency"],
            api_key=api_key,
            observation_start=observation_start,
            observation_end=observation_end,
            realtime_start=window_start,
            realtime_end=window_end,
            output_type=spec["api_output_type"],
        )
        chunks.append(chunk)
        print(
            f"  {name} {window_start}..{window_end}: {len(chunk):,} raw vintage rows",
            flush=True,
        )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def baseline_snapshot(
    session,
    *,
    name: str,
    spec: dict,
    api_key: str,
    start: str,
    observation_start: str,
) -> pd.DataFrame:
    """Get the raw level state that was known at the beginning of the backtest."""
    return raw_window(
        session,
        name=name,
        source_id=spec["source_id"],
        frequency=spec["frequency"],
        api_key=api_key,
        observation_start=observation_start,
        observation_end=start,
        realtime_start=start,
        realtime_end=start,
        output_type=1,
    )


def output_row(
    *,
    name: str,
    spec: dict,
    observation_period: pd.Timestamp,
    vintage: pd.Timestamp,
    value: float,
    method: str,
    realtime_end: object = None,
) -> dict:
    available = vintage + pd.Timedelta(days=1)
    return {
        "series_id": name,
        "source": "FRED/ALFRED",
        "observation_period": observation_period,
        "available_from_utc": available,
        "value": float(value),
        "vintage_date": vintage,
        "frequency": spec["frequency"],
        "source_series_id": spec["source_id"],
        "units_transform": spec["transform"],
        "source_units": "lin",
        "realtime_end": realtime_end,
        "availability_method": method,
        "fred_output_type": spec["api_output_type"],
        "model_eligible": True,
    }


def local_transform(state: dict[pd.Timestamp, float], period: pd.Timestamp, spec: dict) -> float | None:
    transform = spec["transform"]
    if transform == "lin":
        return state.get(period)
    lag_months = int(spec["lag_months"])
    lag_period = period - pd.DateOffset(months=lag_months)
    current = state.get(period)
    previous = state.get(lag_period)
    if current is None or previous is None:
        return None
    if transform == "chg":
        return current - previous
    if transform == "pc1":
        if previous == 0:
            return None
        return ((current / previous) - 1.0) * 100.0
    raise ValueError(f"Unsupported local transform: {transform}")


def reconstruct_series(
    name: str,
    spec: dict,
    bootstrap: pd.DataFrame,
    events: pd.DataFrame,
    start: str,
) -> pd.DataFrame:
    """Reconstruct a point-in-time transformed event stream from raw levels."""
    if bootstrap.empty:
        raise RuntimeError(f"FRED/ALFRED bootstrap snapshot returned no usable rows for {name}")

    bootstrap = bootstrap.sort_values(["observation_period", "vintage_date"]).drop_duplicates(
        "observation_period", keep="last"
    )
    state = {
        row.observation_period: float(row.raw_value)
        for row in bootstrap.itertuples()
    }
    baseline_vintage = pd.Timestamp(date.fromisoformat(start), tz="UTC")
    rows: list[dict] = []
    last_emitted: dict[pd.Timestamp, float] = {}

    # Conservative baseline: values known on the start date are made usable on
    # the following UTC day, rather than pretending we know their exact intraday
    # publication time.
    for period in sorted(state):
        transformed = local_transform(state, period, spec)
        if transformed is None or not np.isfinite(transformed):
            continue
        rows.append(
            output_row(
                name=name,
                spec=spec,
                observation_period=period,
                vintage=baseline_vintage,
                value=transformed,
                method=f"alfred_bootstrap_asof_plus_1d_local_{spec['transform']}",
            )
        )
        last_emitted[period] = float(transformed)

    if events.empty:
        return pd.DataFrame(rows)

    events = events.sort_values(["vintage_date", "observation_period"]).drop_duplicates(
        ["vintage_date", "observation_period", "raw_value"], keep="last"
    )
    lag_months = int(spec.get("lag_months", 0))

    for vintage, group in events.groupby("vintage_date", sort=True):
        impacted: set[pd.Timestamp] = set()
        for event in group.itertuples():
            period = event.observation_period
            state[period] = float(event.raw_value)
            impacted.add(period)
            if lag_months:
                impacted.add(period + pd.DateOffset(months=lag_months))

        for period in sorted(impacted):
            transformed = local_transform(state, period, spec)
            if transformed is None or not np.isfinite(transformed):
                continue
            previous = last_emitted.get(period)
            if previous is not None and np.isclose(previous, transformed, rtol=0.0, atol=1e-12):
                continue
            rows.append(
                output_row(
                    name=name,
                    spec=spec,
                    observation_period=period,
                    vintage=vintage,
                    value=transformed,
                    method=f"alfred_raw_vintage_plus_1d_local_{spec['transform']}",
                )
            )
            last_emitted[period] = float(transformed)

    return pd.DataFrame(rows)


def fetch_series(name: str, spec: dict, api_key: str, start: str, end: str) -> pd.DataFrame:
    session = http_session()
    transform = spec["transform"]
    output_type = spec["api_output_type"]

    # Initial-release-only daily series already provide the event stream directly.
    if output_type == 4 and transform == "lin":
        raw = fetch_yearly_events(
            session,
            name=name,
            spec=spec,
            api_key=api_key,
            observation_start=start,
            observation_end=end,
            realtime_start=start,
            realtime_end=end,
        )
        rows = [
            output_row(
                name=name,
                spec=spec,
                observation_period=row.observation_period,
                vintage=row.vintage_date,
                value=row.raw_value,
                method="alfred_initial_release_date_plus_1d",
                realtime_end=row.raw_realtime_end,
            )
            for row in raw.itertuples()
        ]
        return pd.DataFrame(rows)

    # Revision-prone level/transformed series need a state snapshot at the start
    # plus the subsequent new/revised raw-level events.
    lookback_months = max(24, int(spec.get("lag_months", 0)) + 12)
    start_timestamp = pd.Timestamp(date.fromisoformat(start))
    lookback_start = (start_timestamp - pd.DateOffset(months=lookback_months)).date().isoformat()
    bootstrap = baseline_snapshot(
        session,
        name=name,
        spec=spec,
        api_key=api_key,
        start=start,
        observation_start=lookback_start,
    )
    event_start = (date.fromisoformat(start) + timedelta(days=1)).isoformat()
    events = fetch_yearly_events(
        session,
        name=name,
        spec=spec,
        api_key=api_key,
        observation_start=lookback_start,
        observation_end=end,
        realtime_start=event_start,
        realtime_end=end,
    )
    return reconstruct_series(name, spec, bootstrap, events, start)


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
