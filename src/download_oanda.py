"""Download complete EUR_USD H1 midpoint, bid, and ask candles from OANDA."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "oanda" / "eurusd_h1.parquet"
INSTRUMENT = "EUR_USD"
GRANULARITY = "H1"
PRICE = "MBA"
DEFAULT_START = "2016-01-01T00:00:00Z"
MAX_BATCH_DAYS = 180  # Safely below the endpoint's 5,000-candle maximum.
OUTPUT_COLUMNS = [
    "timestamp",
    "mid_open", "mid_high", "mid_low", "mid_close",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
    "volume", "complete", "spread_close",
]


def parse_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def rfc3339(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def api_base(environment: str) -> str:
    normalized = environment.strip().lower()
    if normalized in {"practice", "demo", "fxpractice"}:
        return "https://api-fxpractice.oanda.com"
    if normalized in {"live", "trade", "fxtrade"}:
        return "https://api-fxtrade.oanda.com"
    raise ValueError("OANDA_ENV must be 'practice' or 'live'")


def make_session(token: str) -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept-Datetime-Format": "RFC3339",
            "User-Agent": "MarketFusion-AI-V0.2",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def parse_candle(candle: dict) -> dict:
    missing_components = [name for name in ("mid", "bid", "ask") if name not in candle]
    if missing_components:
        raise ValueError(f"OANDA candle is missing price components: {missing_components}")

    row: dict[str, object] = {
        "timestamp": pd.to_datetime(candle["time"], utc=True),
        "volume": int(candle["volume"]),
        "complete": bool(candle["complete"]),
    }
    for source, prefix in (("mid", "mid"), ("bid", "bid"), ("ask", "ask")):
        values = candle[source]
        for api_name, output_name in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
            row[f"{prefix}_{output_name}"] = float(values[api_name])
    row["spread_close"] = row["ask_close"] - row["bid_close"]
    return row


def download(start: pd.Timestamp, end: pd.Timestamp, output: Path, batch_days: int, timeout: int) -> pd.DataFrame:
    load_dotenv(ROOT / ".env")
    token = os.getenv("OANDA_API_TOKEN", "").strip()
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    environment = os.getenv("OANDA_ENV", "").strip()
    missing = [
        name
        for name, value in (
            ("OANDA_API_TOKEN", token),
            ("OANDA_ACCOUNT_ID", account_id),
            ("OANDA_ENV", environment),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
    if start >= end:
        raise ValueError("--start must be earlier than --end")
    if not 1 <= batch_days <= MAX_BATCH_DAYS:
        raise ValueError(f"--batch-days must be between 1 and {MAX_BATCH_DAYS}")

    base_url = api_base(environment)
    endpoint = (
        f"{base_url}/v3/accounts/{quote(account_id, safe='-')}/instruments/"
        f"{INSTRUMENT}/candles"
    )
    session = make_session(token)
    rows: list[dict] = []
    cursor = start
    batch_number = 0

    while cursor < end:
        batch_number += 1
        batch_end = min(cursor + pd.Timedelta(days=batch_days), end)
        params = {
            "price": PRICE,
            "granularity": GRANULARITY,
            "smooth": "false",
            "from": rfc3339(cursor),
            "to": rfc3339(batch_end),
            "includeFirst": "true",
        }
        response = session.get(endpoint, params=params, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            request_id = response.headers.get("RequestID", "unavailable")
            message = ""
            try:
                message = response.json().get("errorMessage", "")
            except ValueError:
                pass
            raise RuntimeError(
                f"OANDA request failed with HTTP {response.status_code}; "
                f"request ID={request_id}; message={message or 'not provided'}"
            ) from exc
        payload = response.json()
        candles = payload.get("candles", [])
        complete_rows = [parse_candle(candle) for candle in candles if candle.get("complete") is True]
        rows.extend(complete_rows)
        print(
            f"Batch {batch_number}: {cursor.date()} to {batch_end.date()} - "
            f"{len(complete_rows):,} complete candles"
        )
        cursor = batch_end

    if not rows:
        raise RuntimeError("OANDA returned no complete candles for the requested range")

    frame = pd.DataFrame(rows)
    frame = frame[OUTPUT_COLUMNS].sort_values("timestamp")
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    frame = frame.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    frame["complete"] = frame["complete"].astype(bool)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(output)
    print(f"Removed {duplicate_count:,} duplicated batch-boundary timestamps")
    print(f"Saved {len(frame):,} complete candles to: {output}")
    print(f"Range: {frame['timestamp'].min()} to {frame['timestamp'].max()}")
    return frame


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive UTC start timestamp")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat(),
        help="UTC end timestamp",
    )
    parser.add_argument("--batch-days", type=int, default=MAX_BATCH_DAYS)
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    try:
        download(parse_utc(args.start), parse_utc(args.end), args.output, args.batch_days, args.timeout)
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
