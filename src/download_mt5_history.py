"""Download read-only EURUSD H1 and H4 history from the logged-in MT5 terminal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "data" / "mt5" / "history"
SYMBOL = "EURUSD"
REQUESTED_START = datetime(2016, 1, 1, tzinfo=timezone.utc)
TIMEFRAMES = {
    "H1": (mt5.TIMEFRAME_H1, timedelta(hours=1)),
    "H4": (mt5.TIMEFRAME_H4, timedelta(hours=4)),
}
OUTPUT_COLUMNS = [
    "timestamp_utc", "open", "high", "low", "close",
    "tick_volume", "spread_points", "real_volume",
]


def rates_to_frame(rates) -> pd.DataFrame:
    raw = pd.DataFrame(rates)
    required = {"time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("MT5 rate response is missing fields: " + ", ".join(missing))
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(raw["time"], unit="s", utc=True),
            "open": pd.to_numeric(raw["open"], errors="raise"),
            "high": pd.to_numeric(raw["high"], errors="raise"),
            "low": pd.to_numeric(raw["low"], errors="raise"),
            "close": pd.to_numeric(raw["close"], errors="raise"),
            "tick_volume": pd.to_numeric(raw["tick_volume"], errors="raise").astype("int64"),
            "spread_points": pd.to_numeric(raw["spread"], errors="raise").astype("int64"),
            "real_volume": pd.to_numeric(raw["real_volume"], errors="raise").astype("int64"),
        }
    )
    return frame[OUTPUT_COLUMNS]


def download_timeframe(
    label: str,
    timeframe: int,
    duration: timedelta,
    requested_end: datetime,
) -> pd.DataFrame:
    rates = mt5.copy_rates_range(SYMBOL, timeframe, REQUESTED_START, requested_end)
    api_error = mt5.last_error()
    print(f"{label} mt5.last_error(): {api_error}")
    if rates is None:
        raise RuntimeError(f"copy_rates_range failed for {label}: {api_error}")
    if len(rates) == 0:
        raise RuntimeError(f"copy_rates_range returned no {label} bars: {api_error}")

    frame = rates_to_frame(rates)
    completion_cutoff = pd.Timestamp(requested_end)
    complete = frame["timestamp_utc"] + pd.Timedelta(duration) <= completion_cutoff
    excluded = int((~complete).sum())
    frame = frame[complete].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No completed {label} bars were returned")

    destination = OUTPUT_DIRECTORY / f"{SYMBOL}_{label}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(destination)

    first = frame["timestamp_utc"].iloc[0]
    last = frame["timestamp_utc"].iloc[-1]
    print(f"{label} requested start/end: {REQUESTED_START.isoformat()} to {requested_end.isoformat()}")
    print(f"{label} actual first timestamp: {first.isoformat()}")
    print(f"{label} actual last timestamp: {last.isoformat()}")
    print(f"{label} completed bars: {len(frame):,}")
    print(f"{label} current incomplete bars excluded: {excluded:,}")
    if first > pd.Timestamp(REQUESTED_START):
        print(f"{label} NOTE: requested start is unavailable; actual history begins at {first.isoformat()}")
    print(f"{label} saved: {destination}")
    return frame


def main() -> None:
    requested_end = datetime.now(timezone.utc)
    try:
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"mt5.symbol_select({SYMBOL!r}, True) failed: {mt5.last_error()}")
        for label, (timeframe, duration) in TIMEFRAMES.items():
            download_timeframe(label, timeframe, duration, requested_end)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
