"""Read-only recoverable EURUSD tick collector for an open MetaTrader 5 terminal."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TICK_DIRECTORY = ROOT / "data" / "mt5" / "ticks"
SYMBOL = "EURUSD"
PIP_SIZE = 0.0001
TICK_COLUMNS = [
    "timestamp_utc", "time_msc", "bid", "ask", "mid", "spread",
    "spread_pips", "flags", "volume", "volume_real",
]
UNIQUE_KEY = ["time_msc", "bid", "ask", "flags", "volume", "volume_real"]


def configure_logging(verbose: bool) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    return logging.getLogger("marketfusion.mt5_feed")


def milliseconds_to_utc(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000, tz=timezone.utc)


def tick_key(row: pd.Series) -> tuple:
    return tuple(row[column] for column in UNIQUE_KEY)


def ticks_to_frame(ticks: np.ndarray) -> pd.DataFrame:
    if ticks is None or len(ticks) == 0:
        return pd.DataFrame(columns=TICK_COLUMNS)
    raw = pd.DataFrame(ticks)
    required = {"time_msc", "bid", "ask", "flags", "volume", "volume_real"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("MT5 tick response is missing fields: " + ", ".join(missing))
    frame = pd.DataFrame()
    frame["time_msc"] = pd.to_numeric(raw["time_msc"], errors="raise").astype("int64")
    frame["timestamp_utc"] = pd.to_datetime(frame["time_msc"], unit="ms", utc=True)
    for column in ("bid", "ask", "volume", "volume_real"):
        frame[column] = pd.to_numeric(raw[column], errors="raise")
    frame["flags"] = pd.to_numeric(raw["flags"], errors="raise").astype("int64")
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["spread_pips"] = frame["spread"] / PIP_SIZE
    return (
        frame[TICK_COLUMNS]
        .drop_duplicates(subset=UNIQUE_KEY, keep="last")
        .sort_values(["time_msc", "bid", "ask", "flags"])
        .reset_index(drop=True)
    )


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def storage_state(directory: Path) -> tuple[int | None, set[tuple], int]:
    files = sorted(directory.glob(f"{SYMBOL}_ticks_*.parquet"))
    if not files:
        return None, set(), 0
    total_rows = sum(parquet_row_count(path) for path in files)
    latest = pd.read_parquet(files[-1], engine="pyarrow")
    if latest.empty:
        return None, set(), total_rows
    latest["time_msc"] = pd.to_numeric(latest["time_msc"], errors="raise").astype("int64")
    last_time_msc = int(latest["time_msc"].max())
    boundary = latest[latest["time_msc"] == last_time_msc]
    keys = {tick_key(row) for _, row in boundary.iterrows()}
    return last_time_msc, keys, total_rows


def write_daily_files(frame: pd.DataFrame, directory: Path) -> int:
    if frame.empty:
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["utc_day"] = frame["timestamp_utc"].dt.strftime("%Y-%m-%d")
    written = 0
    for day, day_frame in frame.groupby("utc_day", sort=True):
        day_frame = day_frame.drop(columns="utc_day")
        destination = directory / f"{SYMBOL}_ticks_{day}.parquet"
        if destination.exists():
            existing = pd.read_parquet(destination, engine="pyarrow")
            day_frame = pd.concat([existing, day_frame], ignore_index=True)
        before = len(day_frame)
        day_frame = (
            day_frame.drop_duplicates(subset=UNIQUE_KEY, keep="last")
            .sort_values(["time_msc", "bid", "ask", "flags"])
            .reset_index(drop=True)
        )
        written += len(day_frame) - max(0, before - len(frame[frame["utc_day"] == day]))
        temporary = destination.with_suffix(".tmp.parquet")
        day_frame[TICK_COLUMNS].to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(destination)
    return written


class MT5TickCollector:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.buffer: list[pd.DataFrame] = []
        self.buffer_rows = 0
        self.connected = False
        self.last_time_msc, self.boundary_keys, self.total_ticks = storage_state(args.output_dir)
        now = datetime.now(timezone.utc)
        if self.last_time_msc is None:
            self.scan_cursor = now - timedelta(seconds=args.initial_lookback_seconds)
        else:
            self.scan_cursor = milliseconds_to_utc(self.last_time_msc) - timedelta(seconds=1)
        self.last_flush_monotonic = time.monotonic()
        self.last_live_status_monotonic = 0.0
        self.last_idle_status_monotonic = 0.0

    def connect(self) -> bool:
        mt5.shutdown()
        if not mt5.initialize():
            self.connected = False
            self.logger.warning("MT5 initialize failed: %s", mt5.last_error())
            return False
        terminal = mt5.terminal_info()
        if terminal is None or not terminal.connected:
            self.connected = False
            self.logger.warning("MT5 terminal is not connected: %s", mt5.last_error())
            mt5.shutdown()
            return False
        if not mt5.symbol_select(SYMBOL, True):
            self.connected = False
            self.logger.warning("Could not select %s: %s", SYMBOL, mt5.last_error())
            mt5.shutdown()
            return False
        self.connected = True
        self.logger.info("Connected to MT5; recovering ticks from %s", self.scan_cursor.isoformat())
        return True

    def connection_healthy(self) -> bool:
        if not self.connected:
            return False
        terminal = mt5.terminal_info()
        return terminal is not None and bool(terminal.connected)

    def filter_new(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        if self.last_time_msc is not None:
            after = frame["time_msc"] > self.last_time_msc
            at_boundary = frame["time_msc"] == self.last_time_msc
            boundary_new = pd.Series(False, index=frame.index)
            for index, row in frame[at_boundary].iterrows():
                boundary_new.loc[index] = tick_key(row) not in self.boundary_keys
            frame = frame[after | boundary_new].copy()
        if frame.empty:
            return frame
        frame = frame.drop_duplicates(subset=UNIQUE_KEY, keep="last")
        newest_msc = int(frame["time_msc"].max())
        newest = frame[frame["time_msc"] == newest_msc]
        if self.last_time_msc == newest_msc:
            self.boundary_keys.update(tick_key(row) for _, row in newest.iterrows())
        else:
            self.last_time_msc = newest_msc
            self.boundary_keys = {tick_key(row) for _, row in newest.iterrows()}
        return frame.sort_values(["time_msc", "bid", "ask", "flags"]).reset_index(drop=True)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        ticks = mt5.copy_ticks_range(SYMBOL, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None:
            self.logger.warning("copy_ticks_range failed: %s", mt5.last_error())
            self.connected = False
            return None
        return ticks_to_frame(ticks)

    def add(self, frame: pd.DataFrame) -> None:
        new_frame = self.filter_new(frame)
        if new_frame.empty:
            return
        self.buffer.append(new_frame)
        self.buffer_rows += len(new_frame)
        self.total_ticks += len(new_frame)
        now_monotonic = time.monotonic()
        if now_monotonic - self.last_live_status_monotonic >= self.args.live_status_seconds:
            latest = new_frame.iloc[-1]
            stamp = latest["timestamp_utc"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logger.info(
                "%s | %s | %.5f | %.5f | %.1f pip | %s ticks",
                stamp,
                SYMBOL,
                latest["bid"],
                latest["ask"],
                latest["spread_pips"],
                f"{self.total_ticks:,}",
            )
            self.last_live_status_monotonic = now_monotonic

    def flush(self) -> None:
        if not self.buffer:
            self.last_flush_monotonic = time.monotonic()
            return
        combined = pd.concat(self.buffer, ignore_index=True)
        write_daily_files(combined, self.args.output_dir)
        self.logger.info("Flushed %s ticks to %s", f"{len(combined):,}", self.args.output_dir)
        self.buffer.clear()
        self.buffer_rows = 0
        self.last_flush_monotonic = time.monotonic()

    def idle_status(self) -> None:
        now_monotonic = time.monotonic()
        if now_monotonic - self.last_idle_status_monotonic < self.args.idle_status_seconds:
            return
        now = datetime.now(timezone.utc)
        if self.last_time_msc is None:
            detail = "no tick received"
        else:
            age = (now - milliseconds_to_utc(self.last_time_msc)).total_seconds()
            detail = f"no fresh tick | last tick age {age:,.1f}s"
        self.logger.info(
            "%s | %s | %s | %s ticks",
            now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            SYMBOL,
            detail,
            f"{self.total_ticks:,}",
        )
        self.last_idle_status_monotonic = now_monotonic

    def run(self) -> None:
        started = time.monotonic()
        reconnect_delay = 1.0
        while True:
            if self.args.duration_seconds and time.monotonic() - started >= self.args.duration_seconds:
                return
            if not self.connection_healthy():
                if not self.connect():
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, self.args.max_reconnect_seconds)
                    continue
                reconnect_delay = 1.0

            now = datetime.now(timezone.utc)
            chunk_start = self.scan_cursor
            received_any = False
            failed = False
            while chunk_start < now:
                chunk_end = min(
                    chunk_start + timedelta(seconds=self.args.recovery_chunk_seconds), now
                )
                frame = self.fetch_range(chunk_start, chunk_end)
                if frame is None:
                    failed = True
                    break
                before = self.total_ticks
                self.add(frame)
                received_any = received_any or self.total_ticks > before
                chunk_start = chunk_end
            if failed:
                mt5.shutdown()
                time.sleep(reconnect_delay)
                continue

            # Re-read a small overlap next cycle; time_msc+quote deduplication
            # prevents boundary duplicates while covering polling-cycle races.
            self.scan_cursor = max(self.scan_cursor, now - timedelta(seconds=1))
            if not received_any:
                self.idle_status()
            if (
                self.buffer_rows >= self.args.max_buffer_ticks
                or time.monotonic() - self.last_flush_monotonic >= self.args.flush_seconds
            ):
                self.flush()
            time.sleep(self.args.poll_seconds)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TICK_DIRECTORY)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--flush-seconds", type=float, default=60.0)
    parser.add_argument("--max-buffer-ticks", type=int, default=50_000)
    parser.add_argument("--initial-lookback-seconds", type=int, default=300)
    parser.add_argument("--recovery-chunk-seconds", type=int, default=3_600)
    parser.add_argument("--max-reconnect-seconds", type=float, default=30.0)
    parser.add_argument("--live-status-seconds", type=float, default=1.0)
    parser.add_argument("--idle-status-seconds", type=float, default=30.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0,
        help="Stop after N seconds; zero runs until Ctrl+C",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.flush_seconds <= 0 or args.recovery_chunk_seconds <= 0:
        parser.error("poll, flush, and recovery chunk values must be positive")
    if args.max_buffer_ticks <= 0 or args.initial_lookback_seconds < 0:
        parser.error("buffer size must be positive and lookback cannot be negative")
    return args


def main() -> None:
    args = arguments()
    logger = configure_logging(args.verbose)
    collector = MT5TickCollector(args, logger)
    try:
        collector.run()
    except KeyboardInterrupt:
        logger.info("Ctrl+C received; stopping collector")
    except (OSError, ValueError) as exc:
        logger.error("Collector stopped: %s", exc)
        raise SystemExit(1) from exc
    finally:
        try:
            collector.flush()
        finally:
            mt5.shutdown()
        logger.info("Collector stopped cleanly; total ticks=%s", f"{collector.total_ticks:,}")


if __name__ == "__main__":
    main()
