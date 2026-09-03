"""Efficient read-only MT5 latest-quote collector for dashboard presentation.

This process never builds features, persists tick history, or exposes trading APIs.
Its replaceable JSON snapshot is IPC for latest-value UI state only.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from src.assets.contracts import normalize_asset_id
from src.marketdata.v05a_contract import ROOT, SYMBOL

RUNTIME_DIR = ROOT / "data" / "runtime" / "realtime"
LATEST_QUOTE_FILE = RUNTIME_DIR / f"{SYMBOL}_quote.json"
DEFAULT_POLL_SECONDS = 0.25
MIN_POLL_SECONDS = 0.10
MAX_POLL_SECONDS = 1.00
MAX_CLOCK_ERROR_SECONDS = 5.0
MAX_BROKER_OFFSET_HOURS = 14
STALE_AFTER_SECONDS = 3.0
RECONNECT_DELAY_SECONDS = 1.0
TELEMETRY_WINDOW = 512


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def normalize_tick_timestamp(raw_time_msc: int, received_at: datetime) -> tuple[datetime, int]:
    """Normalize a broker epoch that may encode a whole-hour server offset.

    MT5 brokers sometimes expose server-wall-clock seconds through ``time_msc``.
    Only a whole-hour correction producing a timestamp close to receipt is
    accepted. No future timestamp is allowed into the published contract.
    """
    if raw_time_msc <= 0:
        raise ValueError("MT5 tick time_msc must be positive")
    received = received_at.astimezone(timezone.utc)
    raw = datetime.fromtimestamp(raw_time_msc / 1000.0, tz=timezone.utc)
    candidates: list[tuple[float, int, datetime]] = []
    for offset_hours in range(-MAX_BROKER_OFFSET_HOURS, MAX_BROKER_OFFSET_HOURS + 1):
        candidate = raw - timedelta(hours=offset_hours)
        error = abs((received - candidate).total_seconds())
        candidates.append((error, offset_hours, candidate))
    error, offset, normalized = min(candidates, key=lambda item: item[0])
    if error > MAX_CLOCK_ERROR_SECONDS:
        raise ValueError(f"MT5 tick timestamp differs from receipt by {error:.3f}s after timezone normalization")
    if normalized > received + timedelta(seconds=1):
        raise ValueError("MT5 tick timestamp is in the future")
    return normalized, offset


class RollingTelemetry:
    def __init__(self, window: int = TELEMETRY_WINDOW) -> None:
        self.values: dict[str, deque[float]] = {}
        self.window = window

    def observe(self, name: str, value_ms: float) -> None:
        self.values.setdefault(name, deque(maxlen=self.window)).append(float(value_ms))

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for name, values in self.values.items():
            ordered = sorted(values)
            if not ordered:
                result[name] = {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
                continue
            p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
            middle = len(ordered) // 2
            median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
            result[name] = {
                "count": len(ordered), "p50_ms": round(median, 3),
                "p95_ms": round(ordered[p95_index], 3), "max_ms": round(ordered[-1], 3),
            }
        return result


class PartialM1Candle:
    """UI-only partial candle. It is deliberately outside V0.5A contracts."""

    def __init__(self) -> None:
        self.minute: datetime | None = None
        self.open = self.high = self.low = self.close = 0.0
        self.tick_count = 0

    def update(self, timestamp: datetime, mid: float) -> dict[str, object]:
        minute = timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
        if self.minute != minute:
            self.minute = minute
            self.open = self.high = self.low = self.close = mid
            self.tick_count = 1
        else:
            self.high = max(self.high, mid)
            self.low = min(self.low, mid)
            self.close = mid
            self.tick_count += 1
        return {
            "time": _iso(minute), "open": self.open, "high": self.high,
            "low": self.low, "close": self.close, "volume": None,
            "tick_count": self.tick_count, "is_complete": False,
            "usage": "DASHBOARD_ONLY_NOT_CAUSAL",
        }


def quote_snapshot(tick: Any, point_size: float, received_at: datetime, partial: PartialM1Candle, symbol: str = SYMBOL) -> dict[str, object]:
    bid, ask = float(tick.bid), float(tick.ask)
    if bid <= 0 or ask <= 0 or ask < bid or point_size <= 0:
        raise ValueError("Invalid MT5 quote")
    raw_time_msc = int(getattr(tick, "time_msc", 0) or int(tick.time) * 1000)
    normalized, broker_offset = normalize_tick_timestamp(raw_time_msc, received_at)
    mid = (bid + ask) / 2.0
    age_ms = max(0.0, (received_at - normalized).total_seconds() * 1000.0)
    return {
        "contract_version": "realtime-quote-v1", "symbol": normalize_asset_id(symbol),
        "connection": "LIVE", "freshness": "LIVE" if age_ms <= STALE_AFTER_SECONDS * 1000 else "STALE",
        "received_at_utc": _iso(received_at),
        "raw_tick_time": _iso(datetime.fromtimestamp(raw_time_msc / 1000.0, tz=timezone.utc)),
        "normalized_tick_utc": _iso(normalized), "broker_utc_offset_hours": broker_offset,
        "bid": bid, "ask": ask, "last": float(getattr(tick, "last", 0.0) or 0.0) or None,
        "mid": mid, "spread": ask - bid, "spread_points": (ask - bid) / point_size,
        "quote_age_ms_at_collection": round(age_ms, 3),
        "partial_m1": partial.update(normalized, mid),
        "trading_enabled": False,
    }


def atomic_write_json(payload: dict[str, object], path: Path = LATEST_QUOTE_FILE, retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    try:
        for attempt in range(retries):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt + 1 == retries:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class RealtimeQuoteCollector:
    def __init__(self, poll_seconds: float = DEFAULT_POLL_SECONDS, clock: Callable[[], datetime] = utc_now, symbol: str = SYMBOL, output_path: Path | None = None) -> None:
        if not MIN_POLL_SECONDS <= poll_seconds <= MAX_POLL_SECONDS:
            raise ValueError(f"poll_seconds must be between {MIN_POLL_SECONDS} and {MAX_POLL_SECONDS}")
        self.poll_seconds = poll_seconds
        self.clock = clock
        self.symbol = normalize_asset_id(symbol)
        self.output_path = RUNTIME_DIR / f"{self.symbol}_quote.json" if output_path is None else output_path
        self.partial = PartialM1Candle()
        self.telemetry = RollingTelemetry()
        self.sequence = 0
        self.last_key: tuple[object, ...] | None = None
        self.last_publish = 0.0

    def collect(self, mt5: Any, point_size: float) -> dict[str, object] | None:
        started = time.perf_counter()
        tick = mt5.symbol_info_tick(self.symbol)
        poll_ms = (time.perf_counter() - started) * 1000
        self.telemetry.observe("quote_poll_duration", poll_ms)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({self.symbol}) failed: {mt5.last_error()}")
        received = self.clock()
        payload = quote_snapshot(tick, point_size, received, self.partial, self.symbol)
        self.telemetry.observe("quote_age", float(payload["quote_age_ms_at_collection"]))
        key = (payload["normalized_tick_utc"], payload["bid"], payload["ask"])
        now_monotonic = time.monotonic()
        if key == self.last_key and now_monotonic - self.last_publish < 1.0:
            return None
        self.sequence += 1
        payload["sequence"] = self.sequence
        payload["poll_interval_ms"] = round(self.poll_seconds * 1000)
        payload["performance"] = self.telemetry.snapshot()
        persisted = time.perf_counter()
        atomic_write_json(payload, self.output_path)
        persist_ms = (time.perf_counter() - persisted) * 1000
        self.telemetry.observe("snapshot_persist_duration", persist_ms)
        self.last_key, self.last_publish = key, now_monotonic
        return payload


def _connect_mt5(mt5: Any, collector: RealtimeQuoteCollector) -> float:
    """Reconnect the display-only MT5 session and return point size."""
    try:
        mt5.shutdown()
    except Exception:
        pass
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    if not mt5.symbol_select(collector.symbol, True):
        raise RuntimeError(f"mt5.symbol_select({collector.symbol!r}, True) failed: {mt5.last_error()}")
    info = mt5.symbol_info(collector.symbol)
    if info is None or float(info.point) <= 0:
        raise RuntimeError(f"Could not read symbol info for {collector.symbol}")
    return float(info.point)


def _degraded_payload(collector: RealtimeQuoteCollector, exc: Exception) -> dict[str, object]:
    return {
        "contract_version": "realtime-quote-v1", "symbol": collector.symbol,
        "connection": "DEGRADED", "freshness": "UNAVAILABLE",
        "received_at_utc": _iso(utc_now()), "normalized_tick_utc": None,
        "bid": None, "ask": None, "mid": None, "spread_points": None,
        "error": f"{type(exc).__name__}: {exc}",
        "sequence": collector.sequence, "performance": collector.telemetry.snapshot(),
        "trading_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-ms", type=int, default=250)
    parser.add_argument("--symbol", default=SYMBOL, choices=("EURUSD", "XAUUSD"))
    args = parser.parse_args()
    collector = RealtimeQuoteCollector(args.poll_ms / 1000.0, symbol=args.symbol)
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"MetaTrader5 unavailable: {exc}")

    point_size: float | None = None
    next_poll = time.monotonic()
    try:
        while True:
            try:
                if point_size is None:
                    point_size = _connect_mt5(mt5, collector)
                collector.collect(mt5, point_size)
            except (RuntimeError, ValueError, OSError) as exc:
                atomic_write_json(_degraded_payload(collector, exc), collector.output_path)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                point_size = None
                time.sleep(RECONNECT_DELAY_SECONDS)
                next_poll = time.monotonic()
            next_poll += collector.poll_seconds
            time.sleep(max(0.0, next_poll - time.monotonic()))
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
