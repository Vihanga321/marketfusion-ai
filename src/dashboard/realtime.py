"""Latest-value runtime quote access for the localhost dashboard API."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

from src.assets.contracts import normalize_asset_id
from src.marketdata.realtime_quote import LATEST_QUOTE_FILE, RUNTIME_DIR


class LatestValueBuffer:
    """A single-slot sequence buffer; superseded UI values are never queued."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._version = 0
        self._value: dict[str, Any] | None = None

    def publish(self, value: dict[str, Any]) -> int:
        with self._lock:
            self._version += 1
            self._value = deepcopy(value)
            return self._version

    def latest(self, after_version: int = -1) -> tuple[int, dict[str, Any] | None]:
        with self._lock:
            if self._version == after_version or self._value is None:
                return self._version, None
            return self._version, deepcopy(self._value)


@lru_cache(maxsize=4)
def _read_snapshot(path_text: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_quote_snapshot(path: Path | None = None, symbol: str = "EURUSD") -> dict[str, Any]:
    asset = normalize_asset_id(symbol)
    path = (LATEST_QUOTE_FILE if asset == "EURUSD" else RUNTIME_DIR / f"{asset}_quote.json") if path is None else path
    try:
        return deepcopy(_read_snapshot(str(path.resolve()), path.stat().st_mtime_ns))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {
            "contract_version": "realtime-quote-v1", "symbol": asset,
            "connection": "DISCONNECTED", "freshness": "UNAVAILABLE",
            "received_at_utc": None, "normalized_tick_utc": None,
            "bid": None, "ask": None, "mid": None, "spread_points": None,
            "partial_m1": None, "sequence": 0, "trading_enabled": False,
        }


def prepare_stream_payload(snapshot: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    sent = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    payload = deepcopy(snapshot)
    payload["api_sent_at_utc"] = sent.isoformat()
    try:
        received = datetime.fromisoformat(str(snapshot["received_at_utc"]).replace("Z", "+00:00"))
        payload["feed_delivery_ms"] = round(max(0.0, (sent - received).total_seconds() * 1000), 3)
    except (KeyError, TypeError, ValueError):
        payload["feed_delivery_ms"] = None
    return payload


class RollingStreamTelemetry:
    def __init__(self, window: int = 512) -> None:
        self._values: deque[float] = deque(maxlen=window)
        self._lock = Lock()

    def observe(self, value: float | None) -> None:
        if value is None:
            return
        with self._lock:
            self._values.append(float(value))

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            ordered = sorted(self._values)
        if not ordered:
            return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
        return {"count": len(ordered), "p50_ms": round(median, 3), "p95_ms": round(p95, 3), "max_ms": round(ordered[-1], 3)}


STREAM_TELEMETRY = RollingStreamTelemetry()
