"""Allowlisted local V0.5A candle access for the V0.7 API."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.assets.contracts import asset_paths, normalize_asset_id
from src.dashboard.v07_contract import MAX_CANDLE_LIMIT, TIMEFRAMES


@lru_cache(maxsize=8)
def _cached_candle_frame(path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    required = ["bar_open_utc", "open", "high", "low", "close", "tick_volume"]
    return pd.read_parquet(Path(path_text), columns=required)


def load_candles(timeframe: str, limit: int, symbol: str = "EURUSD") -> dict[str, object]:
    label = timeframe.upper()
    if label not in TIMEFRAMES:
        raise ValueError("timeframe must be one of M1, M5, M15, H1")
    bounded = max(1, min(int(limit), MAX_CANDLE_LIMIT))
    asset = normalize_asset_id(symbol)
    path = TIMEFRAMES[label] if asset == "EURUSD" else asset_paths(asset).bar_file(label)
    if not path.exists():
        return {"symbol": asset, "timeframe": label, "count": 0, "candles": [], "status": "DATA_UNAVAILABLE"}
    required = ["bar_open_utc", "open", "high", "low", "close", "tick_volume"]
    frame = _cached_candle_frame(str(path.resolve()), path.stat().st_mtime_ns).tail(bounded).copy()
    frame["bar_open_utc"] = pd.to_datetime(frame["bar_open_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=required[:5])
    candles: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        values = [float(row[name]) for name in ("open", "high", "low", "close")]
        if not all(math.isfinite(value) for value in values):
            continue
        volume = float(row["tick_volume"]) if pd.notna(row["tick_volume"]) else None
        candles.append({
            "time": row["bar_open_utc"].isoformat(), "open": values[0],
            "high": values[1], "low": values[2], "close": values[3],
            "volume": volume, "is_complete": True,
        })
    return {"symbol": asset, "timeframe": label, "count": len(candles), "candles": candles, "status": "PASS"}
