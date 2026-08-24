"""Allowlisted local V0.5A candle access for the V0.7 API."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.dashboard.v07_contract import MAX_CANDLE_LIMIT, TIMEFRAMES


def load_candles(timeframe: str, limit: int) -> dict[str, object]:
    label = timeframe.upper()
    if label not in TIMEFRAMES:
        raise ValueError("timeframe must be one of M1, M5, M15, H1")
    bounded = max(1, min(int(limit), MAX_CANDLE_LIMIT))
    path = TIMEFRAMES[label]
    if not path.exists():
        return {"timeframe": label, "count": 0, "candles": [], "status": "DATA_UNAVAILABLE"}
    required = ["bar_open_utc", "open", "high", "low", "close", "tick_volume"]
    frame = pd.read_parquet(path, columns=required).tail(bounded).copy()
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
            "volume": volume,
        })
    return {"timeframe": label, "count": len(candles), "candles": candles, "status": "PASS"}
