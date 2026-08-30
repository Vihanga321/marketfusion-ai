"""Mechanical liquidity observations: equal extrema, sweeps, and three-bar FVGs."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from src.engines.contract import SwingPoint
from src.engines.swings import average_true_range, causal_bars, utc


def _id(*parts: object) -> str:
    return sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def liquidity_analysis(
    frame: pd.DataFrame,
    timeframe: str,
    swings: list[SwingPoint],
    *,
    as_of: object | None = None,
    limit: int = 800,
) -> dict[str, Any]:
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if bars.empty:
        return {"equal_levels": [], "sweeps": [], "fair_value_gaps": [], "nearest_liquidity": None}
    atr = average_true_range(bars)
    current_atr = float(atr.iloc[-1])
    current = float(bars.iloc[-1]["close"])
    spread = float(pd.to_numeric(bars.get("spread_points", pd.Series([0.0])), errors="coerce").fillna(0).iloc[-1]) * 0.00001
    tolerance = max(current_atr * 0.12, spread * 2.0, current * 0.00004)
    times = pd.to_datetime(bars["bar_close_utc"], utc=True)
    time_values = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    equal_levels: list[dict[str, Any]] = []
    for kind in ("HIGH", "LOW"):
        same = [item for item in swings if item["kind"] == kind]
        for left, right in zip(same, same[1:]):
            if abs(float(left["price"]) - float(right["price"])) > tolerance:
                continue
            price = float(np.mean([left["price"], right["price"]]))
            available = max(left["confirmed_at_utc"], right["confirmed_at_utc"])
            equal_levels.append({
                "level_id": _id(timeframe, "EQUAL", kind, left["swing_id"], right["swing_id"]),
                "event_type": "EQUAL_HIGHS" if kind == "HIGH" else "EQUAL_LOWS",
                "timeframe": timeframe, "kind": kind, "price": price,
                "detected_at_utc": available, "status": "CONFIRMED",
                "tolerance": tolerance, "distance_price": abs(current - price),
                "distance_atr": abs(current - price) / max(current_atr, 1e-12),
                "swing_ids": [left["swing_id"], right["swing_id"]],
            })
    sweeps: list[dict[str, Any]] = []
    for level in equal_levels:
        start = int(np.searchsorted(time_values, utc(level["detected_at_utc"]).value, side="right"))
        if level["kind"] == "HIGH":
            matches = np.flatnonzero((highs[start:] > level["price"] + tolerance) & (closes[start:] < level["price"]))
        else:
            matches = np.flatnonzero((lows[start:] < level["price"] - tolerance) & (closes[start:] > level["price"]))
        if len(matches):
            index = start + int(matches[0])
            high_sweep = level["kind"] == "HIGH"
            detected = times.iloc[index].isoformat()
            sweeps.append({"event_id": _id(timeframe, "SWEEP", level["level_id"], detected), "event_type": "BUY_SIDE_LIQUIDITY_SWEEP" if high_sweep else "SELL_SIDE_LIQUIDITY_SWEEP", "timeframe": timeframe, "detected_at_utc": detected, "status": "CONFIRMED", "direction": "BEARISH" if high_sweep else "BULLISH", "level_id": level["level_id"], "level": level["price"], "extreme": float(highs[index] if high_sweep else lows[index])})
    gaps: list[dict[str, Any]] = []
    for i in range(2, len(bars)):
        direction: str | None = None
        lower = upper = 0.0
        if lows[i] > highs[i - 2]:
            direction, lower, upper = "BULLISH", float(highs[i - 2]), float(lows[i])
        elif highs[i] < lows[i - 2]:
            direction, lower, upper = "BEARISH", float(highs[i]), float(lows[i - 2])
        if direction is None:
            continue
        detected = times.iloc[i]
        status, filled_at, fill_fraction = "OPEN", None, 0.0
        if direction == "BULLISH":
            full = np.flatnonzero(lows[i + 1:] <= lower)
            stop = i + 1 + int(full[0]) if len(full) else len(bars)
            partial_values = lows[i + 1:stop]
            if len(full):
                status, filled_at, fill_fraction = "FILLED", times.iloc[stop].isoformat(), 1.0
            elif len(partial_values) and float(partial_values.min()) < upper:
                status, fill_fraction = "PARTIALLY_FILLED", float((upper - partial_values.min()) / max(upper - lower, 1e-12))
        else:
            full = np.flatnonzero(highs[i + 1:] >= upper)
            stop = i + 1 + int(full[0]) if len(full) else len(bars)
            partial_values = highs[i + 1:stop]
            if len(full):
                status, filled_at, fill_fraction = "FILLED", times.iloc[stop].isoformat(), 1.0
            elif len(partial_values) and float(partial_values.max()) > lower:
                status, fill_fraction = "PARTIALLY_FILLED", float((partial_values.max() - lower) / max(upper - lower, 1e-12))
        gaps.append({
            "event_id": _id(timeframe, "FVG", direction, detected.isoformat(), round(lower, 8), round(upper, 8)),
            "event_type": "FAIR_VALUE_GAP", "timeframe": timeframe,
            "detected_at_utc": detected.isoformat(), "status": status, "direction": direction,
            "lower_boundary": lower, "upper_boundary": upper, "size": upper - lower,
            "middle_bar_time_utc": times.iloc[i - 1].isoformat(),
            "filled_at_utc": filled_at, "fill_fraction": float(np.clip(fill_fraction, 0.0, 1.0)),
        })
    candidates = [*equal_levels, *[item for item in gaps if item["status"] != "FILLED"]]
    nearest = None
    if candidates:
        def candidate_price(item: dict[str, Any]) -> float:
            if "price" in item:
                return float(item["price"])
            return float((item["lower_boundary"] + item["upper_boundary"]) / 2.0)
        nearest = min(candidates, key=lambda item: abs(current - candidate_price(item)))
    return {
        "equal_levels": equal_levels[-10:], "sweeps": sweeps[-10:], "fair_value_gaps": gaps[-12:],
        "nearest_liquidity": nearest, "tolerance": tolerance,
    }
