"""Deterministic chart-pattern geometry over confirmed causal swings."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from src.engines.contract import PatternRecord, SwingPoint
from src.engines.swings import alternating_swings, average_true_range, causal_bars, utc

PATTERN_TYPES = (
    "DOUBLE_TOP", "DOUBLE_BOTTOM", "TRIPLE_TOP", "TRIPLE_BOTTOM",
    "HEAD_AND_SHOULDERS", "INVERSE_HEAD_AND_SHOULDERS",
    "RISING_WEDGE", "FALLING_WEDGE", "BULL_FLAG", "BEAR_FLAG",
    "BULL_PENNANT", "BEAR_PENNANT", "ASCENDING_TRIANGLE", "DESCENDING_TRIANGLE",
    "SYMMETRICAL_TRIANGLE", "EXPANDING_TRIANGLE", "RECTANGLE_RANGE",
)


def _pattern_id(timeframe: str, name: str, points: list[SwingPoint], detected: str) -> str:
    identity = "|".join([timeframe, name, detected, *[item["swing_id"] for item in points]])
    return sha256(identity.encode()).hexdigest()[:24]


def _lifecycle(
    bars: pd.DataFrame, detected_at: str, direction: str, upper: float | None, lower: float | None,
    tolerance: float,
) -> tuple[str, str | None, str]:
    future = bars.loc[bars["bar_close_utc"].gt(utc(detected_at))]
    for _, row in future.iterrows():
        broke_up = upper is not None and row["close"] > upper + tolerance
        broke_down = lower is not None and row["close"] < lower - tolerance
        if not broke_up and not broke_down:
            continue
        actual = "BULLISH" if broke_up else "BEARISH"
        status = "CONFIRMED" if direction == "NEUTRAL" or actual == direction else "BROKEN"
        return status, pd.Timestamp(row["bar_close_utc"]).isoformat(), actual
    return "FORMING", None, direction


def _record(
    timeframe: str,
    name: str,
    points: list[SwingPoint],
    bars: pd.DataFrame,
    direction: str,
    score: float,
    upper: float | None,
    lower: float | None,
    tolerance: float,
    evidence: dict[str, Any],
) -> PatternRecord:
    detected = max(item["confirmed_at_utc"] for item in points)
    status, confirmed, resolved_direction = _lifecycle(bars, detected, direction, upper, lower, tolerance)
    if upper is not None and lower is not None and upper <= lower:
        status = "INVALIDATED"
    breakout = upper if resolved_direction == "BULLISH" else lower if resolved_direction == "BEARISH" else None
    invalidation = lower if resolved_direction == "BULLISH" else upper if resolved_direction == "BEARISH" else None
    return {
        "pattern_id": _pattern_id(timeframe, name, points, detected),
        "pattern_type": name, "timeframe": timeframe, "detected_at_utc": detected,
        "confirmed_at_utc": confirmed, "status": status, "direction": resolved_direction,
        "score": float(np.clip(score, 0.0, 1.0)),
        # Confidence means geometry completeness/fit only, never outcome probability.
        "confidence": float(np.clip(score, 0.0, 1.0)),
        "start_time_utc": min(item["pivot_time_utc"] for item in points),
        "end_time_utc": max(item["pivot_time_utc"] for item in points),
        "upper_boundary": upper, "lower_boundary": lower,
        "breakout_level": breakout, "invalidation_level": invalidation,
        "evidence": {**evidence, "score_semantics": "STRUCTURAL_GEOMETRY_FIT_NOT_TRADING_PROBABILITY", "swing_ids": [item["swing_id"] for item in points]},
    }


def _extrema_patterns(swings: list[SwingPoint], bars: pd.DataFrame, timeframe: str, tolerance: float) -> list[PatternRecord]:
    result: list[PatternRecord] = []
    ordered = alternating_swings(swings)[-24:]
    for i in range(2, len(ordered)):
        three = ordered[i - 2:i + 1]
        if [item["kind"] for item in three] == ["HIGH", "LOW", "HIGH"]:
            error = abs(three[0]["price"] - three[2]["price"])
            depth = np.mean([three[0]["price"], three[2]["price"]]) - three[1]["price"]
            if error <= tolerance and depth >= tolerance * 2:
                result.append(_record(timeframe, "DOUBLE_TOP", three, bars, "BEARISH", 1 - error / max(tolerance, 1e-12), float(max(three[0]["price"], three[2]["price"])), float(three[1]["price"]), tolerance, {"peak_error": error, "neckline_depth": depth}))
        if [item["kind"] for item in three] == ["LOW", "HIGH", "LOW"]:
            error = abs(three[0]["price"] - three[2]["price"])
            height = three[1]["price"] - np.mean([three[0]["price"], three[2]["price"]])
            if error <= tolerance and height >= tolerance * 2:
                result.append(_record(timeframe, "DOUBLE_BOTTOM", three, bars, "BULLISH", 1 - error / max(tolerance, 1e-12), float(three[1]["price"]), float(min(three[0]["price"], three[2]["price"])), tolerance, {"trough_error": error, "neckline_height": height}))
    for i in range(4, len(ordered)):
        five = ordered[i - 4:i + 1]
        kinds = [item["kind"] for item in five]
        if kinds == ["HIGH", "LOW", "HIGH", "LOW", "HIGH"]:
            highs = np.array([five[j]["price"] for j in (0, 2, 4)], dtype=float)
            spread = float(highs.max() - highs.min())
            neck = float(min(five[1]["price"], five[3]["price"]))
            if spread <= tolerance and highs.mean() - neck >= tolerance * 2:
                result.append(_record(timeframe, "TRIPLE_TOP", five, bars, "BEARISH", 1 - spread / max(tolerance, 1e-12), float(highs.max()), neck, tolerance, {"peak_spread": spread}))
            shoulder_error = abs(five[0]["price"] - five[4]["price"])
            head_prominence = five[2]["price"] - max(five[0]["price"], five[4]["price"])
            if shoulder_error <= tolerance * 1.5 and head_prominence >= tolerance * 1.5:
                result.append(_record(timeframe, "HEAD_AND_SHOULDERS", five, bars, "BEARISH", min(1.0, head_prominence / max(tolerance * 3, 1e-12)), float(five[2]["price"]), neck, tolerance, {"shoulder_error": shoulder_error, "head_prominence": head_prominence}))
        if kinds == ["LOW", "HIGH", "LOW", "HIGH", "LOW"]:
            lows = np.array([five[j]["price"] for j in (0, 2, 4)], dtype=float)
            spread = float(lows.max() - lows.min())
            neck = float(max(five[1]["price"], five[3]["price"]))
            if spread <= tolerance and neck - lows.mean() >= tolerance * 2:
                result.append(_record(timeframe, "TRIPLE_BOTTOM", five, bars, "BULLISH", 1 - spread / max(tolerance, 1e-12), neck, float(lows.min()), tolerance, {"trough_spread": spread}))
            shoulder_error = abs(five[0]["price"] - five[4]["price"])
            head_prominence = min(five[0]["price"], five[4]["price"]) - five[2]["price"]
            if shoulder_error <= tolerance * 1.5 and head_prominence >= tolerance * 1.5:
                result.append(_record(timeframe, "INVERSE_HEAD_AND_SHOULDERS", five, bars, "BULLISH", min(1.0, head_prominence / max(tolerance * 3, 1e-12)), neck, float(five[2]["price"]), tolerance, {"shoulder_error": shoulder_error, "head_prominence": head_prominence}))
    return result


def _line(points: list[SwingPoint]) -> tuple[float, float, float]:
    x = np.arange(len(points), dtype=float)
    y = np.array([item["price"] for item in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_total = float(np.square(y - y.mean()).sum())
    r2 = 1.0 if ss_total <= 1e-20 else 1.0 - float(np.square(y - fitted).sum()) / ss_total
    return float(slope), float(intercept), float(np.clip(r2, 0.0, 1.0))


def _consolidation_patterns(swings: list[SwingPoint], bars: pd.DataFrame, timeframe: str, tolerance: float) -> list[PatternRecord]:
    recent = sorted(swings, key=lambda item: item["pivot_time_utc"])[-14:]
    highs = [item for item in recent if item["kind"] == "HIGH"][-4:]
    lows = [item for item in recent if item["kind"] == "LOW"][-4:]
    if len(highs) < 3 or len(lows) < 3:
        return []
    high_slope, _, high_r2 = _line(highs)
    low_slope, _, low_r2 = _line(lows)
    scale = max(tolerance, 1e-12)
    hs, ls = high_slope / scale, low_slope / scale
    flat = 0.45
    upper, lower = float(np.mean([item["price"] for item in highs[-2:]])), float(np.mean([item["price"] for item in lows[-2:]]))
    points = sorted([*highs, *lows], key=lambda item: item["pivot_time_utc"])
    fit = float(np.mean([high_r2, low_r2]))
    name = direction = None
    if abs(hs) <= flat and ls > flat:
        name, direction = "ASCENDING_TRIANGLE", "BULLISH"
    elif abs(ls) <= flat and hs < -flat:
        name, direction = "DESCENDING_TRIANGLE", "BEARISH"
    elif hs < -flat and ls > flat:
        name, direction = "SYMMETRICAL_TRIANGLE", "NEUTRAL"
    elif hs > flat and ls < -flat:
        name, direction = "EXPANDING_TRIANGLE", "NEUTRAL"
    elif hs > flat and ls > flat and ls > hs + flat:
        name, direction = "RISING_WEDGE", "BEARISH"
    elif hs < -flat and ls < -flat and hs > ls + flat:
        name, direction = "FALLING_WEDGE", "BULLISH"
    elif abs(hs) <= flat and abs(ls) <= flat:
        name, direction = "RECTANGLE_RANGE", "NEUTRAL"
    if name is None:
        return []
    return [_record(timeframe, name, points, bars, direction, max(0.4, fit), upper, lower, tolerance, {"upper_slope_atr": hs, "lower_slope_atr": ls, "upper_r2": high_r2, "lower_r2": low_r2})]


def _continuation_patterns(bars: pd.DataFrame, timeframe: str, tolerance: float, swings: list[SwingPoint]) -> list[PatternRecord]:
    if len(bars) < 24 or len(swings) < 2:
        return []
    window = bars.tail(24)
    impulse = window.iloc[:8]
    consolidation = window.iloc[8:]
    impulse_move = float(impulse.iloc[-1]["close"] - impulse.iloc[0]["open"])
    atr = float(average_true_range(window).iloc[-1])
    if abs(impulse_move) < atr * 2.5:
        return []
    x = np.arange(len(consolidation), dtype=float)
    high_slope = float(np.polyfit(x, consolidation["high"].to_numpy(), 1)[0]) / max(atr, 1e-12)
    low_slope = float(np.polyfit(x, consolidation["low"].to_numpy(), 1)[0]) / max(atr, 1e-12)
    direction = "BULLISH" if impulse_move > 0 else "BEARISH"
    points = sorted(swings[-8:], key=lambda item: item["pivot_time_utc"])
    upper, lower = float(consolidation["high"].max()), float(consolidation["low"].min())
    parallel = abs(high_slope - low_slope) <= 0.08
    converging = high_slope < -0.01 and low_slope > 0.01
    name = None
    if direction == "BULLISH" and parallel and high_slope < 0.02:
        name = "BULL_FLAG"
    elif direction == "BEARISH" and parallel and low_slope > -0.02:
        name = "BEAR_FLAG"
    elif converging:
        name = "BULL_PENNANT" if direction == "BULLISH" else "BEAR_PENNANT"
    if name is None:
        return []
    return [_record(timeframe, name, points, bars, direction, min(1.0, abs(impulse_move) / max(atr * 5, 1e-12)), upper, lower, tolerance, {"impulse_atr": impulse_move / max(atr, 1e-12), "upper_slope_atr": high_slope, "lower_slope_atr": low_slope})]


def detect_patterns(
    frame: pd.DataFrame,
    timeframe: str,
    swings: list[SwingPoint],
    *,
    as_of: object | None = None,
    limit: int = 1200,
) -> list[PatternRecord]:
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if bars.empty or len(swings) < 3:
        return []
    atr = float(average_true_range(bars).iloc[-1])
    spread = float(pd.to_numeric(bars.get("spread_points", pd.Series([0.0])), errors="coerce").fillna(0).iloc[-1]) * 0.00001
    tolerance = max(atr * 0.25, spread * 2.0, float(bars.iloc[-1]["close"]) * 0.00008)
    patterns = [
        *_extrema_patterns(swings, bars, timeframe, tolerance),
        *_consolidation_patterns(swings, bars, timeframe, tolerance),
        *_continuation_patterns(bars, timeframe, tolerance, swings),
    ]
    unique = {item["pattern_id"]: item for item in patterns}
    return sorted(unique.values(), key=lambda item: (item["detected_at_utc"], item["pattern_type"]))


def detect_historical_patterns(
    frame: pd.DataFrame,
    timeframe: str,
    swings: list[SwingPoint],
    *,
    as_of: object | None = None,
    limit: int = 2000,
) -> list[PatternRecord]:
    """Bounded research-only scan; each candidate sees only its prefix."""
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if bars.empty:
        return []
    atr = float(average_true_range(bars).iloc[-1])
    tolerance = max(atr * 0.25, float(bars.iloc[-1]["close"]) * 0.00008)
    ordered = sorted(swings, key=lambda item: item["confirmed_at_utc"])
    found: dict[str, PatternRecord] = {}
    for end in range(3, len(ordered) + 1):
        prefix_swings = ordered[max(0, end - 14):end]
        cutoff = utc(prefix_swings[-1]["confirmed_at_utc"])
        prefix_bars = bars.loc[bars["bar_close_utc"].le(cutoff)]
        if prefix_bars.empty:
            continue
        for item in [*_extrema_patterns(prefix_swings, prefix_bars, timeframe, tolerance), *_consolidation_patterns(prefix_swings, prefix_bars, timeframe, tolerance)]:
            found[item["pattern_id"]] = item
    return sorted(found.values(), key=lambda item: item["detected_at_utc"])
