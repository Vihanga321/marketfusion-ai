"""Deterministic candle geometry and causal breakout observations."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from src.engines.swings import average_true_range, causal_bars


def _event(timeframe: str, name: str, row: pd.Series, direction: str, score: float, evidence: dict[str, Any]) -> dict[str, Any]:
    detected = pd.Timestamp(row["bar_close_utc"]).isoformat()
    identity = sha256(f"{timeframe}|{name}|{detected}".encode()).hexdigest()[:24]
    return {
        "event_id": identity, "event_type": name, "timeframe": timeframe,
        "detected_at_utc": detected, "status": "CONFIRMED", "direction": direction,
        "score": float(np.clip(score, 0.0, 1.0)), "evidence": evidence,
    }


def candle_events(frame: pd.DataFrame, timeframe: str, *, as_of: object | None = None, limit: int = 200) -> list[dict[str, Any]]:
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if bars.empty:
        return []
    body = (bars["close"] - bars["open"]).abs()
    span = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    upper = bars["high"] - bars[["open", "close"]].max(axis=1)
    lower = bars[["open", "close"]].min(axis=1) - bars["low"]
    atr = average_true_range(bars)
    prior_three = bars["close"].shift(1) - bars["close"].shift(4)
    events: list[dict[str, Any]] = []
    for i in range(len(bars)):
        row = bars.iloc[i]
        ratio = float(body.iloc[i] / span.iloc[i]) if np.isfinite(span.iloc[i]) else 0.0
        upper_ratio = float(upper.iloc[i] / span.iloc[i]) if np.isfinite(span.iloc[i]) else 0.0
        lower_ratio = float(lower.iloc[i] / span.iloc[i]) if np.isfinite(span.iloc[i]) else 0.0
        geometry = {"body_ratio": ratio, "upper_wick_ratio": upper_ratio, "lower_wick_ratio": lower_ratio}
        bullish = row["close"] > row["open"]
        bearish = row["close"] < row["open"]
        context_down = i >= 4 and prior_three.iloc[i] < -0.25 * atr.iloc[i]
        context_up = i >= 4 and prior_three.iloc[i] > 0.25 * atr.iloc[i]
        if ratio <= 0.10:
            events.append(_event(timeframe, "DOJI", row, "NEUTRAL", 1.0 - ratio / 0.10, geometry))
        if lower_ratio >= 0.60 and ratio <= 0.35 and upper_ratio <= 0.20:
            name = "HAMMER" if context_down else "HANGING_MAN" if context_up else "BULLISH_PIN_BAR"
            direction = "BULLISH" if name != "HANGING_MAN" else "BEARISH"
            events.append(_event(timeframe, name, row, direction, lower_ratio, {**geometry, "context": "DOWN" if context_down else "UP" if context_up else "NONE"}))
        if upper_ratio >= 0.60 and ratio <= 0.35 and lower_ratio <= 0.20:
            name = "SHOOTING_STAR" if context_up else "INVERTED_HAMMER" if context_down else "BEARISH_PIN_BAR"
            direction = "BEARISH" if name != "INVERTED_HAMMER" else "BULLISH"
            events.append(_event(timeframe, name, row, direction, upper_ratio, {**geometry, "context": "UP" if context_up else "DOWN" if context_down else "NONE"}))
        if i >= 1:
            prior = bars.iloc[i - 1]
            if bullish and prior["close"] < prior["open"] and row["open"] <= prior["close"] and row["close"] >= prior["open"]:
                events.append(_event(timeframe, "BULLISH_ENGULFING", row, "BULLISH", min(1.0, body.iloc[i] / max(body.iloc[i - 1], 1e-12)), geometry))
            if bearish and prior["close"] > prior["open"] and row["open"] >= prior["close"] and row["close"] <= prior["open"]:
                events.append(_event(timeframe, "BEARISH_ENGULFING", row, "BEARISH", min(1.0, body.iloc[i] / max(body.iloc[i - 1], 1e-12)), geometry))
            if row["high"] < prior["high"] and row["low"] > prior["low"]:
                events.append(_event(timeframe, "INSIDE_BAR", row, "NEUTRAL", 1.0 - float(span.iloc[i] / max(span.iloc[i - 1], 1e-12)), geometry))
            if row["high"] > prior["high"] and row["low"] < prior["low"]:
                events.append(_event(timeframe, "OUTSIDE_BAR", row, "BULLISH" if bullish else "BEARISH" if bearish else "NEUTRAL", min(1.0, float(span.iloc[i] / max(span.iloc[i - 1], 1e-12) - 1.0)), geometry))
        if i >= 2:
            three = bars.iloc[i - 2:i + 1]
            bodies = body.iloc[i - 2:i + 1]
            middle = bars.iloc[i - 1]
            first = bars.iloc[i - 2]
            midpoint = (first["open"] + first["close"]) / 2.0
            small_middle = bodies.iloc[1] <= 0.35 * max(span.iloc[i - 1], 1e-12)
            if first["close"] < first["open"] and small_middle and bullish and row["close"] > midpoint:
                events.append(_event(timeframe, "MORNING_STAR", row, "BULLISH", min(1.0, (row["close"] - midpoint) / max(atr.iloc[i], 1e-12) + 0.5), {"middle_body_ratio": float(bodies.iloc[1] / max(span.iloc[i - 1], 1e-12))}))
            if first["close"] > first["open"] and small_middle and bearish and row["close"] < midpoint:
                events.append(_event(timeframe, "EVENING_STAR", row, "BEARISH", min(1.0, (midpoint - row["close"]) / max(atr.iloc[i], 1e-12) + 0.5), {"middle_body_ratio": float(bodies.iloc[1] / max(span.iloc[i - 1], 1e-12))}))
            all_bull = bool((three["close"] > three["open"]).all())
            all_bear = bool((three["close"] < three["open"]).all())
            strong = bool((bodies.to_numpy() >= 0.45 * span.iloc[i - 2:i + 1].to_numpy()).all())
            if all_bull and strong and bool(three["close"].is_monotonic_increasing):
                events.append(_event(timeframe, "THREE_WHITE_SOLDIERS", row, "BULLISH", 0.8, {"consecutive_bars": 3}))
            if all_bear and strong and bool(three["close"].is_monotonic_decreasing):
                events.append(_event(timeframe, "THREE_BLACK_CROWS", row, "BEARISH", 0.8, {"consecutive_bars": 3}))
    return events


def breakout_events(
    frame: pd.DataFrame,
    timeframe: str,
    levels: list[dict[str, Any]],
    *,
    as_of: object | None = None,
    limit: int = 300,
) -> list[dict[str, Any]]:
    """Detect close-confirmed breaks, then later retest/false-break candidates."""
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if len(bars) < 2:
        return []
    atr = average_true_range(bars)
    times = pd.to_datetime(bars["bar_close_utc"], utc=True)
    time_values = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    closes = bars["close"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)
    events: list[dict[str, Any]] = []
    for level in levels:
        price = float(level["price"])
        start = max(1, int(np.searchsorted(time_values, pd.Timestamp(level["available_at_utc"]).value, side="left")))
        if start >= len(bars):
            continue
        kind = str(level["kind"])
        indexes = np.arange(start, len(bars))
        crossed = (closes[indexes - 1] <= price) & (closes[indexes] > price) if kind == "RESISTANCE" else (closes[indexes - 1] >= price) & (closes[indexes] < price)
        hits = indexes[crossed]
        if not len(hits):
            continue
        index = int(hits[0])
        direction = "BULLISH" if kind == "RESISTANCE" else "BEARISH"
        event = _event(timeframe, f"{kind}_BREAKOUT", bars.iloc[index], direction, min(1.0, abs(closes[index] - price) / max(atr_values[index], 1e-12)), {"level_id": level["level_id"], "level": price})
        events.append(event)
        tolerance = max(float(atr_values[index]) * 0.15, price * 0.00005)
        after = np.arange(index + 1, len(bars))
        if not len(after):
            continue
        false_mask = closes[after] < price if direction == "BULLISH" else closes[after] > price
        retest_mask = ((lows[after] <= price + tolerance) & (closes[after] >= price)) if direction == "BULLISH" else ((highs[after] >= price - tolerance) & (closes[after] <= price))
        false_hits, retest_hits = after[false_mask], after[retest_mask]
        false_index = int(false_hits[0]) if len(false_hits) else len(bars) + 1
        retest_index = int(retest_hits[0]) if len(retest_hits) else len(bars) + 1
        if false_index < retest_index:
            events.append(_event(timeframe, "FALSE_BREAK_CANDIDATE", bars.iloc[false_index], "BEARISH" if direction == "BULLISH" else "BULLISH", 0.6, {"breakout_event_id": event["event_id"], "level": price}))
        elif retest_index < len(bars):
            events.append(_event(timeframe, "BREAKOUT_RETEST_CANDIDATE", bars.iloc[retest_index], direction, 0.6, {"breakout_event_id": event["event_id"], "level": price}))
    return events
