"""Causal support/resistance and confirmed-swing market structure."""
from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from src.engines.contract import SwingPoint
from src.engines.swings import average_true_range, causal_bars, utc


def _id(prefix: str, *parts: object) -> str:
    return sha256("|".join([prefix, *map(str, parts)]).encode()).hexdigest()[:24]


def classify_swings(swings: list[SwingPoint], tolerance: float = 0.0) -> list[dict[str, Any]]:
    previous: dict[str, SwingPoint] = {}
    result: list[dict[str, Any]] = []
    for swing in sorted(swings, key=lambda item: item["pivot_time_utc"]):
        prior = previous.get(swing["kind"])
        label = "H" if swing["kind"] == "HIGH" else "L"
        if prior is not None:
            delta = float(swing["price"] - prior["price"])
            if abs(delta) <= tolerance:
                label = "EH" if swing["kind"] == "HIGH" else "EL"
            elif swing["kind"] == "HIGH":
                label = "HH" if delta > 0 else "LH"
            else:
                label = "HL" if delta > 0 else "LL"
        result.append({**swing, "structure_label": label, "prior_same_kind_id": prior["swing_id"] if prior else None})
        previous[swing["kind"]] = swing
    return result


def structure_state(classified: list[dict[str, Any]]) -> str:
    recent = classified[-8:]
    labels = {item["structure_label"] for item in recent}
    if {"HH", "HL"}.issubset(labels) and not ({"LH", "LL"} & labels):
        return "TREND_UP"
    if {"LH", "LL"}.issubset(labels) and not ({"HH", "HL"} & labels):
        return "TREND_DOWN"
    if labels & {"HH", "HL"} and labels & {"LH", "LL"}:
        return "TRANSITION"
    return "RANGE"


def structure_events(frame: pd.DataFrame, timeframe: str, swings: list[SwingPoint], *, as_of: object | None = None, limit: int = 1200) -> list[dict[str, Any]]:
    """Emit BOS/CHOCH only on the first completed close through an available pivot."""
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    classified = classify_swings(swings)
    if len(bars) < 2:
        return []
    times = pd.to_datetime(bars["bar_close_utc"], utc=True).to_numpy(dtype="datetime64[ns]").astype("int64")
    closes = bars["close"].to_numpy(dtype=float)
    candidates: list[dict[str, Any]] = []
    for level in classified:
        start = int(np.searchsorted(times, utc(level["confirmed_at_utc"]).value, side="left"))
        if start >= len(closes):
            continue
        price = float(level["price"])
        indexes = np.arange(max(1, start), len(closes))
        crossed = (closes[indexes - 1] <= price) & (closes[indexes] > price) if level["kind"] == "HIGH" else (closes[indexes - 1] >= price) & (closes[indexes] < price)
        hits = indexes[crossed]
        if len(hits):
            index = int(hits[0])
            candidates.append({"index": index, "level": level, "direction": "BULLISH" if level["kind"] == "HIGH" else "BEARISH"})
    events: list[dict[str, Any]] = []
    trend = "RANGE"
    # Only the latest available same-kind level owns a crossing at a given bar.
    by_bar: dict[tuple[int, str], dict[str, Any]] = {}
    for item in candidates:
        key = (item["index"], item["direction"])
        if key not in by_bar or item["level"]["confirmed_at_utc"] > by_bar[key]["level"]["confirmed_at_utc"]:
            by_bar[key] = item
    for item in sorted(by_bar.values(), key=lambda value: (value["index"], value["direction"])):
        direction, level, index = item["direction"], item["level"], item["index"]
        opposite = trend == "TREND_DOWN" if direction == "BULLISH" else trend == "TREND_UP"
        event_type = "CHOCH" if opposite else "BOS"
        detected = pd.Timestamp(bars.iloc[index]["bar_close_utc"]).isoformat()
        events.append({"event_id": _id(timeframe, event_type, detected, level["swing_id"]), "event_type": event_type, "timeframe": timeframe, "detected_at_utc": detected, "direction": direction, "status": "CONFIRMED", "broken_swing_id": level["swing_id"], "break_level": float(level["price"]), "close": float(closes[index])})
        trend = "TREND_UP" if direction == "BULLISH" else "TREND_DOWN"
    return events


def _cluster_levels(swings: list[SwingPoint], kind: str, tolerance: float, timeframe: str) -> list[dict[str, Any]]:
    clusters: list[list[SwingPoint]] = []
    for swing in [item for item in swings if item["kind"] == kind]:
        target = next((cluster for cluster in clusters if abs(np.mean([x["price"] for x in cluster]) - swing["price"]) <= tolerance), None)
        if target is None:
            clusters.append([swing])
        else:
            target.append(swing)
    result: list[dict[str, Any]] = []
    level_kind = "RESISTANCE" if kind == "HIGH" else "SUPPORT"
    for cluster in clusters:
        price = float(np.mean([item["price"] for item in cluster]))
        available = max(item["confirmed_at_utc"] for item in cluster)
        result.append({
            "level_id": _id(timeframe, level_kind, round(price, 8), available),
            "timeframe": timeframe, "kind": level_kind, "source": "CONFIRMED_SWINGS",
            "price": price, "available_at_utc": available, "touch_count": len(cluster),
            "strength": float(min(1.0, len(cluster) / 4.0)), "swing_ids": [item["swing_id"] for item in cluster],
        })
    return result


def support_resistance(
    frame: pd.DataFrame,
    timeframe: str,
    swings: list[SwingPoint],
    *,
    as_of: object | None = None,
    limit: int = 1200,
) -> dict[str, Any]:
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if bars.empty:
        return {"levels": [], "dynamic": [], "fibonacci": [], "nearest_support": None, "nearest_resistance": None}
    atr = average_true_range(bars)
    current_atr = float(atr.iloc[-1])
    spread_price = float(pd.to_numeric(bars.get("spread_points", pd.Series([0.0])), errors="coerce").fillna(0).iloc[-1]) * 0.00001
    tolerance = max(current_atr * 0.15, spread_price * 2.0, float(bars.iloc[-1]["close"]) * 0.00005)
    levels = _cluster_levels(swings[-80:], "HIGH", tolerance, timeframe) + _cluster_levels(swings[-80:], "LOW", tolerance, timeframe)
    current = float(bars.iloc[-1]["close"])
    for item in levels:
        item["distance_price"] = float(abs(current - item["price"]))
        item["distance_atr"] = float(abs(current - item["price"]) / max(current_atr, 1e-12))
    # Completed previous day/week levels: the current UTC period is excluded.
    close_time = pd.Timestamp(bars.iloc[-1]["bar_close_utc"])
    dates = bars["bar_close_utc"].dt.date
    prior_dates = sorted({value for value in dates if value < close_time.date()})
    calendar_levels: list[dict[str, Any]] = []
    if prior_dates:
        prior_day = bars.loc[dates.eq(prior_dates[-1])]
        for name, value, kind in (("PREVIOUS_DAY_HIGH", prior_day["high"].max(), "RESISTANCE"), ("PREVIOUS_DAY_LOW", prior_day["low"].min(), "SUPPORT")):
            calendar_levels.append({"level_id": _id(timeframe, name, prior_dates[-1]), "timeframe": timeframe, "kind": kind, "source": name, "price": float(value), "available_at_utc": close_time.normalize().isoformat(), "touch_count": 1, "strength": 0.5})
    week_start = close_time.normalize() - pd.Timedelta(days=close_time.dayofweek)
    completed_week = bars.loc[bars["bar_close_utc"].lt(week_start)]
    if not completed_week.empty:
        prior_week_start = completed_week["bar_close_utc"].max().normalize() - pd.Timedelta(days=completed_week["bar_close_utc"].max().dayofweek)
        prior_week = completed_week.loc[completed_week["bar_close_utc"].ge(prior_week_start)]
        for name, value, kind in (("PREVIOUS_WEEK_HIGH", prior_week["high"].max(), "RESISTANCE"), ("PREVIOUS_WEEK_LOW", prior_week["low"].min(), "SUPPORT")):
            calendar_levels.append({"level_id": _id(timeframe, name, prior_week_start), "timeframe": timeframe, "kind": kind, "source": name, "price": float(value), "available_at_utc": week_start.isoformat(), "touch_count": 1, "strength": 0.5})
    levels.extend(calendar_levels)
    dynamic = []
    for period in (20, 50):
        if len(bars) >= period:
            value = float(bars["close"].ewm(span=period, adjust=False).mean().iloc[-1])
            dynamic.append({"name": f"EMA{period}", "price": value, "distance_price": abs(current - value), "available_at_utc": close_time.isoformat()})
    fib: list[dict[str, Any]] = []
    alternating = sorted(swings, key=lambda item: item["pivot_time_utc"])
    if len(alternating) >= 2:
        a, b = alternating[-2], alternating[-1]
        low, high = sorted((float(a["price"]), float(b["price"])))
        direction = "UP_LEG" if b["price"] > a["price"] else "DOWN_LEG"
        for ratio in (0.236, 0.382, 0.5, 0.618, 0.786):
            value = high - (high - low) * ratio if direction == "UP_LEG" else low + (high - low) * ratio
            fib.append({"ratio": ratio, "price": float(value), "leg_direction": direction, "available_at_utc": b["confirmed_at_utc"]})
    supports = [item for item in levels if item["price"] <= current]
    resistances = [item for item in levels if item["price"] >= current]
    return {
        "levels": sorted(levels, key=lambda item: abs(current - item["price"]))[:12],
        "dynamic": dynamic, "fibonacci": fib, "tolerance": tolerance,
        "nearest_support": max(supports, key=lambda item: item["price"], default=None),
        "nearest_resistance": min(resistances, key=lambda item: item["price"], default=None),
    }
