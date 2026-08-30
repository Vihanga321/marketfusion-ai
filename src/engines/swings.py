"""Shared causal swing detection over completed bars.

A pivot at index ``i`` is unavailable until the close of ``i + right_bars``.
Every downstream engine consumes ``confirmed_at_utc`` rather than treating the
pivot timestamp as its information-availability timestamp.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd

from src.engines.contract import SwingPoint

BAR_PRICE_COLUMNS = ("open", "high", "low", "close")


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def causal_bars(frame: pd.DataFrame, as_of: object | None = None, limit: int = 1200, observed_as_of: object | None = None) -> pd.DataFrame:
    """Return sorted, valid, completed bars available at ``as_of``."""
    required = {"bar_open_utc", "bar_close_utc", *BAR_PRICE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Bar frame missing causal fields: " + ", ".join(missing))
    bars = frame.copy()
    bars["bar_open_utc"] = pd.to_datetime(bars["bar_open_utc"], utc=True, errors="raise")
    bars["bar_close_utc"] = pd.to_datetime(bars["bar_close_utc"], utc=True, errors="raise")
    for column in BAR_PRICE_COLUMNS:
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    bars = bars.sort_values("bar_close_utc").drop_duplicates("bar_open_utc", keep="first")
    if as_of is not None:
        bars = bars.loc[bars["bar_close_utc"].le(utc(as_of))]
    if observed_as_of is not None and "first_observed_utc" in bars:
        observed = pd.to_datetime(bars["first_observed_utc"], utc=True, errors="coerce")
        bars = bars.loc[observed.isna() | observed.le(utc(observed_as_of))]
    if (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any() or (bars["low"] > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC enclosure")
    return bars.tail(max(1, int(limit))).reset_index(drop=True)


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = frame["close"].shift(1)
    true_range = pd.concat([
        frame["high"].sub(frame["low"]),
        frame["high"].sub(previous).abs(),
        frame["low"].sub(previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def _identifier(timeframe: str, kind: str, pivot: pd.Timestamp, price: float, left: int, right: int) -> str:
    payload = f"{timeframe}|{kind}|{pivot.isoformat()}|{price:.10f}|{left}|{right}"
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def confirmed_swings(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    as_of: object | None = None,
    left_bars: int = 2,
    right_bars: int = 2,
    limit: int = 1200,
) -> list[SwingPoint]:
    """Detect strict local extrema and timestamp when each became knowable."""
    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing confirmation requires positive left/right windows")
    bars = causal_bars(frame, as_of=as_of, limit=limit)
    if len(bars) < left_bars + right_bars + 1:
        return []
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = pd.to_datetime(bars["bar_close_utc"], utc=True)
    result: list[SwingPoint] = []
    for index in range(left_bars, len(bars) - right_bars):
        high_window = highs[index - left_bars:index + right_bars + 1]
        low_window = lows[index - left_bars:index + right_bars + 1]
        # Strict extrema remove ambiguous equal-price pivot selection. Equal
        # highs/lows are instead modelled by the liquidity engine.
        high_other = np.delete(high_window, left_bars)
        low_other = np.delete(low_window, left_bars)
        pivot_time = closes.iloc[index]
        confirmed_at = closes.iloc[index + right_bars]
        if bool(np.all(highs[index] > high_other)):
            result.append({
                "swing_id": _identifier(timeframe, "HIGH", pivot_time, highs[index], left_bars, right_bars),
                "timeframe": timeframe, "kind": "HIGH",
                "pivot_time_utc": pivot_time.isoformat(), "confirmed_at_utc": confirmed_at.isoformat(),
                "price": float(highs[index]), "left_bars": left_bars, "right_bars": right_bars,
            })
        if bool(np.all(lows[index] < low_other)):
            result.append({
                "swing_id": _identifier(timeframe, "LOW", pivot_time, lows[index], left_bars, right_bars),
                "timeframe": timeframe, "kind": "LOW",
                "pivot_time_utc": pivot_time.isoformat(), "confirmed_at_utc": confirmed_at.isoformat(),
                "price": float(lows[index]), "left_bars": left_bars, "right_bars": right_bars,
            })
    return sorted(result, key=lambda item: (item["pivot_time_utc"], item["kind"]))


def alternating_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Collapse adjacent same-kind extrema to the more extreme confirmed pivot."""
    result: list[SwingPoint] = []
    for swing in sorted(swings, key=lambda item: item["pivot_time_utc"]):
        if not result or result[-1]["kind"] != swing["kind"]:
            result.append(swing)
            continue
        prior = result[-1]
        replace = swing["price"] > prior["price"] if swing["kind"] == "HIGH" else swing["price"] < prior["price"]
        if replace:
            result[-1] = swing
    return result
