"""Causal regime, session, event, spread, and freshness classification."""
from __future__ import annotations

from typing import Any
import math

import numpy as np
import pandas as pd

from src.fusion.v06b_contract import EVENT_POLICY, MIN_REGIME_HISTORY_ROWS, TARGET_EVENT_CODES, TARGET_EVENT_NAMES


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def freshness(observed_at: object | None, now_utc: object, max_age_minutes: float, degraded: bool = False) -> dict[str, object]:
    if observed_at is None or pd.isna(observed_at):
        return {"status": "UNAVAILABLE", "observed_at_utc": None, "age_minutes": None}
    try:
        age = float((utc(now_utc) - utc(observed_at)).total_seconds() / 60.0)
    except Exception:
        return {"status": "UNAVAILABLE", "observed_at_utc": None, "age_minutes": None}
    if age < -0.05 or age > max_age_minutes:
        status = "STALE"
    elif degraded:
        status = "DEGRADED"
    else:
        status = "FRESH"
    return {"status": status, "observed_at_utc": utc(observed_at).isoformat(), "age_minutes": age}


def classify_session(row: pd.Series | dict[str, Any]) -> str:
    if bool(row.get("is_london_ny_overlap", 0)):
        return "OVERLAP"
    if bool(row.get("is_asia_session", 0)):
        return "ASIA"
    if bool(row.get("is_london_session", 0)):
        return "LONDON"
    if bool(row.get("is_new_york_session", 0)):
        return "NEW_YORK"
    return "OFF"


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def classify_market_regime(history: pd.DataFrame, row: pd.Series) -> dict[str, object]:
    decision = utc(row["decision_timestamp_utc"])
    prior = history.copy()
    prior["decision_timestamp_utc"] = pd.to_datetime(prior["decision_timestamp_utc"], utc=True, errors="coerce")
    prior = prior.loc[prior["decision_timestamp_utc"] < decision]
    values = pd.to_numeric(prior.get("m5_volatility_60m"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    current = _finite(row.get("m5_volatility_60m"))
    if current is None or current < 0 or len(values) < MIN_REGIME_HISTORY_ROWS:
        return {"volatility_regime": "UNAVAILABLE", "trend_regime": "RANGE", "history_rows": int(len(values)), "threshold_method": "expanding_prior_quantiles", "thresholds": None}
    q25, q75, q95 = (float(values.quantile(q)) for q in (0.25, 0.75, 0.95))
    regime = "QUIET" if current <= q25 else "NORMAL" if current <= q75 else "HIGH" if current <= q95 else "EXTREME"
    denominator = max(current, float(values.median()), 1e-9)
    returns = [_finite(row.get(name)) for name in ("m5_return_15m", "m5_return_60m", "m5_return_240m")]
    if any(value is None for value in returns):
        trend = "RANGE"
        score = None
    else:
        score = float((0.5 * returns[0] + 0.3 * returns[1] + 0.2 * returns[2]) / denominator)
        trend = "STRONG_UP" if score >= 1.0 else "UP" if score >= 0.25 else "STRONG_DOWN" if score <= -1.0 else "DOWN" if score <= -0.25 else "RANGE"
    return {
        "volatility_regime": regime, "trend_regime": trend, "trend_score": score,
        "history_rows": int(len(values)), "current_volatility": current,
        "threshold_method": "expanding_prior_quantiles",
        "thresholds": {"quiet_q25": q25, "high_q75": q75, "extreme_q95": q95},
    }


def classify_spread(row: pd.Series) -> dict[str, object]:
    current = _finite(row.get("m5_spread_points"))
    median = _finite(row.get("m5_spread_median_60m"))
    if current is None or median is None or current < 0 or median < 0:
        return {"status": "UNAVAILABLE", "current_points": current, "recent_median_points": median, "ratio": None}
    ratio = 1.0 if current == 0 and median == 0 else (float("inf") if median == 0 else current / median)
    status = "NORMAL" if ratio <= 1.5 else "WIDE" if ratio <= 2.5 else "EXTREME"
    return {"status": status, "current_points": current, "recent_median_points": median, "ratio": ratio}


def _is_exact_target(row: pd.Series) -> bool:
    code = str(row.get("event_code", "")).strip().lower()
    name = str(row.get("event_name", "")).strip().lower()
    return code in TARGET_EVENT_CODES or name in TARGET_EVENT_NAMES


def classify_event_risk(events: pd.DataFrame | None, now_utc: object) -> dict[str, object]:
    if events is None:
        return {"status": "NONE", "data_status": "UNAVAILABLE", "nearest_event": None, "minutes_to_event": None, "blocks_direction": True}
    required = {"event_timestamp_utc", "event_name", "event_code"}
    if events.empty:
        return {"status": "NONE", "data_status": "DEGRADED", "nearest_event": None, "minutes_to_event": None, "blocks_direction": True}
    if not required.issubset(events.columns):
        return {"status": "NONE", "data_status": "UNAVAILABLE", "nearest_event": None, "minutes_to_event": None, "blocks_direction": True}
    work = events.copy()
    work = work.loc[work.apply(_is_exact_target, axis=1)].copy()
    if work.empty:
        return {"status": "NONE", "data_status": "DEGRADED", "nearest_event": None, "minutes_to_event": None, "blocks_direction": True}
    work["event_timestamp_utc"] = pd.to_datetime(work["event_timestamp_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["event_timestamp_utc"])
    if work.empty:
        return {"status": "NONE", "data_status": "UNAVAILABLE", "nearest_event": None, "minutes_to_event": None, "blocks_direction": True}
    now = utc(now_utc)
    work["minutes"] = (work["event_timestamp_utc"] - now).dt.total_seconds() / 60.0
    relevant = work.loc[work["minutes"] >= -60].sort_values("event_timestamp_utc")
    nearest = relevant.iloc[0] if not relevant.empty else work.iloc[(work["minutes"].abs()).argmin()]
    minutes = float(nearest["minutes"])
    status = "NONE"
    for label, low, high in EVENT_POLICY:
        if low <= minutes <= high:
            status = label
            break
    event = {
        "event_id": str(nearest.get("event_id", "")) or None,
        "event_code": str(nearest.get("event_code")),
        "event_name": str(nearest.get("event_name")),
        "event_timestamp_utc": utc(nearest["event_timestamp_utc"]).isoformat(),
    }
    return {"status": status, "data_status": "FRESH", "nearest_event": event, "minutes_to_event": minutes, "blocks_direction": status in {"ELEVATED", "BLOCK", "POST_RELEASE_VOLATILITY"}}
