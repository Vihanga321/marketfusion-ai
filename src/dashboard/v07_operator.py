"""Deterministic operational context for the read-only V0.7 dashboard."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.dashboard.v07_contract import (
    DATA_FRESH_MAX_SECONDS, DATA_LIVE_MAX_SECONDS,
    MARKET_TRANSITION_SOON_SECONDS, OPERATOR_CONTRACT_VERSION,
)
from src.inference.v06a_contract import MAX_FEATURE_AGE_MINUTES
from src.marketdata.v05a_contract import QUOTE_DIR
from src.runtime.v06c_contract import LOCAL_TIMEZONE

UTC = timezone.utc
COLOMBO = ZoneInfo(LOCAL_TIMEZONE)
NEW_YORK = ZoneInfo("America/New_York")
FEATURE_STALE_SECONDS = MAX_FEATURE_AGE_MINUTES * 60

SESSION_SCHEDULES = (
    ("SYDNEY", ZoneInfo("Australia/Sydney"), time(8), time(17)),
    ("TOKYO", ZoneInfo("Asia/Tokyo"), time(9), time(18)),
    ("LONDON", ZoneInfo("Europe/London"), time(8), time(17)),
    ("NEW_YORK", NEW_YORK, time(8), time(17)),
)


def _as_utc(value: object | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return None
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return stamp.to_pydatetime()
    except Exception:
        return None


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _age_seconds(value: datetime | None, now: datetime) -> float | None:
    if value is None:
        return None
    return max(0.0, (now - value).total_seconds())


def _local_datetime(day: date, clock: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, clock, tzinfo=zone)


def market_calendar(now_utc: object) -> dict[str, object]:
    """Return the standard retail-FX weekend state without inventing holidays."""
    now = _as_utc(now_utc)
    if now is None:
        return {
            "status": "UNKNOWN", "market_open": False,
            "reason": "CURRENT_TIME_UNAVAILABLE", "next_market_open_utc": None,
            "next_market_close_utc": None,
        }
    local = now.astimezone(NEW_YORK)
    days_since_sunday = (local.weekday() + 1) % 7
    sunday = local.date() - timedelta(days=days_since_sunday)
    week_open = _local_datetime(sunday, time(17), NEW_YORK)
    week_close = _local_datetime(sunday + timedelta(days=5), time(17), NEW_YORK)
    if local < week_open:
        next_open = week_open
        next_close = week_close
        is_open = False
    elif local < week_close:
        next_open = _local_datetime(sunday + timedelta(days=7), time(17), NEW_YORK)
        next_close = week_close
        is_open = True
    else:
        next_open = _local_datetime(sunday + timedelta(days=7), time(17), NEW_YORK)
        next_close = _local_datetime(sunday + timedelta(days=12), time(17), NEW_YORK)
        is_open = False
    if is_open:
        seconds = max(0.0, (next_close - local).total_seconds())
        status = "CLOSING_SOON" if seconds <= MARKET_TRANSITION_SOON_SECONDS else "OPEN"
        reason = "FOREX_TRADING_WEEK_OPEN"
    else:
        seconds = max(0.0, (next_open - local).total_seconds())
        status = "OPENING_SOON" if seconds <= MARKET_TRANSITION_SOON_SECONDS else "CLOSED_WEEKEND"
        reason = "FOREX_WEEKEND_CALENDAR"
    return {
        "status": status, "market_open": is_open, "reason": reason,
        "next_market_open_utc": _iso(next_open.astimezone(UTC)),
        "next_market_close_utc": _iso(next_close.astimezone(UTC)),
    }


def _session_intervals(now: datetime) -> list[tuple[str, datetime, datetime]]:
    intervals: list[tuple[str, datetime, datetime]] = []
    for name, zone, opens, closes in SESSION_SCHEDULES:
        local_day = now.astimezone(zone).date()
        for offset in range(-1, 8):
            candidate = local_day + timedelta(days=offset)
            if candidate.weekday() >= 5:
                continue
            start = _local_datetime(candidate, opens, zone).astimezone(UTC)
            end_day = candidate if closes > opens else candidate + timedelta(days=1)
            end = _local_datetime(end_day, closes, zone).astimezone(UTC)
            intervals.append((name, start, end))
    return intervals


def forex_session_state(now_utc: object, calendar: dict[str, object] | None = None) -> dict[str, object]:
    now = _as_utc(now_utc)
    if now is None:
        return {"current_session": "UNKNOWN", "active_sessions": [], "overlap": False, "next_session": None, "next_session_change_utc": None, "next_transition": "UNAVAILABLE"}
    calendar = market_calendar(now) if calendar is None else calendar
    if not calendar.get("market_open"):
        return {
            "current_session": "WEEKEND", "active_sessions": [], "overlap": False,
            "next_session": "SYDNEY", "next_session_change_utc": calendar.get("next_market_open_utc"),
            "next_transition": "FOREX WEEK OPEN",
        }
    intervals = _session_intervals(now)
    active = [(name, end) for name, start, end in intervals if start <= now < end]
    active_names = [name for name, _ in active]
    starts = sorted(
        (start, name) for name, start, _ in intervals
        if start > now and market_calendar(start + timedelta(seconds=1)).get("market_open")
    )
    transitions = [(end, f"{name} CLOSE") for name, end in active]
    transitions.extend((start, f"{name} OPEN") for start, name in starts)
    next_change, transition = min(transitions, default=(None, "UNAVAILABLE"), key=lambda item: item[0] or datetime.max.replace(tzinfo=UTC))
    return {
        "current_session": " + ".join(active_names) if active_names else "INTERSESSION",
        "active_sessions": active_names, "overlap": len(active_names) > 1,
        "next_session": starts[0][1] if starts else None,
        "next_session_change_utc": _iso(next_change), "next_transition": transition,
    }


def classify_data_freshness(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "UNAVAILABLE"
    if age_seconds <= DATA_LIVE_MAX_SECONDS:
        return "LIVE"
    if age_seconds <= DATA_FRESH_MAX_SECONDS:
        return "FRESH"
    if age_seconds <= FEATURE_STALE_SECONDS:
        return "DELAYED"
    return "STALE"


def latest_quote_timestamp(v05a_status: dict[str, Any]) -> datetime | None:
    value = v05a_status.get("quote_partition")
    if not value:
        return None
    try:
        path = Path(str(value)).resolve()
        path.relative_to(QUOTE_DIR.resolve())
        frame = pd.read_parquet(path, columns=["broker_tick_time_utc"])
        return _as_utc(frame.iloc[-1]["broker_tick_time_utc"]) if len(frame) else None
    except Exception:
        return None


def _data_status(v05a: dict[str, Any], state: dict[str, Any], now: datetime, quote_time: datetime | None, calendar: dict[str, object]) -> dict[str, object]:
    timeframes = v05a.get("timeframes") if isinstance(v05a.get("timeframes"), dict) else {}
    m5 = _as_utc((timeframes.get("M5") or {}).get("last_bar_close_utc"))
    m15 = _as_utc((timeframes.get("M15") or {}).get("last_bar_close_utc"))
    feature = _as_utc(v05a.get("last_dataset_decision_utc"))
    if feature is None:
        feature = _as_utc((((state.get("market") or {}).get("freshness") or {}).get("observed_at_utc")))
    basis = feature or m5 or quote_time
    basis_age = _age_seconds(basis, now)
    status = classify_data_freshness(basis_age)
    closed_context = not bool(calendar.get("market_open"))
    if status == "UNAVAILABLE":
        reason = "MARKET_TIMESTAMPS_UNAVAILABLE"
    elif status == "STALE" and closed_context:
        reason = "STALE_MARKET_CLOSED"
    elif status == "STALE":
        reason = "FEATURE_AGE_EXCEEDS_LIMIT"
    else:
        reason = "FEATURE_AGE_WITHIN_LIMIT"
    return {
        "status": status, "reason": reason, "basis": "FEATURE_ROW" if feature else "M5_CANDLE" if m5 else "MARKET_TICK",
        "latest_market_tick_utc": _iso(quote_time), "latest_completed_m5_utc": _iso(m5),
        "latest_completed_m15_utc": _iso(m15), "feature_row_utc": _iso(feature),
        "market_age_seconds": basis_age, "tick_age_seconds": _age_seconds(quote_time, now),
        "m5_age_seconds": _age_seconds(m5, now), "m15_age_seconds": _age_seconds(m15, now),
        "feature_age_seconds": _age_seconds(feature, now), "market_closed_context": closed_context,
        "thresholds_seconds": {"live": DATA_LIVE_MAX_SECONDS, "fresh": DATA_FRESH_MAX_SECONDS, "stale": FEATURE_STALE_SECONDS},
    }


def _trading_window(state: dict[str, Any], calendar: dict[str, object]) -> dict[str, object]:
    decision = state.get("decision") or {}
    gate = str(decision.get("gate") or "WAIT_SYSTEM_DATA_INVALID")
    runtime_window = state.get("trade_window") or {}
    if not calendar.get("market_open"):
        status, reason = "MARKET_CLOSED", str(calendar.get("reason") or "MARKET_CLOSED")
    elif "STALE" in gate:
        status, reason = "BLOCKED_STALE", gate
    elif "EVENT" in gate:
        status, reason = "BLOCKED_EVENT", gate
    elif any(item in gate for item in ("SPREAD", "VOLATILITY", "RISK")):
        status, reason = "BLOCKED_RISK", gate
    elif any(item in gate for item in ("NO_MODEL", "NO_APPROVED", "INSUFFICIENT_APPROVED")):
        status, reason = "WAITING_FOR_MODEL", gate
    elif any(item in gate for item in ("DATA_INVALID", "DATA_UNAVAILABLE", "RUNTIME_ERROR")):
        status, reason = "WAITING_FOR_DATA", gate
    elif runtime_window.get("status") == "ADVISORY_WINDOW" and decision.get("action") in {"BUY_BIAS", "SELL_BIAS"}:
        status, reason = "POTENTIAL_WINDOW", "BACKEND_ADVISORY_WINDOW"
    else:
        status, reason = "OBSERVATION", gate
    trading_state = "MARKET_CLOSED" if status == "MARKET_CLOSED" else "READY" if status == "POTENTIAL_WINDOW" else "BLOCKED" if status.startswith("BLOCKED") else "WAIT"
    return {
        "status": status, "reason": reason, "trading_state": trading_state,
        "start_utc": runtime_window.get("start_utc"), "end_utc": runtime_window.get("end_utc"),
        "horizon_minutes": runtime_window.get("horizon_minutes"),
        "execution": "DISABLED", "trading_enabled": False, "manual_confirmation_required": True,
    }


def build_operator_status(
    state: dict[str, Any], v05a_status: dict[str, Any], now_utc: object | None = None,
    quote_timestamp: object | None = None, state_valid: bool = True, state_freshness: str = "FRESH",
) -> dict[str, object]:
    now = _as_utc(now_utc) if now_utc is not None else datetime.now(tz=UTC)
    if now is None:
        now = datetime.now(tz=UTC)
    calendar = market_calendar(now)
    session = forex_session_state(now, calendar)
    quote_time = _as_utc(quote_timestamp) if quote_timestamp is not None else latest_quote_timestamp(v05a_status)
    data = _data_status(v05a_status, state, now, quote_time, calendar)
    window = _trading_window(state, calendar)
    if not state_valid:
        system_status, system_reason = "FAIL", "RUNTIME_STATE_CONTRACT_INVALID"
    elif state_freshness != "FRESH":
        system_status, system_reason = "DEGRADED", "RUNTIME_STATE_UPDATE_DELAYED"
    elif not str(v05a_status.get("status", "")).startswith("PASS"):
        system_status, system_reason = "DEGRADED", "V05A_COLLECTOR_STATUS_UNAVAILABLE"
    else:
        system_status, system_reason = "PASS", "SOFTWARE_RUNTIME_HEALTHY"
    decision = state.get("decision") or {}
    reassessment_at = _as_utc((decision.get("next_reassessment") or {}).get("utc"))
    if reassessment_at is None:
        reassessment = {"status": "UNAVAILABLE", "at_utc": None, "reason": "RUNTIME_DID_NOT_PUBLISH_NEXT_REASSESSMENT"}
    else:
        if not calendar.get("market_open"):
            reason = "Runtime rechecks on schedule; a new completed M5 candle is expected only after market open."
        elif data.get("status") in {"STALE", "DELAYED"}:
            reason = "Waiting for the next fresh completed M5 feature row."
        else:
            reason = "Next reassessment published by the V0.6 runtime."
        reassessment = {"status": "SCHEDULED" if reassessment_at >= now else "OVERDUE", "at_utc": _iso(reassessment_at), "reason": reason}
    primary_reason = ((state.get("reasons") or [{}])[0] or {}).get("message") or "No backend decision reason is available."
    if window["status"] == "MARKET_CLOSED" and data["status"] == "STALE":
        wait_explanation = "The Forex weekend calendar confirms the market is closed; the latest completed feature row remains stale until trading resumes."
    else:
        wait_explanation = str(primary_reason)
    return {
        "contract_version": OPERATOR_CONTRACT_VERSION,
        "current_time": {"utc": _iso(now), "asia_colombo": now.astimezone(COLOMBO).isoformat()},
        "market": calendar, "session": session, "data_freshness": data,
        "reassessment": reassessment, "trading_window": window,
        "system_health": {"status": system_status, "reason": system_reason},
        "trading_state": window["trading_state"], "wait_explanation": wait_explanation,
        "runtime_gate": decision.get("gate"), "runtime_status": (state.get("system") or {}).get("status"),
        "mode": (state.get("system") or {}).get("mode", "SHADOW_ADVISORY_ONLY"),
    }
