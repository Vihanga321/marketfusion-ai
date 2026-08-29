"""Canonical read-only FX market calendar and session classification."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
MARKET_TRANSITION_SOON_SECONDS = 3600
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


def _local_datetime(day: date, clock: time, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, clock, tzinfo=zone)


def market_calendar(now_utc: object) -> dict[str, object]:
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
        next_open, next_close, is_open = week_open, week_close, False
    elif local < week_close:
        next_open, next_close, is_open = (
            _local_datetime(sunday + timedelta(days=7), time(17), NEW_YORK), week_close, True
        )
    else:
        next_open, next_close, is_open = (
            _local_datetime(sunday + timedelta(days=7), time(17), NEW_YORK),
            _local_datetime(sunday + timedelta(days=12), time(17), NEW_YORK), False,
        )
    if is_open:
        seconds = max(0.0, (next_close - local).total_seconds())
        status, reason = ("CLOSING_SOON" if seconds <= MARKET_TRANSITION_SOON_SECONDS else "OPEN"), "FOREX_TRADING_WEEK_OPEN"
    else:
        seconds = max(0.0, (next_open - local).total_seconds())
        status, reason = ("OPENING_SOON" if seconds <= MARKET_TRANSITION_SOON_SECONDS else "CLOSED_WEEKEND"), "FOREX_WEEKEND_CALENDAR"
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
            end = _local_datetime(candidate, closes, zone).astimezone(UTC)
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
