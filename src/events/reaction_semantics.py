"""Canonical UTC M1 indexing and event-model eligibility rules."""

from __future__ import annotations

import pandas as pd


REACTION_HORIZONS_MINUTES = (1, 5, 15, 60, 240)


def exact_utc_minute(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="raise")
    if timestamp.second or timestamp.microsecond or timestamp.nanosecond:
        raise ValueError("Event timestamp must be an exact UTC minute boundary")
    return timestamp


def pre_event_bar_open(event_timestamp: object) -> pd.Timestamp:
    """M1 bar whose close is the price immediately before a release at T."""
    return exact_utc_minute(event_timestamp) - pd.Timedelta(minutes=1)


def reaction_bar_open(event_timestamp: object, horizon_minutes: int) -> pd.Timestamp:
    """M1 bar whose close is the H-minute reaction price.

    At a release on minute T, +1m is the close of bar T, +5m is the close
    of bar T+4m, and analogously for longer horizons.
    """
    if horizon_minutes < 1:
        raise ValueError("Reaction horizon must be at least one minute")
    return exact_utc_minute(event_timestamp) + pd.Timedelta(minutes=horizon_minutes - 1)


def requested_times_only(
    times: pd.Series, requested_start: pd.Timestamp, requested_end: pd.Timestamp
) -> tuple[pd.Series, int]:
    """Return exact in-window timestamps and the count quarantined outside it."""
    in_range = times.between(requested_start, requested_end, inclusive="both")
    return times[in_range].reset_index(drop=True), int((~in_range).sum())


def assert_market_reaction_training_gate(frame: pd.DataFrame) -> None:
    required = {"status", "model_eligible_market_reaction"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Market-reaction table is missing gate columns: " + ", ".join(missing))
    rejected = frame["status"].ne("PASS")
    if rejected.any():
        counts = frame.loc[rejected, "status"].value_counts().to_dict()
        raise ValueError(
            "Market-reaction training input contains non-PASS rows: " + str(counts)
        )
    eligible = frame["model_eligible_market_reaction"].map(
        lambda value: value is True or str(value).strip().lower() in {"true", "1"}
    )
    if not eligible.all():
        raise ValueError(
            f"{int((~eligible).sum())} PASS rows lack model_eligible_market_reaction=true"
        )
