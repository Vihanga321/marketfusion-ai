"""Decision-time gates separating pre-release context from release payload."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import pandas as pd


class DecisionMode(str, Enum):
    PRE_RELEASE = "PRE_RELEASE"
    POST_RELEASE_IMMEDIATE = "POST_RELEASE_IMMEDIATE"


IDENTITY_PREFIXES = ("event_id", "event_type", "event_timestamp_utc")
PRE_RELEASE_PREFIXES = ("context_", "pre_release_context_")
POST_RELEASE_PREFIXES = PRE_RELEASE_PREFIXES + ("release_payload_",)


def assert_mode_allows(columns: Iterable[str], mode: DecisionMode) -> None:
    allowed_prefixes = PRE_RELEASE_PREFIXES if mode == DecisionMode.PRE_RELEASE else POST_RELEASE_PREFIXES
    for column in columns:
        if column.startswith("outcome_"):
            raise ValueError(f"Future market outcome is forbidden in every decision mode: {column}")
        if column in IDENTITY_PREFIXES:
            continue
        if not column.startswith(allowed_prefixes):
            raise ValueError(f"{mode.value} does not allow field: {column}")
        if mode == DecisionMode.PRE_RELEASE and column.startswith("release_payload_"):
            raise ValueError(f"PRE_RELEASE cannot use release-time information: {column}")


def project_mode(frame, mode: DecisionMode):
    prefixes = PRE_RELEASE_PREFIXES if mode == DecisionMode.PRE_RELEASE else POST_RELEASE_PREFIXES
    columns = [
        column for column in frame.columns
        if column in IDENTITY_PREFIXES or column.startswith(prefixes)
    ]
    assert_mode_allows(columns, mode)
    if any(column.startswith("outcome_") for column in columns):
        raise AssertionError("Outcome entered decision-mode projection")
    return frame.loc[:, columns].copy()


def validate_decision_time(
    mode: DecisionMode,
    decision_timestamp_utc: object,
    event_timestamp_utc: object,
) -> None:
    decision = pd.Timestamp(decision_timestamp_utc)
    event = pd.Timestamp(event_timestamp_utc)
    if decision.tzinfo is None or event.tzinfo is None:
        raise ValueError("Decision and event timestamps must be timezone-aware")
    decision = decision.tz_convert("UTC")
    event = event.tz_convert("UTC")
    if mode == DecisionMode.PRE_RELEASE and not decision < event:
        raise ValueError("PRE_RELEASE decision timestamp must be strictly before event T")
    if mode == DecisionMode.POST_RELEASE_IMMEDIATE and decision < event:
        raise ValueError("POST_RELEASE_IMMEDIATE decision timestamp cannot precede event T")
