"""Explicit PRE_RELEASE and POST_RELEASE_IMMEDIATE feature policies for V0.4C."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.events.decision_mode_registry import DecisionMode, assert_mode_allows
from src.events.event_value_source_registry import SOURCES
from src.memory.retrieval_feature_registry import FEATURES as V04B_FEATURES


@dataclass(frozen=True)
class PolicyFeature:
    name: str
    decision_mode: str
    category: str
    availability_rule: str
    normalization_method: str
    weight: float
    required: bool
    enabled: bool
    source: str


def _event_feature(measure_id: str, field: str, mode: DecisionMode, enabled: bool) -> PolicyFeature:
    prefix = "pre_release_context" if mode == DecisionMode.PRE_RELEASE else "release_payload"
    availability = (
        "available_from_utc < event_timestamp_utc"
        if mode == DecisionMode.PRE_RELEASE
        else "available_from_utc == event_timestamp_utc"
    )
    return PolicyFeature(
        name=f"{prefix}_event_{measure_id}_{field}",
        decision_mode=mode.value,
        category="event_value",
        availability_rule=availability,
        normalization_method="candidate_history_robust_scale",
        weight=0.25 if "revision" in field else 0.50,
        required=False,
        enabled=enabled,
        source="V0.4C audited normalized event-value table",
    )


def policy_features(mode: DecisionMode) -> tuple[PolicyFeature, ...]:
    base = tuple(
        PolicyFeature(
            name=feature.name,
            decision_mode=mode.value,
            category=feature.category,
            availability_rule=feature.availability_rule,
            normalization_method=feature.normalization_method,
            weight=feature.weight,
            required=feature.required,
            enabled=True,
            source=feature.source,
        )
        for feature in V04B_FEATURES
    )
    pre = tuple(
        item
        for source in SOURCES
        for item in (
            _event_feature(source.measure_id, "previous_value", DecisionMode.PRE_RELEASE, True),
            _event_feature(source.measure_id, "consensus_value", DecisionMode.PRE_RELEASE, False),
        )
    )
    if mode == DecisionMode.PRE_RELEASE:
        return base + pre
    post = tuple(
        item
        for source in SOURCES
        for item in (
            _event_feature(source.measure_id, "actual_value", DecisionMode.POST_RELEASE_IMMEDIATE, True),
            _event_feature(source.measure_id, "revision_raw", DecisionMode.POST_RELEASE_IMMEDIATE, True),
            _event_feature(source.measure_id, "prior_period_newly_released_at_release", DecisionMode.POST_RELEASE_IMMEDIATE, True),
            _event_feature(source.measure_id, "surprise_signed_raw", DecisionMode.POST_RELEASE_IMMEDIATE, False),
            _event_feature(source.measure_id, "surprise_z_robust", DecisionMode.POST_RELEASE_IMMEDIATE, False),
        )
    )
    return base + pre + post


def policy_registry_frame(mode: DecisionMode) -> pd.DataFrame:
    return pd.DataFrame([asdict(feature) for feature in policy_features(mode)])


def approved_policy_columns(mode: DecisionMode) -> tuple[str, ...]:
    return tuple(feature.name for feature in policy_features(mode) if feature.enabled)


def validate_policy_registry(mode: DecisionMode) -> None:
    features = policy_features(mode)
    names = [feature.name for feature in features]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate {mode.value} policy feature")
    assert_mode_allows(names, mode)
    if any(name.startswith("outcome_") for name in names):
        raise ValueError("Historical outcome entered decision feature policy")
    if mode == DecisionMode.PRE_RELEASE and any("actual_value" in name or "surprise" in name for name in names):
        raise ValueError("PRE_RELEASE registry contains release payload")


for _mode in DecisionMode:
    validate_policy_registry(_mode)
