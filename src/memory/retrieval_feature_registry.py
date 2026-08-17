"""Explicit allowlist of point-in-time-safe analogue-retrieval features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class RetrievalFeature:
    name: str
    data_type: str
    category: str
    availability_rule: str
    normalization_method: str
    weight: float
    required: bool
    source: str
    point_in_time_safe: bool = True


def _feature(
    name: str,
    category: str,
    weight: float,
    required: bool,
    source: str,
    availability_rule: str,
) -> RetrievalFeature:
    feature = RetrievalFeature(
        name=name,
        data_type="float64",
        category=category,
        availability_rule=availability_rule,
        normalization_method="candidate_history_robust_scale",
        weight=weight,
        required=required,
        source=source,
    )
    validate_feature(feature)
    return feature


def validate_feature(feature: RetrievalFeature) -> None:
    if not feature.name.startswith("context_") or feature.name.startswith("outcome_"):
        raise ValueError(f"Retrieval feature must be an explicit context field: {feature.name}")
    if not feature.point_in_time_safe:
        raise ValueError(f"Unsafe feature cannot enter retrieval registry: {feature.name}")
    if feature.weight <= 0 or not feature.normalization_method:
        raise ValueError(f"Invalid retrieval feature configuration: {feature.name}")
    if "event_timestamp_utc" not in feature.availability_rule:
        raise ValueError(f"Availability rule is not tied to event time: {feature.name}")


PRICE_SOURCE = "Dukascopy JForex ticks; exact window [T-10m,T); no post-T ticks"
CALENDAR_SOURCE = "Deterministic event timestamp/calendar transformation"
MACRO_SOURCE = "Audited data/macro/macro_safe.parquet point-in-time as-of join"

FEATURES: tuple[RetrievalFeature, ...] = (
    _feature("context_return_5m_pips", "pre_event_price", 1.00, True, PRICE_SOURCE, "source ticks < event_timestamp_utc"),
    _feature("context_return_10m_pips", "pre_event_price", 1.00, True, PRICE_SOURCE, "source ticks < event_timestamp_utc"),
    _feature("context_realized_vol_10m", "pre_event_price", 0.75, True, PRICE_SOURCE, "source ticks < event_timestamp_utc"),
    _feature("context_range_10m_pips", "pre_event_price", 0.75, True, PRICE_SOURCE, "source ticks < event_timestamp_utc"),
    _feature("context_pre_event_spread_pips", "pre_event_liquidity", 0.50, True, PRICE_SOURCE, "closing tick of T-1m < event_timestamp_utc"),
    _feature("context_pre_event_max_spread_10m_pips", "pre_event_liquidity", 0.50, True, PRICE_SOURCE, "source ticks < event_timestamp_utc"),
    _feature("context_trend_5m_code", "pre_event_regime", 0.30, True, PRICE_SOURCE, "derived only from source ticks < event_timestamp_utc"),
    _feature("context_trend_10m_code", "pre_event_regime", 0.40, True, PRICE_SOURCE, "derived only from source ticks < event_timestamp_utc"),
    _feature("context_volatility_regime_code", "pre_event_regime", 0.30, False, PRICE_SOURCE, "thresholds fitted only on events < event_timestamp_utc"),
    _feature("context_month_sin", "calendar", 0.15, True, CALENDAR_SOURCE, "deterministic at event_timestamp_utc"),
    _feature("context_month_cos", "calendar", 0.15, True, CALENDAR_SOURCE, "deterministic at event_timestamp_utc"),
    _feature("context_weekday_sin", "calendar", 0.10, True, CALENDAR_SOURCE, "deterministic at event_timestamp_utc"),
    _feature("context_weekday_cos", "calendar", 0.10, True, CALENDAR_SOURCE, "deterministic at event_timestamp_utc"),
    _feature("context_release_local_hour", "calendar", 0.10, True, CALENDAR_SOURCE, "America/New_York conversion at event_timestamp_utc"),
    _feature("context_us_dst", "calendar", 0.10, True, CALENDAR_SOURCE, "America/New_York DST state at event_timestamp_utc"),
    _feature("context_days_since_previous_same_type", "calendar", 0.20, False, CALENDAR_SOURCE, "previous same-type event_timestamp_utc < event_timestamp_utc"),
    *tuple(
        _feature(
            f"context_macro_{name}", "macro", 0.20, False, MACRO_SOURCE,
            "macro available_from_utc <= event_timestamp_utc; latest known observation/vintage",
        )
        for name in (
            "fed_funds_rate", "us_2y_yield", "us_10y_yield", "us_cpi_yoy",
            "us_core_cpi_yoy", "us_unemployment", "us_payroll_growth", "us_gdp_growth",
            "us_pce", "us_core_pce", "us_retail_sales_yoy", "ecb_rate",
            "yield_curve_10y_2y", "rate_differential",
        )
    ),
)


def approved_feature_names() -> tuple[str, ...]:
    return tuple(feature.name for feature in FEATURES)


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(feature) for feature in FEATURES])


def validate_registry() -> None:
    names = approved_feature_names()
    if len(names) != len(set(names)):
        raise ValueError("Retrieval registry contains duplicate feature names")
    for feature in FEATURES:
        validate_feature(feature)


validate_registry()
