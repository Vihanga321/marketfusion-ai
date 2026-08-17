"""Audited source registry for normalized V0.4C event-value measures."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class EventValueSource:
    event_type: str
    measure_id: str
    measure_name: str
    authority: str
    source_name: str
    source_series_id: str
    source_url_or_reference: str
    point_in_time_capability: str
    vintage_behavior: str
    release_timestamp_semantics: str
    revision_semantics: str
    consensus_capability: str
    unit: str
    frequency: str
    transformation_rules: str
    economic_direction_semantics: str
    known_limitations: str


SOURCES: tuple[EventValueSource, ...] = (
    EventValueSource(
        event_type="us_cpi_release",
        measure_id="headline_cpi_yoy_sa",
        measure_name="Headline CPI year-over-year, seasonally adjusted",
        authority="U.S. Bureau of Labor Statistics",
        source_name="FRED/ALFRED point-in-time BLS series",
        source_series_id="CPIAUCSL",
        source_url_or_reference="https://fred.stlouisfed.org/series/CPIAUCSL",
        point_in_time_capability="ALFRED real-time periods preserve historical vintages",
        vintage_behavior="revisions retained by observation period and vintage date",
        release_timestamp_semantics="ALFRED first-vintage date paired with validated BLS 08:30 America/New_York event timestamp",
        revision_semantics="same-release vintage for the prior observation is retained separately when present",
        consensus_capability="NONE",
        unit="percent_change_year_ago",
        frequency="M",
        transformation_rules="100 * (point-in-time CPIAUCSL[t] / CPIAUCSL[t-12] - 1)",
        economic_direction_semantics="higher means hotter inflation; no hard-coded EUR/USD direction",
        known_limitations="derived SA ALFRED measure; not claimed to be the exact NSA headline printed in every historical release table",
    ),
    EventValueSource(
        event_type="us_cpi_release",
        measure_id="core_cpi_yoy_sa",
        measure_name="Core CPI year-over-year, seasonally adjusted",
        authority="U.S. Bureau of Labor Statistics",
        source_name="FRED/ALFRED point-in-time BLS series",
        source_series_id="CPILFESL",
        source_url_or_reference="https://fred.stlouisfed.org/series/CPILFESL",
        point_in_time_capability="ALFRED real-time periods preserve historical vintages",
        vintage_behavior="revisions retained by observation period and vintage date",
        release_timestamp_semantics="ALFRED first-vintage date paired with validated BLS 08:30 America/New_York event timestamp",
        revision_semantics="same-release vintage for the prior observation is retained separately when present",
        consensus_capability="NONE",
        unit="percent_change_year_ago",
        frequency="M",
        transformation_rules="100 * (point-in-time CPILFESL[t] / CPILFESL[t-12] - 1)",
        economic_direction_semantics="higher means hotter core inflation; no hard-coded EUR/USD direction",
        known_limitations="derived SA ALFRED measure; not claimed to be an archived consensus-provider field",
    ),
    EventValueSource(
        event_type="us_employment_situation",
        measure_id="nonfarm_payroll_change",
        measure_name="Total nonfarm payroll employment monthly change",
        authority="U.S. Bureau of Labor Statistics",
        source_name="FRED/ALFRED point-in-time BLS series",
        source_series_id="PAYEMS",
        source_url_or_reference="https://fred.stlouisfed.org/series/PAYEMS",
        point_in_time_capability="ALFRED real-time periods preserve historical vintages",
        vintage_behavior="initial and revised monthly payroll levels retained",
        release_timestamp_semantics="ALFRED first-vintage date paired with validated BLS 08:30 America/New_York event timestamp",
        revision_semantics="prior-month transformed value before T and any same-release revised value are separate",
        consensus_capability="NONE",
        unit="thousands_of_jobs_change",
        frequency="M",
        transformation_rules="point-in-time PAYEMS[t] - PAYEMS[t-1], in thousands",
        economic_direction_semantics="higher arithmetic surprise means stronger payroll growth; separate from FX response",
        known_limitations="individual transformed prior-period changes are preserved; raw payroll-level revision arithmetic requires a separate raw-level source",
    ),
    EventValueSource(
        event_type="us_employment_situation",
        measure_id="unemployment_rate",
        measure_name="U-3 unemployment rate",
        authority="U.S. Bureau of Labor Statistics",
        source_name="FRED/ALFRED point-in-time BLS series",
        source_series_id="UNRATE",
        source_url_or_reference="https://fred.stlouisfed.org/series/UNRATE",
        point_in_time_capability="ALFRED real-time periods preserve historical vintages",
        vintage_behavior="initial and revised monthly rates retained when supplied",
        release_timestamp_semantics="ALFRED first-vintage date paired with validated BLS 08:30 America/New_York event timestamp",
        revision_semantics="same-release prior-period vintage retained separately when present",
        consensus_capability="NONE",
        unit="percent",
        frequency="M",
        transformation_rules="point-in-time UNRATE monthly level",
        economic_direction_semantics="lower arithmetic surprise means stronger labor conditions; raw arithmetic is never inverted",
        known_limitations="consensus is unavailable; household-survey measure is distinct from payroll employment",
    ),
)


def source_registry_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(source) for source in SOURCES])


def source_for(event_type: str, measure_id: str) -> EventValueSource:
    matches = [s for s in SOURCES if s.event_type == event_type and s.measure_id == measure_id]
    if len(matches) != 1:
        raise KeyError(f"Expected one source for {event_type}/{measure_id}; found {len(matches)}")
    return matches[0]


def validate_source_registry() -> None:
    keys = [(source.event_type, source.measure_id) for source in SOURCES]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate event-value source registry identity")
    for source in SOURCES:
        if source.consensus_capability != "NONE":
            raise ValueError("No historical consensus provider has been audited for V0.4C")
        if source.authority != "U.S. Bureau of Labor Statistics":
            raise ValueError(f"Non-authoritative event-value source: {source.measure_id}")


validate_source_registry()
