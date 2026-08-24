"""Strict candidate mappings from MarketFusion measures to consensus-provider calendar rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ConsensusMeasureMapping:
    measure_id: str
    event_type: str
    provider: str
    country: str
    acceptable_categories: tuple[str, ...]
    acceptable_events: tuple[str, ...]
    acceptable_provider_units: tuple[str, ...]
    canonical_unit: str
    forecast_value_scale: float
    provider_ticker: str | None
    provider_symbol: str | None
    mapping_status: str
    notes: str


MAPPINGS: tuple[ConsensusMeasureMapping, ...] = (
    ConsensusMeasureMapping(
        measure_id="headline_cpi_yoy_sa",
        event_type="us_cpi_release",
        provider="Trading Economics",
        country="United States",
        acceptable_categories=("Inflation Rate YoY", "CPI YoY"),
        acceptable_events=("Inflation Rate YoY", "CPI YoY"),
        acceptable_provider_units=("%", "percent"),
        canonical_unit="percent_change_year_ago",
        forecast_value_scale=1.0,
        provider_ticker=None,
        provider_symbol=None,
        mapping_status="CANDIDATE_UNVERIFIED",
        notes="Provider ticker/symbol must be locked only after a successful live mapping audit.",
    ),
    ConsensusMeasureMapping(
        measure_id="core_cpi_yoy_sa",
        event_type="us_cpi_release",
        provider="Trading Economics",
        country="United States",
        acceptable_categories=("Core Inflation Rate YoY", "Core CPI YoY"),
        acceptable_events=("Core Inflation Rate YoY", "Core CPI YoY"),
        acceptable_provider_units=("%", "percent"),
        canonical_unit="percent_change_year_ago",
        forecast_value_scale=1.0,
        provider_ticker=None,
        provider_symbol=None,
        mapping_status="CANDIDATE_UNVERIFIED",
        notes="Exact category/event identity must be confirmed from provider payloads.",
    ),
    ConsensusMeasureMapping(
        measure_id="nonfarm_payroll_change",
        event_type="us_employment_situation",
        provider="Trading Economics",
        country="United States",
        acceptable_categories=("Non Farm Payrolls", "Nonfarm Payrolls"),
        acceptable_events=("Non Farm Payrolls", "Nonfarm Payrolls"),
        acceptable_provider_units=("K", "Thousand", "thousand"),
        canonical_unit="thousands_of_jobs_change",
        forecast_value_scale=0.001,
        provider_ticker=None,
        provider_symbol=None,
        mapping_status="CANDIDATE_UNVERIFIED",
        notes=(
            "ForecastValue is commonly expressed as absolute jobs while the canonical MarketFusion "
            "measure is thousands; raw Forecast strings with K are preserved and parsed directly."
        ),
    ),
    ConsensusMeasureMapping(
        measure_id="unemployment_rate",
        event_type="us_employment_situation",
        provider="Trading Economics",
        country="United States",
        acceptable_categories=("Unemployment Rate",),
        acceptable_events=("Unemployment Rate",),
        acceptable_provider_units=("%", "percent"),
        canonical_unit="percent",
        forecast_value_scale=1.0,
        provider_ticker=None,
        provider_symbol=None,
        mapping_status="CANDIDATE_UNVERIFIED",
        notes="Raw arithmetic surprise is actual minus consensus; labor interpretation is separate.",
    ),
)


def mapping_for(measure_id: str, provider: str = "Trading Economics") -> ConsensusMeasureMapping:
    matches = [m for m in MAPPINGS if m.measure_id == measure_id and m.provider == provider]
    if len(matches) != 1:
        raise KeyError(f"Expected one mapping for {provider}/{measure_id}; found {len(matches)}")
    return matches[0]


def mapping_registry_frame() -> pd.DataFrame:
    rows = []
    for mapping in MAPPINGS:
        row = asdict(mapping)
        for name in ("acceptable_categories", "acceptable_events", "acceptable_provider_units"):
            row[name] = " | ".join(row[name])
        rows.append(row)
    return pd.DataFrame(rows)


def validate_mapping_registry() -> None:
    keys = [(m.provider, m.measure_id) for m in MAPPINGS]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate consensus measure mapping")
    expected = {
        "headline_cpi_yoy_sa",
        "core_cpi_yoy_sa",
        "nonfarm_payroll_change",
        "unemployment_rate",
    }
    if {m.measure_id for m in MAPPINGS} != expected:
        raise ValueError("V0.4D mapping registry must cover exactly the four audited V0.4C measures")
    if any(m.mapping_status == "APPROVED" for m in MAPPINGS):
        raise ValueError("No provider mapping may be pre-approved before the live audit")


validate_mapping_registry()
