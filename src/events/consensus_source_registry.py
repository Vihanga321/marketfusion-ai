"""Provider registry for V0.4D historical consensus intelligence.

A provider is not model-eligible merely because it exposes a historical
``Forecast`` field. MarketFusion requires evidence that the consensus was
available strictly before the corresponding release timestamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd


class ApprovalStatus(str, Enum):
    APPROVED_POINT_IN_TIME = "APPROVED_POINT_IN_TIME"
    CONDITIONALLY_APPROVED = "CONDITIONALLY_APPROVED"
    INSUFFICIENT_METADATA = "INSUFFICIENT_METADATA"
    REJECTED_POST_RELEASE_RISK = "REJECTED_POST_RELEASE_RISK"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class ConsensusSource:
    provider_name: str
    dataset_name: str
    access_method: str
    consensus_field: str
    provider_forecast_field: str
    timestamp_field: str
    timestamp_semantics: str
    point_in_time_method: str
    historical_snapshot_supported: bool
    revision_history_supported: bool
    historical_start: str
    consensus_definition: str
    poll_statistic: str
    source_documentation: str
    point_in_time_safe: bool
    known_limitations: str
    approval_status: ApprovalStatus


SOURCES: tuple[ConsensusSource, ...] = (
    ConsensusSource(
        provider_name="Trading Economics",
        dataset_name="Economic Calendar",
        access_method="HTTPS API",
        consensus_field="Forecast / ForecastValue",
        provider_forecast_field="TEForecast / TEForecastValue",
        timestamp_field="No dedicated consensus-availability timestamp documented",
        timestamp_semantics=(
            "Date is the release time; LastUpdate is the most recent event-record update and must not "
            "be treated as consensus availability"
        ),
        point_in_time_method="Economic Calendar Point-in-Time endpoint",
        historical_snapshot_supported=True,
        revision_history_supported=True,
        historical_start="provider-dependent",
        consensus_definition="Consensus forecast from a representative group of economists",
        poll_statistic="provider consensus; exact statistic must be verified from entitlement metadata",
        source_documentation="https://docs.tradingeconomics.com/economic_calendar/point-in-time/",
        point_in_time_safe=False,
        known_limitations=(
            "Public schema distinguishes Forecast from TEForecast but does not document a dedicated "
            "timestamp proving when each historical Forecast first became available. Live audit must "
            "therefore remain fail-closed until pre-release availability is evidenced."
        ),
        approval_status=ApprovalStatus.CONDITIONALLY_APPROVED,
    ),
    ConsensusSource(
        provider_name="LSEG Reuters Polls",
        dataset_name="Reuters Polls Consensus",
        access_method="LSEG entitlement / API / bulk service",
        consensus_field="Reuters poll consensus fields",
        provider_forecast_field="N/A",
        timestamp_field="provider polling point timestamp",
        timestamp_semantics="LSEG documents estimates traceable to specific polling points in time",
        point_in_time_method="historical poll snapshots / forecast revisions",
        historical_snapshot_supported=True,
        revision_history_supported=True,
        historical_start="1999",
        consensus_definition="Consensus estimates built from Reuters polling contributors",
        poll_statistic="provider-defined consensus statistic",
        source_documentation=(
            "https://www.lseg.com/en/data-catalogue/economics/economic-macro-forecasts/"
            "reuters-polls-consensus"
        ),
        point_in_time_safe=False,
        known_limitations=(
            "Documentation supports point-in-time traceability, but MarketFusion has not verified a "
            "user entitlement or concrete field-level payload. No synthetic LSEG records are allowed."
        ),
        approval_status=ApprovalStatus.CONDITIONALLY_APPROVED,
    ),
)


def source_registry_frame() -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        row = asdict(source)
        row["approval_status"] = source.approval_status.value
        rows.append(row)
    return pd.DataFrame(rows)


def source_for(provider_name: str) -> ConsensusSource:
    matches = [source for source in SOURCES if source.provider_name == provider_name]
    if len(matches) != 1:
        raise KeyError(f"Expected one consensus source named {provider_name!r}; found {len(matches)}")
    return matches[0]


def validate_source_registry() -> None:
    names = [source.provider_name for source in SOURCES]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate consensus-provider registry identity")
    te = source_for("Trading Economics")
    if "Forecast" not in te.consensus_field or "TEForecast" not in te.provider_forecast_field:
        raise ValueError("Trading Economics Forecast/TEForecast separation is missing")
    if te.point_in_time_safe or te.approval_status == ApprovalStatus.APPROVED_POINT_IN_TIME:
        raise ValueError("Trading Economics may not be pre-approved without live point-in-time evidence")


validate_source_registry()
