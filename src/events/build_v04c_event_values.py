"""Build normalized, point-in-time V0.4C economic release values from local ALFRED data."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.events.event_value_source_registry import SOURCES, source_registry_frame
from src.events.v04c_contract import (
    EVENT_VALUE_CONTRACT,
    EXPECTED_EVENT_VALUE_ROWS,
    ROOT,
    V04B_CONTRACT,
    V04B_MEMORY_PATH,
    verify_v04b_memory,
)


FRED_PATH = ROOT / "data" / "macro" / "fred_macro.parquet"
OUTPUT = ROOT / "data" / "processed" / "v04c_event_values.parquet"
REVISION_COMPONENT_OUTPUT = ROOT / "data" / "processed" / "v04c_event_value_revision_components.parquet"
FINGERPRINT_REPORT = ROOT / "reports" / "v04b_historical_event_memory_fingerprint.txt"
SAMPLE_REPORT = ROOT / "reports" / "v04c_event_values_sample.csv"
QUALITY_REPORT = ROOT / "reports" / "v04c_event_values_quality.txt"
SCHEMA_REPORT = ROOT / "reports" / "v04c_event_values_schema.csv"
COVERAGE_REPORT = ROOT / "reports" / "v04c_event_value_source_coverage.csv"
SURPRISE_COVERAGE_REPORT = ROOT / "reports" / "v04c_surprise_coverage.csv"
SOURCE_REGISTRY_REPORT = ROOT / "reports" / "v04c_event_value_source_registry.csv"
REVISION_COMPONENT_SAMPLE = ROOT / "reports" / "v04c_event_value_revision_components_sample.csv"

LOCAL_SERIES = {
    "CPIAUCSL": "us_cpi_yoy",
    "CPILFESL": "us_core_cpi_yoy",
    "PAYEMS": "us_payroll_growth",
    "UNRATE": "us_unemployment",
}
MIN_NORMALIZATION_HISTORY = 10


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False, engine="pyarrow")
    else:
        frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_fingerprint(metadata: dict[str, object]) -> None:
    counts = metadata["event_type_counts"]
    lines = [
        "MARKETFUSION V0.4B HISTORICAL EVENT MEMORY FINGERPRINT",
        f"file: {V04B_MEMORY_PATH.relative_to(ROOT).as_posix()}",
        f"sha256: {metadata['sha256']}",
        f"rows: {metadata['rows']}",
        f"columns: {metadata['columns']}",
        f"event_count: {metadata['event_count']}",
        f"us_cpi_release: {counts.get('us_cpi_release', 0)}",
        f"us_employment_situation: {counts.get('us_employment_situation', 0)}",
        f"min_timestamp_utc: {metadata['min_timestamp']}",
        f"max_timestamp_utc: {metadata['max_timestamp']}",
        f"retrieval_contract: {metadata['retrieval_contract']}",
        "FINGERPRINT_STATUS: PASS",
        "",
    ]
    _atomic_text("\n".join(lines), FINGERPRINT_REPORT)


def load_alfred(path: Path = FRED_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing audited local ALFRED data: {path}")
    frame = pd.read_parquet(path, engine="pyarrow")
    required = {
        "series_id", "observation_period", "available_from_utc", "value",
        "vintage_date", "source_series_id", "availability_method", "model_eligible",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"ALFRED data lacks fields: {sorted(required - set(frame.columns))}")
    frame = frame[frame["model_eligible"].fillna(False).astype(bool)].copy()
    for column in ("observation_period", "available_from_utc", "vintage_date"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise RuntimeError("ALFRED source contains non-finite values")
    return frame


def _last(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values(["available_from_utc", "vintage_date"]).iloc[-1]


def _direction(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return "UP" if value > 0 else ("DOWN" if value < 0 else "UNCHANGED")


def _surprise_semantics(measure_id: str, raw: float | None) -> tuple[str | None, str | None]:
    if raw is None or pd.isna(raw):
        return None, None
    if measure_id == "nonfarm_payroll_change":
        return ("STRONGER_LABOR" if raw > 0 else ("WEAKER_LABOR" if raw < 0 else "AT_CONSENSUS")), None
    if measure_id == "unemployment_rate":
        return ("STRONGER_LABOR" if raw < 0 else ("WEAKER_LABOR" if raw > 0 else "AT_CONSENSUS")), None
    return None, ("HOTTER" if raw > 0 else ("COOLER" if raw < 0 else "AT_CONSENSUS"))


def calculate_surprise_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    both = result["actual_value"].notna() & result["consensus_value"].notna()
    result["surprise_signed_raw"] = np.where(
        both, result["actual_value"] - result["consensus_value"], np.nan
    )
    result["surprise_abs_raw"] = result["surprise_signed_raw"].abs()
    result["surprise_status"] = np.where(both, "AVAILABLE", "CONSENSUS_UNAVAILABLE")
    semantics = [
        _surprise_semantics(measure_id, raw)
        for measure_id, raw in zip(result["measure_id"], result["surprise_signed_raw"])
    ]
    result["surprise_strength_direction"] = [item[0] for item in semantics]
    result["surprise_inflation_direction"] = [item[1] for item in semantics]
    return result


def add_causal_surprise_normalization(
    frame: pd.DataFrame,
    minimum_history: int = MIN_NORMALIZATION_HISTORY,
) -> pd.DataFrame:
    result = frame.sort_values(["event_timestamp_utc", "event_id", "measure_id"]).copy()
    normalized: dict[int, float] = {}
    counts: dict[int, int] = {}
    methods: dict[int, str] = {}
    cutoffs: dict[int, object] = {}
    buckets: dict[int, str | None] = {}
    histories: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for index, row in result.iterrows():
        history = histories.setdefault(str(row["measure_id"]), [])
        raw = row["surprise_signed_raw"]
        counts[index] = len(history)
        cutoffs[index] = history[-1][0] if history else pd.NaT
        normalized[index] = np.nan
        buckets[index] = None
        if pd.isna(raw):
            methods[index] = "UNAVAILABLE_RAW_SURPRISE"
            continue
        if len(history) < minimum_history:
            methods[index] = f"INSUFFICIENT_HISTORY_MIN_{minimum_history}"
        else:
            prior = np.asarray([value for _, value in history], dtype=float)
            median = float(np.median(prior))
            q1, q3 = np.quantile(prior, [0.25, 0.75])
            scale = float(q3 - q1)
            method = "prior_only_median_iqr"
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.std(prior, ddof=0))
                method = "prior_only_median_std_fallback"
            if np.isfinite(scale) and scale > 0:
                value = (float(raw) - median) / scale
                normalized[index] = value
                methods[index] = method
                buckets[index] = (
                    "LARGE_POSITIVE" if value >= 1.0 else
                    "SMALL_POSITIVE" if value >= 0.25 else
                    "NEAR_CONSENSUS" if value > -0.25 else
                    "SMALL_NEGATIVE" if value > -1.0 else
                    "LARGE_NEGATIVE"
                )
            else:
                methods[index] = "INSUFFICIENT_VARIATION"
        history.append((pd.Timestamp(row["event_timestamp_utc"]), float(raw)))
    result["surprise_z_robust"] = pd.Series(normalized)
    result["normalization_history_count"] = pd.Series(counts, dtype="int64")
    result["normalization_method"] = pd.Series(methods)
    result["normalization_timestamp_cutoff"] = pd.to_datetime(pd.Series(cutoffs), utc=True)
    result["surprise_bucket_causal"] = pd.Series(buckets)
    return result.sort_values(["event_timestamp_utc", "event_id", "measure_id"]).reset_index(drop=True)


def build_event_values(memory: pd.DataFrame, alfred: pd.DataFrame, retrieval_time: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in memory.sort_values("event_timestamp_utc").itertuples(index=False):
        event_time = pd.Timestamp(event.event_timestamp_utc).tz_convert("UTC")
        reference_period = pd.Timestamp(event.reference_period).tz_convert("UTC")
        event_date = event_time.date()
        for source in (item for item in SOURCES if item.event_type == event.event_type):
            local_series = LOCAL_SERIES[source.source_series_id]
            series = alfred[alfred["series_id"].eq(local_series)]
            actual_candidates = series[
                series["observation_period"].eq(reference_period)
                & series["vintage_date"].dt.date.eq(event_date)
            ]
            if len(actual_candidates) != 1:
                raise RuntimeError(
                    f"{event.event_id}/{source.measure_id}: expected one same-release actual; "
                    f"found {len(actual_candidates)}"
                )
            actual = actual_candidates.iloc[0]

            previous_period = reference_period - pd.DateOffset(months=1)
            prior_candidates = series[
                series["observation_period"].eq(previous_period)
                & (series["available_from_utc"] < event_time)
                & (series["vintage_date"].dt.date < event_date)
            ]
            previous = _last(prior_candidates)
            revision_candidates = series[
                series["observation_period"].eq(previous_period)
                & series["vintage_date"].dt.date.eq(event_date)
            ]
            if len(revision_candidates) > 1:
                raise RuntimeError(f"Ambiguous same-release revision: {event.event_id}/{source.measure_id}")
            revised = _last(revision_candidates)
            previous_value = float(previous["value"]) if previous is not None else np.nan
            prior_period_new_value = (
                float(revised["value"]) if revised is not None and previous is None else np.nan
            )
            revised_value = (
                float(revised["value"]) if revised is not None and previous is not None else np.nan
            )
            revision_raw = revised_value - previous_value if previous is not None and revised is not None else np.nan
            value_status = (
                "PRIOR_PERIOD_FIRST_RELEASE_AT_T" if not pd.isna(prior_period_new_value) else
                "ACTUAL_ONLY" if previous is None else
                "COMPLETE" if revised is not None else
                "REVISION_UNAVAILABLE"
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_timestamp_utc": event_time,
                    "event_type": event.event_type,
                    "measure_id": source.measure_id,
                    "measure_name": source.measure_name,
                    "unit": source.unit,
                    "frequency": source.frequency,
                    "reference_period": reference_period,
                    "actual_value": float(actual["value"]),
                    "previous_value_pre_release": previous_value,
                    "previous_value_revised_at_release": revised_value,
                    "consensus_value": np.nan,
                    "actual_available_from_utc": event_time,
                    "previous_available_from_utc": previous["available_from_utc"] if previous is not None else pd.NaT,
                    "revision_available_from_utc": (
                        event_time if revised is not None and previous is not None else pd.NaT
                    ),
                    "prior_period_newly_released_at_release": prior_period_new_value,
                    "prior_period_new_release_available_from_utc": (
                        event_time if not pd.isna(prior_period_new_value) else pd.NaT
                    ),
                    "consensus_available_from_utc": pd.NaT,
                    "source": source.source_name,
                    "source_authority": source.authority,
                    "source_record_id": (
                        f"{source.source_series_id}:{reference_period.date().isoformat()}:"
                        f"{pd.Timestamp(actual['vintage_date']).date().isoformat()}"
                    ),
                    "source_vintage": actual["vintage_date"],
                    "source_url_or_reference": source.source_url_or_reference,
                    "retrieval_timestamp_utc": retrieval_time,
                    "source_date_availability_utc": actual["available_from_utc"],
                    "availability_refinement": source.release_timestamp_semantics,
                    "transformation_rules": source.transformation_rules,
                    "raw_source_value": float(actual["value"]),
                    "raw_source_unit": source.unit,
                    "value_status": value_status,
                    "consensus_status": "UNAVAILABLE",
                    "point_in_time_verified": True,
                    "event_revision_raw": revision_raw,
                    "event_revision_abs": abs(revision_raw) if not pd.isna(revision_raw) else np.nan,
                    "event_revision_direction": _direction(revision_raw),
                    "economic_positive_direction": source.economic_direction_semantics,
                    "event_value_contract_version": EVENT_VALUE_CONTRACT,
                    "source_v04b_contract_version": V04B_CONTRACT,
                }
            )
    values = calculate_surprise_fields(pd.DataFrame(rows))
    return add_causal_surprise_normalization(values)


def build_revision_components(memory: pd.DataFrame, alfred: pd.DataFrame) -> pd.DataFrame:
    """Preserve every older observation changed or first published at release T."""
    rows: list[dict[str, object]] = []
    for event in memory.sort_values("event_timestamp_utc").itertuples(index=False):
        event_time = pd.Timestamp(event.event_timestamp_utc).tz_convert("UTC")
        reference_period = pd.Timestamp(event.reference_period).tz_convert("UTC")
        event_date = event_time.date()
        for source in (item for item in SOURCES if item.event_type == event.event_type):
            local_series = LOCAL_SERIES[source.source_series_id]
            series = alfred[alfred["series_id"].eq(local_series)]
            released = series[
                (series["observation_period"] < reference_period)
                & series["vintage_date"].dt.date.eq(event_date)
            ].sort_values("observation_period")
            for component in released.itertuples(index=False):
                prior = _last(
                    series[
                        series["observation_period"].eq(component.observation_period)
                        & (series["available_from_utc"] < event_time)
                        & (series["vintage_date"].dt.date < event_date)
                    ]
                )
                pre_value = float(prior["value"]) if prior is not None else np.nan
                revised_value = float(component.value)
                rows.append(
                    {
                        "event_id": event.event_id,
                        "event_timestamp_utc": event_time,
                        "event_type": event.event_type,
                        "measure_id": source.measure_id,
                        "revision_reference_period": component.observation_period,
                        "value_pre_release": pre_value,
                        "value_released_at_T": revised_value,
                        "revision_raw": revised_value - pre_value if prior is not None else np.nan,
                        "pre_release_available_from_utc": prior["available_from_utc"] if prior is not None else pd.NaT,
                        "released_available_from_utc": event_time,
                        "source_series_id": source.source_series_id,
                        "source_vintage": component.vintage_date,
                        "component_status": "REVISION" if prior is not None else "PRIOR_PERIOD_FIRST_RELEASE_AT_T",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["event_timestamp_utc", "event_id", "measure_id", "revision_reference_period"]
    ).reset_index(drop=True)


def validate_revision_components(components: pd.DataFrame) -> None:
    if components.empty:
        return
    if components.duplicated(["event_id", "measure_id", "revision_reference_period"]).any():
        raise RuntimeError("Duplicate raw revision component identity")
    if any(column.startswith("outcome_") for column in components):
        raise RuntimeError("Market outcome entered revision component table")
    event_times = pd.to_datetime(components["event_timestamp_utc"], utc=True)
    released = pd.to_datetime(components["released_available_from_utc"], utc=True)
    if not released.eq(event_times).all():
        raise RuntimeError("Revision component availability is not release T")
    prior_present = components["value_pre_release"].notna()
    prior_available = pd.to_datetime(components["pre_release_available_from_utc"], utc=True)
    if (prior_present & ~(prior_available < event_times)).any():
        raise RuntimeError("Revision component prior value was not available before T")
    revision = components["component_status"].eq("REVISION")
    expected = components["value_released_at_T"] - components["value_pre_release"]
    if not np.allclose(
        components.loc[revision, "revision_raw"], expected.loc[revision], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Raw revision component arithmetic is inconsistent")


def validate_event_values(values: pd.DataFrame, expected_rows: int | None = EXPECTED_EVENT_VALUE_ROWS) -> None:
    if expected_rows is not None and len(values) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} normalized event values; found {len(values)}")
    if values.duplicated(["event_id", "measure_id"]).any():
        raise RuntimeError("Normalized event/measure identity is not unique")
    if any(column.startswith("outcome_") for column in values.columns):
        raise RuntimeError("Future market outcome entered release-value construction")
    event_times = pd.to_datetime(values["event_timestamp_utc"], utc=True, errors="raise")
    actual_available = pd.to_datetime(values["actual_available_from_utc"], utc=True, errors="raise")
    if not actual_available.eq(event_times).all():
        raise RuntimeError("Actual value availability must equal official event T")
    if values["actual_value"].isna().any() or not np.isfinite(values["actual_value"].to_numpy(float)).all():
        raise RuntimeError("Audited actual value is missing/non-finite")
    if not np.allclose(values["actual_value"], values["raw_source_value"], rtol=0.0, atol=1e-12):
        raise RuntimeError("Actual value no longer matches preserved raw source value")
    source_vintage = pd.to_datetime(values["source_vintage"], utc=True, errors="raise")
    if not source_vintage.dt.date.eq(event_times.dt.date).all():
        raise RuntimeError("Actual source vintage is not the official release date")
    previous_present = values["previous_value_pre_release"].notna()
    previous_available = pd.to_datetime(values["previous_available_from_utc"], utc=True)
    if (previous_present & ~(previous_available < event_times)).any():
        raise RuntimeError("Previous value was not available strictly before T")
    revision_present = values["previous_value_revised_at_release"].notna()
    revision_available = pd.to_datetime(values["revision_available_from_utc"], utc=True)
    if (revision_present & ~revision_available.eq(event_times)).any():
        raise RuntimeError("Later or mistimed revision entered release payload")
    if (revision_present & ~previous_present).any():
        raise RuntimeError("Release-time revision lacks the preserved pre-release value")
    both_revision = revision_present & previous_present
    expected_revision = values["previous_value_revised_at_release"] - values["previous_value_pre_release"]
    if not np.allclose(
        values.loc[both_revision, "event_revision_raw"], expected_revision.loc[both_revision],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("Release-time revision arithmetic is inconsistent")
    newly_released = values["prior_period_newly_released_at_release"].notna()
    newly_available = pd.to_datetime(
        values["prior_period_new_release_available_from_utc"], utc=True
    )
    if (newly_released & ~newly_available.eq(event_times)).any():
        raise RuntimeError("Newly released prior-period value is not timestamped at T")
    if (newly_released & (previous_present | revision_present)).any():
        raise RuntimeError("Newly released prior-period value was misclassified as previous/revision")
    consensus_present = values["consensus_value"].notna()
    consensus_available = pd.to_datetime(values["consensus_available_from_utc"], utc=True)
    if (consensus_present & ~(consensus_available < event_times)).any():
        raise RuntimeError("Consensus must be available strictly before T")
    expected_raw = values["actual_value"] - values["consensus_value"]
    raw_present = values["surprise_signed_raw"].notna()
    if not np.allclose(
        values.loc[raw_present, "surprise_signed_raw"], expected_raw.loc[raw_present],
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("Raw surprise arithmetic is inconsistent")
    if (~consensus_present & raw_present).any():
        raise RuntimeError("Surprise exists without trustworthy consensus")
    cutoff = pd.to_datetime(values["normalization_timestamp_cutoff"], utc=True)
    normalized = values["surprise_z_robust"].notna()
    if (normalized & ~(cutoff < event_times)).any():
        raise RuntimeError("Surprise normalization used query/future history")
    if (normalized & (values["normalization_history_count"] < MIN_NORMALIZATION_HISTORY)).any():
        raise RuntimeError("Normalized surprise violates minimum prior-history gate")
    registry = {(source.event_type, source.measure_id): source for source in SOURCES}
    for row in values.itertuples(index=False):
        source = registry.get((row.event_type, row.measure_id))
        if source is None:
            raise RuntimeError(f"Official event mapping absent: {row.event_type}/{row.measure_id}")
        if row.unit != source.unit or row.raw_source_unit != source.unit:
            raise RuntimeError(f"Unit mismatch: {row.event_id}/{row.measure_id}")
    if not values["point_in_time_verified"].fillna(False).astype(bool).all():
        raise RuntimeError("Unverified value entered audited event table")


def coverage_frame(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_type, measure_id), group in values.groupby(["event_type", "measure_id"], sort=True):
        rows.append(
            {
                "event_type": event_type,
                "measure_id": measure_id,
                "rows": len(group),
                "first_event_utc": group["event_timestamp_utc"].min(),
                "last_event_utc": group["event_timestamp_utc"].max(),
                "actual_count": int(group["actual_value"].notna().sum()),
                "previous_count": int(group["previous_value_pre_release"].notna().sum()),
                "revision_count": int(group["previous_value_revised_at_release"].notna().sum()),
                "prior_period_first_release_count": int(group["prior_period_newly_released_at_release"].notna().sum()),
                "consensus_count": int(group["consensus_value"].notna().sum()),
                "source": group["source"].iloc[0],
                "source_authority": group["source_authority"].iloc[0],
                "point_in_time_verified_count": int(group["point_in_time_verified"].sum()),
            }
        )
    return pd.DataFrame(rows)


def surprise_coverage_frame(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_type, measure_id), group in values.groupby(["event_type", "measure_id"], sort=True):
        consensus = int(group["consensus_value"].notna().sum())
        rows.append(
            {
                "event_type": event_type,
                "measure_id": measure_id,
                "N": len(group),
                "time_span_start": group["event_timestamp_utc"].min(),
                "time_span_end": group["event_timestamp_utc"].max(),
                "consensus_count": consensus,
                "raw_surprise_count": int(group["surprise_signed_raw"].notna().sum()),
                "normalized_surprise_count": int(group["surprise_z_robust"].notna().sum()),
                "research_status": "AVAILABLE" if consensus else "INSUFFICIENT_SAMPLE",
            }
        )
    return pd.DataFrame(rows)


def schema_frame(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in values.columns:
        if column in {"event_id", "event_timestamp_utc", "event_type", "measure_id", "measure_name", "unit", "frequency", "reference_period"}:
            group, semantics = "identity", "scheduled release/normalized measure identity"
        elif column.startswith("actual_") or column in {"actual_value", "raw_source_value", "raw_source_unit"}:
            group, semantics = "release_payload", "available at official release T; POST_RELEASE_IMMEDIATE only"
        elif column.startswith("previous_"):
            group, semantics = "pre_release_or_revision", "pre-release value is <T; revised-at-release value is T"
        elif column.startswith("prior_period_new"):
            group, semantics = "release_payload", "older period first published at T; not classified as a revision"
        elif column.startswith("consensus_"):
            group, semantics = "pre_release_context", "must be strictly before T; unavailable in current audited source"
        elif column.startswith("surprise_") or column.startswith("normalization_"):
            group, semantics = "release_payload", "requires actual at T and prior consensus; causal history only"
        elif column.startswith("event_revision_") or column == "revision_available_from_utc":
            group, semantics = "release_payload", "revision published at T; never overwrites prior value"
        else:
            group, semantics = "provenance_status", "audit metadata; not an analogue outcome"
        rows.append({"column": column, "dtype": str(values[column].dtype), "logical_group": group, "availability_semantics": semantics})
    return pd.DataFrame(rows)


def write_outputs(
    values: pd.DataFrame,
    components: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    _atomic_frame(values, OUTPUT)
    _atomic_frame(components, REVISION_COMPONENT_OUTPUT)
    _atomic_frame(values.head(24), SAMPLE_REPORT)
    _atomic_frame(schema_frame(values), SCHEMA_REPORT)
    coverage = coverage_frame(values)
    _atomic_frame(coverage, COVERAGE_REPORT)
    _atomic_frame(surprise_coverage_frame(values), SURPRISE_COVERAGE_REPORT)
    _atomic_frame(source_registry_frame(), SOURCE_REGISTRY_REPORT)
    _atomic_frame(components.head(30), REVISION_COMPONENT_SAMPLE)
    counts = values["event_type"].value_counts().to_dict()
    lines = [
        "MARKETFUSION V0.4C EVENT VALUES QUALITY",
        f"source_v04b_sha256: {metadata['sha256']}",
        f"event_value_contract: {EVENT_VALUE_CONTRACT}",
        f"events: {values['event_id'].nunique()}",
        f"rows: {len(values)}",
        f"measures: {values['measure_id'].nunique()}",
        f"cpi_rows: {counts.get('us_cpi_release', 0)}",
        f"employment_rows: {counts.get('us_employment_situation', 0)}",
        f"actual_values: {int(values['actual_value'].notna().sum())}",
        f"previous_values: {int(values['previous_value_pre_release'].notna().sum())}",
        f"release_time_revisions: {int(values['previous_value_revised_at_release'].notna().sum())}",
        f"prior_period_values_first_released_at_T: {int(values['prior_period_newly_released_at_release'].notna().sum())}",
        f"normalized_revision_component_rows: {len(components)}",
        f"revision_components: {int(components['component_status'].eq('REVISION').sum())}",
        f"prior_period_component_first_releases: {int(components['component_status'].eq('PRIOR_PERIOD_FIRST_RELEASE_AT_T').sum())}",
        f"consensus_values: {int(values['consensus_value'].notna().sum())}",
        f"raw_surprises: {int(values['surprise_signed_raw'].notna().sum())}",
        f"normalized_surprises: {int(values['surprise_z_robust'].notna().sum())}",
        "point_in_time_violations: 0",
        "revision_chronology_violations: 0",
        "post_release_consensus_violations: 0",
        "unit_mismatches: 0",
        "ambiguous_reference_periods: 0",
        "",
        "SOURCE INTERPRETATION",
        "- Actuals are official BLS series values preserved in ALFRED real-time vintages.",
        "- Same-day ALFRED vintage dates are refined to the already-validated BLS release timestamp T.",
        "- The original conservative next-day macro availability is retained as source_date_availability_utc.",
        "- Prior values use only rows available strictly before T; same-release revisions remain separate.",
        "- CPI values are point-in-time SA year-over-year transforms, not claimed as exact NSA press-table fields.",
        "- Historical consensus has no audited provider and remains NULL/UNAVAILABLE.",
        "",
        "SURPRISE_RESEARCH_STATUS: INSUFFICIENT_SAMPLE",
        "EVENT_VALUE_QUALITY_STATUS: PASS",
        "",
    ]
    _atomic_text("\n".join(lines), QUALITY_REPORT)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    memory, metadata = verify_v04b_memory()
    write_fingerprint(metadata)
    alfred = load_alfred()
    retrieval_time = pd.Timestamp(datetime.fromtimestamp(FRED_PATH.stat().st_mtime, tz=timezone.utc))
    values = build_event_values(memory, alfred, retrieval_time)
    components = build_revision_components(memory, alfred)
    validate_event_values(values)
    validate_revision_components(components)
    component_counts = components.groupby(["event_id", "measure_id"]).size().rename("revision_component_count")
    values = values.merge(component_counts, on=["event_id", "measure_id"], how="left")
    values["revision_component_count"] = values["revision_component_count"].fillna(0).astype("int64")
    write_outputs(values, components, metadata)
    print("V04C_EVENT_VALUE_STATUS: PASS")
    print(f"Events: {values['event_id'].nunique()}")
    print(f"Rows: {len(values)}")
    print(f"Actuals: {values['actual_value'].notna().sum()}")
    print("Consensus: 0 (audited provider unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
