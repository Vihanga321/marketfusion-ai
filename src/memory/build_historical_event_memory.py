"""Build the leakage-safe V0.4B Historical Economic Event Memory."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Support both ``python -m src.memory...`` and the repository's documented
# ``python src/memory/...py`` invocation on Windows.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from retrieval_feature_registry import FEATURES, approved_feature_names, registry_frame, validate_registry
    from v04b_contract import (
        MEMORY_CONTRACT,
        PIP_SIZE,
        REACTION_COLUMNS,
        REACTION_CONTRACT,
        REACTION_EVENT_TYPE_COUNTS,
        REACTION_MAX_TIMESTAMP,
        REACTION_MIN_TIMESTAMP,
        REACTION_PATH,
        REACTION_ROWS,
        REACTION_SHA256,
        ROOT,
        sha256_file,
        verify_reaction_dataset,
    )
except ImportError:  # pragma: no cover
    from src.memory.retrieval_feature_registry import FEATURES, approved_feature_names, registry_frame, validate_registry
    from src.memory.v04b_contract import (
        MEMORY_CONTRACT,
        PIP_SIZE,
        REACTION_COLUMNS,
        REACTION_CONTRACT,
        REACTION_EVENT_TYPE_COUNTS,
        REACTION_MAX_TIMESTAMP,
        REACTION_MIN_TIMESTAMP,
        REACTION_PATH,
        REACTION_ROWS,
        REACTION_SHA256,
        ROOT,
        sha256_file,
        verify_reaction_dataset,
    )

from src.events.build_v04a_event_reactions import resolve_validated_window, validate_window
from src.events.v04a_contract import verify_frozen_evidence


STATUS_PATH = ROOT / "data" / "dukascopy" / "v04a_reaction" / "reaction_extraction_status.tsv"
MACRO_PATH = ROOT / "data" / "macro" / "macro_safe.parquet"
OUTPUT = ROOT / "data" / "processed" / "v04b_historical_event_memory.parquet"
FINGERPRINT_REPORT = ROOT / "reports" / "v04a_event_reaction_fingerprint.txt"
SAMPLE_REPORT = ROOT / "reports" / "v04b_historical_event_memory_sample.csv"
QUALITY_REPORT = ROOT / "reports" / "v04b_historical_event_memory_quality.txt"
SCHEMA_REPORT = ROOT / "reports" / "v04b_historical_event_memory_schema.csv"
REGISTRY_REPORT = ROOT / "reports" / "v04b_retrieval_feature_registry.csv"

BASE_MACRO_SERIES = (
    "fed_funds_rate", "us_2y_yield", "us_10y_yield", "us_cpi_yoy",
    "us_core_cpi_yoy", "us_unemployment", "us_payroll_growth", "us_gdp_growth",
    "us_pce", "us_core_pce", "us_retail_sales_yoy", "ecb_rate", "euro_hicp",
    "euro_core_hicp", "euro_unemployment", "euro_gdp_growth", "euro_2y_yield",
    "euro_10y_yield",
)
IDENTITY_COLUMNS = (
    "event_id", "event_type", "event_timestamp_utc", "reference_period",
    "timestamp_precision", "timestamp_provenance", "market_data_source",
    "production_status", "adjudication_class", "reaction_contract_version",
)


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


def write_fingerprint_report(metadata: dict[str, object]) -> None:
    counts = metadata["event_type_counts"]
    lines = [
        "MARKETFUSION V0.4A REACTION DATASET FINGERPRINT",
        f"file: {REACTION_PATH.relative_to(ROOT).as_posix()}",
        f"sha256: {metadata['sha256']}",
        f"row_count: {metadata['rows']}",
        f"column_count: {metadata['columns']}",
        f"contract_version: {metadata['contract_version']}",
        f"min_timestamp_utc: {metadata['min_timestamp']}",
        f"max_timestamp_utc: {metadata['max_timestamp']}",
        f"us_cpi_release: {counts.get('us_cpi_release', 0)}",
        f"us_employment_situation: {counts.get('us_employment_situation', 0)}",
        "FINGERPRINT_STATUS: PASS",
        "",
    ]
    _atomic_text("\n".join(lines), FINGERPRINT_REPORT)


def _sign_code(value: float) -> float:
    return 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)


def build_price_context(reactions: pd.DataFrame, statuses: pd.DataFrame) -> pd.DataFrame:
    if statuses["event_id"].duplicated().any() or len(statuses) != REACTION_ROWS:
        raise RuntimeError("Reaction extraction status must contain 242 unique events")
    if not statuses["retrieval_status"].eq("COMPLETE").all():
        raise RuntimeError("Historical memory requires every V0.4A reaction extraction to be COMPLETE")
    status_lookup = statuses.set_index("event_id", drop=False)
    rows: list[dict[str, object]] = []
    for event in reactions.sort_values("event_timestamp_utc").itertuples(index=False):
        event_series = pd.Series(event._asdict())
        status = status_lookup.loc[event.event_id]
        path = resolve_validated_window(status, event_series)
        raw = pd.read_csv(path, sep="\t")
        event_time = pd.Timestamp(event.event_timestamp_utc)
        event_time = event_time.tz_convert("UTC")
        window = validate_window(raw, event.event_id, event_time)
        pre = window.loc[window["minute_utc"] < event_time].copy()
        if len(pre) != 10 or pre["minute_utc"].max() != event_time - pd.Timedelta(minutes=1):
            raise RuntimeError(f"{event.event_id}: exact pre-T ten-minute window unavailable")
        last_five = pre.tail(5)
        return_5m = (float(last_five.iloc[-1]["mid_close"]) - float(last_five.iloc[0]["mid_open"])) / PIP_SIZE
        return_10m = (float(pre.iloc[-1]["mid_close"]) - float(pre.iloc[0]["mid_open"])) / PIP_SIZE
        one_minute_path = np.concatenate(
            [[float(pre.iloc[0]["mid_open"])], pre["mid_close"].to_numpy(dtype=float)]
        )
        minute_returns_pips = np.diff(one_minute_path) / PIP_SIZE
        rows.append(
            {
                "event_id": event.event_id,
                "context_price_window_start_utc": pre.iloc[0]["minute_utc"],
                "context_price_available_through_utc": pre.iloc[-1]["minute_utc"] + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
                "context_return_5m_pips": return_5m,
                "context_return_10m_pips": return_10m,
                "context_realized_vol_10m": float(np.std(minute_returns_pips, ddof=0)),
                "context_range_10m_pips": float((pre["mid_high"].max() - pre["mid_low"].min()) / PIP_SIZE),
                "context_pre_event_max_spread_10m_pips": float(pre["spread_high"].max() / PIP_SIZE),
                "context_trend_5m": "UP" if return_5m > 0 else ("DOWN" if return_5m < 0 else "FLAT"),
                "context_trend_10m": "UP" if return_10m > 0 else ("DOWN" if return_10m < 0 else "FLAT"),
                "context_trend_5m_code": _sign_code(return_5m),
                "context_trend_10m_code": _sign_code(return_10m),
                "context_price_source": "Dukascopy JForex ticks; exact [T-10m,T) window",
                "context_price_window_sha256": str(status["data_sha256"]),
            }
        )
    return pd.DataFrame(rows)


def add_calendar_context(memory: pd.DataFrame) -> pd.DataFrame:
    result = memory.sort_values("event_timestamp_utc").copy()
    utc = pd.to_datetime(result["event_timestamp_utc"], utc=True)
    month_angle = 2.0 * math.pi * (utc.dt.month - 1) / 12.0
    weekday_angle = 2.0 * math.pi * utc.dt.weekday / 7.0
    result["context_month"] = utc.dt.month.astype("int8")
    result["context_month_sin"] = np.sin(month_angle)
    result["context_month_cos"] = np.cos(month_angle)
    result["context_weekday"] = utc.dt.weekday.astype("int8")
    result["context_weekday_sin"] = np.sin(weekday_angle)
    result["context_weekday_cos"] = np.cos(weekday_angle)
    local = utc.dt.tz_convert(ZoneInfo("America/New_York"))
    result["context_release_local_hour"] = local.dt.hour + local.dt.minute / 60.0
    result["context_us_dst"] = local.map(lambda value: 1.0 if value.dst().total_seconds() else 0.0)
    result["context_days_since_previous_same_type"] = (
        result.groupby("event_type")["event_timestamp_utc"].diff().dt.total_seconds() / 86_400.0
    )
    result["context_calendar_source"] = "event_timestamp_utc and America/New_York zoneinfo"
    return result


def add_causal_volatility_regime(memory: pd.DataFrame, minimum_history: int = 10) -> pd.DataFrame:
    result = memory.sort_values("event_timestamp_utc").copy()
    labels: dict[int, str] = {}
    codes: dict[int, float] = {}
    lower_bounds: dict[int, float] = {}
    upper_bounds: dict[int, float] = {}
    history: dict[str, list[float]] = {}
    for index, row in result.iterrows():
        prior = history.setdefault(str(row["event_type"]), [])
        value = float(row["context_realized_vol_10m"])
        if len(prior) < minimum_history:
            labels[index] = "UNKNOWN"
            codes[index] = np.nan
            lower_bounds[index] = np.nan
            upper_bounds[index] = np.nan
        else:
            lower, upper = np.quantile(np.asarray(prior, dtype=float), [1.0 / 3.0, 2.0 / 3.0])
            lower_bounds[index] = float(lower)
            upper_bounds[index] = float(upper)
            if value < lower:
                labels[index], codes[index] = "LOW", -1.0
            elif value > upper:
                labels[index], codes[index] = "HIGH", 1.0
            else:
                labels[index], codes[index] = "NORMAL", 0.0
        prior.append(value)
    result["context_volatility_regime"] = pd.Series(labels)
    result["context_volatility_regime_code"] = pd.Series(codes)
    result["context_volatility_regime_lower"] = pd.Series(lower_bounds)
    result["context_volatility_regime_upper"] = pd.Series(upper_bounds)
    result["context_volatility_regime_rule"] = (
        f"same-event-type expanding terciles from strictly earlier events; minimum_history={minimum_history}"
    )
    return result


def latest_observation_events(series: pd.DataFrame) -> pd.DataFrame:
    ordered = series.sort_values(["available_from_utc", "observation_period", "vintage_date"]).copy()
    state: dict[pd.Timestamp, pd.Series] = {}
    events: list[dict[str, object]] = []
    previous: tuple[object, ...] | None = None
    for available, updates in ordered.groupby("available_from_utc", sort=True):
        for _, row in updates.iterrows():
            state[pd.Timestamp(row["observation_period"])] = row
        latest_period = max(state)
        selected = state[latest_period]
        current = (latest_period, float(selected["value"]), pd.Timestamp(selected["vintage_date"]))
        if current != previous:
            events.append(
                {
                    "available_from_utc": pd.Timestamp(available),
                    "observation_period": latest_period,
                    "value": float(selected["value"]),
                    "vintage_date": pd.Timestamp(selected["vintage_date"]),
                    "source": selected["source"],
                    "availability_method": selected["availability_method"],
                }
            )
            previous = current
    return pd.DataFrame(events)


def add_macro_context(memory: pd.DataFrame, macro_path: Path = MACRO_PATH) -> tuple[pd.DataFrame, dict[str, float]]:
    if not macro_path.is_file():
        raise RuntimeError(f"Missing audited point-in-time macro data: {macro_path}")
    macro = pd.read_parquet(macro_path, engine="pyarrow")
    required = {
        "series_id", "source", "observation_period", "available_from_utc", "value",
        "vintage_date", "availability_method", "model_eligible",
    }
    if not required.issubset(macro.columns):
        raise RuntimeError(f"Macro safe data lacks columns: {sorted(required - set(macro.columns))}")
    macro = macro[macro["model_eligible"].fillna(False).astype(bool)].copy()
    for column in ("observation_period", "available_from_utc", "vintage_date"):
        macro[column] = pd.to_datetime(macro[column], utc=True, errors="raise")
    macro["value"] = pd.to_numeric(macro["value"], errors="raise")
    if not np.isfinite(macro["value"].to_numpy(dtype=float)).all():
        raise RuntimeError("Macro safe data contains non-finite values")
    if (macro["available_from_utc"] < macro["observation_period"]).any():
        raise RuntimeError("Macro safe data contains values available before their observation period")
    if macro.duplicated(["series_id", "observation_period", "available_from_utc", "value"]).any():
        raise RuntimeError("Macro safe data contains duplicate release/vintage rows")

    result = memory.sort_values("event_timestamp_utc").copy()
    left = pd.DataFrame({"event_timestamp_utc": pd.to_datetime(result["event_timestamp_utc"], utc=True)})
    coverage: dict[str, float] = {}
    for series_id in BASE_MACRO_SERIES:
        source = macro[macro["series_id"].eq(series_id)]
        prefix = f"context_macro_{series_id}"
        if source.empty:
            result[prefix] = np.nan
            result[f"{prefix}__available_from_utc"] = pd.NaT
            result[f"{prefix}__observation_period"] = pd.NaT
            result[f"{prefix}__vintage_date"] = pd.NaT
            result[f"{prefix}__source"] = ""
            result[f"{prefix}__availability_method"] = ""
            coverage[series_id] = 0.0
            continue
        events = latest_observation_events(source).sort_values("available_from_utc")
        joined = pd.merge_asof(
            left.sort_values("event_timestamp_utc"), events,
            left_on="event_timestamp_utc", right_on="available_from_utc",
            direction="backward", allow_exact_matches=True,
        ).sort_index()
        result[prefix] = joined["value"].to_numpy()
        for field in ("available_from_utc", "observation_period", "vintage_date", "source", "availability_method"):
            result[f"{prefix}__{field}"] = joined[field].to_numpy()
        availability = pd.to_datetime(result[f"{prefix}__available_from_utc"], utc=True)
        leak = (availability > pd.to_datetime(result["event_timestamp_utc"], utc=True)).fillna(False)
        if leak.any():
            raise RuntimeError(f"Point-in-time macro leak in {series_id}: {int(leak.sum())} rows")
        coverage[series_id] = float(result[prefix].notna().mean())

    result["context_macro_yield_curve_10y_2y"] = (
        result["context_macro_us_10y_yield"] - result["context_macro_us_2y_yield"]
    )
    result["context_macro_yield_curve_10y_2y__available_from_utc"] = pd.concat(
        [
            pd.to_datetime(result["context_macro_us_10y_yield__available_from_utc"], utc=True),
            pd.to_datetime(result["context_macro_us_2y_yield__available_from_utc"], utc=True),
        ],
        axis=1,
    ).max(axis=1)
    result["context_macro_yield_curve_10y_2y__source"] = (
        "derived from point-in-time US 10-year and 2-year Treasury yields"
    )
    result["context_macro_yield_curve_10y_2y__availability_method"] = (
        "available when both components are public; timestamp=max(component availability)"
    )
    result["context_macro_rate_differential"] = (
        result["context_macro_fed_funds_rate"] - result["context_macro_ecb_rate"]
    )
    result["context_macro_rate_differential__available_from_utc"] = pd.concat(
        [
            pd.to_datetime(result["context_macro_fed_funds_rate__available_from_utc"], utc=True),
            pd.to_datetime(result["context_macro_ecb_rate__available_from_utc"], utc=True),
        ],
        axis=1,
    ).max(axis=1)
    result["context_macro_rate_differential__source"] = (
        "derived from point-in-time Federal Funds and ECB policy rates"
    )
    result["context_macro_rate_differential__availability_method"] = (
        "available when both components are public; timestamp=max(component availability)"
    )
    for feature in ("yield_curve_10y_2y", "rate_differential"):
        available = pd.to_datetime(
            result[f"context_macro_{feature}__available_from_utc"], utc=True
        )
        if (available > pd.to_datetime(result["event_timestamp_utc"], utc=True)).fillna(False).any():
            raise RuntimeError(f"Point-in-time macro leak in derived feature: {feature}")
    result["context_macro_source_file_sha256"] = sha256_file(macro_path)
    return result, coverage


def validate_eligible_identity(memory: pd.DataFrame, expected_count: int = REACTION_ROWS) -> None:
    if len(memory) != expected_count or memory["event_id"].duplicated().any():
        raise RuntimeError(f"Memory must contain exactly {expected_count} unique event records")
    if not memory["production_status"].eq("PASS").all():
        raise RuntimeError("Non-PASS event entered historical memory")
    if not memory["adjudication_class"].eq("STRICT_PASS").all():
        raise RuntimeError("Quarantined event entered historical memory")
    if not memory["reaction_extraction_status"].eq("COMPLETE").all():
        raise RuntimeError("Memory contains an event without validated historical outcomes")
    if not memory["model_eligible_market_reaction"].fillna(False).astype(bool).all():
        raise RuntimeError("Memory contains a V0.4A event not eligible for market-reaction research")


def validate_context_availability(memory: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(memory["event_timestamp_utc"], utc=True, errors="raise")
    if any(
        pd.Timestamp(value).tzinfo is None
        or pd.Timestamp(value).utcoffset() is None
        or pd.Timestamp(value).utcoffset().total_seconds() != 0
        for value in memory["event_timestamp_utc"]
    ):
        raise RuntimeError("Event timestamps must be timezone-aware UTC")
    if (pd.to_datetime(memory["context_price_available_through_utc"], utc=True) >= timestamps).any():
        raise RuntimeError("Pre-event price context uses data at or after event T")
    for column in (name for name in memory if name.endswith("__available_from_utc")):
        available = pd.to_datetime(memory[column], utc=True, errors="raise")
        if (available > timestamps).fillna(False).any():
            raise RuntimeError(f"Context availability after event T: {column}")
    for column in (name for name in memory if name.endswith("__vintage_date")):
        vintage = pd.to_datetime(memory[column], utc=True, errors="raise")
        if (vintage > timestamps).fillna(False).any():
            raise RuntimeError(f"Macro vintage after event T: {column}")


def validate_memory(memory: pd.DataFrame) -> None:
    validate_eligible_identity(memory)
    validate_context_availability(memory)
    outcome_columns = [column for column in memory if column.startswith("outcome_")]
    if not outcome_columns:
        raise RuntimeError("Historical outcome payload is missing")
    if memory[outcome_columns].isna().any().any():
        raise RuntimeError("Required historical outcomes contain missing values")
    numeric_outcomes = memory[outcome_columns].select_dtypes(include=[np.number])
    if not np.isfinite(numeric_outcomes.to_numpy(dtype=float)).all():
        raise RuntimeError("Required historical outcomes contain non-finite values")
    validate_registry()
    for feature in FEATURES:
        if feature.name not in memory:
            raise RuntimeError(f"Registered retrieval feature absent from memory: {feature.name}")
        values = pd.to_numeric(memory[feature.name], errors="coerce")
        if feature.required and (values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all()):
            raise RuntimeError(f"Required retrieval context is missing/non-finite: {feature.name}")
    registered = set(approved_feature_names())
    if any(name.startswith("outcome_") for name in registered):
        raise RuntimeError("Outcome leakage into retrieval feature registry")


def memory_schema(memory: pd.DataFrame) -> pd.DataFrame:
    registry = {feature.name: feature for feature in FEATURES}
    rows = []
    for column in memory.columns:
        if column in IDENTITY_COLUMNS or column in {"memory_contract_version", "reaction_extraction_status"}:
            side = "identity_provenance"
            availability = "known at or before event T"
        elif column.startswith("context_"):
            side = "retrieval_context" if column in registry else "context_audit_or_descriptor"
            availability = registry[column].availability_rule if column in registry else "documented pre-T context/audit metadata"
        elif column.startswith("outcome_"):
            side = "historical_outcomes"
            availability = "available only after event T; prohibited from retrieval"
        else:
            side = "provenance"
            availability = "metadata only"
        rows.append(
            {
                "column": column,
                "dtype": str(memory[column].dtype),
                "logical_side": side,
                "retrieval_approved": column in registry,
                "availability_semantics": availability,
                "source": registry[column].source if column in registry else "see column family/provenance fields",
                "required": registry[column].required if column in registry else (side != "context_audit_or_descriptor"),
            }
        )
    for placeholder in ("actual", "previous", "consensus", "forecast", "surprise"):
        rows.append(
            {
                "column": placeholder, "dtype": "not_populated", "logical_side": "future_event_values",
                "retrieval_approved": False, "availability_semantics": "absent until separately audited",
                "source": "none", "required": False,
            }
        )
    return pd.DataFrame(rows)


def build_memory() -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    verify_frozen_evidence()
    reactions, fingerprint = verify_reaction_dataset()
    write_fingerprint_report(fingerprint)
    reactions["event_timestamp_utc"] = pd.to_datetime(reactions["event_timestamp_utc"], utc=True)
    reactions["reference_period"] = pd.to_datetime(reactions["reference_period"], utc=True)
    if reactions["event_id"].duplicated().any() or not reactions["production_status"].eq("PASS").all():
        raise RuntimeError("Frozen reaction dataset is not the strict eligible universe")
    if not reactions["adjudication_class"].eq("STRICT_PASS").all():
        raise RuntimeError("Frozen reaction dataset contains quarantined events")
    statuses = pd.read_csv(STATUS_PATH, sep="\t", dtype=str, keep_default_na=False)
    price = build_price_context(reactions, statuses)
    memory = reactions.merge(price, on="event_id", how="left", validate="one_to_one")
    memory = add_calendar_context(memory)
    memory = add_causal_volatility_regime(memory)
    memory, macro_coverage = add_macro_context(memory)
    memory["memory_contract_version"] = MEMORY_CONTRACT
    memory["source_reaction_dataset_sha256"] = fingerprint["sha256"]
    memory = memory.sort_values(["event_timestamp_utc", "event_type", "event_id"]).reset_index(drop=True)
    validate_memory(memory)
    return memory, fingerprint, macro_coverage


def write_outputs(memory: pd.DataFrame, fingerprint: dict[str, object], macro_coverage: dict[str, float]) -> None:
    _atomic_frame(memory, OUTPUT)
    _atomic_frame(memory.head(20), SAMPLE_REPORT)
    _atomic_frame(memory_schema(memory), SCHEMA_REPORT)
    _atomic_frame(registry_frame(), REGISTRY_REPORT)
    outcomes = [column for column in memory if column.startswith("outcome_")]
    context = list(approved_feature_names())
    lines = [
        "MARKETFUSION V0.4B HISTORICAL EVENT MEMORY QUALITY",
        f"memory_contract: {MEMORY_CONTRACT}",
        f"source_reaction_sha256: {fingerprint['sha256']}",
        f"memory_records: {len(memory)}",
        f"unique_event_ids: {memory['event_id'].nunique()}",
        f"us_cpi_release: {int(memory['event_type'].eq('us_cpi_release').sum())}",
        f"us_employment_situation: {int(memory['event_type'].eq('us_employment_situation').sum())}",
        f"retrieval_context_features: {len(context)}",
        f"historical_outcome_columns: {len(outcomes)}",
        "quarantined_events: 0",
        "non_pass_events: 0",
        "",
        "PRE-EVENT PRICE CONTEXT",
        "- exact Dukascopy [T-10m,T) ticks only: 5m/10m returns, 10m volatility/range/spread",
        "- 15m/30m/60m/4h/1d omitted: the frozen reaction cache does not contain those pre-T lookbacks",
        "- no interpolation, forward fill, cross-broker splice, or post-T tick use",
        "",
        "CAUSAL REGIMES",
        "- volatility terciles use only strictly earlier same-event-type contexts",
        "- first ten events per type are UNKNOWN/optional, never fitted with future events",
        "",
        "POINT-IN-TIME MACRO COVERAGE",
    ]
    lines.extend(f"- {series}: {coverage:.1%}" for series, coverage in macro_coverage.items())
    lines.extend(
        [
            "",
            "LEAKAGE / OUTCOME ISOLATION",
            "- retrieval uses an explicit context-only allowlist",
            "- outcome_* fields remain attached only as historical payload after ranking",
            "- every joined macro available_from_utc is <= event_timestamp_utc",
            "- actual/previous/consensus/forecast/surprise are not populated",
            "",
            "MEMORY_QUALITY_STATUS: PASS",
            "",
        ]
    )
    _atomic_text("\n".join(lines), QUALITY_REPORT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    memory, fingerprint, macro_coverage = build_memory()
    write_outputs(memory, fingerprint, macro_coverage)
    print("V04B_MEMORY_STATUS: PASS")
    print(f"Records: {len(memory)}")
    print(f"Retrieval features: {len(FEATURES)}")
    print(f"Saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
