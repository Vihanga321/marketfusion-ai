"""Build leakage-safe V0.4A event reactions from validated tick-rebuilt M1 windows."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from v04a_contract import (
        EXPECTED_ELIGIBLE_EVENTS,
        FLAT_TOLERANCE_PIPS,
        HORIZON_BAR_OFFSETS,
        PIP_SIZE,
        REACTION_CONTRACT_VERSION,
        REACTION_SOURCE_ID,
        ROOT,
        WINDOW_AFTER_MINUTES,
        WINDOW_BEFORE_MINUTES,
        sha256_file,
        verify_frozen_evidence,
    )
except ImportError:  # pragma: no cover
    from src.events.v04a_contract import (
        EXPECTED_ELIGIBLE_EVENTS,
        FLAT_TOLERANCE_PIPS,
        HORIZON_BAR_OFFSETS,
        PIP_SIZE,
        REACTION_CONTRACT_VERSION,
        REACTION_SOURCE_ID,
        ROOT,
        WINDOW_AFTER_MINUTES,
        WINDOW_BEFORE_MINUTES,
        sha256_file,
        verify_frozen_evidence,
    )


ELIGIBLE = ROOT / "data" / "processed" / "v04a_eligible_event_manifest.parquet"
STATUS = ROOT / "data" / "dukascopy" / "v04a_reaction" / "reaction_extraction_status.tsv"
REACTIONS = ROOT / "data" / "processed" / "v04a_event_reactions.parquet"
REACTIONS_LONG = ROOT / "data" / "processed" / "v04a_event_reactions_long.parquet"
SAMPLE = ROOT / "reports" / "v04a_event_reaction_sample.csv"
COVERAGE = ROOT / "reports" / "v04a_event_reaction_coverage.csv"
QUALITY = ROOT / "reports" / "v04a_event_reaction_quality.txt"
MEMORY_SCHEMA = ROOT / "reports" / "v04a_historical_event_memory_schema.csv"

BASE_WINDOW_COLUMNS = [
    "event_id", "event_timestamp_utc", "minute_utc",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
    "mid_open", "mid_high", "mid_low", "mid_close",
    "spread_open", "spread_high", "spread_low", "spread_close", "tick_count",
]


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


def reaction_bar_start(event_timestamp: pd.Timestamp, horizon_minutes: int) -> pd.Timestamp:
    if horizon_minutes not in HORIZON_BAR_OFFSETS:
        raise ValueError(f"Unsupported reaction horizon: {horizon_minutes}")
    return event_timestamp + pd.Timedelta(minutes=HORIZON_BAR_OFFSETS[horizon_minutes])


def pre_event_bar_start(event_timestamp: pd.Timestamp) -> pd.Timestamp:
    return event_timestamp - pd.Timedelta(minutes=1)


def pips(price_difference: float) -> float:
    return price_difference / PIP_SIZE


def direction(reaction_pips: float) -> str:
    if not math.isfinite(reaction_pips):
        raise ValueError("Direction requires a finite reaction")
    if reaction_pips > FLAT_TOLERANCE_PIPS:
        return "UP"
    if reaction_pips < -FLAT_TOLERANCE_PIPS:
        return "DOWN"
    return "FLAT"


def continuation_reversal(first: str, later: str) -> tuple[bool, bool]:
    allowed = {"UP", "DOWN", "FLAT"}
    if first not in allowed or later not in allowed:
        raise ValueError("Unknown direction label")
    active = first != "FLAT" and later != "FLAT"
    return active and first == later, active and first != later


def assert_no_post_event_outcomes(feature_columns: Iterable[str]) -> None:
    leaked = sorted(column for column in feature_columns if column.startswith("outcome_"))
    if leaked:
        raise ValueError(f"Post-event outcomes cannot enter the pre-event feature matrix: {leaked}")


def resolve_validated_window(status: pd.Series, event: pd.Series, root: Path = ROOT) -> Path:
    event_time = pd.Timestamp(event["event_timestamp_utc"])
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    expected_from = event_time - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES)
    expected_to = event_time + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES)
    if str(status["reaction_contract_version"]) != REACTION_CONTRACT_VERSION:
        raise ValueError("stale reaction contract in status")
    if str(status["source"]) != REACTION_SOURCE_ID:
        raise ValueError("unexpected reaction market-data source")
    if pd.Timestamp(status["event_timestamp_utc"]) != event_time:
        raise ValueError("status event timestamp mismatch")
    if pd.Timestamp(status["window_start_utc"]) != expected_from or pd.Timestamp(status["window_end_utc"]) != expected_to:
        raise ValueError("status window bounds mismatch")
    if int(status["minute_count"]) != 261:
        raise ValueError("status does not attest complete 261-minute coverage")
    path = (root / str(status["window_file"])).resolve()
    allowed_root = (root / "data" / "dukascopy" / "v04a_reaction" / "windows").resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError("window path escapes the reaction cache") from error
    if not path.is_file():
        raise ValueError("fingerprinted reaction window file is missing")
    if sha256_file(path).lower() != str(status["data_sha256"]).lower():
        raise ValueError("reaction window SHA-256 mismatch")
    return path


def validate_window(window: pd.DataFrame, event_id: str, event_time: pd.Timestamp) -> pd.DataFrame:
    missing = sorted(set(BASE_WINDOW_COLUMNS).difference(window.columns))
    if missing:
        raise ValueError(f"window lacks columns: {missing}")
    frame = window[BASE_WINDOW_COLUMNS].copy()
    if len(frame) != WINDOW_BEFORE_MINUTES + WINDOW_AFTER_MINUTES + 1:
        raise ValueError(f"minute count is {len(frame)}, expected 261")
    if frame["event_id"].nunique() != 1 or frame["event_id"].iloc[0] != event_id:
        raise ValueError("window event_id mismatch")
    frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="raise")
    frame["minute_utc"] = pd.to_datetime(frame["minute_utc"], utc=True, errors="raise")
    if not frame["event_timestamp_utc"].eq(event_time).all():
        raise ValueError("window event timestamp mismatch")
    if frame["minute_utc"].duplicated().any() or not frame["minute_utc"].is_monotonic_increasing:
        raise ValueError("window minutes must be unique and sorted")
    expected = pd.date_range(
        event_time - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES),
        event_time + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES),
        freq="min",
        tz="UTC",
    )
    if not frame["minute_utc"].reset_index(drop=True).equals(pd.Series(expected, name="minute_utc")):
        raise ValueError("window has a missing, extra, or unexpected minute; interpolation is forbidden")

    numeric = [column for column in BASE_WINDOW_COLUMNS if column not in {"event_id", "event_timestamp_utc", "minute_utc"}]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    values = frame[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("window contains NaN or infinite market values")
    price_columns = [column for column in numeric if column.startswith(("bid_", "ask_", "mid_"))]
    if (frame[price_columns] < 0.0).any().any() or (frame["tick_count"] <= 0).any():
        raise ValueError("window contains negative market values or a non-positive tick count")
    for side in ("bid", "ask", "mid"):
        side_columns = [f"{side}_open", f"{side}_high", f"{side}_low", f"{side}_close"]
        if (frame[side_columns] <= 0).any().any():
            raise ValueError(f"{side} contains non-positive prices")
        if (frame[f"{side}_high"] < frame[[f"{side}_open", f"{side}_close", f"{side}_low"]].max(axis=1)).any():
            raise ValueError(f"{side} high does not enclose OHLC")
        if (frame[f"{side}_low"] > frame[[f"{side}_open", f"{side}_close", f"{side}_high"]].min(axis=1)).any():
            raise ValueError(f"{side} low does not enclose OHLC")
    if (frame["ask_open"] < frame["bid_open"]).any() or (frame["ask_close"] < frame["bid_close"]).any():
        raise ValueError("negative spread detected at open/close")
    if (frame[["spread_open", "spread_high", "spread_low", "spread_close"]] < 0).any().any():
        raise ValueError("negative reconstructed spread")
    if not np.allclose(
        frame["mid_open"], (frame["bid_open"] + frame["ask_open"]) / 2.0, rtol=0.0, atol=5.0e-10
    ) or not np.allclose(
        frame["mid_close"], (frame["bid_close"] + frame["ask_close"]) / 2.0, rtol=0.0, atol=5.0e-10
    ):
        raise ValueError("MID open/close is inconsistent with actual BID/ASK ticks")
    if not np.allclose(
        frame["spread_open"], frame["ask_open"] - frame["bid_open"], rtol=0.0, atol=5.0e-10
    ) or not np.allclose(
        frame["spread_close"], frame["ask_close"] - frame["bid_close"], rtol=0.0, atol=5.0e-10
    ):
        raise ValueError("spread open/close is inconsistent with actual BID/ASK ticks")
    if (frame["spread_high"] < frame[["spread_open", "spread_close", "spread_low"]].max(axis=1)).any():
        raise ValueError("spread high does not enclose spread values")
    if (frame["spread_low"] > frame[["spread_open", "spread_close", "spread_high"]].min(axis=1)).any():
        raise ValueError("spread low does not enclose spread values")
    return frame


def extract_event(window: pd.DataFrame, event: pd.Series) -> tuple[dict[str, object], list[dict[str, object]]]:
    event_time = pd.Timestamp(event["event_timestamp_utc"])
    if event_time.tzinfo is None:
        event_time = event_time.tz_localize("UTC")
    else:
        event_time = event_time.tz_convert("UTC")
    if event_time.second or event_time.microsecond or event_time.nanosecond:
        raise ValueError("event timestamp is not an exact UTC minute")
    frame = validate_window(window, str(event["event_id"]), event_time).set_index("minute_utc")
    pre_time = pre_event_bar_start(event_time)
    pre = frame.loc[pre_time]
    pre_mid = float(pre["mid_close"])
    pre_spread_pips = pips(float(pre["spread_close"]))

    row: dict[str, object] = {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "event_timestamp_utc": event_time,
        "reference_period": event["reference_period"],
        "timestamp_precision": event["timestamp_precision"],
        "timestamp_provenance": event["timestamp_source"],
        "market_data_source": event["market_data_source"],
        "production_status": event["production_status"],
        "adjudication_class": event["adjudication_class"],
        "model_eligible_market_reaction": bool(event["model_eligible_market_reaction"]),
        "reaction_contract_version": event["reaction_contract_version"],
        "reaction_extraction_status": "COMPLETE",
        "context_pre_bar_timestamp_utc": pre_time,
        "context_pre_mid_close": pre_mid,
        "context_pre_event_spread_pips": pre_spread_pips,
    }
    long_rows: list[dict[str, object]] = []
    directions: dict[int, str] = {}
    for horizon, offset in HORIZON_BAR_OFFSETS.items():
        horizon_time = reaction_bar_start(event_time, horizon)
        horizon_bar = frame.loc[horizon_time]
        post_path = frame.loc[event_time:horizon_time]
        horizon_mid = float(horizon_bar["mid_close"])
        reaction = pips(horizon_mid - pre_mid)
        label = direction(reaction)
        directions[horizon] = label
        maximum_up = max(0.0, pips(float(post_path["mid_high"].max()) - pre_mid))
        maximum_down = min(0.0, pips(float(post_path["mid_low"].min()) - pre_mid))
        path_range = pips(float(post_path["mid_high"].max() - post_path["mid_low"].min()))
        spread_close = pips(float(horizon_bar["spread_close"]))
        max_spread = pips(float(post_path["spread_high"].max()))
        prefix = f"outcome_"
        row[f"{prefix}bar_timestamp_{horizon}m_utc"] = horizon_time
        row[f"{prefix}mid_close_{horizon}m"] = horizon_mid
        row[f"{prefix}reaction_{horizon}m_pips"] = reaction
        row[f"{prefix}abs_reaction_{horizon}m_pips"] = abs(reaction)
        row[f"{prefix}direction_{horizon}m"] = label
        row[f"{prefix}spread_{horizon}m_pips"] = spread_close
        row[f"{prefix}max_spread_{horizon}m_pips"] = max_spread
        row[f"{prefix}max_up_{horizon}m_pips"] = maximum_up
        row[f"{prefix}max_down_{horizon}m_pips"] = maximum_down
        row[f"{prefix}range_{horizon}m_pips"] = path_range
        if horizon in {1, 5, 15}:
            row[f"{prefix}spread_expansion_{horizon}m_pips"] = max_spread - pre_spread_pips
        long_rows.append(
            {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "event_timestamp_utc": event_time,
                "horizon_minutes": horizon,
                "pre_mid": pre_mid,
                "horizon_mid": horizon_mid,
                "reaction_pips": reaction,
                "absolute_reaction_pips": abs(reaction),
                "direction": label,
                "spread_pips": spread_close,
                "max_spread_pips": max_spread,
                "max_up_pips": maximum_up,
                "max_down_pips": maximum_down,
                "range_pips": path_range,
                "reaction_contract_version": REACTION_CONTRACT_VERSION,
            }
        )

    for first, later in ((1, 5), (5, 15), (15, 60)):
        continued, reversed_ = continuation_reversal(directions[first], directions[later])
        row[f"outcome_continued_{first}m_to_{later}m"] = continued
        row[f"outcome_reversed_{first}m_to_{later}m"] = reversed_
    return row, long_rows


def _empty_reactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_columns = [
        "event_id", "event_type", "event_timestamp_utc", "reference_period",
        "timestamp_precision", "timestamp_provenance", "market_data_source",
        "production_status", "adjudication_class", "model_eligible_market_reaction",
        "reaction_contract_version", "reaction_extraction_status",
        "context_pre_bar_timestamp_utc", "context_pre_mid_close", "context_pre_event_spread_pips",
    ]
    for horizon in HORIZON_BAR_OFFSETS:
        wide_columns.extend(
            [
                f"outcome_bar_timestamp_{horizon}m_utc", f"outcome_mid_close_{horizon}m",
                f"outcome_reaction_{horizon}m_pips", f"outcome_abs_reaction_{horizon}m_pips",
                f"outcome_direction_{horizon}m", f"outcome_spread_{horizon}m_pips",
                f"outcome_max_spread_{horizon}m_pips", f"outcome_max_up_{horizon}m_pips",
                f"outcome_max_down_{horizon}m_pips", f"outcome_range_{horizon}m_pips",
            ]
        )
        if horizon in {1, 5, 15}:
            wide_columns.append(f"outcome_spread_expansion_{horizon}m_pips")
    for first, later in ((1, 5), (5, 15), (15, 60)):
        wide_columns.extend(
            [f"outcome_continued_{first}m_to_{later}m", f"outcome_reversed_{first}m_to_{later}m"]
        )
    long_columns = [
        "event_id", "event_type", "event_timestamp_utc", "horizon_minutes", "pre_mid",
        "horizon_mid", "reaction_pips", "absolute_reaction_pips", "direction", "spread_pips",
        "max_spread_pips", "max_up_pips", "max_down_pips", "range_pips",
        "reaction_contract_version",
    ]
    return pd.DataFrame(columns=wide_columns), pd.DataFrame(columns=long_columns)


def historical_memory_schema() -> pd.DataFrame:
    rows = [
        ("EVENT_IDENTITY", "event_id", "available_now"),
        ("EVENT_IDENTITY", "event_type", "available_now"),
        ("EVENT_IDENTITY", "event_timestamp_utc", "available_now"),
        ("EVENT_IDENTITY", "reference_period", "available_now"),
        ("PRE_EVENT_CONTEXT", "context_pre_mid_close", "available_after_market_extraction"),
        ("PRE_EVENT_CONTEXT", "context_pre_event_spread_pips", "available_after_market_extraction"),
        ("PRE_EVENT_CONTEXT", "macro_regime", "future_stage_not_populated"),
        ("PRE_EVENT_CONTEXT", "rates_regime", "future_stage_not_populated"),
        ("PRE_EVENT_CONTEXT", "price_trend", "future_stage_not_populated"),
        ("PRE_EVENT_CONTEXT", "volatility_regime", "future_stage_not_populated"),
        ("PRE_EVENT_CONTEXT", "cross_market_regime", "future_stage_not_populated"),
        ("PRE_EVENT_CONTEXT", "central_bank_context", "future_stage_not_populated"),
        ("EVENT_VALUES", "actual", "future_stage_not_populated"),
        ("EVENT_VALUES", "previous", "future_stage_not_populated"),
        ("EVENT_VALUES", "consensus", "future_stage_not_populated"),
        ("EVENT_VALUES", "surprise", "future_stage_not_populated"),
        ("MARKET_REACTION", "outcome_reaction_{1,5,15,60,240}m_pips", "available_after_market_extraction"),
        ("SPREAD_LIQUIDITY", "outcome_spread_{1,5,15,60,240}m_pips", "available_after_market_extraction"),
        ("SPREAD_LIQUIDITY", "outcome_spread_expansion_{1,5,15}m_pips", "available_after_market_extraction"),
        ("PATH_INFORMATION", "outcome_max_up/max_down/range_*m_pips", "available_after_market_extraction"),
        ("PATH_INFORMATION", "outcome_continued/reversed_*", "available_after_market_extraction"),
        ("PROVENANCE", "market_data_source", "available_now"),
        ("PROVENANCE", "timestamp_provenance", "available_now"),
        ("PROVENANCE", "production_status", "available_now"),
        ("PROVENANCE", "adjudication_class", "available_now"),
        ("PROVENANCE", "reaction_contract_version", "available_now"),
    ]
    return pd.DataFrame(rows, columns=["memory_group", "field_or_pattern", "population_status"])


def build_reactions(
    eligible_path: Path = ELIGIBLE,
    status_path: Path = STATUS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], list[str]]:
    eligible = pd.read_parquet(eligible_path)
    if len(eligible) != EXPECTED_ELIGIBLE_EVENTS:
        raise RuntimeError(f"Eligible manifest count changed: {len(eligible)}")
    if not eligible["production_status"].eq("PASS").all() or not eligible["model_eligible_market_reaction"].all():
        raise RuntimeError("Eligible manifest contains a quarantined event")
    if eligible["event_id"].duplicated().any():
        raise RuntimeError("Eligible event IDs are not unique")
    statuses = pd.DataFrame(columns=["event_id", "retrieval_status", "window_file"])
    if status_path.is_file():
        statuses = pd.read_csv(status_path, sep="\t", dtype=str, keep_default_na=False)
        required = {
            "event_id", "event_timestamp_utc", "retrieval_status", "window_file",
            "reaction_contract_version", "source", "window_start_utc", "window_end_utc",
            "minute_count", "data_sha256",
        }
        if not required.issubset(statuses.columns):
            raise RuntimeError(f"Reaction extraction status lacks columns: {sorted(required - set(statuses.columns))}")
        if statuses["event_id"].duplicated().any():
            raise RuntimeError("Reaction extraction status contains duplicate event IDs")
        if not set(statuses["event_id"]).issubset(set(eligible["event_id"])):
            raise RuntimeError("Reaction extraction status contains a quarantined or unknown event")
        allowed_statuses = {"COMPLETE", "TEMPORARILY_UNAVAILABLE", "DATA_QUALITY_FAILURE", "ERROR"}
        if not statuses["retrieval_status"].isin(allowed_statuses).all():
            raise RuntimeError("Reaction extraction status contains an unknown state")
    status_lookup = statuses.set_index("event_id", drop=False)

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    effective_status: dict[str, str] = {}
    quality_errors: list[str] = []
    for _, event in eligible.sort_values("event_timestamp_utc").iterrows():
        event_id = str(event["event_id"])
        if event_id not in status_lookup.index:
            effective_status[event_id] = "NOT_RUN"
            continue
        status = status_lookup.loc[event_id]
        retrieval_status = str(status["retrieval_status"])
        if str(status["reaction_contract_version"]) != REACTION_CONTRACT_VERSION:
            effective_status[event_id] = "DATA_QUALITY_FAILURE"
            quality_errors.append(f"{event_id}: stale reaction contract in status")
            continue
        if retrieval_status != "COMPLETE":
            effective_status[event_id] = retrieval_status
            continue
        try:
            path = resolve_validated_window(status, event)
            window = pd.read_csv(path, sep="\t")
            row, normalized = extract_event(window, event)
            wide_rows.append(row)
            long_rows.extend(normalized)
            effective_status[event_id] = "COMPLETE"
        except Exception as error:
            effective_status[event_id] = "DATA_QUALITY_FAILURE"
            quality_errors.append(f"{event_id}: {error}")

    empty_wide, empty_long = _empty_reactions()
    wide = pd.DataFrame(wide_rows) if wide_rows else empty_wide
    long = pd.DataFrame(long_rows) if long_rows else empty_long
    if not wide.empty:
        wide = wide[empty_wide.columns]
        long = long[empty_long.columns]
    if wide["event_id"].duplicated().any():
        raise RuntimeError("Reaction output event IDs are not unique")
    assert_no_post_event_outcomes(
        [column for column in wide.columns if column.startswith("context_")]
    )

    eligible_for_coverage = eligible.copy()
    eligible_for_coverage["effective_status"] = eligible_for_coverage["event_id"].map(effective_status)
    eligible_for_coverage["event_year"] = pd.to_datetime(
        eligible_for_coverage["event_timestamp_utc"], utc=True
    ).dt.year
    coverage_rows = []
    for (year, event_type), group in eligible_for_coverage.groupby(["event_year", "event_type"]):
        counts = group["effective_status"].value_counts().to_dict()
        for horizon in ("pre", *HORIZON_BAR_OFFSETS):
            coverage_rows.append(
                {
                    "year": int(year), "event_type": event_type, "horizon_minutes": horizon,
                    "eligible_expected": len(group), "complete": int(counts.get("COMPLETE", 0)),
                    "temporarily_unavailable": int(counts.get("TEMPORARILY_UNAVAILABLE", 0)),
                    "data_quality_failures": int(counts.get("DATA_QUALITY_FAILURE", 0)),
                    "errors": int(counts.get("ERROR", 0)), "not_run": int(counts.get("NOT_RUN", 0)),
                }
            )
    totals = pd.Series(effective_status).value_counts().to_dict()
    total_counts = {
        "eligible": len(eligible),
        "complete": int(totals.get("COMPLETE", 0)),
        "unavailable": int(totals.get("TEMPORARILY_UNAVAILABLE", 0)),
        "data_quality_failures": int(totals.get("DATA_QUALITY_FAILURE", 0)),
        "errors": int(totals.get("ERROR", 0)),
        "not_run": int(totals.get("NOT_RUN", 0)),
        "invalid_ticks": 0,
        "negative_spread_ticks": 0,
        "duplicate_ticks": 0,
        "missing_minutes": 0,
    }
    if not statuses.empty:
        for source_column, target in (
            ("invalid_tick_count", "invalid_ticks"),
            ("negative_spread_tick_count", "negative_spread_ticks"),
            ("duplicate_tick_count", "duplicate_ticks"),
        ):
            if source_column in statuses:
                total_counts[target] = int(pd.to_numeric(statuses[source_column], errors="raise").sum())
        if "minute_count" in statuses:
            minutes = pd.to_numeric(statuses["minute_count"], errors="raise").astype(int)
            total_counts["missing_minutes"] = int((len(statuses) * 261) - minutes.sum())
    return wide, long, pd.DataFrame(coverage_rows), total_counts, quality_errors


def write_outputs(
    wide: pd.DataFrame,
    long: pd.DataFrame,
    coverage: pd.DataFrame,
    counts: dict[str, int],
    quality_errors: list[str],
) -> None:
    _atomic_frame(wide, REACTIONS)
    _atomic_frame(long, REACTIONS_LONG)
    _atomic_frame(wide.head(20), SAMPLE)
    _atomic_frame(coverage, COVERAGE)
    _atomic_frame(historical_memory_schema(), MEMORY_SCHEMA)
    status = "PASS" if counts["complete"] == counts["eligible"] and not quality_errors else (
        "REACTION_EXTRACTION_NOT_RUN" if counts["not_run"] == counts["eligible"] else "INCOMPLETE"
    )
    lines = [
        "MARKETFUSION V0.4A EVENT REACTION QUALITY",
        f"Reaction contract: {REACTION_CONTRACT_VERSION}",
        f"Eligible strict PASS events: {counts['eligible']}",
        f"Reaction extraction complete: {counts['complete']}",
        f"Unavailable during extraction: {counts['unavailable']}",
        f"Data quality failures: {counts['data_quality_failures']}",
        f"Errors: {counts['errors']}",
        f"Not run: {counts['not_run']}",
        f"Invalid ticks: {counts['invalid_ticks']}",
        f"Negative-spread ticks: {counts['negative_spread_ticks']}",
        f"Exact duplicate ticks removed: {counts['duplicate_ticks']}",
        f"Missing reconstructed minutes: {counts['missing_minutes']}",
        "",
        "Horizon semantics: pre=T-1; +1=T; +5=T+4; +15=T+14; +60=T+59; +240=T+239.",
        "PIP_SIZE: 0.0001",
        "MID: reconstructed from each valid tick midpoint, never averaged BID/ASK OHLC.",
        "Spread: reconstructed from each actual ask-bid tick spread; no assumed spread.",
        "Missing minutes: rejected; no interpolation or forward filling.",
        "Leakage gate: context_* is pre-event; outcome_* is post-event and forbidden from feature matrices.",
        f"Exact required horizon coverage: {'PASS' if counts['complete'] == counts['eligible'] else 'INCOMPLETE'}",
        "BID/ASK/MID validation: " + (
            "PASS" if counts["complete"] == counts["eligible"] else (
                "FAIL" if counts["data_quality_failures"] else "NOT_TESTED"
            )
        ),
        "Spread validation: " + (
            "PASS" if counts["complete"] == counts["eligible"] and counts["negative_spread_ticks"] == 0 else (
                "FAIL" if counts["negative_spread_ticks"] else "NOT_TESTED"
            )
        ),
        "",
    ]
    lines.append("BY YEAR AND EVENT TYPE:")
    if coverage.empty:
        lines.append("- no eligible groups")
    else:
        grouped = coverage.drop_duplicates(["year", "event_type"])
        for item in grouped.itertuples(index=False):
            lines.append(
                f"- {item.year} {item.event_type}: eligible={item.eligible_expected}; "
                f"complete={item.complete}; unavailable={item.temporarily_unavailable}; "
                f"quality_failures={item.data_quality_failures}; errors={item.errors}; not_run={item.not_run}"
            )
    lines.append("")
    if quality_errors:
        lines.append("QUALITY ERRORS:")
        lines.extend(f"- {error}" for error in quality_errors)
        lines.append("")
    lines.append(f"REACTION_DATASET_STATUS: {status}")
    lines.append("")
    _atomic_text("\n".join(lines), QUALITY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify_frozen_evidence()
    wide, long, coverage, counts, errors = build_reactions()
    write_outputs(wide, long, coverage, counts, errors)
    if counts["not_run"] == counts["eligible"]:
        print("REACTION_EXTRACTION_NOT_RUN")
    else:
        print(f"REACTION_EXTRACTION_COMPLETE: {counts['complete']}/{counts['eligible']}")
    print(f"Counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
