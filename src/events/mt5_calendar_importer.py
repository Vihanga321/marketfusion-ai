"""Fail-closed audit tools for the free MetaTrader 5 Economic Calendar path.

Historical MT5 ``forecast_value`` fields are useful research candidates, but they
are deliberately not marked point-in-time safe because a historical row does not
prove when that forecast first became available before release T.

Live snapshot exports are different: ``captured_at_gmt`` records when MarketFusion
observed the forecast. A snapshot can become point-in-time eligible only after its
event/measure identity is independently matched to the frozen MarketFusion event
and the capture timestamp is strictly earlier than that event timestamp.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

HISTORY_REQUIRED_COLUMNS = {
    "value_id", "event_id", "event_time_server", "reference_period_server", "revision",
    "actual_value", "forecast_value", "prev_value", "revised_prev_value", "impact_type",
    "event_name", "event_code", "country_code", "currency", "unit", "importance",
    "multiplier", "digits", "time_mode", "sector", "frequency", "source_url",
    "exported_at_server",
}
SNAPSHOT_REQUIRED_COLUMNS = HISTORY_REQUIRED_COLUMNS | {"captured_at_gmt"}
MEASURE_IDS = (
    "headline_cpi_yoy_sa",
    "core_cpi_yoy_sa",
    "nonfarm_payroll_change",
    "unemployment_rate",
)


def _norm(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = value.casefold().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value).strip()


def classify_candidate_measure(event_name: object, event_code: object) -> tuple[str | None, str]:
    """Return only an audit candidate mapping, never a model-eligible mapping."""
    text = f"{_norm(event_name)} {_norm(event_code)}"
    is_yoy = any(token in text for token in ("y/y", "yoy", "year over year", "year on year"))
    is_cpi = "cpi" in text or "consumer price index" in text
    if is_cpi and "core" in text and is_yoy:
        return "core_cpi_yoy_sa", "CANDIDATE_NAME_CODE_MATCH"
    if is_cpi and "core" not in text and is_yoy:
        return "headline_cpi_yoy_sa", "CANDIDATE_NAME_CODE_MATCH"
    if ("nonfarm" in text or "non farm" in text) and "payroll" in text:
        return "nonfarm_payroll_change", "CANDIDATE_NAME_CODE_MATCH"
    if "unemployment rate" in text:
        return "unemployment_rate", "CANDIDATE_NAME_CODE_MATCH"
    return None, "UNMAPPED"


def historical_consensus_status(forecast_value: object) -> str:
    return (
        "UNAVAILABLE"
        if pd.isna(forecast_value)
        else "HISTORICAL_FORECAST_PRESENT_UNVERIFIED_SNAPSHOT_TIME"
    )


def strictly_pre_release(captured_at_utc: object, event_timestamp_utc: object) -> bool:
    captured = pd.Timestamp(captured_at_utc)
    event = pd.Timestamp(event_timestamp_utc)
    if captured.tzinfo is None or event.tzinfo is None:
        raise ValueError("capture and event timestamps must be timezone-aware")
    return captured.tz_convert("UTC") < event.tz_convert("UTC")


def _read_tsv(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"MT5 calendar export missing columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError("MT5 calendar export is empty")
    for column in ("value_id", "event_id", "revision", "digits"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("actual_value", "forecast_value", "prev_value", "revised_prev_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("event_time_server", "reference_period_server", "exported_at_server"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if "captured_at_gmt" in frame:
        frame["captured_at_gmt"] = pd.to_datetime(frame["captured_at_gmt"], utc=True, errors="raise")
    if not frame["country_code"].fillna("").eq("US").all():
        raise RuntimeError("Calendar export contains non-US rows")
    if not frame["currency"].fillna("").eq("USD").all():
        raise RuntimeError("Calendar export contains non-USD rows")
    return frame


def read_mt5_calendar_history_tsv(path: Path) -> pd.DataFrame:
    frame = _read_tsv(path, HISTORY_REQUIRED_COLUMNS)
    if frame["value_id"].duplicated().any():
        dupes = frame.loc[frame["value_id"].duplicated(keep=False), "value_id"].tolist()
        raise RuntimeError(f"Duplicate historical MT5 calendar value_id rows: {dupes[:10]}")
    return frame


def read_mt5_calendar_snapshot_tsv(path: Path) -> pd.DataFrame:
    # The same value/event may appear in many snapshots, so duplicates are expected.
    return _read_tsv(path, SNAPSHOT_REQUIRED_COLUMNS)


def _add_candidate_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mappings = [
        classify_candidate_measure(name, code)
        for name, code in zip(result["event_name"], result["event_code"])
    ]
    result["candidate_measure_id"] = [item[0] for item in mappings]
    result["mapping_status"] = [item[1] for item in mappings]
    return result


def build_historical_audit(frame: pd.DataFrame) -> pd.DataFrame:
    result = _add_candidate_mapping(frame)
    result["consensus_status"] = [
        historical_consensus_status(value) for value in result["forecast_value"]
    ]
    result["consensus_snapshot_timestamp_utc"] = pd.NaT
    result["consensus_available_from_utc"] = pd.NaT
    result["point_in_time_verified"] = False
    result["model_eligible_consensus"] = False
    result["server_time_conversion_status"] = "UNVERIFIED_BROKER_SERVER_TIME"
    return result


def build_snapshot_audit(frame: pd.DataFrame) -> pd.DataFrame:
    result = _add_candidate_mapping(frame)
    result["consensus_snapshot_timestamp_utc"] = result["captured_at_gmt"]
    result["consensus_available_from_utc"] = result["captured_at_gmt"]
    result["snapshot_capture_timestamp_verified"] = True
    result["event_identity_verified"] = False
    result["point_in_time_verified"] = False
    result["model_eligible_consensus"] = False
    result["consensus_status"] = result["forecast_value"].apply(
        lambda value: "SNAPSHOT_FORECAST_PRESENT_IDENTITY_UNVERIFIED"
        if pd.notna(value)
        else "UNAVAILABLE"
    )
    return result


def quality_lines(history: pd.DataFrame | None, snapshots: pd.DataFrame | None) -> list[str]:
    lines = ["MARKETFUSION V0.4D FREE MT5 ECONOMIC CALENDAR AUDIT"]
    if history is not None:
        candidates = history["candidate_measure_id"].notna()
        lines.extend([
            f"historical_rows: {len(history)}",
            f"historical_candidate_measure_rows: {int(candidates.sum())}",
            f"historical_forecast_present_rows: {int(history['forecast_value'].notna().sum())}",
            "historical_point_in_time_verified_rows: 0",
            "historical_model_eligible_consensus_rows: 0",
        ])
    if snapshots is not None:
        lines.extend([
            f"snapshot_rows: {len(snapshots)}",
            f"snapshot_forecast_present_rows: {int(snapshots['forecast_value'].notna().sum())}",
            f"snapshot_capture_timestamp_verified_rows: {int(snapshots['snapshot_capture_timestamp_verified'].sum())}",
            "snapshot_event_identity_verified_rows: 0",
            "snapshot_model_eligible_consensus_rows: 0",
        ])
    lines.extend([
        "policy_historical: FORECAST_RESEARCH_ONLY_UNTIL_PRE_RELEASE_SNAPSHOT_TIME_IS_PROVEN",
        "policy_snapshots: CAPTURE_TIME_PROVEN_BUT_EVENT_IDENTITY_MUST_BE_AUDITED_BEFORE_ELIGIBILITY",
        "policy_server_time: KEEP_BROKER_SERVER_TIME_UNCONVERTED_UNTIL_EVENT_DAY_OFFSET_IS_AUDITED",
        "V04D_FREE_MT5_CALENDAR_STATUS: PASS_FAIL_CLOSED",
        "",
    ])
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--history-audit", type=Path, default=Path("data/mt5/calendar/v04d_history_audit.csv"))
    parser.add_argument("--snapshot-audit", type=Path, default=Path("data/mt5/calendar/v04d_snapshot_audit.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("reports/v04d_mt5_calendar_quality.txt"))
    args = parser.parse_args()
    if args.history is None and args.snapshots is None:
        parser.error("provide --history and/or --snapshots")

    history_audit = None
    snapshot_audit = None
    if args.history is not None:
        history_audit = build_historical_audit(read_mt5_calendar_history_tsv(args.history))
        args.history_audit.parent.mkdir(parents=True, exist_ok=True)
        history_audit.to_csv(args.history_audit, index=False, lineterminator="\n")
    if args.snapshots is not None:
        snapshot_audit = build_snapshot_audit(read_mt5_calendar_snapshot_tsv(args.snapshots))
        args.snapshot_audit.parent.mkdir(parents=True, exist_ok=True)
        snapshot_audit.to_csv(args.snapshot_audit, index=False, lineterminator="\n")

    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.quality_report.write_text(
        "\n".join(quality_lines(history_audit, snapshot_audit)), encoding="utf-8", newline="\n"
    )
    print("V04D_FREE_MT5_CALENDAR_AUDIT: PASS_FAIL_CLOSED")
    if history_audit is not None:
        print(f"Historical rows: {len(history_audit)}")
    if snapshot_audit is not None:
        print(f"Snapshot rows: {len(snapshot_audit)}")
    print("Model-eligible consensus: 0 (identity/PIT approval intentionally pending)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
