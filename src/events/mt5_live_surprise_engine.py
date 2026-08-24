"""Build verified live raw surprises from MarketFusion's MT5 target snapshots.

Only a causally verified pre-release consensus row and a verified post-release actual
for the same exact MT5 release may be combined.  The earliest observed post-release
actual is used so later revisions cannot overwrite the original release payload.
This module is research/data infrastructure only; it does not place trades.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.events.mt5_calendar_identity_audit import TARGET_EVENT_IDS
from src.events.mt5_live_consensus_gate import (
    _as_mixed_datetime,
    _event_time_to_utc,
    _identity_verified,
    derive_server_offset_seconds,
    offset_is_verified,
)


def build_live_surprises(
    snapshots: pd.DataFrame,
    canonical_consensus: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_snapshot = {
        "event_id", "event_time_server", "event_name", "event_code", "country_code",
        "currency", "unit", "multiplier", "actual_value", "exported_at_server",
        "captured_at_gmt",
    }
    missing = required_snapshot - set(snapshots.columns)
    if missing:
        raise RuntimeError(f"Snapshot audit missing columns: {sorted(missing)}")

    required_consensus = {
        "event_id", "event_timestamp_utc", "event_name", "event_code", "forecast_value",
        "captured_at_gmt", "consensus_status",
    }
    missing_consensus = required_consensus - set(canonical_consensus.columns)
    if missing_consensus:
        raise RuntimeError(f"Canonical consensus missing columns: {sorted(missing_consensus)}")

    snap = snapshots.copy()
    snap["event_id"] = pd.to_numeric(snap["event_id"], errors="raise").astype("int64")
    snap["actual_value"] = pd.to_numeric(snap["actual_value"], errors="coerce")
    snap["captured_at_gmt"] = _as_mixed_datetime(snap["captured_at_gmt"], utc=True)
    snap["server_offset_seconds"] = derive_server_offset_seconds(
        snap["exported_at_server"], snap["captured_at_gmt"]
    )
    snap["server_offset_verified"] = offset_is_verified(snap["server_offset_seconds"])
    snap["event_timestamp_utc"] = _event_time_to_utc(
        snap["event_time_server"], snap["server_offset_seconds"]
    )
    snap["event_identity_verified"] = _identity_verified(snap)
    snap["post_release"] = snap["captured_at_gmt"] >= snap["event_timestamp_utc"]
    snap["verified_post_release_actual"] = (
        snap["server_offset_verified"]
        & snap["event_identity_verified"]
        & snap["post_release"]
        & snap["actual_value"].notna()
    )

    actuals = snap.loc[snap["verified_post_release_actual"]].copy()
    if not actuals.empty:
        actuals = actuals.sort_values(["event_id", "event_timestamp_utc", "captured_at_gmt"])
        actuals = actuals.groupby(["event_id", "event_timestamp_utc"], as_index=False, group_keys=False).head(1)

    consensus = canonical_consensus.copy()
    if consensus.empty:
        empty = pd.DataFrame(columns=[
            "event_id", "measure_id", "event_name", "event_timestamp_utc", "unit", "multiplier",
            "forecast_value", "forecast_captured_at_gmt", "actual_value", "actual_captured_at_gmt",
            "actual_capture_latency_seconds", "raw_surprise", "surprise_status",
        ])
        return snap, empty

    consensus["event_id"] = pd.to_numeric(consensus["event_id"], errors="raise").astype("int64")
    consensus["forecast_value"] = pd.to_numeric(consensus["forecast_value"], errors="raise")
    consensus["event_timestamp_utc"] = _as_mixed_datetime(consensus["event_timestamp_utc"], utc=True)
    consensus["captured_at_gmt"] = _as_mixed_datetime(consensus["captured_at_gmt"], utc=True)
    consensus = consensus.loc[consensus["consensus_status"].eq("VERIFIED_PRE_RELEASE_CONSENSUS")].copy()
    consensus = consensus.rename(columns={"captured_at_gmt": "forecast_captured_at_gmt"})

    if actuals.empty or consensus.empty:
        empty = pd.DataFrame(columns=[
            "event_id", "measure_id", "event_name", "event_timestamp_utc", "unit", "multiplier",
            "forecast_value", "forecast_captured_at_gmt", "actual_value", "actual_captured_at_gmt",
            "actual_capture_latency_seconds", "raw_surprise", "surprise_status",
        ])
        return snap, empty

    actuals = actuals.rename(columns={"captured_at_gmt": "actual_captured_at_gmt"})
    merged = consensus.merge(
        actuals[[
            "event_id", "event_timestamp_utc", "event_name", "event_code", "unit", "multiplier",
            "actual_value", "actual_captured_at_gmt",
        ]],
        on=["event_id", "event_timestamp_utc", "event_name", "event_code"],
        how="inner",
        validate="one_to_one",
    )

    merged["measure_id"] = merged["event_id"].map(
        {event_id: meta["measure_id"] for event_id, meta in TARGET_EVENT_IDS.items()}
    )
    merged["actual_capture_latency_seconds"] = (
        merged["actual_captured_at_gmt"] - merged["event_timestamp_utc"]
    ).dt.total_seconds()
    merged["raw_surprise"] = merged["actual_value"] - merged["forecast_value"]
    merged["surprise_status"] = "VERIFIED_LIVE_RAW_SURPRISE"

    surprises = merged[[
        "event_id", "measure_id", "event_name", "event_timestamp_utc", "unit", "multiplier",
        "forecast_value", "forecast_captured_at_gmt", "actual_value", "actual_captured_at_gmt",
        "actual_capture_latency_seconds", "raw_surprise", "surprise_status",
    ]].sort_values(["event_timestamp_utc", "event_id"]).reset_index(drop=True)
    return snap, surprises


def quality_report(gated_snapshots: pd.DataFrame, surprises: pd.DataFrame) -> str:
    lines = [
        "MARKETFUSION V0.4D LIVE MT5 SURPRISE ENGINE",
        f"snapshot_rows: {len(gated_snapshots)}",
        f"verified_post_release_actual_rows: {int(gated_snapshots['verified_post_release_actual'].sum())}",
        f"verified_live_surprise_rows: {len(surprises)}",
        "consensus_rule: latest causally verified pre-release forecast from canonical consensus gate",
        "actual_rule: earliest causally verified post-release MT5 actual for the exact same release",
        "revision_policy: later actual revisions never overwrite the first observed release actual",
        "normalization_policy: raw surprises only; no z-score/bucket activation until sufficient audited sample",
        "trading_policy: research/data infrastructure only; no order execution",
        "V04D_LIVE_MT5_SURPRISE_STATUS: PASS_FAIL_CLOSED",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-audit", type=Path, default=Path("data/mt5/calendar/v04d_snapshot_audit.csv"))
    parser.add_argument("--canonical-consensus", type=Path, default=Path("data/mt5/calendar/v04d_live_verified_consensus.csv"))
    parser.add_argument("--snapshot-gate-output", type=Path, default=Path("data/mt5/calendar/v04d_live_actual_gate.csv"))
    parser.add_argument("--surprise-output", type=Path, default=Path("data/mt5/calendar/v04d_live_verified_surprises.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("reports/v04d_live_mt5_surprise_engine.txt"))
    args = parser.parse_args()

    snapshots = pd.read_csv(args.snapshot_audit)
    canonical = pd.read_csv(args.canonical_consensus)
    gated, surprises = build_live_surprises(snapshots, canonical)

    for path in (args.snapshot_gate_output, args.surprise_output, args.quality_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    gated.to_csv(args.snapshot_gate_output, index=False, lineterminator="\n")
    surprises.to_csv(args.surprise_output, index=False, lineterminator="\n")
    text = quality_report(gated, surprises)
    args.quality_report.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    if len(surprises):
        print(surprises.to_string(index=False))
    else:
        print("No verified live surprise yet; waiting for a target release actual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
