"""Verify live MT5 calendar snapshots for MarketFusion V0.4D.

This gate uses the simultaneously captured MT5 trade-server wall clock and GMT clock
to derive the server offset for each snapshot.  It then converts event_time_server to
UTC, accepts only the four exact audited MT5 event identities, and requires the
forecast to have been captured strictly before the release timestamp.

Historical MT5 forecast rows are deliberately outside this gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.events.mt5_calendar_identity_audit import TARGET_EVENT_IDS


def _as_mixed_datetime(series: pd.Series, *, utc: bool = False) -> pd.Series:
    return pd.to_datetime(series, errors="raise", format="mixed", utc=utc)


def derive_server_offset_seconds(exported_at_server: pd.Series, captured_at_gmt: pd.Series) -> pd.Series:
    server = _as_mixed_datetime(exported_at_server, utc=False)
    captured = _as_mixed_datetime(captured_at_gmt, utc=True).dt.tz_convert(None)
    return (server - captured).dt.total_seconds()


def offset_is_verified(offset_seconds: pd.Series) -> pd.Series:
    # TimeTradeServer()/TimeGMT() are second-resolution values sampled back-to-back.
    # Permit at most 5 seconds of sampling skew around a whole-minute UTC offset.
    nearest_minute = (offset_seconds / 60.0).round() * 60.0
    residual_ok = (offset_seconds - nearest_minute).abs().le(5.0)
    range_ok = offset_seconds.abs().le(14 * 3600)
    return residual_ok & range_ok


def _event_time_to_utc(event_time_server: pd.Series, offset_seconds: pd.Series) -> pd.Series:
    server = _as_mixed_datetime(event_time_server, utc=False)
    utc_naive = server - pd.to_timedelta(offset_seconds, unit="s")
    return utc_naive.dt.tz_localize("UTC")


def _identity_verified(frame: pd.DataFrame) -> pd.Series:
    verified = pd.Series(False, index=frame.index, dtype=bool)
    numeric_ids = pd.to_numeric(frame["event_id"], errors="raise").astype("int64")
    for event_id, expected in TARGET_EVENT_IDS.items():
        mask = numeric_ids.eq(event_id)
        mask &= frame["event_code"].fillna("").eq(expected["event_code"])
        mask &= frame["event_name"].fillna("").eq(expected["event_name"])
        mask &= frame["country_code"].fillna("").eq("US")
        mask &= frame["currency"].fillna("").eq("USD")
        verified |= mask
    return verified


def build_gate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "value_id", "event_id", "event_time_server", "event_name", "event_code",
        "country_code", "currency", "forecast_value", "exported_at_server", "captured_at_gmt",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Snapshot audit missing columns: {sorted(missing)}")

    result = frame.copy()
    result["event_id"] = pd.to_numeric(result["event_id"], errors="raise").astype("int64")
    result["forecast_value"] = pd.to_numeric(result["forecast_value"], errors="coerce")
    result["captured_at_gmt"] = _as_mixed_datetime(result["captured_at_gmt"], utc=True)
    result["server_offset_seconds"] = derive_server_offset_seconds(
        result["exported_at_server"], result["captured_at_gmt"]
    )
    result["server_offset_verified"] = offset_is_verified(result["server_offset_seconds"])
    result["event_timestamp_utc"] = _event_time_to_utc(
        result["event_time_server"], result["server_offset_seconds"]
    )
    result["event_identity_verified"] = _identity_verified(result)
    result["strictly_pre_release"] = result["captured_at_gmt"] < result["event_timestamp_utc"]
    result["point_in_time_verified"] = (
        result["server_offset_verified"]
        & result["event_identity_verified"]
        & result["strictly_pre_release"]
    )
    result["model_eligible_consensus"] = result["point_in_time_verified"] & result["forecast_value"].notna()

    def status(row: pd.Series) -> str:
        if not bool(row["event_identity_verified"]):
            return "NON_TARGET_OR_IDENTITY_UNVERIFIED"
        if not bool(row["server_offset_verified"]):
            return "SERVER_OFFSET_UNVERIFIED"
        if not bool(row["strictly_pre_release"]):
            return "NOT_STRICTLY_PRE_RELEASE"
        if pd.isna(row["forecast_value"]):
            return "FORECAST_UNAVAILABLE"
        return "VERIFIED_PRE_RELEASE_CONSENSUS"

    result["consensus_status"] = result.apply(status, axis=1)

    eligible = result.loc[result["model_eligible_consensus"]].copy()
    if eligible.empty:
        canonical = pd.DataFrame(columns=[
            "event_id", "event_timestamp_utc", "event_name", "event_code", "forecast_value",
            "captured_at_gmt", "server_offset_seconds", "consensus_status",
        ])
    else:
        # Latest verified snapshot strictly before each concrete release wins.
        eligible = eligible.sort_values(["event_id", "event_timestamp_utc", "captured_at_gmt"])
        canonical = eligible.groupby(["event_id", "event_timestamp_utc"], as_index=False, group_keys=False).tail(1)
        canonical = canonical[[
            "event_id", "event_timestamp_utc", "event_name", "event_code", "forecast_value",
            "captured_at_gmt", "server_offset_seconds", "consensus_status",
        ]].reset_index(drop=True)
    return result, canonical


def quality_report(gated: pd.DataFrame, canonical: pd.DataFrame) -> str:
    target = gated["event_identity_verified"]
    offsets = sorted({round(float(v) / 3600.0, 6) for v in gated.loc[gated["server_offset_verified"], "server_offset_seconds"]})
    lines = [
        "MARKETFUSION V0.4D LIVE MT5 CONSENSUS GATE",
        f"snapshot_rows: {len(gated)}",
        f"server_offset_verified_rows: {int(gated['server_offset_verified'].sum())}",
        f"verified_server_offsets_hours: {offsets}",
        f"exact_target_identity_rows: {int(target.sum())}",
        f"target_forecast_present_rows: {int((target & gated['forecast_value'].notna()).sum())}",
        f"strict_pre_release_target_rows: {int((target & gated['strictly_pre_release']).sum())}",
        f"model_eligible_consensus_snapshot_rows: {int(gated['model_eligible_consensus'].sum())}",
        f"canonical_verified_consensus_rows: {len(canonical)}",
        "eligibility_rule: exact audited MT5 identity + verified server/GMT offset + capture strictly before release + forecast present",
        "historical_policy: historical MT5 forecasts remain non-model-eligible",
        "V04D_LIVE_MT5_CONSENSUS_GATE_STATUS: PASS_FAIL_CLOSED",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-audit", type=Path, default=Path("data/mt5/calendar/v04d_snapshot_audit.csv"))
    parser.add_argument("--gated-output", type=Path, default=Path("data/mt5/calendar/v04d_live_consensus_gate.csv"))
    parser.add_argument("--canonical-output", type=Path, default=Path("data/mt5/calendar/v04d_live_verified_consensus.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("reports/v04d_live_mt5_consensus_gate.txt"))
    args = parser.parse_args()

    frame = pd.read_csv(args.snapshot_audit)
    gated, canonical = build_gate(frame)
    for path in (args.gated_output, args.canonical_output, args.quality_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    gated.to_csv(args.gated_output, index=False, lineterminator="\n")
    canonical.to_csv(args.canonical_output, index=False, lineterminator="\n")
    text = quality_report(gated, canonical)
    args.quality_report.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
