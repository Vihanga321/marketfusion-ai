"""Audit stable MetaTrader 5 Economic Calendar identities for MarketFusion V0.4D.

This stage is diagnostic and fail-closed.  It narrows the broad name-based candidate
set to the four exact US calendar events observed in the user's 2015-2026 MetaQuotes
history export.  It deliberately does *not* make historical forecasts model-eligible:
we still lack proof of each historical forecast's pre-release snapshot time, and MT5
trade-server timestamps still require an independent UTC offset audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_EVENT_IDS: dict[int, dict[str, str]] = {
    840030007: {"measure_id": "headline_cpi_yoy_sa", "event_code": "consumer-price-index-yy", "event_name": "CPI y/y"},
    840030008: {"measure_id": "core_cpi_yoy_sa", "event_code": "consumer-price-index-ex-food-energy-yy", "event_name": "Core CPI y/y"},
    840030015: {"measure_id": "unemployment_rate", "event_code": "unemployment-rate", "event_name": "Unemployment Rate"},
    840030016: {"measure_id": "nonfarm_payroll_change", "event_code": "nonfarm-payrolls", "event_name": "Nonfarm Payrolls"},
}

REJECTED_VARIANT_EVENT_IDS: dict[int, str] = {
    840030024: "U6 unemployment rate is not U-3 unemployment rate",
    840030023: "Private Nonfarm Payrolls is not Total Nonfarm Payrolls",
}

REQUIRED_COLUMNS = {
    "event_id", "event_code", "event_name", "event_time_server",
    "reference_period_server", "actual_value", "forecast_value", "prev_value",
    "revised_prev_value", "country_code", "currency", "unit", "importance",
    "multiplier", "digits", "time_mode", "frequency",
}


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"event_id": "Int64"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"V0.4D history audit is missing columns: {sorted(missing)}")
    frame["event_id"] = pd.to_numeric(frame["event_id"], errors="raise").astype("int64")
    for column in ("actual_value", "forecast_value", "prev_value", "revised_prev_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("event_time_server", "reference_period_server"):
        # Export/audit CSVs can contain both date-only strings and full timestamps.
        # Parse each value independently so pandas does not lock onto the first format.
        frame[column] = pd.to_datetime(frame[column], errors="raise", format="mixed")
    return frame


def audit_identity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for event_id, expected in TARGET_EVENT_IDS.items():
        subset = frame.loc[frame["event_id"].eq(event_id)].copy()
        codes = sorted(subset["event_code"].dropna().astype(str).unique().tolist())
        names = sorted(subset["event_name"].dropna().astype(str).unique().tolist())
        units = sorted(subset["unit"].dropna().astype(str).unique().tolist())
        multipliers = sorted(subset["multiplier"].dropna().astype(str).unique().tolist())
        frequencies = sorted(subset["frequency"].dropna().astype(str).unique().tolist())
        countries = sorted(subset["country_code"].dropna().astype(str).unique().tolist())
        currencies = sorted(subset["currency"].dropna().astype(str).unique().tolist())
        exact_code = codes == [expected["event_code"]]
        exact_name = names == [expected["event_name"]]
        us_usd = countries == ["US"] and currencies == ["USD"]
        stable_metadata = bool(exact_code and exact_name and us_usd and len(units) == 1 and len(frequencies) == 1)
        rows.append({
            "event_id": event_id,
            "measure_id": expected["measure_id"],
            "rows": len(subset),
            "actual_rows": int(subset["actual_value"].notna().sum()),
            "forecast_rows": int(subset["forecast_value"].notna().sum()),
            "previous_rows": int(subset["prev_value"].notna().sum()),
            "revised_previous_rows": int(subset["revised_prev_value"].notna().sum()),
            "min_event_time_server": subset["event_time_server"].min() if len(subset) else pd.NaT,
            "max_event_time_server": subset["event_time_server"].max() if len(subset) else pd.NaT,
            "event_codes": " | ".join(codes),
            "event_names": " | ".join(names),
            "units": " | ".join(units),
            "multipliers": " | ".join(multipliers),
            "frequencies": " | ".join(frequencies),
            "countries": " | ".join(countries),
            "currencies": " | ".join(currencies),
            "exact_code": exact_code,
            "exact_name": exact_name,
            "us_usd": us_usd,
            "stable_metadata": stable_metadata,
            "identity_status": "EXACT_EVENT_ID_CANDIDATE" if stable_metadata and len(subset) else "IDENTITY_REVIEW_REQUIRED",
            "historical_pit_verified": False,
            "model_eligible_consensus": False,
        })

    rejected_rows: list[dict[str, object]] = []
    for event_id, reason in REJECTED_VARIANT_EVENT_IDS.items():
        subset = frame.loc[frame["event_id"].eq(event_id)]
        rejected_rows.append({
            "event_id": event_id,
            "rows": len(subset),
            "event_codes": " | ".join(sorted(subset["event_code"].dropna().astype(str).unique().tolist())),
            "event_names": " | ".join(sorted(subset["event_name"].dropna().astype(str).unique().tolist())),
            "rejection_reason": reason,
            "model_eligible": False,
        })
    return pd.DataFrame(rows), pd.DataFrame(rejected_rows)


def _month_key(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, utc=True, errors="raise", format="mixed")
    return values.dt.strftime("%Y-%m")


def compare_with_v04c(frame: pd.DataFrame, v04c_path: Path) -> pd.DataFrame:
    """Diagnostic actual/reference-period comparison without using MT5 server time as UTC."""
    if not v04c_path.is_file():
        return pd.DataFrame()
    v04c = pd.read_parquet(v04c_path, engine="pyarrow")
    required = {"measure_id", "reference_period", "actual_value"}
    if not required.issubset(v04c.columns):
        raise RuntimeError(f"V0.4C event values missing: {sorted(required - set(v04c.columns))}")
    v04c = v04c[list(required)].copy()
    v04c["reference_month"] = _month_key(v04c["reference_period"])
    v04c["actual_value"] = pd.to_numeric(v04c["actual_value"], errors="raise")

    comparison_rows: list[pd.DataFrame] = []
    for event_id, expected in TARGET_EVENT_IDS.items():
        mt5 = frame.loc[frame["event_id"].eq(event_id), ["reference_period_server", "actual_value"]].copy()
        mt5["measure_id"] = expected["measure_id"]
        mt5["reference_month"] = mt5["reference_period_server"].dt.strftime("%Y-%m")
        mt5 = mt5.rename(columns={"actual_value": "mt5_actual_value"})
        rhs = v04c.loc[v04c["measure_id"].eq(expected["measure_id"]), ["reference_month", "actual_value"]].copy()
        rhs = rhs.rename(columns={"actual_value": "v04c_actual_value"})
        merged = mt5.merge(rhs, on="reference_month", how="inner")
        if merged.empty:
            continue
        merged["event_id"] = event_id
        merged["actual_difference"] = merged["mt5_actual_value"] - merged["v04c_actual_value"]
        comparison_rows.append(merged)
    if not comparison_rows:
        return pd.DataFrame()
    return pd.concat(comparison_rows, ignore_index=True)


def quality_report(identity: pd.DataFrame, rejected: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "MARKETFUSION V0.4D MT5 EVENT IDENTITY AUDIT",
        f"target_event_ids: {len(identity)}",
        f"target_rows: {int(identity['rows'].sum())}",
        f"target_forecast_rows: {int(identity['forecast_rows'].sum())}",
        f"stable_exact_identity_candidates: {int(identity['stable_metadata'].sum())}/{len(identity)}",
        f"rejected_variant_ids: {len(rejected)}",
        f"rejected_variant_rows: {int(rejected['rows'].sum()) if len(rejected) else 0}",
        "historical_point_in_time_verified_rows: 0",
        "historical_model_eligible_consensus_rows: 0",
        "server_time_to_utc_status: UNVERIFIED",
    ]
    if len(comparison):
        lines.extend([
            f"v04c_actual_comparison_rows: {len(comparison)}",
            f"v04c_actual_exact_match_rows_1e-9: {int(comparison['actual_difference'].abs().le(1e-9).sum())}",
            f"v04c_actual_max_abs_difference: {comparison['actual_difference'].abs().max():.12g}",
        ])
        for measure_id, group in comparison.groupby("measure_id", sort=True):
            lines.append(
                f"actual_compare_{measure_id}: rows={len(group)} exact={int(group['actual_difference'].abs().le(1e-9).sum())} "
                f"max_abs_diff={group['actual_difference'].abs().max():.12g}"
            )
    else:
        lines.append("v04c_actual_comparison: NOT_RUN_OR_NO_MATCHES")
    lines.extend([
        "identity_policy: exact event ID/code/name is necessary but not sufficient for model eligibility",
        "pit_policy: historical MT5 forecast rows remain research-only without a proven pre-release snapshot time",
        "V04D_MT5_IDENTITY_STATUS: PASS_FAIL_CLOSED",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-audit", type=Path, default=Path("data/mt5/calendar/v04d_history_audit.csv"))
    parser.add_argument("--v04c", type=Path, default=Path("data/processed/v04c_event_values.parquet"))
    parser.add_argument("--identity-output", type=Path, default=Path("reports/v04d_mt5_event_identity.csv"))
    parser.add_argument("--rejected-output", type=Path, default=Path("reports/v04d_mt5_rejected_variants.csv"))
    parser.add_argument("--actual-output", type=Path, default=Path("reports/v04d_mt5_v04c_actual_comparison.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("reports/v04d_mt5_identity_quality.txt"))
    args = parser.parse_args()

    frame = _read(args.history_audit)
    identity, rejected = audit_identity(frame)
    comparison = compare_with_v04c(frame, args.v04c)

    for path in (args.identity_output, args.rejected_output, args.actual_output, args.quality_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    identity.to_csv(args.identity_output, index=False, lineterminator="\n")
    rejected.to_csv(args.rejected_output, index=False, lineterminator="\n")
    if len(comparison):
        comparison.to_csv(args.actual_output, index=False, lineterminator="\n")
    args.quality_report.write_text(quality_report(identity, rejected, comparison), encoding="utf-8", newline="\n")

    print(identity[["event_id", "measure_id", "rows", "forecast_rows", "units", "multipliers", "identity_status"]].to_string(index=False))
    print()
    print("Rejected similar-but-wrong event IDs:")
    print(rejected[["event_id", "rows", "event_names", "rejection_reason"]].to_string(index=False))
    print()
    print(quality_report(identity, rejected, comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
