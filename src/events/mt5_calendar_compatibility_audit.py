"""Measure-compatibility audit for MarketFusion V0.4D free MT5 calendar data.

This stage explains whether MT5 calendar actuals are numerically compatible with the
frozen V0.4C actual series. It is diagnostic only. Historical MT5 forecasts remain
non-model-eligible because their original pre-release snapshot time is not proven.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RULES = {
    "headline_cpi_yoy_sa": {"display_decimals": 1, "mode": "rounding", "note": "V0.4C is a PIT SA YoY transformation; MT5 is a one-decimal calendar CPI y/y field."},
    "core_cpi_yoy_sa": {"display_decimals": 1, "mode": "rounding", "note": "V0.4C is a PIT SA YoY transformation; MT5 is a one-decimal calendar Core CPI y/y field."},
    "unemployment_rate": {"display_decimals": 1, "mode": "exact", "note": "U-3 unemployment rate identity; exact cross-source comparison expected."},
    "nonfarm_payroll_change": {"display_decimals": 0, "mode": "exact", "note": "Total Nonfarm Payrolls, thousands of jobs; mismatches are quarantined rather than repaired."},
}


def audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"measure_id", "reference_month", "mt5_actual_value", "v04c_actual_value", "actual_difference"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing comparison columns: {sorted(missing)}")
    for col in ("mt5_actual_value", "v04c_actual_value", "actual_difference"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    summary_rows = []
    mismatch_rows = []
    for measure_id, rule in RULES.items():
        group = frame.loc[frame["measure_id"].eq(measure_id)].copy()
        decimals = int(rule["display_decimals"])
        group["exact_match"] = group["actual_difference"].abs().le(1e-9)
        group["v04c_rounded_to_mt5_precision"] = group["v04c_actual_value"].round(decimals)
        group["precision_match"] = (group["mt5_actual_value"] - group["v04c_rounded_to_mt5_precision"]).abs().le(1e-9)

        exact = int(group["exact_match"].sum())
        precision = int(group["precision_match"].sum())
        n = len(group)
        if rule["mode"] == "rounding":
            status = "ROUNDING_COMPATIBLE_DIAGNOSTIC" if n and precision == n else "COMPATIBILITY_REVIEW_REQUIRED"
        else:
            if n and exact == n:
                status = "EXACT_COMPATIBLE"
            elif n and exact > 0:
                status = "PARTIAL_EXACT_WITH_QUARANTINED_MISMATCHES"
            else:
                status = "COMPATIBILITY_REVIEW_REQUIRED"

        summary_rows.append({
            "measure_id": measure_id,
            "rows": n,
            "exact_matches": exact,
            "precision_matches": precision,
            "display_decimals": decimals,
            "max_abs_raw_difference": group["actual_difference"].abs().max() if n else float("nan"),
            "compatibility_status": status,
            "cross_source_historical_surprise_eligible": False,
            "historical_mt5_forecast_pit_verified": False,
            "note": rule["note"],
        })

        bad = group.loc[~group["precision_match"]].copy()
        if len(bad):
            bad["compatibility_status"] = "QUARANTINED_MISMATCH"
            mismatch_rows.append(bad)

    summary = pd.DataFrame(summary_rows)
    mismatches = pd.concat(mismatch_rows, ignore_index=True) if mismatch_rows else pd.DataFrame()
    return summary, mismatches


def report(summary: pd.DataFrame, mismatches: pd.DataFrame) -> str:
    lines = [
        "MARKETFUSION V0.4D MT5 MEASURE COMPATIBILITY AUDIT",
        f"measures: {len(summary)}",
        f"comparison_rows: {int(summary['rows'].sum())}",
        f"exact_matches: {int(summary['exact_matches'].sum())}",
        f"precision_matches: {int(summary['precision_matches'].sum())}",
        f"quarantined_precision_mismatches: {len(mismatches)}",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.measure_id}: rows={row.rows} exact={row.exact_matches} precision_match={row.precision_matches} status={row.compatibility_status}"
        )
    lines.extend([
        "historical_consensus_policy: MT5 historical forecast values remain NON-MODEL-ELIGIBLE without proven pre-release snapshot timing",
        "cross_source_policy: do not subtract an MT5 forecast from a differently-defined V0.4C actual merely because values are numerically close",
        "future_policy: once a live MT5 forecast snapshot is causally verified, prefer provider-consistent MT5 forecast + MT5 actual for raw surprise, while retaining official-source actuals for independent validation",
        "V04D_MT5_COMPATIBILITY_STATUS: PASS_FAIL_CLOSED",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("reports/v04d_mt5_v04c_actual_comparison.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/v04d_mt5_measure_compatibility.csv"))
    parser.add_argument("--mismatch-output", type=Path, default=Path("reports/v04d_mt5_measure_mismatches.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("reports/v04d_mt5_measure_compatibility.txt"))
    args = parser.parse_args()

    frame = pd.read_csv(args.comparison)
    summary, mismatches = audit(frame)
    for path in (args.summary_output, args.mismatch_output, args.quality_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False, lineterminator="\n")
    if len(mismatches):
        mismatches.to_csv(args.mismatch_output, index=False, lineterminator="\n")
    elif args.mismatch_output.exists():
        args.mismatch_output.unlink()
    text = report(summary, mismatches)
    args.quality_report.write_text(text, encoding="utf-8", newline="\n")
    print(summary.to_string(index=False))
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
