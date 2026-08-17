"""Audit whether MT5 can supply EURUSD M1 bars around validated V0.4A events.

This is a read-only availability test before building event reactions. It requests
small windows around each historical event instead of downloading the full M1
history. No orders, positions, or account changes are made.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

try:
    from .reaction_semantics import pre_event_bar_open, reaction_bar_open, requested_times_only
except ImportError:  # Support direct execution.
    from reaction_semantics import pre_event_bar_open, reaction_bar_open, requested_times_only


ROOT = Path(__file__).resolve().parents[2]
EVENT_FILE = ROOT / "data" / "events" / "bls_release_events.parquet"
REPORT_FILE = ROOT / "reports" / "mt5_event_m1_coverage.csv"
SUMMARY_FILE = ROOT / "reports" / "mt5_event_m1_coverage_summary.txt"
SYMBOL = "EURUSD"
HORIZONS_MINUTES = (1, 5, 15, 60, 240)
WINDOW_BEFORE = timedelta(minutes=10)
WINDOW_AFTER = timedelta(hours=4, minutes=10)


def _to_py_utc(ts: pd.Timestamp) -> datetime:
    return ts.to_pydatetime().astimezone(timezone.utc)


def _nearest_distance_seconds(times: pd.Series, target: pd.Timestamp) -> float | None:
    if times.empty:
        return None
    distances = (times - target).abs().dt.total_seconds()
    return float(distances.min())


def _bar_point_available(times: pd.Series, bar_open: pd.Timestamp) -> bool:
    # Exact M1 coverage is preferred. A one-second tolerance only protects against
    # representation quirks; it does not permit a neighboring minute.
    distance = _nearest_distance_seconds(times, bar_open)
    return distance is not None and distance <= 1.0


def select_events(frame: pd.DataFrame, sample_per_year: int | None) -> pd.DataFrame:
    eligible = frame[frame["model_eligible"].fillna(False).astype(bool)].copy()
    eligible["event_timestamp_utc"] = pd.to_datetime(
        eligible["event_timestamp_utc"], utc=True, errors="coerce"
    )
    eligible = eligible.dropna(subset=["event_timestamp_utc"])
    eligible["event_year"] = eligible["event_timestamp_utc"].dt.year

    if sample_per_year is None:
        return eligible.sort_values("event_timestamp_utc").reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for (_, _), group in eligible.groupby(["event_year", "event_type"], sort=True):
        group = group.sort_values("event_timestamp_utc")
        count = min(sample_per_year, len(group))
        if count == len(group):
            chosen = group
        elif count == 1:
            chosen = group.iloc[[len(group) // 2]]
        else:
            indices = pd.Series(range(len(group))).quantile(
                [i / (count - 1) for i in range(count)], interpolation="nearest"
            ).astype(int).unique()
            chosen = group.iloc[indices]
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True).sort_values("event_timestamp_utc").reset_index(drop=True)


def audit_event(event: pd.Series) -> dict:
    event_ts = pd.Timestamp(event["event_timestamp_utc"])
    start = _to_py_utc(event_ts - WINDOW_BEFORE)
    end = _to_py_utc(event_ts + WINDOW_AFTER)

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)
    api_error = mt5.last_error()

    row = {
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "reference_period": event["reference_period"],
        "event_timestamp_utc": event_ts,
        "api_error_code": api_error[0] if isinstance(api_error, tuple) and api_error else None,
        "api_error_message": api_error[1] if isinstance(api_error, tuple) and len(api_error) > 1 else None,
        "raw_bars_returned": 0,
        "bars_returned": 0,
        "out_of_range_bars": 0,
        "first_bar_utc": pd.NaT,
        "last_bar_utc": pd.NaT,
        "pre_event_m1": False,
    }
    for horizon in HORIZONS_MINUTES:
        row[f"after_{horizon}m_m1"] = False

    if rates is None or len(rates) == 0:
        row["coverage_complete"] = False
        return row

    bars = pd.DataFrame(rates)
    if "time" not in bars.columns:
        row["coverage_complete"] = False
        row["api_error_message"] = "MT5 response missing time field"
        return row

    raw_times = pd.to_datetime(bars["time"], unit="s", utc=True).drop_duplicates().sort_values()
    row["raw_bars_returned"] = int(len(raw_times))
    times, out_of_range = requested_times_only(
        raw_times,
        event_ts - pd.Timedelta(WINDOW_BEFORE),
        event_ts + pd.Timedelta(WINDOW_AFTER),
    )
    row["out_of_range_bars"] = out_of_range
    if out_of_range:
        row["api_error_message"] = (
            f"MT5 returned {out_of_range} cached bar(s) outside the requested window; quarantined"
        )
    if times.empty:
        row["coverage_complete"] = False
        return row
    row["bars_returned"] = int(len(times))
    row["first_bar_utc"] = times.iloc[0]
    row["last_bar_utc"] = times.iloc[-1]

    # Price immediately before a minute-aligned release is represented by the M1
    # bar that opens one minute earlier and closes at the release minute.
    row["pre_event_m1"] = _bar_point_available(times, pre_event_bar_open(event_ts))

    for horizon in HORIZONS_MINUTES:
        # For a horizon H, the M1 bar opening at event+H-1m closes at event+H.
        target_open = reaction_bar_open(event_ts, horizon)
        row[f"after_{horizon}m_m1"] = _bar_point_available(times, target_open)

    required = ["pre_event_m1"] + [f"after_{horizon}m_m1" for horizon in HORIZONS_MINUTES]
    row["coverage_complete"] = all(bool(row[column]) for column in required)
    return row


def run(sample_per_year: int | None = None) -> pd.DataFrame:
    if not EVENT_FILE.exists():
        raise RuntimeError(f"Missing validated event file: {EVENT_FILE}")

    events = pd.read_parquet(EVENT_FILE, engine="pyarrow")
    required = {"event_id", "event_type", "event_timestamp_utc", "reference_period", "model_eligible"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise RuntimeError("Event file is missing columns: " + ", ".join(missing))

    selected = select_events(events, sample_per_year)
    if selected.empty:
        raise RuntimeError("No model-eligible events available for MT5 coverage audit")

    try:
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
        if not mt5.symbol_select(SYMBOL, True):
            raise RuntimeError(f"mt5.symbol_select({SYMBOL!r}, True) failed: {mt5.last_error()}")

        rows: list[dict] = []
        for index, event in selected.iterrows():
            rows.append(audit_event(event))
            completed = index + 1
            if completed % 25 == 0 or completed == len(selected):
                print(f"Audited {completed:,}/{len(selected):,} event windows", flush=True)
    finally:
        mt5.shutdown()

    result = pd.DataFrame(rows).sort_values("event_timestamp_utc").reset_index(drop=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = REPORT_FILE.with_suffix(".tmp.csv")
    result.to_csv(temporary_report, index=False)
    temporary_report.replace(REPORT_FILE)

    complete = int(result["coverage_complete"].sum())
    out_of_range_rows = int(result["out_of_range_bars"].gt(0).sum())
    out_of_range_bars = int(result["out_of_range_bars"].sum())
    total = len(result)
    by_type = (
        result.groupby("event_type")["coverage_complete"]
        .agg(["count", "sum"])
        .rename(columns={"count": "events", "sum": "complete"})
    )
    by_type["coverage_pct"] = (100.0 * by_type["complete"] / by_type["events"]).round(2)

    lines = [
        "MARKETFUSION AI V0.4A - MT5 M1 EVENT WINDOW COVERAGE AUDIT",
        "=" * 72,
        f"Events audited: {total:,}",
        f"Complete M1 reaction windows: {complete:,}/{total:,} ({100.0 * complete / total:.2f}%)",
        f"MT5 out-of-range cache responses quarantined: {out_of_range_rows:,} events / {out_of_range_bars:,} bars",
        "Required points: pre-event plus 1m, 5m, 15m, 60m, 240m post-event closes",
        "",
        "BY EVENT TYPE",
        by_type.to_string(),
        "",
        "INTERPRETATION",
        "- This is a data-availability audit only; it does not calculate returns or claim an edge.",
        "- Incomplete windows must not be silently filled or interpolated.",
        "- Bars outside the requested copy_rates_range interval are quarantined as MT5 cache artifacts.",
        "- If old M1 windows are unavailable from MT5, another trustworthy historical intraday source is required for short-horizon reactions.",
    ]
    temporary_summary = SUMMARY_FILE.with_suffix(".tmp.txt")
    temporary_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_summary.replace(SUMMARY_FILE)

    print(f"Complete M1 windows: {complete:,}/{total:,} ({100.0 * complete / total:.2f}%)")
    print(f"Coverage CSV: {REPORT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-per-year",
        type=int,
        default=None,
        help="Audit N events per year per event type instead of all eligible events.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    try:
        if args.sample_per_year is not None and args.sample_per_year < 1:
            raise ValueError("--sample-per-year must be at least 1")
        run(args.sample_per_year)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
