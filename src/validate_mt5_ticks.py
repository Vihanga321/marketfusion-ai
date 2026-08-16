"""Validate stored MarketFusion EURUSD MetaTrader 5 tick Parquet files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TICK_DIRECTORY = ROOT / "data" / "mt5" / "ticks"
REPORT_FILE = ROOT / "reports" / "mt5_tick_quality_report.txt"
SYMBOL = "EURUSD"
PIP_SIZE = 0.0001
REQUIRED_COLUMNS = [
    "timestamp_utc", "time_msc", "bid", "ask", "mid", "spread",
    "spread_pips", "flags", "volume", "volume_real",
]
UNIQUE_KEY = ["time_msc", "bid", "ask", "flags", "volume", "volume_real"]


def count_pct(mask: pd.Series) -> str:
    if not len(mask):
        return "0 (n/a)"
    return f"{int(mask.sum()):,} ({mask.mean():.6%})"


def normal_fx_market_closed(timestamp: pd.Timestamp) -> bool:
    weekday = timestamp.weekday()
    hour = timestamp.hour
    return weekday == 5 or (weekday == 4 and hour >= 21) or (weekday == 6 and hour < 22)


def normal_weekend_gap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    duration = (end - start).total_seconds()
    return (
        start.weekday() == 4
        and end.weekday() in {6, 0}
        and 36 * 3_600 <= duration <= 75 * 3_600
    )


def maximum_true_run(mask: pd.Series) -> int:
    if mask.empty or not mask.any():
        return 0
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return int(mask.groupby(groups).sum().max())


def write_early_failure(report_path: Path, tick_dir: Path, reason: str) -> str:
    report = "\n".join(
        [
            "MARKETFUSION AI - MT5 EURUSD TICK QUALITY REPORT",
            "=" * 78,
            f"Tick directory: {tick_dir}",
            "Total ticks: 0",
            "VALIDATION_STATUS: FAIL",
            "FAILURES:",
            f"- {reason}",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


def validate(tick_dir: Path = TICK_DIRECTORY, report_path: Path = REPORT_FILE) -> tuple[bool, str]:
    files = sorted(tick_dir.glob(f"{SYMBOL}_ticks_*.parquet"))
    if not files:
        report = write_early_failure(report_path, tick_dir, "No EURUSD tick Parquet files found")
        return False, report

    failures: list[str] = []
    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    unreadable: list[str] = []
    utc_dtype_files: list[str] = []

    for path in files:
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except Exception as exc:
            unreadable.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if "timestamp_utc" in frame:
            dtype = frame["timestamp_utc"].dtype
            if not (isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"):
                utc_dtype_files.append(path.name)
        frame["_source_file"] = path.name
        frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)

    if unreadable:
        failures.extend("Unreadable Parquet file: " + value for value in unreadable)
    if not frames:
        report = write_early_failure(report_path, tick_dir, failures[0] if failures else "No readable tick files")
        return False, report

    df = pd.concat(frames, ignore_index=True)
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing_columns:
        failures.append("Missing required columns: " + ", ".join(missing_columns))
        report = write_early_failure(report_path, tick_dir, failures[-1])
        return False, report

    if utc_dtype_files:
        failures.append("Timestamp is not stored as timezone-aware UTC in: " + ", ".join(utc_dtype_files))

    timestamps = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "timestamp_utc"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    time_msc = numeric["time_msc"]
    bid = numeric["bid"]
    ask = numeric["ask"]
    mid = numeric["mid"]
    spread = numeric["spread"]
    spread_pips = numeric["spread_pips"]

    invalid_timestamp = timestamps.isna()
    nan_required = df[REQUIRED_COLUMNS].isna().any(axis=1)
    infinite = pd.Series(np.isinf(numeric.to_numpy()).any(axis=1), index=df.index)
    bid_invalid = bid <= 0
    ask_invalid = ask <= 0
    crossed = ask < bid
    negative_spread = spread < -1e-12
    zero_spread = np.isclose(spread, 0, rtol=0, atol=1e-12)
    mid_mismatch = ~np.isclose(mid, (bid + ask) / 2, rtol=0, atol=1e-12, equal_nan=True)
    spread_mismatch = ~np.isclose(spread, ask - bid, rtol=0, atol=1e-12, equal_nan=True)
    spread_pips_mismatch = ~np.isclose(
        spread_pips, spread / PIP_SIZE, rtol=0, atol=1e-9, equal_nan=True
    )
    duplicate_tick = df.duplicated(subset=UNIQUE_KEY, keep=False)
    duplicate_time_msc = time_msc.duplicated(keep=False)

    timestamp_from_msc = pd.to_datetime(time_msc, unit="ms", utc=True, errors="coerce")
    timestamp_mismatch = timestamps.sub(timestamp_from_msc).abs().gt(pd.Timedelta(milliseconds=0.001))
    original_time_diff = time_msc.diff()
    out_of_order = original_time_diff < 0
    timestamps_sorted = bool(time_msc.is_monotonic_increasing)

    for source, group in df.assign(_time_msc=time_msc).groupby("_source_file", sort=False):
        if not group["_time_msc"].is_monotonic_increasing:
            failures.append(f"Ticks are out of order inside {source}")

    filename_day_mismatch = pd.Series(False, index=df.index)
    for source, indices in df.groupby("_source_file").groups.items():
        expected_day = source.removeprefix(f"{SYMBOL}_ticks_").removesuffix(".parquet")
        actual_day = timestamps.loc[indices].dt.strftime("%Y-%m-%d")
        filename_day_mismatch.loc[indices] = actual_day.ne(expected_day)

    sorted_frame = pd.DataFrame(
        {"timestamp": timestamps, "time_msc": time_msc, "bid": bid, "ask": ask}
    ).dropna(subset=["timestamp", "time_msc"]).sort_values(["time_msc", "bid", "ask"])
    gaps = sorted_frame["timestamp"].diff()
    large_gap = gaps > pd.Timedelta(minutes=5)
    gap_rows = []
    for index in sorted_frame.index[large_gap.fillna(False)]:
        position = sorted_frame.index.get_loc(index)
        start = sorted_frame.iloc[position - 1]["timestamp"]
        end = sorted_frame.iloc[position]["timestamp"]
        gap_rows.append((start, end, end - start, normal_weekend_gap(start, end)))
    weekend_gaps = [row for row in gap_rows if row[3]]
    unexpected_gaps = [row for row in gap_rows if not row[3]]

    repeated_quote = sorted_frame["bid"].eq(sorted_frame["bid"].shift()) & sorted_frame["ask"].eq(
        sorted_frame["ask"].shift()
    )
    max_repeated_run = maximum_true_run(repeated_quote)
    saturday_ticks = timestamps.dt.weekday.eq(5)

    positive_spread = spread[(spread >= 0) & np.isfinite(spread)]
    median_spread = float(positive_spread.median()) if len(positive_spread) else np.nan
    p95_spread = float(positive_spread.quantile(0.95)) if len(positive_spread) else np.nan
    p99_spread = float(positive_spread.quantile(0.99)) if len(positive_spread) else np.nan
    maximum_spread = float(positive_spread.max()) if len(positive_spread) else np.nan
    extreme_threshold = max(0.0020, median_spread * 10) if np.isfinite(median_spread) else 0.0020
    extreme_spread = spread > extreme_threshold

    critical_masks = {
        "invalid timestamps": invalid_timestamp,
        "rows with NaN": nan_required,
        "rows with infinity": infinite,
        "non-positive bid": bid_invalid,
        "non-positive ask": ask_invalid,
        "ask below bid": crossed,
        "negative spread": negative_spread,
        "incorrect mid": pd.Series(mid_mismatch, index=df.index),
        "incorrect spread": pd.Series(spread_mismatch, index=df.index),
        "incorrect spread_pips": pd.Series(spread_pips_mismatch, index=df.index),
        "timestamp/time_msc mismatch": timestamp_mismatch,
        "duplicate exact ticks": duplicate_tick,
        "filename UTC-day mismatch": filename_day_mismatch,
    }
    invalid_rows = pd.Series(False, index=df.index)
    for label, mask in critical_masks.items():
        mask = pd.Series(mask, index=df.index).fillna(True)
        invalid_rows |= mask
        if mask.any():
            failures.append(f"{label}: {int(mask.sum()):,} rows")
    if not timestamps_sorted or out_of_order.any():
        failures.append(f"Out-of-order ticks: {int(out_of_order.sum()):,}")
    if unexpected_gaps:
        failures.append(f"Unexpected market-hours gaps over 5 minutes: {len(unexpected_gaps):,}")
    if zero_spread.mean() > 0.01:
        failures.append(f"Zero spreads exceed 1%: {int(zero_spread.sum()):,} rows")
    elif zero_spread.any():
        warnings.append(f"Zero spreads observed: {int(zero_spread.sum()):,} rows")
    if duplicate_time_msc.mean() > 0.05:
        warnings.append("More than 5% of rows share time_msc values; review broker timestamp resolution")
    if extreme_spread.any():
        warnings.append(f"Extreme spreads above {extreme_threshold / PIP_SIZE:.1f} pips: {int(extreme_spread.sum()):,}")
    if saturday_ticks.any():
        warnings.append(f"Saturday UTC ticks observed: {int(saturday_ticks.sum()):,}")
    if max_repeated_run >= 1_000 or (len(repeated_quote) >= 1_000 and repeated_quote.mean() > 0.95):
        failures.append("Suspiciously long or frequent repeated bid/ask prices")

    valid_times = timestamps.dropna()
    first_timestamp = valid_times.min() if len(valid_times) else pd.NaT
    last_timestamp = valid_times.max() if len(valid_times) else pd.NaT
    trading_days = int(valid_times.dt.date.nunique()) if len(valid_times) else 0
    now = pd.Timestamp.now(tz="UTC")
    age_seconds = float((now - last_timestamp).total_seconds()) if pd.notna(last_timestamp) else np.nan
    market_closed = normal_fx_market_closed(now)
    if np.isfinite(age_seconds) and age_seconds > 900 and not market_closed:
        warnings.append(f"Latest tick is stale during expected market hours: {age_seconds:,.1f} seconds old")

    lines = [
        "MARKETFUSION AI - MT5 EURUSD TICK QUALITY REPORT",
        "=" * 78,
        f"Tick directory: {tick_dir}",
        f"Files read: {len(files):,}",
        f"Total ticks: {len(df):,}",
        f"First timestamp: {first_timestamp}",
        f"Last timestamp: {last_timestamp}",
        f"Latest tick age seconds: {age_seconds:,.3f}",
        f"FX market currently classified closed: {market_closed}",
        f"Number of trading days: {trading_days:,}",
        "",
        "TIMESTAMP AND DUPLICATE CHECKS",
        "-" * 78,
        f"Timestamps sorted globally: {timestamps_sorted}",
        f"Out-of-order transitions: {count_pct(out_of_order.fillna(False))}",
        f"Duplicate exact ticks: {count_pct(duplicate_tick)}",
        f"Duplicate percentage: {duplicate_tick.mean():.6%}",
        f"Rows sharing duplicate time_msc: {count_pct(duplicate_time_msc)}",
        f"Timestamp/time_msc mismatches: {count_pct(timestamp_mismatch.fillna(True))}",
        f"File-name UTC-day mismatches: {count_pct(filename_day_mismatch)}",
        "",
        "PRICE AND SPREAD CHECKS",
        "-" * 78,
        f"Bid <= 0: {count_pct(bid_invalid)}",
        f"Ask <= 0: {count_pct(ask_invalid)}",
        f"Ask < bid: {count_pct(crossed)}",
        f"Negative spreads: {count_pct(negative_spread)}",
        f"Zero spreads: {count_pct(pd.Series(zero_spread, index=df.index))}",
        f"Incorrect mid calculations: {count_pct(pd.Series(mid_mismatch, index=df.index))}",
        f"Incorrect spread calculations: {count_pct(pd.Series(spread_mismatch, index=df.index))}",
        f"Median spread: {median_spread:.8f} ({median_spread / PIP_SIZE:.3f} pips)",
        f"95th percentile spread: {p95_spread:.8f} ({p95_spread / PIP_SIZE:.3f} pips)",
        f"99th percentile spread: {p99_spread:.8f} ({p99_spread / PIP_SIZE:.3f} pips)",
        f"Maximum spread: {maximum_spread:.8f} ({maximum_spread / PIP_SIZE:.3f} pips)",
        f"Extreme spread threshold: {extreme_threshold:.8f} ({extreme_threshold / PIP_SIZE:.3f} pips)",
        f"Extreme spread rows: {count_pct(extreme_spread)}",
        "",
        "GAPS, WEEKENDS, AND REPEATED PRICES",
        "-" * 78,
        f"Gaps over 5 minutes: {len(gap_rows):,}",
        f"Normal weekend gaps: {len(weekend_gaps):,}",
        f"Unexpected gaps over 5 minutes: {len(unexpected_gaps):,}",
        f"Saturday UTC ticks: {count_pct(saturday_ticks)}",
        f"Consecutive repeated bid/ask rows: {count_pct(repeated_quote)}",
        f"Maximum consecutive repeated-price run: {max_repeated_run:,}",
    ]
    if unexpected_gaps:
        lines.append("Largest unexpected gaps:")
        for start, end, duration, _ in sorted(unexpected_gaps, key=lambda row: row[2], reverse=True)[:20]:
            lines.append(f"- {start} to {end}: {duration}")
    lines.extend(
        [
            "",
            f"Invalid rows: {int(invalid_rows.sum()):,}",
            f"VALIDATION_STATUS: {'PASS' if not failures else 'FAIL'}",
        ]
    )
    if failures:
        lines.append("FAILURES:")
        lines.extend(f"- {reason}" for reason in dict.fromkeys(failures))
    if warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {reason}" for reason in dict.fromkeys(warnings))
    if not failures:
        lines.append("No blocking tick-integrity problem was detected.")

    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return not failures, report


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tick-dir", type=Path, default=TICK_DIRECTORY)
    parser.add_argument("--report", type=Path, default=REPORT_FILE)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    passed, report = validate(args.tick_dir, args.report)
    print(report)
    print(f"Saved report: {args.report}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
