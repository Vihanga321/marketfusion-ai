"""Validate the MarketFusion AI V0.2 OANDA EUR/USD H1 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "oanda" / "eurusd_h1.parquet"
REPORT_FILE = ROOT / "reports" / "oanda_data_quality_report.txt"
PRICE_GROUPS = ("mid", "bid", "ask")
OHLC_PARTS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = [
    "timestamp",
    *[f"{group}_{part}" for group in PRICE_GROUPS for part in OHLC_PARTS],
    "volume", "complete", "spread_close",
]
ANOMALY_MIN_SAMPLES = 500


@dataclass
class ValidationResult:
    passed: bool
    failures: list[str]
    warnings: list[str]
    report: str


def count_pct(mask: pd.Series) -> str:
    return f"{int(mask.sum()):,} ({mask.mean():.4%})" if len(mask) else "0 (n/a)"


def normal_fx_weekend_closure(timestamp: pd.Timestamp) -> bool:
    """Conservative UTC weekend filter accommodating US daylight-saving shifts."""
    weekday = timestamp.weekday()
    hour = timestamp.hour
    return weekday == 5 or (weekday == 4 and hour >= 21) or (weekday == 6 and hour < 22)


def correlation(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    return float(left[valid].corr(right[valid])) if valid.sum() >= 2 else np.nan


def validate(path: Path = DATA_FILE, write_report: bool = True) -> ValidationResult:
    failures: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    def add(value: str = "") -> None:
        lines.append(value)

    add("MARKETFUSION AI V0.2 - OANDA EUR/USD H1 DATA QUALITY REPORT")
    add("=" * 78)
    add(f"Data file: {path}")

    if not path.exists():
        failures.append("OANDA Parquet file does not exist")
        add("Rows: 0")
        add("VALIDATION_STATUS: FAIL")
        add("FAILURES:")
        add("- OANDA Parquet file does not exist")
        report = "\n".join(lines) + "\n"
        if write_report:
            REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
            REPORT_FILE.write_text(report, encoding="utf-8")
        return ValidationResult(False, failures, warnings, report)

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        failures.append(f"Could not read Parquet: {type(exc).__name__}: {exc}")
        add("VALIDATION_STATUS: FAIL")
        add("FAILURES:")
        add(f"- {failures[-1]}")
        report = "\n".join(lines) + "\n"
        if write_report:
            REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
            REPORT_FILE.write_text(report, encoding="utf-8")
        return ValidationResult(False, failures, warnings, report)

    add(f"Rows: {len(df):,}")
    add(f"Columns: {', '.join(df.columns)}")
    if df.empty:
        failures.append("Dataset is empty")
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing_columns:
        failures.append("Missing required columns: " + ", ".join(missing_columns))

    if failures:
        add("VALIDATION_STATUS: FAIL")
        add("FAILURES:")
        for failure in failures:
            add(f"- {failure}")
        report = "\n".join(lines) + "\n"
        if write_report:
            REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
            REPORT_FILE.write_text(report, encoding="utf-8")
        return ValidationResult(False, failures, warnings, report)

    add()
    add("1. TIMESTAMPS AND COVERAGE")
    add("-" * 78)
    original_timestamp = df["timestamp"]
    timestamp_dtype = original_timestamp.dtype
    utc_dtype = isinstance(timestamp_dtype, pd.DatetimeTZDtype) and str(timestamp_dtype.tz) == "UTC"
    timestamps = pd.to_datetime(original_timestamp, errors="coerce", utc=True)
    invalid_timestamps = int(timestamps.isna().sum())
    duplicate_timestamps = int(timestamps.duplicated().sum())
    sorted_ascending = bool(timestamps.is_monotonic_increasing)
    exact_hour = bool(
        ((timestamps.dt.minute == 0) & (timestamps.dt.second == 0) & (timestamps.dt.microsecond == 0)).all()
    )
    add(f"Stored timestamp dtype: {timestamp_dtype}")
    add(f"UTC timezone-aware dtype: {utc_dtype}")
    add(f"Invalid timestamps: {invalid_timestamps:,}")
    add(f"Duplicate timestamps: {duplicate_timestamps:,}")
    add(f"Sorted ascending: {sorted_ascending}")
    add(f"Aligned to exact UTC hour: {exact_hour}")
    if timestamps.notna().any():
        add(f"Range: {timestamps.min()} to {timestamps.max()}")
    if not utc_dtype:
        failures.append("Timestamp column is not stored as timezone-aware UTC")
    if invalid_timestamps:
        failures.append(f"Found {invalid_timestamps} invalid timestamps")
    if duplicate_timestamps:
        failures.append(f"Found {duplicate_timestamps} duplicate timestamps")
    if not sorted_ascending:
        failures.append("Timestamps are not sorted ascending")
    if not exact_hour:
        failures.append("One or more timestamps are not aligned to an exact UTC hour")

    valid_timestamps = timestamps.dropna().sort_values()
    if len(valid_timestamps) >= 2:
        full_grid = pd.date_range(valid_timestamps.min(), valid_timestamps.max(), freq="h", tz="UTC")
        missing_grid = full_grid.difference(pd.DatetimeIndex(valid_timestamps))
        weekend_missing = [stamp for stamp in missing_grid if normal_fx_weekend_closure(stamp)]
        unexpected_missing = [stamp for stamp in missing_grid if not normal_fx_weekend_closure(stamp)]
    else:
        missing_grid = pd.DatetimeIndex([])
        weekend_missing = []
        unexpected_missing = []
    add(f"Missing grid hours, total: {len(missing_grid):,}")
    add(f"Missing hours explained by normal FX weekend closure: {len(weekend_missing):,}")
    add(f"Unexpected missing non-weekend H1 candles: {len(unexpected_missing):,}")
    if unexpected_missing:
        sample = ", ".join(str(value) for value in unexpected_missing[:20])
        add(f"Unexpected missing samples: {sample}")
        failures.append(f"Found {len(unexpected_missing)} missing non-weekend H1 candles")

    add()
    add("2. NUMERIC AND COMPLETENESS CHECKS")
    add("-" * 78)
    numeric_columns = [column for column in REQUIRED_COLUMNS if column not in {"timestamp", "complete"}]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    nan_cells = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    infinite_cells = int(np.isinf(numeric.to_numpy()).sum())
    price_columns = [f"{group}_{part}" for group in PRICE_GROUPS for part in OHLC_PARTS]
    non_positive_prices = int((numeric[price_columns] <= 0).sum().sum())
    complete_values = df["complete"].fillna(False).astype(bool)
    incomplete = int((~complete_values).sum())
    add(f"NaN cells in required columns: {nan_cells:,}")
    add(f"Infinite numeric cells: {infinite_cells:,}")
    add(f"Non-positive price cells: {non_positive_prices:,}")
    add(f"Incomplete candles: {incomplete:,}")
    add(f"Negative volume rows: {count_pct(numeric['volume'] < 0)}")
    add(f"Zero volume rows: {count_pct(numeric['volume'] == 0)}")
    if nan_cells:
        failures.append(f"Found {nan_cells} NaN cells in required columns")
    if infinite_cells:
        failures.append(f"Found {infinite_cells} infinite values")
    if non_positive_prices:
        failures.append(f"Found {non_positive_prices} non-positive prices")
    if incomplete:
        failures.append(f"Found {incomplete} incomplete candles")
    if (numeric["volume"] < 0).any():
        failures.append("Negative volume values found")

    add()
    add("3. OHLC ENCLOSURE AND CANDLE SHAPE")
    add("-" * 78)
    tolerance = 1e-12
    for group in PRICE_GROUPS:
        o = numeric[f"{group}_open"]
        h = numeric[f"{group}_high"]
        low = numeric[f"{group}_low"]
        c = numeric[f"{group}_close"]
        bad = (
            (h + tolerance < low)
            | (h + tolerance < o)
            | (h + tolerance < c)
            | (low - tolerance > o)
            | (low - tolerance > c)
        )
        open_close_equal = np.isclose(o, c, rtol=0, atol=tolerance)
        high_low_equal = np.isclose(h, low, rtol=0, atol=tolerance)
        add(f"{group.upper()} OHLC enclosure violations: {count_pct(bad)}")
        add(f"{group.upper()} open == close: {count_pct(pd.Series(open_close_equal, index=df.index))}")
        add(f"{group.upper()} high == low: {count_pct(pd.Series(high_low_equal, index=df.index))}")
        if bad.any():
            failures.append(f"Found {int(bad.sum())} {group} OHLC enclosure violations")
        if pd.Series(open_close_equal).mean() > 0.25:
            warnings.append(f"{group} open==close frequency exceeds 25%")
        if pd.Series(high_low_equal).mean() > 0.01:
            warnings.append(f"{group} zero-range frequency exceeds 1%")

    add()
    add("4. BID/ASK SPREAD")
    add("-" * 78)
    spread = numeric["spread_close"]
    recomputed_spread = numeric["ask_close"] - numeric["bid_close"]
    spread_mismatch = ~np.isclose(spread, recomputed_spread, rtol=0, atol=1e-12)
    negative_spread = spread < -tolerance
    q1, q3 = spread.quantile([0.25, 0.75])
    extreme_threshold = max(float(q3 + 10 * (q3 - q1)), 0.0010)
    extreme_spread = spread > extreme_threshold
    add(f"Stored spread mismatches ask_close - bid_close: {count_pct(pd.Series(spread_mismatch, index=df.index))}")
    add(f"Negative close spreads: {count_pct(negative_spread)}")
    add(f"Median spread: {spread.median():.8f} ({spread.median() * 10_000:.3f} pips)")
    add(f"95th percentile spread: {spread.quantile(0.95):.8f} ({spread.quantile(0.95) * 10_000:.3f} pips)")
    add(f"99th percentile spread: {spread.quantile(0.99):.8f} ({spread.quantile(0.99) * 10_000:.3f} pips)")
    add(f"Maximum spread: {spread.max():.8f} ({spread.max() * 10_000:.3f} pips)")
    add(f"Extreme-spread threshold: {extreme_threshold:.8f} ({extreme_threshold * 10_000:.3f} pips)")
    add(f"Extreme spread rows: {count_pct(extreme_spread)}")
    if spread_mismatch.any():
        failures.append(f"Found {int(spread_mismatch.sum())} incorrect stored spreads")
    if negative_spread.any():
        failures.append(f"Found {int(negative_spread.sum())} negative spreads")
    if extreme_spread.any():
        warnings.append(f"Found {int(extreme_spread.sum())} extreme spreads requiring review")

    add()
    add("5. NEXT-CANDLE AND GEOMETRY ANOMALY CHECKS")
    add("-" * 78)
    mid_o = numeric["mid_open"]
    mid_h = numeric["mid_high"]
    mid_l = numeric["mid_low"]
    mid_c = numeric["mid_close"]
    next_c = mid_c.shift(-1)
    consecutive = timestamps.shift(-1).sub(timestamps).eq(pd.Timedelta(hours=1))
    future_return = next_c / mid_c - 1
    upper_wick = mid_h - pd.concat([mid_o, mid_c], axis=1).max(axis=1)
    lower_wick = pd.concat([mid_o, mid_c], axis=1).min(axis=1) - mid_l
    wick_prediction = (upper_wick > lower_wick).astype(int)
    next_direction = (next_c > mid_c).astype(int)
    anomaly_rows = consecutive & next_c.notna()
    anomaly_count = int(anomaly_rows.sum())
    wick_accuracy = float((wick_prediction[anomaly_rows] == next_direction[anomaly_rows]).mean()) if anomaly_count else np.nan
    upper_corr = correlation(upper_wick.where(anomaly_rows), future_return.where(anomaly_rows))
    lower_corr = correlation(lower_wick.where(anomaly_rows), future_return.where(anomaly_rows))
    next_inside = ((next_c >= mid_l) & (next_c <= mid_h) & anomaly_rows)
    next_open_equals_close = np.isclose(mid_o.shift(-1), mid_c, rtol=0, atol=1e-12) & anomaly_rows
    add(f"Consecutive one-hour pairs tested: {anomaly_count:,}")
    add(f"Next mid close inside current mid high-low: {count_pct(next_inside[anomaly_rows])}")
    add(f"Current wick-imbalance next-direction accuracy: {wick_accuracy:.4f}")
    add(f"Current upper-wick correlation with next return: {upper_corr:.4f}")
    add(f"Current lower-wick correlation with next return: {lower_corr:.4f}")
    add(f"Next mid open exactly equals current mid close: {count_pct(pd.Series(next_open_equals_close[anomaly_rows]))}")
    if anomaly_count >= ANOMALY_MIN_SAMPLES and (
        wick_accuracy >= 0.65 or abs(upper_corr) >= 0.25 or abs(lower_corr) >= 0.25
    ):
        failures.append("Suspicious current-candle geometry predicts the next candle")

    status = "PASS" if not failures else "FAIL"
    add()
    add(f"VALIDATION_STATUS: {status}")
    if failures:
        add("FAILURES:")
        for failure in failures:
            add(f"- {failure}")
    if warnings:
        add("WARNINGS:")
        for warning in warnings:
            add(f"- {warning}")
    if not failures:
        add("No blocking integrity or predictive-geometry anomaly was detected.")

    report = "\n".join(lines) + "\n"
    if write_report:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(report, encoding="utf-8")
    return ValidationResult(not failures, failures, warnings, report)


def main() -> None:
    result = validate()
    print(result.report)
    print(f"Saved report: {REPORT_FILE}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
