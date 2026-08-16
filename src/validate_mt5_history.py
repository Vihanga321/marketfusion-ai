"""Validate MarketFusion EURUSD MT5 H1 and H4 historical bar datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIRECTORY = ROOT / "data" / "mt5" / "history"
REPORT_DIRECTORY = ROOT / "reports"
SYMBOL = "EURUSD"
REQUIRED_COLUMNS = [
    "timestamp_utc", "open", "high", "low", "close",
    "tick_volume", "spread_points", "real_volume",
]
TIMEFRAME_HOURS = {"H1": 1, "H4": 4}
ANOMALY_MIN_SAMPLES = 500


@dataclass
class HistoryValidation:
    timeframe: str
    passed: bool
    failures: list[str]
    warnings: list[str]
    report: str


def count_pct(mask: pd.Series) -> str:
    return f"{int(mask.sum()):,} ({mask.mean():.4%})" if len(mask) else "0 (n/a)"


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    return float(left[valid].corr(right[valid])) if valid.sum() >= 2 else np.nan


def early_failure(timeframe: str, path: Path, reason: str) -> HistoryValidation:
    report = "\n".join(
        [
            f"MARKETFUSION AI - MT5 EURUSD {timeframe} HISTORY QUALITY REPORT",
            "=" * 78,
            f"Data file: {path}",
            "Bars: 0",
            "VALIDATION_STATUS: FAIL",
            "FAILURES:",
            f"- {reason}",
            "",
        ]
    )
    return HistoryValidation(timeframe, False, [reason], [], report)


def validate_timeframe(timeframe: str, write_report: bool = True) -> HistoryValidation:
    if timeframe not in TIMEFRAME_HOURS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    hours = TIMEFRAME_HOURS[timeframe]
    duration = pd.Timedelta(hours=hours)
    path = HISTORY_DIRECTORY / f"{SYMBOL}_{timeframe}.parquet"
    report_path = REPORT_DIRECTORY / f"mt5_history_quality_{timeframe}.txt"
    if not path.exists():
        result = early_failure(timeframe, path, "History Parquet file does not exist")
        if write_report:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(result.report, encoding="utf-8")
        return result
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        result = early_failure(timeframe, path, f"Could not read Parquet: {type(exc).__name__}: {exc}")
        if write_report:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(result.report, encoding="utf-8")
        return result
    if df.empty:
        result = early_failure(timeframe, path, "Dataset is empty")
        if write_report:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(result.report, encoding="utf-8")
        return result
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(df.columns))
    if missing_columns:
        result = early_failure(timeframe, path, "Missing columns: " + ", ".join(missing_columns))
        if write_report:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(result.report, encoding="utf-8")
        return result

    failures: list[str] = []
    warnings: list[str] = []
    stored_dtype = df["timestamp_utc"].dtype
    utc_stored = isinstance(stored_dtype, pd.DatetimeTZDtype) and str(stored_dtype.tz) == "UTC"
    timestamps = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "timestamp_utc"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    o, high, low, close = (numeric[column] for column in ("open", "high", "low", "close"))

    invalid_timestamp = timestamps.isna()
    duplicate_timestamp = timestamps.duplicated(keep=False)
    sorted_timestamps = bool(timestamps.is_monotonic_increasing)
    exact_alignment = (
        timestamps.dt.minute.eq(0)
        & timestamps.dt.second.eq(0)
        & timestamps.dt.microsecond.eq(0)
        & (timestamps.dt.hour.mod(hours).eq(0) if hours > 1 else True)
    )
    nan_rows = df[REQUIRED_COLUMNS].isna().any(axis=1)
    infinite_rows = pd.Series(np.isinf(numeric.to_numpy()).any(axis=1), index=df.index)
    non_positive = (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
    high_violation = (high < o) | (high < close) | (high < low)
    low_violation = (low > o) | (low > close) | (low > high)
    high_below_low = high < low
    zero_range = np.isclose(high, low, rtol=0, atol=1e-12)
    open_equals_close = np.isclose(o, close, rtol=0, atol=1e-12)
    negative_spread = numeric["spread_points"] < 0

    critical = {
        "invalid timestamps": invalid_timestamp,
        "duplicate timestamps": duplicate_timestamp,
        "misaligned timeframe timestamps": ~pd.Series(exact_alignment, index=df.index),
        "NaN rows": nan_rows,
        "infinite rows": infinite_rows,
        "non-positive prices": non_positive,
        "high enclosure violations": high_violation,
        "low enclosure violations": low_violation,
        "high below low": high_below_low,
        "negative spread_points": negative_spread,
    }
    invalid_rows = pd.Series(False, index=df.index)
    if not utc_stored:
        failures.append("timestamp_utc is not stored as timezone-aware UTC")
    if not sorted_timestamps:
        failures.append("Timestamps are not sorted ascending")
    for label, mask in critical.items():
        mask = pd.Series(mask, index=df.index).fillna(True)
        invalid_rows |= mask
        if mask.any():
            failures.append(f"{label}: {int(mask.sum()):,} rows")

    valid_times = timestamps.dropna().sort_values()
    grid = pd.date_range(valid_times.min(), valid_times.max(), freq=f"{hours}h", tz="UTC")
    missing = grid.difference(pd.DatetimeIndex(valid_times))
    expected_weekend_missing = [stamp for stamp in missing if stamp.weekday() >= 5]
    unexpected_missing = [stamp for stamp in missing if stamp.weekday() < 5]
    if unexpected_missing:
        warnings.append(
            f"{len(unexpected_missing):,} missing weekday bars; likely holidays/session closures, and target creation must exclude them"
        )

    returns = close.pct_change()
    extreme_return = returns.abs() > 0.05
    if extreme_return.any():
        warnings.append(f"Extreme absolute returns above 5%: {int(extreme_return.sum()):,}")

    spread = numeric["spread_points"]
    spread_q1, spread_q3 = spread.quantile([0.25, 0.75])
    spread_extreme_threshold = max(float(spread_q3 + 10 * (spread_q3 - spread_q1)), 100.0)
    extreme_spread = spread > spread_extreme_threshold
    if extreme_spread.any():
        warnings.append(
            f"Extreme spread_points above {spread_extreme_threshold:.1f}: {int(extreme_spread.sum()):,}"
        )
    if pd.Series(zero_range).mean() > 0.01:
        warnings.append("Zero-range candle frequency exceeds 1%")
    if pd.Series(open_equals_close).mean() > 0.25:
        warnings.append("Open==close frequency exceeds 25%")

    next_close = close.shift(-1)
    consecutive = timestamps.shift(-1).sub(timestamps).eq(duration)
    next_return = next_close / close - 1
    upper_wick = high - pd.concat([o, close], axis=1).max(axis=1)
    lower_wick = pd.concat([o, close], axis=1).min(axis=1) - low
    wick_prediction = (upper_wick > lower_wick).astype(int)
    next_direction = (next_close > close).astype(int)
    anomaly_rows = consecutive & next_close.notna()
    anomaly_count = int(anomaly_rows.sum())
    wick_accuracy = (
        float((wick_prediction[anomaly_rows] == next_direction[anomaly_rows]).mean())
        if anomaly_count
        else np.nan
    )
    upper_corr = safe_corr(upper_wick.where(anomaly_rows), next_return.where(anomaly_rows))
    lower_corr = safe_corr(lower_wick.where(anomaly_rows), next_return.where(anomaly_rows))
    next_inside = (next_close >= low) & (next_close <= high) & anomaly_rows
    next_open_equals_close = np.isclose(o.shift(-1), close, rtol=0, atol=1e-12) & anomaly_rows
    if anomaly_count >= ANOMALY_MIN_SAMPLES and (
        wick_accuracy >= 0.65 or abs(upper_corr) >= 0.25 or abs(lower_corr) >= 0.25
    ):
        failures.append("Yahoo-style current-wick predictive anomaly detected")

    now = pd.Timestamp.now(tz="UTC")
    last_timestamp = valid_times.max()
    current_excluded = bool(last_timestamp + duration <= now)
    if not current_excluded:
        failures.append("Current incomplete candle is present")

    lines = [
        f"MARKETFUSION AI - MT5 EURUSD {timeframe} HISTORY QUALITY REPORT",
        "=" * 78,
        f"Data file: {path}",
        f"Bars: {len(df):,}",
        f"First timestamp: {valid_times.min()}",
        f"Last timestamp: {last_timestamp}",
        f"Stored timestamp dtype: {stored_dtype}",
        f"UTC timezone-aware storage: {utc_stored}",
        f"Sorted timestamps: {sorted_timestamps}",
        f"Duplicate timestamps: {count_pct(duplicate_timestamp)}",
        f"Current incomplete candle excluded: {current_excluded}",
        "",
        "PRICE AND CANDLE CHECKS",
        "-" * 78,
        f"NaN rows: {count_pct(nan_rows)}",
        f"Infinite rows: {count_pct(infinite_rows)}",
        f"Non-positive price rows: {count_pct(non_positive)}",
        f"High enclosure violations: {count_pct(high_violation)}",
        f"Low enclosure violations: {count_pct(low_violation)}",
        f"High < low: {count_pct(high_below_low)}",
        f"Zero-range candles: {count_pct(pd.Series(zero_range, index=df.index))}",
        f"Open == close: {count_pct(pd.Series(open_equals_close, index=df.index))}",
        f"Extreme absolute returns > 5%: {count_pct(extreme_return.fillna(False))}",
        "",
        "MISSING BARS AND SPREAD",
        "-" * 78,
        f"Missing grid bars: {len(missing):,}",
        f"Expected weekend missing bars: {len(expected_weekend_missing):,}",
        f"Missing weekday/session bars: {len(unexpected_missing):,}",
        "Missing weekday samples: " + (
            ", ".join(str(value) for value in unexpected_missing[:20]) if unexpected_missing else "none"
        ),
        f"Negative spread_points: {count_pct(negative_spread)}",
        f"Median spread_points: {spread.median():.3f}",
        f"95th percentile spread_points: {spread.quantile(0.95):.3f}",
        f"99th percentile spread_points: {spread.quantile(0.99):.3f}",
        f"Maximum spread_points: {spread.max():.3f}",
        f"Extreme spread threshold: {spread_extreme_threshold:.3f}",
        f"Extreme spread rows: {count_pct(extreme_spread)}",
        "",
        "NEXT-BAR ARTIFACT CHECKS",
        "-" * 78,
        f"Consecutive bar pairs tested: {anomaly_count:,}",
        f"Next close inside current high-low: {count_pct(next_inside[anomaly_rows])}",
        f"Current wick-imbalance next-direction accuracy: {wick_accuracy:.4f}",
        f"Upper-wick correlation with next return: {upper_corr:.4f}",
        f"Lower-wick correlation with next return: {lower_corr:.4f}",
        f"Next open exactly equals current close: {count_pct(pd.Series(next_open_equals_close[anomaly_rows]))}",
        "",
        f"Invalid rows: {int(invalid_rows.sum()):,}",
        f"VALIDATION_STATUS: {'PASS' if not failures else 'FAIL'}",
    ]
    if failures:
        lines.append("FAILURES:")
        lines.extend(f"- {reason}" for reason in dict.fromkeys(failures))
    if warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {reason}" for reason in dict.fromkeys(warnings))
    if not failures:
        lines.append("No blocking integrity or Yahoo-style predictive artifact was detected.")
    report = "\n".join(lines) + "\n"
    result = HistoryValidation(timeframe, not failures, failures, warnings, report)
    if write_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    return result


def main() -> None:
    passed = True
    for timeframe in TIMEFRAME_HOURS:
        result = validate_timeframe(timeframe, write_report=True)
        print(result.report)
        print(f"Saved report: {REPORT_DIRECTORY / f'mt5_history_quality_{timeframe}.txt'}")
        passed &= result.passed
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
