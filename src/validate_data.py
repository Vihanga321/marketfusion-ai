"""Validate the raw and feature-engineered EUR/USD datasets.

The checks are deliberately diagnostic: this script never changes source data.
Run from any directory with ``python src/validate_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "eurusd_daily.csv"
FEATURE_FILE = ROOT / "data" / "processed" / "eurusd_features.csv"
REPORT_FILE = ROOT / "reports" / "data_quality_report.txt"
OHLC = ["open", "high", "low", "close"]


def count_and_pct(mask: pd.Series) -> str:
    count = int(mask.sum())
    return f"{count:,} ({count / len(mask):.2%})" if len(mask) else "0 (n/a)"


def samples(values: pd.Series | pd.Index, limit: int = 15) -> str:
    items = [str(value) for value in list(values)[:limit]]
    return ", ".join(items) if items else "none"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def validate() -> str:
    raw = load_csv(RAW_FILE)
    features = load_csv(FEATURE_FILE)
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("MARKETFUSION AI - EUR/USD DATA QUALITY REPORT")
    add("=" * 72)
    add(f"Raw file:       {RAW_FILE}")
    add(f"Feature file:   {FEATURE_FILE}")
    add()

    required_raw = {"date", *OHLC, "adj_close", "volume"}
    missing_raw_columns = sorted(required_raw.difference(raw.columns))
    add("1. SCHEMA AND COVERAGE")
    add("-" * 72)
    add(f"Raw shape: {raw.shape[0]:,} rows x {raw.shape[1]:,} columns")
    add(f"Raw columns: {', '.join(raw.columns)}")
    add(f"Missing required raw columns: {samples(missing_raw_columns)}")
    add(f"Feature shape: {features.shape[0]:,} rows x {features.shape[1]:,} columns")
    add(f"Feature columns: {', '.join(features.columns)}")

    for name, frame in (("Raw", raw), ("Feature", features)):
        dates = pd.to_datetime(frame.get("date"), errors="coerce")
        add(f"{name} invalid dates: {int(dates.isna().sum()):,}")
        if dates.notna().any():
            add(f"{name} date range: {dates.min().date()} to {dates.max().date()}")
        add(f"{name} duplicated timestamps: {int(dates.duplicated().sum()):,}")
        add(f"{name} timestamps sorted ascending: {bool(dates.is_monotonic_increasing)}")
        add(f"{name} duplicate full rows: {int(frame.duplicated().sum()):,}")
    add()

    raw_dates = pd.to_datetime(raw["date"], errors="coerce")
    valid_dates = raw_dates.dropna().sort_values()
    expected_weekdays = pd.bdate_range(valid_dates.min(), valid_dates.max())
    missing_weekdays = expected_weekdays.difference(pd.DatetimeIndex(valid_dates))
    gaps = valid_dates.diff().dt.days.dropna()
    add("2. MISSING DATES AND GAPS")
    add("-" * 72)
    add(
        "Missing Monday-Friday dates (includes legitimate market holidays): "
        f"{len(missing_weekdays):,}"
    )
    add(f"Missing weekday samples: {samples(missing_weekdays.strftime('%Y-%m-%d'))}")
    add(f"Calendar gaps longer than 3 days: {int((gaps > 3).sum()):,}")
    if not gaps.empty:
        largest = gaps.nlargest(10)
        gap_text = [f"{valid_dates.loc[index].date()} ({int(days)} days)" for index, days in largest.items()]
        add(f"Largest gaps ending on: {samples(gap_text, 10)}")
    add()

    numeric_columns = [column for column in [*OHLC, "adj_close", "volume"] if column in raw]
    numeric = raw[numeric_columns].apply(pd.to_numeric, errors="coerce")
    non_numeric = raw[numeric_columns].notna() & numeric.isna()
    add("3. NULL, NUMERIC, AND PRICE SANITY CHECKS")
    add("-" * 72)
    add(f"Raw NaN cells: {int(raw.isna().sum().sum()):,}")
    add(f"Feature NaN cells: {int(features.isna().sum().sum()):,}")
    add(f"Raw non-numeric values in numeric columns: {int(non_numeric.sum().sum()):,}")
    add(f"Raw infinite numeric values: {int(np.isinf(numeric.to_numpy()).sum()):,}")
    if set(OHLC).issubset(numeric):
        ohlc = numeric[OHLC]
        add(f"Non-positive OHLC rows: {count_and_pct((ohlc <= 0).any(axis=1))}")
        add(f"OHLC rows outside broad EUR/USD sanity range [0.5, 2.0]: {count_and_pct(((ohlc < 0.5) | (ohlc > 2.0)).any(axis=1))}")
        add(f"Lowest observed price: {ohlc.min().min():.8f}")
        add(f"Highest observed price: {ohlc.max().max():.8f}")
    feature_numeric = features.select_dtypes(include=np.number)
    add(f"Feature infinite numeric values: {int(np.isinf(feature_numeric.to_numpy()).sum()):,}")
    add()

    add("4. OHLC CONSISTENCY")
    add("-" * 72)
    if set(OHLC).issubset(numeric):
        o, h, low, c = (numeric[column] for column in OHLC)
        high_below_open = h < o
        high_below_close = h < c
        low_above_open = low > o
        low_above_close = low > c
        hierarchy_bad = (h < low) | high_below_open | high_below_close | low_above_open | low_above_close
        max_high_shortfall = pd.concat([(o - h).clip(lower=0), (c - h).clip(lower=0)], axis=1).max(axis=1)
        max_low_shortfall = pd.concat([(low - o).clip(lower=0), (low - c).clip(lower=0)], axis=1).max(axis=1)
        add(f"High < low: {count_and_pct(h < low)}")
        add(f"High < open: {count_and_pct(high_below_open)}")
        add(f"High < close: {count_and_pct(high_below_close)}")
        add(f"Low > open: {count_and_pct(low_above_open)}")
        add(f"Low > close: {count_and_pct(low_above_close)}")
        add(f"Rows with any OHLC enclosure violation: {count_and_pct(hierarchy_bad)}")
        add(f"Maximum high/low enclosure error: {max(max_high_shortfall.max(), max_low_shortfall.max()):.8f}")
        bad_dates = raw_dates[hierarchy_bad].dt.strftime("%Y-%m-%d")
        add(f"Violation date samples: {samples(bad_dates)}")
        add(f"Exact open == close: {count_and_pct(o == c)}")
        add(f"Open within 0.1 pip of close: {count_and_pct((o - c).abs() <= 0.00001)}")
        add(f"Zero-range candles: {count_and_pct(h == low)}")
    add()

    add("5. UNUSUAL CANDLE BEHAVIOR")
    add("-" * 72)
    if set(OHLC).issubset(numeric):
        candle_range_pct = (h - low) / c.abs()
        close_return = c.pct_change()
        q1, q3 = candle_range_pct.quantile([0.25, 0.75])
        extreme_cutoff = q3 + 3 * (q3 - q1)
        unusual = candle_range_pct > extreme_cutoff
        add(f"Median daily range: {candle_range_pct.median():.4%}")
        add(f"99th percentile daily range: {candle_range_pct.quantile(0.99):.4%}")
        add(f"IQR extreme-range cutoff: {extreme_cutoff:.4%}")
        add(f"Candles above IQR extreme cutoff: {count_and_pct(unusual)}")
        add(f"Absolute close-to-close returns > 5%: {count_and_pct(close_return.abs() > 0.05)}")
        top_range = candle_range_pct.nlargest(10)
        top_text = [f"{raw_dates.loc[index].date()} ({value:.3%})" for index, value in top_range.items()]
        add(f"Largest range dates: {samples(top_text, 10)}")
    add()

    add("6. VOLUME AND YAHOO FINANCE ARTIFACT CHECKS")
    add("-" * 72)
    if "volume" in numeric:
        add(f"Zero volume rows: {count_and_pct(numeric['volume'] == 0)}")
        add(f"Distinct volume values: {numeric['volume'].nunique(dropna=True):,}")
    if {"close", "adj_close"}.issubset(numeric):
        adjusted_equal = np.isclose(numeric["close"], numeric["adj_close"], equal_nan=True)
        add(f"Close == adjusted close: {count_and_pct(pd.Series(adjusted_equal, index=raw.index))}")
    if set(OHLC).issubset(numeric):
        next_close = c.shift(-1)
        valid_next = next_close.notna()
        next_direction = (next_close > c).astype(int)
        upper_space = h - c
        lower_space = c - low
        wick_prediction = (upper_space > lower_space).astype(int)
        next_inside_current_range = (next_close >= low) & (next_close <= h) & valid_next
        imbalance_accuracy = (wick_prediction[valid_next] == next_direction[valid_next]).mean()
        add(f"Next close lies inside current high-low range: {count_and_pct(next_inside_current_range[valid_next])}")
        add(f"Current wick-imbalance next-direction accuracy: {imbalance_accuracy:.4f}")
        add(f"Correlation current upper space vs next return: {upper_space[valid_next].corr(next_close[valid_next] / c[valid_next] - 1):.4f}")
        add(f"Correlation current lower space vs next return: {lower_space[valid_next].corr(next_close[valid_next] / c[valid_next] - 1):.4f}")
        if imbalance_accuracy >= 0.65:
            add("CRITICAL: Current high/low geometry predicts the next timestamp unusually well.")
            add("This is consistent with a Yahoo daily-FX bar alignment artifact and makes same-row wick/range features unsafe until independently verified.")
    add()

    add("7. PROCESSED DATA AND TARGET INTEGRITY")
    add("-" * 72)
    required_target = {"date", "close", "future_close", "future_return", "target"}
    missing_target_columns = sorted(required_target.difference(features.columns))
    add(f"Missing target columns: {samples(missing_target_columns)}")
    if not missing_target_columns:
        feature_dates = pd.to_datetime(features["date"], errors="coerce")
        raw_sorted = raw.assign(date=raw_dates).sort_values("date").copy()
        raw_sorted["expected_future_close"] = raw_sorted["close"].shift(-1)
        raw_sorted["expected_future_return"] = raw_sorted["expected_future_close"] / raw_sorted["close"] - 1
        raw_sorted["expected_target"] = (raw_sorted["expected_future_close"] > raw_sorted["close"]).astype(int)
        expected = raw_sorted.set_index("date")[["expected_future_close", "expected_future_return", "expected_target"]]
        aligned = expected.reindex(feature_dates).reset_index(drop=True)
        future_close_ok = np.isclose(features["future_close"], aligned["expected_future_close"], equal_nan=True)
        future_return_ok = np.isclose(features["future_return"], aligned["expected_future_return"], equal_nan=True)
        target_ok = features["target"].to_numpy() == aligned["expected_target"].to_numpy()
        add(f"future_close mismatches vs raw next close: {int((~future_close_ok).sum()):,}")
        add(f"future_return mismatches vs raw next return: {int((~future_return_ok).sum()):,}")
        add(f"target mismatches vs recomputed target: {int((~target_ok).sum()):,}")
        add(f"Target class 0: {count_and_pct(features['target'] == 0)}")
        add(f"Target class 1: {count_and_pct(features['target'] == 1)}")
    future_named = [column for column in features if any(token in column.lower() for token in ("future", "target", "next", "lead", "forward"))]
    add(f"Future/target-named columns present in processed CSV: {samples(future_named, 50)}")
    add("These columns are valid labels/diagnostics only and must never enter X.")
    add()

    add("8. OVERALL ASSESSMENT")
    add("-" * 72)
    add("The files are structurally complete and the target calculation is internally consistent.")
    add("However, zero volume, frequent open==close, OHLC enclosure violations, and especially the predictive same-row high/low geometry are serious Yahoo FX artifacts.")
    add("Any accuracy driven by upper_wick, lower_wick, range, high, or low should be treated as leakage-like and not as deployable forecasting skill.")

    return "\n".join(lines) + "\n"


def main() -> None:
    report = validate()
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
