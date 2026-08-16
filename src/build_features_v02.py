"""Build causal MarketFusion AI V0.2 features after OANDA validation passes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validate_oanda_data import DATA_FILE, validate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "processed" / "eurusd_h1_features_v02.parquet"
RETURN_HORIZONS = (1, 3, 6, 12)
VOLATILITY_WINDOWS = (6, 12, 24)
MA_WINDOWS = (6, 12, 24, 72)
MOMENTUM_HORIZONS = (3, 6, 12)
TARGET_HORIZONS = (1, 4)

FEATURES_V02 = [
    "return_1h", "return_3h", "return_6h", "return_12h",
    "volatility_6h", "volatility_12h", "volatility_24h",
    "price_vs_ma_6h", "price_vs_ma_12h", "price_vs_ma_24h", "price_vs_ma_72h",
    "rsi_14h",
    "momentum_3h", "momentum_6h", "momentum_12h",
    "hour_of_day", "day_of_week", "spread_close",
]


def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    change = series.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    result = result.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return result.mask((average_loss == 0) & (average_gain == 0), 50.0)


def exact_horizon(series: pd.Series, timestamps: pd.Series, periods: int) -> pd.Series:
    result = series.shift(periods)
    contiguous = timestamps.sub(timestamps.shift(periods)).eq(pd.Timedelta(hours=periods))
    return result.where(contiguous)


def build_features() -> pd.DataFrame:
    validation = validate(DATA_FILE, write_report=True)
    if not validation.passed:
        details = "; ".join(validation.failures)
        raise RuntimeError(f"OANDA data failed validation; feature generation blocked: {details}")

    raw = pd.read_parquet(DATA_FILE, engine="pyarrow").sort_values("timestamp").reset_index(drop=True)
    df = raw[["timestamp", "mid_close", "bid_close", "ask_close", "spread_close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # OANDA stamps an H1 candle at its start. Its complete OHLC is actionable
    # only after the interval ends.
    df["decision_timestamp"] = df["timestamp"] + pd.Timedelta(hours=1)
    close = df["mid_close"]

    for hours in RETURN_HORIZONS:
        prior_close = exact_horizon(close, df["timestamp"], hours)
        df[f"return_{hours}h"] = close / prior_close - 1

    for hours in VOLATILITY_WINDOWS:
        df[f"volatility_{hours}h"] = df["return_1h"].rolling(hours, min_periods=hours).std()

    for hours in MA_WINDOWS:
        moving_average = close.rolling(hours, min_periods=hours).mean()
        df[f"price_vs_ma_{hours}h"] = close / moving_average - 1

    df["rsi_14h"] = rsi_wilder(close, 14)

    for hours in MOMENTUM_HORIZONS:
        prior_close = exact_horizon(close, df["timestamp"], hours)
        df[f"momentum_{hours}h"] = close - prior_close

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    for hours in TARGET_HORIZONS:
        future_timestamp = df["timestamp"].shift(-hours)
        future_close = close.shift(-hours)
        exact = future_timestamp.sub(df["timestamp"]).eq(pd.Timedelta(hours=hours))
        df[f"future_timestamp_{hours}h"] = future_timestamp.where(exact)
        df[f"future_return_{hours}h"] = (future_close / close - 1).where(exact)
        df[f"target_{hours}h"] = (df[f"future_return_{hours}h"] > 0).astype("Int8").where(exact)

    required = FEATURES_V02 + [
        "future_timestamp_1h", "future_return_1h", "target_1h",
        "future_timestamp_4h", "future_return_4h", "target_4h",
    ]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    numeric = df[FEATURES_V02 + ["future_return_1h", "future_return_4h"]]
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite values found in V0.2 features")
    if any(name in FEATURES_V02 for name in ("upper_wick", "lower_wick")):
        raise AssertionError("Forbidden wick feature entered FEATURES_V02")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".tmp.parquet")
    df.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OUTPUT_FILE)
    print(f"OANDA validation: PASS")
    print(f"Input rows: {before:,}")
    print(f"Labeled feature rows: {len(df):,}")
    print(f"Feature count: {len(FEATURES_V02)}")
    print(f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved: {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    try:
        build_features()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
