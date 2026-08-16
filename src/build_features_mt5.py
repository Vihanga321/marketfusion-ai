"""Build causal EURUSD MT5 H1 features only after H1 history validation passes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validate_mt5_history import HISTORY_DIRECTORY, SYMBOL, validate_timeframe


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = HISTORY_DIRECTORY / f"{SYMBOL}_H1.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "eurusd_mt5_features.parquet"
RETURN_HORIZONS = (1, 3, 6, 12)
VOLATILITY_WINDOWS = (6, 12, 24, 72)
MA_WINDOWS = (6, 12, 24, 72)
MOMENTUM_HORIZONS = (3, 6, 12)
TARGET_HORIZONS = (1, 4)

FEATURES_MT5 = [
    "return_1h", "return_3h", "return_6h", "return_12h",
    "volatility_6h", "volatility_12h", "volatility_24h", "volatility_72h",
    "price_vs_ma_6h", "price_vs_ma_12h", "price_vs_ma_24h", "price_vs_ma_72h",
    "rsi_14h",
    "momentum_3h", "momentum_6h", "momentum_12h",
    "hour_of_day", "day_of_week", "spread_points", "tick_volume",
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


def exact_prior(series: pd.Series, timestamps: pd.Series, hours: int) -> pd.Series:
    prior = series.shift(hours)
    exact = timestamps.sub(timestamps.shift(hours)).eq(pd.Timedelta(hours=hours))
    return prior.where(exact)


def build_features() -> pd.DataFrame:
    validation = validate_timeframe("H1", write_report=True)
    if not validation.passed:
        raise RuntimeError(
            "MT5 H1 history failed validation; feature generation blocked: "
            + "; ".join(validation.failures)
        )

    raw = pd.read_parquet(INPUT_FILE, engine="pyarrow").sort_values("timestamp_utc").reset_index(drop=True)
    df = raw[["timestamp_utc", "close", "spread_points", "tick_volume"]].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["decision_timestamp"] = df["timestamp_utc"] + pd.Timedelta(hours=1)
    close = df["close"]

    for hours in RETURN_HORIZONS:
        df[f"return_{hours}h"] = close / exact_prior(close, df["timestamp_utc"], hours) - 1
    for hours in VOLATILITY_WINDOWS:
        df[f"volatility_{hours}h"] = df["return_1h"].rolling(hours, min_periods=hours).std()
    for hours in MA_WINDOWS:
        average = close.rolling(hours, min_periods=hours).mean()
        df[f"price_vs_ma_{hours}h"] = close / average - 1
    df["rsi_14h"] = rsi_wilder(close, 14)
    for hours in MOMENTUM_HORIZONS:
        df[f"momentum_{hours}h"] = close - exact_prior(close, df["timestamp_utc"], hours)
    df["hour_of_day"] = df["timestamp_utc"].dt.hour
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek

    for hours in TARGET_HORIZONS:
        future_timestamp = df["timestamp_utc"].shift(-hours)
        future_close = close.shift(-hours)
        exact = future_timestamp.sub(df["timestamp_utc"]).eq(pd.Timedelta(hours=hours))
        future_return = (future_close / close - 1).where(exact)
        df[f"future_timestamp_{hours}h"] = future_timestamp.where(exact)
        df[f"future_return_{hours}h"] = future_return
        df[f"target_{hours}h"] = (future_return > 0).astype("Int8").where(exact)

    before = len(df)
    df = df.dropna(subset=FEATURES_MT5).reset_index(drop=True)
    if np.isinf(df[FEATURES_MT5].to_numpy()).any():
        raise ValueError("Infinite values found in MT5 features")
    forbidden = {"upper_wick", "lower_wick"}.intersection(FEATURES_MT5)
    if forbidden:
        raise AssertionError(f"Forbidden wick features entered model feature list: {sorted(forbidden)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".tmp.parquet")
    df.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OUTPUT_FILE)
    print("MT5 H1 validation: PASS")
    print(f"Input H1 bars: {before:,}")
    print(f"Feature rows: {len(df):,}")
    print(f"Features: {len(FEATURES_MT5)}")
    print(f"1H labeled rows: {int(df['target_1h'].notna().sum()):,}")
    print(f"4H labeled rows: {int(df['target_4h'].notna().sum()):,}")
    print(f"Saved: {OUTPUT_FILE}")
    return df


def main() -> None:
    try:
        build_features()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
