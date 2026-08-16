import pandas as pd
import numpy as np
from pathlib import Path


INPUT_FILE = Path("data/raw/eurusd_daily.csv")
OUTPUT_FILE = Path("data/processed/eurusd_features.csv")


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def build_features():

    print("=" * 60)
    print("MARKETFUSION AI - FEATURE ENGINE")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------
    # PRICE RETURNS
    # -----------------------------

    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)


    # -----------------------------
    # CANDLE INFORMATION
    # -----------------------------

    df["range"] = df["high"] - df["low"]

    df["body"] = df["close"] - df["open"]

    df["upper_wick"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )


    # -----------------------------
    # MOVING AVERAGES
    # -----------------------------

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    df["price_vs_ma10"] = (
        df["close"] / df["ma_10"] - 1
    )

    df["price_vs_ma20"] = (
        df["close"] / df["ma_20"] - 1
    )

    df["ma10_vs_ma20"] = (
        df["ma_10"] / df["ma_20"] - 1
    )


    # -----------------------------
    # RSI
    # -----------------------------

    df["rsi_14"] = rsi(df["close"], 14)


    # -----------------------------
    # VOLATILITY
    # -----------------------------

    df["volatility_5"] = (
        df["return_1d"].rolling(5).std()
    )

    df["volatility_10"] = (
        df["return_1d"].rolling(10).std()
    )

    df["volatility_20"] = (
        df["return_1d"].rolling(20).std()
    )


    # -----------------------------
    # ATR
    # -----------------------------

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = abs(
        df["high"] - previous_close
    )

    tr3 = abs(
        df["low"] - previous_close
    )

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr_14"] = (
        true_range.rolling(14).mean()
    )


    # -----------------------------
    # MOMENTUM
    # -----------------------------

    df["momentum_3"] = (
        df["close"] - df["close"].shift(3)
    )

    df["momentum_5"] = (
        df["close"] - df["close"].shift(5)
    )

    df["momentum_10"] = (
        df["close"] - df["close"].shift(10)
    )


    # -----------------------------
    # CALENDAR
    # -----------------------------

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month


    # -----------------------------
    # FUTURE TARGET
    # -----------------------------

    df["future_close"] = df["close"].shift(-1)

    df["future_return"] = (
        df["future_close"] / df["close"] - 1
    )

    df["target"] = (
        df["future_close"] > df["close"]
    ).astype(int)


    # Remove incomplete rows

    df = df.dropna().reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Rows ready: {len(df):,}")
    print(f"Features: {len(df.columns)}")

    print()
    print("Target distribution:")

    print(
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print()
    print(df.tail())

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print("Feature dataset ready ✅")


if __name__ == "__main__":
    build_features()