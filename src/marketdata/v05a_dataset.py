"""Build the causal V0.5A multi-timeframe EURUSD feature/label dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.marketdata.mt5_continuous_store import atomic_write_parquet, read_bar_store
from src.marketdata.v05a_contract import (
    CONTINUOUS_DIR,
    FEATURE_COLUMNS,
    FEATURE_FILE,
    OUTCOME_COLUMNS,
    PREDICTION_HORIZONS_MINUTES,
    SYMBOL,
    TIMEFRAMES,
)


def _exact_return(close: pd.Series, end_time: pd.Series, bars: int, minutes: int) -> pd.Series:
    prior_close = close.shift(bars)
    prior_time = end_time.shift(bars)
    exact = end_time.sub(prior_time).eq(pd.Timedelta(minutes=bars * minutes))
    return (close / prior_close - 1.0).where(exact)


def _prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    spec = TIMEFRAMES[label]
    df = frame.copy().sort_values("bar_close_utc").reset_index(drop=True)
    df["bar_open_utc"] = pd.to_datetime(df["bar_open_utc"], utc=True, errors="raise")
    df["bar_close_utc"] = pd.to_datetime(df["bar_close_utc"], utc=True, errors="raise")
    close = pd.to_numeric(df["close"], errors="raise")
    spread = pd.to_numeric(df["spread_points"], errors="raise")
    volume = pd.to_numeric(df["tick_volume"], errors="raise")
    one = _exact_return(close, df["bar_close_utc"], 1, spec.minutes)
    out = pd.DataFrame({"available_from_utc": df["bar_close_utc"]})

    if label == "M1":
        out["m1_return_1m"] = one
        out["m1_return_5m"] = _exact_return(close, df["bar_close_utc"], 5, 1)
        out["m1_volatility_15m"] = one.rolling(15, min_periods=15).std()
        out["m1_spread_points"] = spread
        out["m1_tick_volume"] = volume
    elif label == "M5":
        out["m5_close"] = close
        out["m5_return_5m"] = one
        out["m5_return_15m"] = _exact_return(close, df["bar_close_utc"], 3, 5)
        out["m5_return_60m"] = _exact_return(close, df["bar_close_utc"], 12, 5)
        out["m5_return_240m"] = _exact_return(close, df["bar_close_utc"], 48, 5)
        out["m5_volatility_15m"] = one.rolling(3, min_periods=3).std()
        out["m5_volatility_60m"] = one.rolling(12, min_periods=12).std()
        out["m5_volatility_240m"] = one.rolling(48, min_periods=48).std()
        out["m5_spread_points"] = spread
        out["m5_spread_median_60m"] = spread.rolling(12, min_periods=12).median()
        out["m5_spread_max_60m"] = spread.rolling(12, min_periods=12).max()
        out["m5_tick_volume"] = volume
        out["m5_tick_volume_mean_60m"] = volume.rolling(12, min_periods=12).mean()
    elif label == "M15":
        out["m15_return_15m"] = one
        out["m15_return_60m"] = _exact_return(close, df["bar_close_utc"], 4, 15)
        out["m15_return_240m"] = _exact_return(close, df["bar_close_utc"], 16, 15)
        out["m15_volatility_60m"] = one.rolling(4, min_periods=4).std()
        out["m15_volatility_240m"] = one.rolling(16, min_periods=16).std()
    elif label == "H1":
        out["h1_return_60m"] = one
        out["h1_return_240m"] = _exact_return(close, df["bar_close_utc"], 4, 60)
        out["h1_volatility_240m"] = one.rolling(4, min_periods=4).std()
        out["h1_volatility_1440m"] = one.rolling(24, min_periods=24).std()
    else:
        raise ValueError(f"Unsupported timeframe: {label}")
    return out


def _add_sessions(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    utc = pd.to_datetime(result["decision_timestamp_utc"], utc=True)
    london = utc.dt.tz_convert("Europe/London")
    new_york = utc.dt.tz_convert("America/New_York")
    result["utc_hour"] = utc.dt.hour.astype("int16")
    result["day_of_week"] = utc.dt.dayofweek.astype("int8")
    result["is_asia_session"] = utc.dt.hour.between(0, 8).astype("int8")
    result["is_london_session"] = london.dt.hour.between(8, 16).astype("int8")
    result["is_new_york_session"] = new_york.dt.hour.between(8, 16).astype("int8")
    result["is_london_ny_overlap"] = (
        result["is_london_session"].eq(1) & result["is_new_york_session"].eq(1)
    ).astype("int8")
    return result


def _add_matured_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    decision = pd.to_datetime(result["decision_timestamp_utc"], utc=True)
    close_by_decision = pd.Series(
        pd.to_numeric(result["m5_close"], errors="raise").to_numpy(),
        index=pd.DatetimeIndex(decision),
    )
    current_close = pd.to_numeric(result["m5_close"], errors="raise")
    for minutes in PREDICTION_HORIZONS_MINUTES:
        future_timestamp = decision + pd.Timedelta(minutes=minutes)
        future_close = future_timestamp.map(close_by_decision)
        available = future_close.notna()
        future_return = (pd.Series(future_close, index=result.index) / current_close - 1.0).where(available.to_numpy())
        result[f"outcome_future_timestamp_{minutes}m"] = future_timestamp.where(available.to_numpy())
        result[f"outcome_future_return_{minutes}m"] = future_return
        result[f"outcome_direction_{minutes}m"] = (future_return > 0).astype("Int8").where(future_return.notna())
        result[f"outcome_matured_at_utc_{minutes}m"] = future_timestamp.where(available.to_numpy())
    return result


def build_dataset_from_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    missing = sorted(set(TIMEFRAMES) - set(frames))
    if missing:
        raise ValueError("Missing timeframe frames: " + ", ".join(missing))
    if any(frames[label].empty for label in TIMEFRAMES):
        empty = [label for label in TIMEFRAMES if frames[label].empty]
        raise ValueError("Empty timeframe frames: " + ", ".join(empty))

    prepared = {label: _prepare(frames[label], label) for label in TIMEFRAMES}
    base = prepared["M5"].rename(columns={"available_from_utc": "decision_timestamp_utc"}).copy()
    base["m5_available_from_utc"] = base["decision_timestamp_utc"]

    for label in ("M1", "M15", "H1"):
        right = prepared[label].copy().sort_values("available_from_utc")
        availability_name = f"{label.lower()}_available_from_utc"
        right = right.rename(columns={"available_from_utc": availability_name})
        base = pd.merge_asof(
            base.sort_values("decision_timestamp_utc"),
            right.sort_values(availability_name),
            left_on="decision_timestamp_utc",
            right_on=availability_name,
            direction="backward",
            allow_exact_matches=True,
        )

    base = _add_sessions(base)
    base = _add_matured_outcomes(base)
    feature_missing = base[FEATURE_COLUMNS].isna().any(axis=1)
    base["feature_complete"] = ~feature_missing
    base["contract_version"] = "v0.5a-continuous-market-data-v1"
    ordered = [
        "decision_timestamp_utc",
        "m5_close",
        "m1_available_from_utc", "m5_available_from_utc", "m15_available_from_utc", "h1_available_from_utc",
        *FEATURE_COLUMNS,
        "feature_complete",
        *OUTCOME_COLUMNS,
        "contract_version",
    ]
    return base[ordered].sort_values("decision_timestamp_utc").reset_index(drop=True)


def load_continuous_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for label, spec in TIMEFRAMES.items():
        path = CONTINUOUS_DIR / spec.filename
        frame = read_bar_store(path)
        if frame.empty:
            raise FileNotFoundError(f"Missing/empty V0.5A bar store: {path}")
        frames[label] = frame
    return frames


def build_and_save_dataset(output: Path = FEATURE_FILE) -> pd.DataFrame:
    dataset = build_dataset_from_frames(load_continuous_frames())
    atomic_write_parquet(dataset, output)
    return dataset


def model_feature_view(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return only point-in-time features; outcomes are physically excluded."""
    forbidden = [column for column in FEATURE_COLUMNS if column.startswith("outcome_")]
    if forbidden:
        raise RuntimeError(f"Outcome fields entered feature registry: {forbidden}")
    return dataset[["decision_timestamp_utc", *FEATURE_COLUMNS, "feature_complete"]].copy()
