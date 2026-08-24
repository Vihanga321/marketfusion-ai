"""Create finalized daily EURUSD history summaries from the V0.5A M5 store."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.marketdata.mt5_continuous_store import atomic_write_parquet
from src.marketdata.v05a_contract import ROOT

DAILY_HISTORY_FILE = ROOT / "data" / "processed" / "v05a_eurusd_daily_market_history.parquet"


def build_daily_history(m5: pd.DataFrame, now_utc: object | None = None) -> pd.DataFrame:
    if m5.empty:
        raise ValueError("Cannot build daily history from empty M5 data")
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    df = m5.copy().sort_values("bar_open_utc")
    df["bar_open_utc"] = pd.to_datetime(df["bar_open_utc"], utc=True, errors="raise")
    df["bar_close_utc"] = pd.to_datetime(df["bar_close_utc"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "tick_volume", "spread_points"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["date_utc"] = df["bar_close_utc"].dt.strftime("%Y-%m-%d")
    df["return_5m"] = df["close"].pct_change()
    consecutive = df["bar_close_utc"].diff().eq(pd.Timedelta(minutes=5))
    df.loc[~consecutive, "return_5m"] = pd.NA

    rows: list[dict[str, object]] = []
    today = now.strftime("%Y-%m-%d")
    for day, group in df.groupby("date_utc", sort=True):
        group = group.sort_values("bar_open_utc")
        returns = pd.to_numeric(group["return_5m"], errors="coerce")
        rows.append({
            "date_utc": day,
            "first_bar_open_utc": group["bar_open_utc"].iloc[0],
            "last_bar_close_utc": group["bar_close_utc"].iloc[-1],
            "open": float(group["open"].iloc[0]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group["close"].iloc[-1]),
            "close_to_close_return": float(group["close"].iloc[-1] / group["open"].iloc[0] - 1.0),
            "realized_volatility_5m": float(returns.std()) if returns.notna().sum() >= 2 else None,
            "mean_spread_points": float(group["spread_points"].mean()),
            "max_spread_points": float(group["spread_points"].max()),
            "tick_volume_total": int(group["tick_volume"].sum()),
            "m5_bar_count": int(len(group)),
            "day_status": "PROVISIONAL_CURRENT_UTC_DAY" if day == today else "FINALIZED_UTC_DAY",
            "available_from_utc": group["bar_close_utc"].iloc[-1],
            "source": "MT5_V05A_M5",
        })
    return pd.DataFrame(rows)


def finalized_daily_history(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.loc[daily["day_status"].eq("FINALIZED_UTC_DAY")].copy().reset_index(drop=True)


def build_and_save_daily_history(m5: pd.DataFrame, output: Path = DAILY_HISTORY_FILE, now_utc: object | None = None) -> pd.DataFrame:
    daily = build_daily_history(m5, now_utc)
    atomic_write_parquet(daily, output)
    return daily
