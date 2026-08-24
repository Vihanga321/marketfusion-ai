"""Read-only MT5 adapters and immutable/atomic V0.5A storage helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.marketdata.v05a_contract import (
    BAR_COLUMNS,
    CONFLICT_DIR,
    CONTINUOUS_DIR,
    QUOTE_COLUMNS,
    QUOTE_DIR,
    STABILIZATION_SECONDS,
    SYMBOL,
    TimeframeSpec,
)

IMMUTABLE_BAR_FIELDS = [
    "bar_close_utc", "open", "high", "low", "close",
    "tick_volume", "spread_points", "real_volume",
]


def _utc_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def rates_to_completed_frame(
    rates: Any,
    spec: TimeframeSpec,
    captured_at_utc: object,
    stabilization_seconds: int = STABILIZATION_SECONDS,
) -> pd.DataFrame:
    """Convert MT5 rates to completed bars only.

    MT5 rate timestamps are bar-open timestamps. A bar is admitted only when its
    scheduled close is at least ``stabilization_seconds`` behind capture time.
    """
    raw = pd.DataFrame(rates)
    required = {"time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError("MT5 rate response missing fields: " + ", ".join(missing))
    if raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    captured = _utc_timestamp(captured_at_utc)
    frame = pd.DataFrame({
        "bar_open_utc": pd.to_datetime(raw["time"], unit="s", utc=True),
        "open": pd.to_numeric(raw["open"], errors="raise"),
        "high": pd.to_numeric(raw["high"], errors="raise"),
        "low": pd.to_numeric(raw["low"], errors="raise"),
        "close": pd.to_numeric(raw["close"], errors="raise"),
        "tick_volume": pd.to_numeric(raw["tick_volume"], errors="raise").astype("int64"),
        "spread_points": pd.to_numeric(raw["spread"], errors="raise").astype("int64"),
        "real_volume": pd.to_numeric(raw["real_volume"], errors="raise").astype("int64"),
    })
    frame["bar_close_utc"] = frame["bar_open_utc"] + pd.Timedelta(minutes=spec.minutes)
    cutoff = captured - pd.Timedelta(seconds=stabilization_seconds)
    frame = frame.loc[frame["bar_close_utc"].le(cutoff)].copy()
    frame["first_observed_utc"] = captured
    frame["source"] = "MT5"
    frame = frame[BAR_COLUMNS].sort_values("bar_open_utc").reset_index(drop=True)
    validate_bar_frame(frame, spec)
    return frame


def validate_bar_frame(frame: pd.DataFrame, spec: TimeframeSpec) -> None:
    missing = sorted(set(BAR_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Bar frame missing columns: " + ", ".join(missing))
    if frame.empty:
        return
    opens = pd.to_datetime(frame["bar_open_utc"], utc=True, errors="raise")
    closes = pd.to_datetime(frame["bar_close_utc"], utc=True, errors="raise")
    observed = pd.to_datetime(frame["first_observed_utc"], utc=True, errors="raise")
    expected_close = opens + pd.Timedelta(minutes=spec.minutes)
    if not closes.eq(expected_close).all():
        raise ValueError(f"{spec.label} bar_close_utc does not equal open + timeframe")
    if opens.duplicated().any():
        raise ValueError(f"Duplicate {spec.label} bar_open_utc values")
    if not opens.is_monotonic_increasing:
        raise ValueError(f"{spec.label} bars are not sorted")
    numeric = frame[["open", "high", "low", "close", "tick_volume", "spread_points", "real_volume"]].apply(
        pd.to_numeric, errors="raise"
    )
    if (numeric[["open", "high", "low", "close"]] <= 0).any(axis=None):
        raise ValueError(f"{spec.label} contains non-positive prices")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError(f"{spec.label} contains high enclosure violations")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError(f"{spec.label} contains low enclosure violations")
    if (numeric["spread_points"] < 0).any():
        raise ValueError(f"{spec.label} contains negative spread")
    if (observed < closes).any():
        raise ValueError(f"{spec.label} contains bars observed before scheduled close")


def _different(existing: pd.Series, incoming: pd.Series, field: str) -> pd.Series:
    if field in {"open", "high", "low", "close"}:
        left = pd.to_numeric(existing, errors="raise")
        right = pd.to_numeric(incoming, errors="raise")
        return (left - right).abs().gt(1e-12)
    if field.endswith("_utc"):
        return pd.to_datetime(existing, utc=True).ne(pd.to_datetime(incoming, utc=True))
    return existing.astype(str).ne(incoming.astype(str))


def merge_immutable_bars(existing: pd.DataFrame, incoming: pd.DataFrame, spec: TimeframeSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append new completed bars while refusing to rewrite already-observed bars."""
    if existing.empty:
        merged = incoming.copy().sort_values("bar_open_utc").reset_index(drop=True)
        validate_bar_frame(merged, spec)
        return merged, pd.DataFrame()
    if incoming.empty:
        validate_bar_frame(existing, spec)
        return existing.copy(), pd.DataFrame()

    left = existing.copy()
    right = incoming.copy()
    left["bar_open_utc"] = pd.to_datetime(left["bar_open_utc"], utc=True)
    right["bar_open_utc"] = pd.to_datetime(right["bar_open_utc"], utc=True)
    overlap = left.merge(right, on="bar_open_utc", how="inner", suffixes=("_stored", "_incoming"))
    conflict_mask = pd.Series(False, index=overlap.index)
    for field in IMMUTABLE_BAR_FIELDS:
        conflict_mask |= _different(overlap[f"{field}_stored"], overlap[f"{field}_incoming"], field)
    conflicts = overlap.loc[conflict_mask].copy()
    blocked_keys = set(overlap.loc[conflict_mask, "bar_open_utc"].tolist())

    existing_keys = set(left["bar_open_utc"].tolist())
    appendable = right.loc[~right["bar_open_utc"].isin(existing_keys | blocked_keys)].copy()
    merged = pd.concat([left, appendable], ignore_index=True).sort_values("bar_open_utc").reset_index(drop=True)
    validate_bar_frame(merged, spec)
    return merged[BAR_COLUMNS], conflicts


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def read_bar_store(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BAR_COLUMNS)
    frame = pd.read_parquet(path, engine="pyarrow")
    for column in ("bar_open_utc", "bar_close_utc", "first_observed_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return frame[BAR_COLUMNS].sort_values("bar_open_utc").reset_index(drop=True)


def persist_conflicts(conflicts: pd.DataFrame, spec: TimeframeSpec, captured_at_utc: object) -> Path | None:
    if conflicts.empty:
        return None
    CONFLICT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_timestamp(captured_at_utc).strftime("%Y%m%dT%H%M%SZ")
    path = CONFLICT_DIR / f"{SYMBOL}_{spec.label}_{stamp}.csv"
    conflicts.to_csv(path, index=False, lineterminator="\n")
    return path


def quote_to_frame(tick: Any, point_size: float, captured_at_utc: object) -> pd.DataFrame:
    captured = _utc_timestamp(captured_at_utc)
    bid = float(tick.bid)
    ask = float(tick.ask)
    if bid <= 0 or ask <= 0 or ask < bid or point_size <= 0:
        raise ValueError("Invalid MT5 quote")
    time_msc = getattr(tick, "time_msc", None)
    if time_msc:
        broker_tick_time = pd.to_datetime(int(time_msc), unit="ms", utc=True)
    else:
        broker_tick_time = pd.to_datetime(int(tick.time), unit="s", utc=True)
    spread = ask - bid
    row = {
        "captured_at_utc": captured,
        "broker_tick_time_utc": broker_tick_time,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "spread_price": spread,
        "spread_points": spread / point_size,
        "point_size": point_size,
        "source": "MT5",
    }
    return pd.DataFrame([row], columns=QUOTE_COLUMNS)


def append_quote_partition(frame: pd.DataFrame) -> Path:
    if frame.empty:
        raise ValueError("Cannot append an empty quote frame")
    captured = pd.to_datetime(frame["captured_at_utc"], utc=True, errors="raise")
    day = captured.iloc[0].strftime("%Y-%m-%d")
    path = QUOTE_DIR / f"{SYMBOL}_{day}.parquet"
    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    combined["captured_at_utc"] = pd.to_datetime(combined["captured_at_utc"], utc=True)
    combined["broker_tick_time_utc"] = pd.to_datetime(combined["broker_tick_time_utc"], utc=True)
    combined = combined.drop_duplicates(
        subset=["captured_at_utc", "broker_tick_time_utc", "bid", "ask"], keep="first"
    ).sort_values("captured_at_utc").reset_index(drop=True)
    atomic_write_parquet(combined[QUOTE_COLUMNS], path)
    return path


class Mt5ReadOnlySession:
    """Minimal read-only MT5 session. Trading APIs are intentionally not exposed."""

    def __init__(self, symbol: str = SYMBOL) -> None:
        self.symbol = symbol
        self.mt5 = None

    def __enter__(self) -> "Mt5ReadOnlySession":
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise RuntimeError("MetaTrader5 package is not installed in this Python environment") from exc
        self.mt5 = mt5
        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise RuntimeError(f"mt5.symbol_select({self.symbol!r}, True) failed: {mt5.last_error()}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()

    def symbol_point(self) -> float:
        info = self.mt5.symbol_info(self.symbol)
        if info is None or float(info.point) <= 0:
            raise RuntimeError(f"Could not read symbol info for {self.symbol}")
        return float(info.point)

    def latest_quote(self, captured_at_utc: object) -> pd.DataFrame:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({self.symbol}) failed: {self.mt5.last_error()}")
        return quote_to_frame(tick, self.symbol_point(), captured_at_utc)

    def fetch_rates(self, spec: TimeframeSpec, start_utc: datetime, end_utc: datetime):
        timeframe = getattr(self.mt5, spec.mt5_attribute)
        rates = self.mt5.copy_rates_range(self.symbol, timeframe, start_utc, end_utc)
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed for {spec.label}: {self.mt5.last_error()}")
        return rates


def sync_timeframe(
    session: Mt5ReadOnlySession,
    spec: TimeframeSpec,
    captured_at_utc: datetime,
    overlap_bars: int,
) -> dict[str, object]:
    path = CONTINUOUS_DIR / spec.filename
    existing = read_bar_store(path)
    if existing.empty:
        start = captured_at_utc - timedelta(days=spec.bootstrap_days)
    else:
        last_open = pd.Timestamp(existing["bar_open_utc"].iloc[-1]).to_pydatetime()
        start = last_open - timedelta(minutes=spec.minutes * max(overlap_bars, 1))
    rates = session.fetch_rates(spec, start, captured_at_utc)
    incoming = rates_to_completed_frame(rates, spec, captured_at_utc)
    merged, conflicts = merge_immutable_bars(existing, incoming, spec)
    atomic_write_parquet(merged, path)
    conflict_path = persist_conflicts(conflicts, spec, captured_at_utc)
    before = len(existing)
    return {
        "timeframe": spec.label,
        "rows_before": before,
        "rows_after": len(merged),
        "new_rows": max(len(merged) - before, 0),
        "incoming_rows": len(incoming),
        "conflict_rows": len(conflicts),
        "conflict_file": str(conflict_path) if conflict_path else None,
        "first_bar_utc": merged["bar_open_utc"].iloc[0].isoformat() if len(merged) else None,
        "last_bar_close_utc": merged["bar_close_utc"].iloc[-1].isoformat() if len(merged) else None,
        "path": str(path),
    }
