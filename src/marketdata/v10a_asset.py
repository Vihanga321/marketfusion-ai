"""Read-only V1.0A MT5 asset collector, audit, features, and target research."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from src.assets.contracts import BrokerSymbolSpec, asset_paths, discover_broker_symbol, normalize_asset_id
from src.marketdata.mt5_continuous_store import atomic_write_parquet, merge_immutable_bars, rates_to_completed_frame
from src.marketdata.v05a_contract import TimeframeSpec
from src.research.v10a_targets import baseline_report, build_target_research, causal_atr


@dataclass(frozen=True)
class CollectionSpec:
    label: str
    mt5_attribute: str
    minutes: int
    rows: int


COLLECTION_SPECS = (
    CollectionSpec("M1", "TIMEFRAME_M1", 1, 100_000),
    CollectionSpec("M5", "TIMEFRAME_M5", 5, 100_000),
    CollectionSpec("M15", "TIMEFRAME_M15", 15, 80_000),
    CollectionSpec("H1", "TIMEFRAME_H1", 60, 50_000),
    CollectionSpec("H4", "TIMEFRAME_H4", 240, 20_000),
    CollectionSpec("D1", "TIMEFRAME_D1", 1440, 10_000),
)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _audit_frame(frame: pd.DataFrame, spec: CollectionSpec) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "first_timestamp": None, "last_timestamp": None, "duplicates": 0, "missing_periods": None, "weekend_gaps": 0, "broker_breaks": 0}
    opens = pd.to_datetime(frame["bar_open_utc"], utc=True)
    gaps = opens.diff().dropna()
    expected = pd.Timedelta(minutes=spec.minutes)
    large = gaps.gt(expected * 1.5)
    weekend = pd.Series(opens.shift(1).dt.dayofweek.ge(4).to_numpy(), index=opens.index) & opens.dt.dayofweek.le(0)
    return {
        "rows": len(frame), "first_timestamp": opens.iloc[0].isoformat(),
        "last_timestamp": opens.iloc[-1].isoformat(), "duplicates": int(opens.duplicated().sum()),
        "missing_periods": int(sum(max(0, round(value / expected) - 1) for value in gaps[large])),
        "weekend_gaps": int((large & weekend.reindex(large.index, fill_value=False)).sum()),
        "broker_breaks": int((large & ~weekend.reindex(large.index, fill_value=False)).sum()),
    }


def build_asset_features(m5: pd.DataFrame, output: Path) -> pd.DataFrame:
    frame = m5.sort_values("bar_close_utc").reset_index(drop=True).copy()
    result = pd.DataFrame({"decision_timestamp_utc": pd.to_datetime(frame["bar_close_utc"], utc=True)})
    result["close"] = frame["close"].astype(float)
    result["m5_return_5m"] = frame["close"].pct_change(1)
    result["m5_return_15m"] = frame["close"].pct_change(3)
    result["m5_return_60m"] = frame["close"].pct_change(12)
    result["m5_return_240m"] = frame["close"].pct_change(48)
    result["m5_volatility_60m"] = result["m5_return_5m"].rolling(12, min_periods=12).std()
    result["m5_volatility_240m"] = result["m5_return_5m"].rolling(48, min_periods=48).std()
    result["m5_spread_points"] = frame["spread_points"].astype(float)
    result["m5_spread_price"] = frame["spread_points"].astype(float) * float(frame.attrs.get("point", 1.0))
    result["atr_14"] = causal_atr(frame)
    result["spread_relative_to_price"] = result["m5_spread_price"] / result["close"]
    result["spread_relative_to_atr"] = result["m5_spread_price"] / result["atr_14"].replace(0, pd.NA)
    result["utc_hour"] = result["decision_timestamp_utc"].dt.hour
    result["day_of_week"] = result["decision_timestamp_utc"].dt.dayofweek
    result["feature_complete"] = result[["m5_return_240m", "m5_volatility_240m", "atr_14"]].notna().all(axis=1)
    atomic_write_parquet(result, output)
    return result


def collect_once(mt5: Any, asset_id: str = "XAUUSD", *, include_optional: bool = True, reuse_existing: bool = False, incremental: bool = False) -> dict[str, object]:
    asset = normalize_asset_id(asset_id)
    broker = discover_broker_symbol(mt5, asset)
    if not mt5.symbol_select(broker.broker_symbol, True):
        raise RuntimeError(f"MT5 symbol_select({broker.broker_symbol!r}) failed: {mt5.last_error()}")
    paths = asset_paths(asset)
    captured = datetime.now(timezone.utc)
    bars: dict[str, pd.DataFrame] = {}
    audit: dict[str, object] = {}
    selected = [item for item in COLLECTION_SPECS if include_optional or item.label in {"M1", "M5", "M15", "H1"}]
    for item in selected:
        output = paths.bar_file(item.label, broker.broker_symbol)
        if reuse_existing and output.exists():
            frame = pd.read_parquet(output)
        else:
            requested_rows = min(item.rows, 500) if incremental else item.rows
            rates = mt5.copy_rates_from_pos(broker.broker_symbol, getattr(mt5, item.mt5_attribute), 1, requested_rows)
            if rates is None:
                raise RuntimeError(f"MT5 copy_rates_from_pos failed for {item.label}: {mt5.last_error()}")
            contract = TimeframeSpec(item.label, item.mt5_attribute, item.minutes, 0)
            frame = rates_to_completed_frame(rates, contract, captured)
            if incremental and output.exists():
                existing = pd.read_parquet(output)
                merged, conflicts = merge_immutable_bars(existing, frame, contract)
                if not conflicts.empty:
                    raise RuntimeError(f"Immutable {asset} {item.label} bar conflict detected")
                frame = merged
        frame.attrs["point"] = broker.point
        bars[item.label] = frame
        if not (reuse_existing and output.exists()):
            atomic_write_parquet(frame, output)
        audit[item.label] = _audit_frame(frame, item)
    for item in COLLECTION_SPECS:
        if item.label not in audit:
            audit[item.label] = {"rows": 0, "status": "NOT_COLLECTED_BROKER_REQUEST_BLOCKED_DURING_LIVE_AUDIT"}
    tick = mt5.symbol_info_tick(broker.broker_symbol)
    quote = None
    if tick is not None and float(tick.bid) > 0 and float(tick.ask) >= float(tick.bid):
        spread_price = float(tick.ask) - float(tick.bid)
        mid = (float(tick.bid) + float(tick.ask)) / 2
        quote = {
            "bid": float(tick.bid), "ask": float(tick.ask), "mid": mid,
            "spread_price": spread_price, "spread_points": broker.price_to_points(spread_price),
            "spread_relative_to_price": spread_price / mid,
        }
    feature_path = paths.features / "features.parquet"
    bars["M5"].attrs["point"] = broker.point
    features = build_asset_features(bars["M5"], feature_path)
    target_frame, targets = build_target_research(bars["M5"], broker)
    baseline = baseline_report(target_frame)
    atomic_write_parquet(target_frame, paths.features / "target_research.parquet")
    result = {
        "contract_version": "v1.0a-asset-data-v1", "status": "PASS", "asset_id": asset,
        "asset_class": "PRECIOUS_METAL" if asset == "XAUUSD" else "FOREX",
        "captured_at_utc": captured.isoformat(), "broker_contract": broker.to_dict(),
        "timezone": "UTC_NORMALIZED_FROM_MT5_EPOCH", "completed_bars_only": True,
        "data_quality": audit, "quote": quote,
        "spread_diagnostics": {
            "absolute_price_latest": None if quote is None else quote["spread_price"],
            "points_latest": None if quote is None else quote["spread_points"],
            "relative_to_price_latest": None if quote is None else quote["spread_relative_to_price"],
            "relative_to_atr_latest": None if features.empty else features["spread_relative_to_atr"].dropna().iloc[-1] if features["spread_relative_to_atr"].notna().any() else None,
            "historical_points_percentiles": {str(q): float(bars["M5"]["spread_points"].quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)} if not bars["M5"].empty else {},
        },
        "targets": targets, "baselines": baseline,
        "models": {str(h): {"model_status": "NO_APPROVED_MODEL", "model_id": None} for h in (15, 60, 240)},
        "execution": "DISABLED", "runtime": "SHADOW_ADVISORY_ONLY", "manual_confirmation": "REQUIRED",
    }
    _atomic_json(result, paths.runtime / "asset_status.json")
    _atomic_json(broker.to_dict(), paths.runtime / "broker_contract.json")
    _atomic_json(targets, paths.reports / "target_research.json")
    _atomic_json(baseline, paths.reports / "baseline_research.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD", choices=("EURUSD", "XAUUSD"))
    parser.add_argument("--skip-optional", action="store_true", help="Skip H4/D1 if the broker history request is unavailable or blocking")
    parser.add_argument("--reuse-existing", action="store_true", help="Audit existing completed bars without re-downloading them")
    parser.add_argument("--incremental", action="store_true", help="Merge a bounded recent window into the immutable symbol store")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"MetaTrader5 unavailable: {exc}")
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    try:
        while True:
            payload = collect_once(mt5, args.symbol, include_optional=not args.skip_optional, reuse_existing=args.reuse_existing, incremental=args.incremental)
            print(json.dumps(payload, indent=2, default=str))
            if not args.continuous:
                return 0
            time.sleep(max(30, args.interval_seconds))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
