"""Future-only MFE/MAE evaluation for advisory windows actually emitted live."""
from __future__ import annotations

import pandas as pd


def evaluate_window(prediction: pd.Series, bars: pd.DataFrame, now_utc: object) -> dict[str, object] | None:
    start_value, end_value = prediction.get("suggested_start_utc"), prediction.get("suggested_end_utc")
    action = prediction.get("advisory")
    if not start_value or not end_value or action not in {"BUY_BIAS", "SELL_BIAS"}:
        return None
    start, end = pd.Timestamp(start_value), pd.Timestamp(end_value)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    now = pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if now < end:
        return None
    frame = bars.copy()
    frame["bar_open_utc"] = pd.to_datetime(frame["bar_open_utc"], utc=True, errors="raise")
    frame = frame.loc[frame["bar_open_utc"].ge(start) & frame["bar_open_utc"].lt(end)].sort_values("bar_open_utc")
    expected = int((end - start).total_seconds() // 60)
    if len(frame) != expected:
        return None
    entry = float(prediction["market_mid"])
    direction = 1.0 if action == "BUY_BIAS" else -1.0
    favorable = direction * (pd.to_numeric(frame["high" if direction > 0 else "low"]) - entry)
    adverse = direction * (pd.to_numeric(frame["low" if direction > 0 else "high"]) - entry)
    exit_price = float(frame.iloc[-1]["close"])
    return {
        "prediction_id": prediction["prediction_id"], "direction": action,
        "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
        "realized_move_pips": direction * (exit_price - entry) / 0.0001,
        "mfe_pips": float(favorable.max() / 0.0001), "mae_pips": float(adverse.min() / 0.0001),
        "mfe_return": float(favorable.max() / entry), "mae_return": float(adverse.min() / entry),
        "evaluation_label": "SHADOW WINDOW EVALUATION",
    }
