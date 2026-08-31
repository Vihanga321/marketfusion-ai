"""Causal, cost-aware XAUUSD target research for 15m/60m/240m horizons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.assets.contracts import BrokerSymbolSpec

HORIZONS = (15, 60, 240)
CLASS_LABELS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


@dataclass(frozen=True)
class TargetBand:
    name: str
    cost_multiplier: float
    atr_fraction: float


CANDIDATE_BANDS = (
    TargetBand("COST_1X_ATR_05", 1.0, 0.05),
    TargetBand("COST_1_5X_ATR_10", 1.5, 0.10),
    TargetBand("COST_2X_ATR_15", 2.0, 0.15),
)
FINAL_BAND = CANDIDATE_BANDS[1]


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous).abs(),
        (frame["low"] - previous).abs(),
    ], axis=1).max(axis=1)


def causal_atr(frame: pd.DataFrame, periods: int = 14) -> pd.Series:
    """ATR known at the completed decision bar; no future values are used."""
    return true_range(frame).rolling(periods, min_periods=periods).mean()


def neutral_band_return(
    close: pd.Series,
    spread_points: pd.Series,
    atr: pd.Series,
    spec: BrokerSymbolSpec,
    band: TargetBand = FINAL_BAND,
) -> pd.Series:
    price = pd.to_numeric(close, errors="coerce")
    spread_price = pd.to_numeric(spread_points, errors="coerce").clip(lower=0) * spec.point
    cost_return = spread_price * band.cost_multiplier / price
    atr_return = pd.to_numeric(atr, errors="coerce") * band.atr_fraction / price
    return pd.concat([cost_return, atr_return], axis=1).max(axis=1)


def target_classes(future_return: pd.Series, neutral_band: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=future_return.index, dtype="Int64")
    valid = future_return.notna() & neutral_band.notna()
    result.loc[valid & future_return.lt(-neutral_band)] = 0
    result.loc[valid & future_return.abs().le(neutral_band)] = 1
    result.loc[valid & future_return.gt(neutral_band)] = 2
    return result


def build_target_research(
    completed_m5: pd.DataFrame,
    spec: BrokerSymbolSpec,
    bands: Iterable[TargetBand] = CANDIDATE_BANDS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = completed_m5.sort_values("bar_close_utc").reset_index(drop=True).copy()
    frame["atr_14"] = causal_atr(frame)
    report: dict[str, object] = {
        "asset_id": spec.asset_id,
        "broker_symbol": spec.broker_symbol,
        "construction": "COMPLETED_M5_DECISION_TO_EXACT_FUTURE_COMPLETED_M5_CLOSE",
        "target_leakage": "PASS",
        "selection_rule": "cost realism + ATR meaning + class stability; never optimized for model accuracy",
        "selected_band": FINAL_BAND.name,
        "candidates": {},
    }
    for horizon in HORIZONS:
        steps = horizon // 5
        future_timestamp = frame["bar_close_utc"].shift(-steps)
        expected_timestamp = pd.to_datetime(frame["bar_close_utc"], utc=True) + pd.Timedelta(minutes=horizon)
        exact = pd.to_datetime(future_timestamp, utc=True).eq(expected_timestamp)
        future_return = (frame["close"].shift(-steps) / frame["close"] - 1.0).where(exact)
        frame[f"outcome_future_timestamp_{horizon}m"] = future_timestamp.where(exact)
        frame[f"outcome_future_return_{horizon}m"] = future_return
        report["candidates"][str(horizon)] = {}
        for band in bands:
            neutral = neutral_band_return(frame["close"], frame["spread_points"], frame["atr_14"], spec, band)
            classes = target_classes(future_return, neutral)
            counts = classes.value_counts().to_dict()
            total = int(classes.notna().sum())
            distribution = {CLASS_LABELS[index]: {"rows": int(counts.get(index, 0)), "fraction": (float(counts.get(index, 0) / total) if total else None)} for index in CLASS_LABELS}
            report["candidates"][str(horizon)][band.name] = {"rows": total, "distribution": distribution}
            if band == FINAL_BAND:
                frame[f"target_neutral_band_{horizon}m"] = neutral
                frame[f"target_class_{horizon}m"] = classes
    return frame, report


def baseline_report(targets: pd.DataFrame) -> dict[str, object]:
    """Evaluate non-ML baselines without shuffled or reconstructed predictions."""
    report: dict[str, object] = {}
    for horizon in HORIZONS:
        actual = targets[f"target_class_{horizon}m"].dropna().astype(int)
        if actual.empty:
            report[str(horizon)] = {"status": "INSUFFICIENT_DATA"}
            continue
        majority = int(actual.mode().iloc[0])
        majority_accuracy = float(actual.eq(majority).mean())
        # Causal directional baseline: previous completed M5 return sign, neutral only at zero.
        previous_return = targets.loc[actual.index, "close"].pct_change().shift(1)
        predicted = pd.Series(np.where(previous_return > 0, 2, np.where(previous_return < 0, 0, 1)), index=actual.index)
        valid = previous_return.notna()
        report[str(horizon)] = {
            "rows": len(actual), "majority_class": CLASS_LABELS[majority],
            "majority_accuracy": majority_accuracy,
            "class_prior": {CLASS_LABELS[index]: float(actual.eq(index).mean()) for index in CLASS_LABELS},
            "previous_direction_accuracy": float(predicted.loc[valid].eq(actual.loc[valid]).mean()) if valid.any() else None,
            "promotion": "NO_APPROVED_MODEL",
        }
    return report
