"""Deterministic descriptive metrics for true forward-shadow observations."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.v08_contract import (
    DESCRIPTIVE_BOOTSTRAP_SEED, MIN_DIRECTIONAL_CALLS, MIN_FORWARD_SHADOW_ROWS,
)


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def brier_score(classes: Iterable[int], probabilities: np.ndarray) -> float | None:
    actual = np.asarray(list(classes), dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if not len(actual):
        return None
    if probs.shape != (len(actual), 3) or not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must be finite N x 3 rows summing to one")
    expected = np.eye(3)[actual]
    return float(np.mean(np.sum((probs - expected) ** 2, axis=1)))


def multiclass_log_loss(classes: Iterable[int], probabilities: np.ndarray) -> float | None:
    actual = np.asarray(list(classes), dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if not len(actual):
        return None
    if probs.shape != (len(actual), 3):
        raise ValueError("Probabilities must be N x 3")
    clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    return float(-np.mean(np.log(clipped[np.arange(len(actual)), actual])))


def expected_calibration_error(classes: Iterable[int], probabilities: np.ndarray, bins: int = 10) -> tuple[float | None, float | None]:
    actual = np.asarray(list(classes), dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if not len(actual):
        return None, None
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predicted == actual
    ece, mce = 0.0, 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if not selected.any():
            continue
        gap = abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
        ece += float(selected.mean()) * gap
        mce = max(mce, gap)
    return ece, mce


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_interval(values: Iterable[float], samples: int = 1000) -> tuple[float | None, float | None]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 20:
        return None, None
    rng = np.random.default_rng(DESCRIPTIVE_BOOTSTRAP_SEED)
    means = np.asarray([rng.choice(data, len(data), replace=True).mean() for _ in range(samples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if not len(actual):
        return None
    scores = []
    for label in (0, 1, 2):
        tp = int(((actual == label) & (predicted == label)).sum())
        fp = int(((actual != label) & (predicted == label)).sum())
        fn = int(((actual == label) & (predicted != label)).sum())
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    recalls = [float((predicted[actual == label] == label).mean()) for label in (0, 1, 2) if (actual == label).any()]
    return None if not recalls else float(np.mean(recalls))


def evaluate_horizon(joined: pd.DataFrame, horizon: int) -> dict[str, object]:
    frame = joined.loc[joined["horizon_minutes"].eq(horizon)].copy() if not joined.empty else joined.copy()
    total = len(frame)
    action = frame.get("advisory", pd.Series(dtype=str)).astype(str)
    directional = action.isin(["BUY_BIAS", "SELL_BIAS"])
    waits = action.eq("WAIT")
    actual = pd.to_numeric(frame.get("target_class", pd.Series(dtype=float)), errors="coerce")
    predicted_direction = np.where(action.eq("BUY_BIAS"), 2, np.where(action.eq("SELL_BIAS"), 0, 1)).astype(int)
    directional_actual = actual.loc[directional].to_numpy(dtype=int) if directional.any() else np.asarray([], dtype=int)
    directional_predicted = predicted_direction[directional.to_numpy()] if directional.any() else np.asarray([], dtype=int)
    hits = int((directional_actual == directional_predicted).sum())
    low, high = wilson_interval(hits, len(directional_actual))
    probability_columns = [f"h{horizon}_p_down", f"h{horizon}_p_neutral", f"h{horizon}_p_up"]
    probability_columns_available = all(name in frame for name in probability_columns)
    probability_rows = frame[probability_columns].notna().all(axis=1) if probability_columns_available else pd.Series(False, index=frame.index)
    probs = frame.loc[probability_rows, probability_columns].to_numpy(dtype=float) if probability_columns_available else np.empty((0, 3))
    prob_actual = actual.loc[probability_rows].to_numpy(dtype=int)
    ece, mce = expected_calibration_error(prob_actual, probs) if len(probs) else (None, None)
    signed = np.where(action.eq("BUY_BIAS"), 1.0, np.where(action.eq("SELL_BIAS"), -1.0, 0.0))
    returns = pd.to_numeric(frame.get("raw_return", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if total else np.asarray([])
    costs = pd.to_numeric(frame.get("cost_band", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if total else np.asarray([])
    cost_metric = float(np.mean(signed[directional.to_numpy()] * returns[directional.to_numpy()] - costs[directional.to_numpy()])) if directional.any() else None
    return {
        "horizon_minutes": horizon, "matured_count": total, "wait_count": int(waits.sum()),
        "directional_calls": int(directional.sum()), "advisory_coverage": safe_rate(int(directional.sum()), total),
        "wait_rate": safe_rate(int(waits.sum()), total), "buy_rate": safe_rate(int(action.eq("BUY_BIAS").sum()), total),
        "sell_rate": safe_rate(int(action.eq("SELL_BIAS").sum()), total),
        "directional_accuracy": safe_rate(hits, len(directional_actual)),
        "directional_accuracy_ci_low": low, "directional_accuracy_ci_high": high,
        "balanced_accuracy": _balanced_accuracy(directional_actual, directional_predicted),
        "macro_f1": _macro_f1(directional_actual, directional_predicted),
        "brier": brier_score(prob_actual, probs) if len(probs) else None,
        "log_loss": multiclass_log_loss(prob_actual, probs) if len(probs) else None,
        "ece": ece, "mce": mce,
        "mean_future_return": float(np.mean(returns)) if total else None,
        "median_future_return": float(np.median(returns)) if total else None,
        "mean_move_pips": float(pd.to_numeric(frame["move_pips"]).mean()) if total else None,
        "median_move_pips": float(pd.to_numeric(frame["move_pips"]).median()) if total else None,
        "cost_aware_research_metric": cost_metric,
        "sample_status": "PASS" if total >= MIN_FORWARD_SHADOW_ROWS else "INSUFFICIENT_DATA",
        "directional_status": "PASS" if int(directional.sum()) >= MIN_DIRECTIONAL_CALLS else "INSUFFICIENT_DATA",
    }


def wait_analysis(joined: pd.DataFrame) -> dict[str, object]:
    if joined.empty:
        return {"matured": 0, "waits": 0, "wait_rate": None, "neutral_band_rate": None, "large_move_rate": None, "no_model_waits": 0, "event_waits": 0, "risk_waits": 0}
    waits = joined.loc[joined["advisory"].eq("WAIT")].copy()
    inside = waits["raw_return"].abs().le(waits["cost_band"])
    large = waits["raw_return"].abs().gt(2 * waits["cost_band"])
    blockers = waits.get("blockers", pd.Series("[]", index=waits.index)).astype(str)
    return {
        "matured": len(joined), "waits": len(waits), "wait_rate": safe_rate(len(waits), len(joined)),
        "neutral_band_rate": safe_rate(int(inside.sum()), len(waits)), "large_move_rate": safe_rate(int(large.sum()), len(waits)),
        "no_model_waits": int(blockers.str.contains("NO_APPROVED_MODEL").sum()),
        "event_waits": int(blockers.str.contains("EVENT").sum()),
        "risk_waits": int(blockers.str.contains("SPREAD|VOLATILITY", regex=True).sum()),
    }


def grouped_performance(joined: pd.DataFrame, group_column: str, minimum: int) -> pd.DataFrame:
    columns = [group_column, "horizon_minutes", "sample_count", "wait_rate", "directional_coverage", "directional_accuracy", "brier", "cost_aware_research_metric", "status"]
    if joined.empty or group_column not in joined:
        return pd.DataFrame(columns=columns)
    rows = []
    for (group, horizon), frame in joined.groupby([group_column, "horizon_minutes"], dropna=False):
        metrics = evaluate_horizon(frame, int(horizon))
        rows.append({
            group_column: group, "horizon_minutes": int(horizon), "sample_count": len(frame),
            "wait_rate": metrics["wait_rate"], "directional_coverage": metrics["advisory_coverage"],
            "directional_accuracy": metrics["directional_accuracy"], "brier": metrics["brier"],
            "cost_aware_research_metric": metrics["cost_aware_research_metric"],
            "status": "PASS" if len(frame) >= minimum else "INSUFFICIENT_SAMPLE",
        })
    return pd.DataFrame(rows, columns=columns)
