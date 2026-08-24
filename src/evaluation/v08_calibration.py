"""Forward-only calibration monitor for approved-model probability rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.v08_contract import CALIBRATION_BINS, HORIZONS, MIN_CALIBRATION_ROWS
from src.evaluation.v08_metrics import brier_score, expected_calibration_error, multiclass_log_loss


def calibration_monitor(joined: pd.DataFrame) -> pd.DataFrame:
    columns = ["horizon_minutes", "bin", "rows", "mean_confidence", "observed_frequency", "brier", "log_loss", "ece", "mce", "status"]
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        names = [f"h{horizon}_p_down", f"h{horizon}_p_neutral", f"h{horizon}_p_up"]
        if joined.empty or not all(name in joined for name in names):
            rows.append({"horizon_minutes": horizon, "bin": "ALL", "rows": 0, "status": "NO_APPROVED_MODEL_DATA"})
            continue
        frame = joined.loc[joined["horizon_minutes"].eq(horizon) & joined[names].notna().all(axis=1)].copy()
        if frame.empty:
            rows.append({"horizon_minutes": horizon, "bin": "ALL", "rows": 0, "status": "NO_APPROVED_MODEL_DATA"})
            continue
        probs = frame[names].to_numpy(float)
        actual = frame["target_class"].to_numpy(int)
        confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
        ece, mce = expected_calibration_error(actual, probs)
        overall = "PASS" if len(frame) >= MIN_CALIBRATION_ROWS else "INSUFFICIENT_DATA"
        rows.append({
            "horizon_minutes": horizon, "bin": "ALL", "rows": len(frame),
            "mean_confidence": float(confidence.mean()), "observed_frequency": float((predicted == actual).mean()),
            "brier": brier_score(actual, probs), "log_loss": multiclass_log_loss(actual, probs),
            "ece": ece, "mce": mce, "status": overall,
        })
        for lower, upper in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
            selected = (confidence >= lower) & (confidence < upper)
            if selected.any():
                rows.append({
                    "horizon_minutes": horizon, "bin": f"{lower:.2f}-{min(upper, 1):.2f}",
                    "rows": int(selected.sum()), "mean_confidence": float(confidence[selected].mean()),
                    "observed_frequency": float((predicted[selected] == actual[selected]).mean()), "status": overall,
                })
    return pd.DataFrame(rows, columns=columns)
