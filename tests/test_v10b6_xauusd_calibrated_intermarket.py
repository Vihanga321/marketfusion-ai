from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.v10b6_xauusd_calibrated_intermarket import (
    CANDIDATES,
    apply_platt_calibrator,
    evidence_status,
    fit_platt_calibrator,
    temporal_calibration_split,
)


def test_candidate_set_is_small_and_predeclared() -> None:
    observed = {
        (
            item.horizon_minutes,
            item.target_contract_name,
            item.baseline,
            item.context_group,
            item.model_family,
        )
        for item in CANDIDATES
    }
    assert observed == {
        (30, "SELECTIVE_MEDIUM", "PRICE_STRUCTURE_VOLATILITY_NO_CLOCK", "FX_USD", "HIST_GRADIENT_BOOSTING"),
        (30, "SELECTIVE_MEDIUM", "PRICE_CORE_NO_CLOCK", "SILVER", "LOGISTIC_REGRESSION"),
        (15, "V10B_REFERENCE", "PRICE_CORE_NO_CLOCK", "SILVER", "LOGISTIC_REGRESSION"),
        (15, "V10B_REFERENCE", "PRICE_CORE_NO_CLOCK", "EQUITY_RISK", "LOGISTIC_REGRESSION"),
    }


def test_temporal_calibration_split_purges_base_target_overlap() -> None:
    decision = pd.date_range("2025-01-01", periods=4_000, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "decision_broker_timestamp": decision,
            "future_timestamp_utc": decision + pd.Timedelta(minutes=30),
            "directional_target": np.arange(len(decision)) % 2,
        }
    )
    base, calibration, info = temporal_calibration_split(frame, 30)
    calibration_start = pd.Timestamp(calibration["decision_broker_timestamp"].iloc[0])
    assert len(base) >= 1_500
    assert len(calibration) >= 400
    assert (pd.to_datetime(base["future_timestamp_utc"], utc=True) < calibration_start).all()
    assert info["base_train_future_overlap"] == 0


def test_platt_calibration_outputs_valid_binary_probabilities() -> None:
    raw = np.array(
        [
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.40, 0.60],
            [0.30, 0.70],
            [0.20, 0.80],
        ],
        dtype=float,
    )
    target = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    calibrator = fit_platt_calibrator(raw, target)
    calibrated = apply_platt_calibrator(calibrator, raw)
    assert calibrated.shape == raw.shape
    assert np.isfinite(calibrated).all()
    assert (calibrated >= 0).all() and (calibrated <= 1).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def _passing_row() -> dict[str, float | int | str]:
    return {
        "status": "EVALUATED",
        "balanced_accuracy_mean": 0.520,
        "recent_fold_balanced_accuracy": 0.515,
        "folds_beating_paired_baseline": 4,
        "balanced_accuracy_delta": 0.010,
        "log_loss_mean": 0.688,
        "baseline_log_loss_mean": 0.692,
        "class_prior_log_loss_mean": 0.691,
        "cost_aware_mean": -0.00010,
        "baseline_cost_aware_mean": -0.00012,
        "calibration_log_loss_gain": 0.004,
    }


def test_evidence_gate_accepts_only_full_calibrated_pass() -> None:
    status, failed = evidence_status(_passing_row())
    assert status == "CALIBRATED_INTRADAY_EVIDENCE_CANDIDATE"
    assert failed == []

    failed_prior = _passing_row()
    failed_prior["log_loss_mean"] = 0.693
    status, failed = evidence_status(failed_prior)
    assert status == "NO_CALIBRATED_INTRADAY_EVIDENCE"
    assert "class_prior_log_loss" in failed


def test_calibration_must_not_worsen_context_log_loss() -> None:
    row = _passing_row()
    row["calibration_log_loss_gain"] = -0.0001
    status, failed = evidence_status(row)
    assert status == "NO_CALIBRATED_INTRADAY_EVIDENCE"
    assert "calibration_log_loss_non_improving" in failed
