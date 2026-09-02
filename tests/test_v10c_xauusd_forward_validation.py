from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.v10c_xauusd_forward_validation import (
    FROZEN_CONTRACTS,
    MIN_ACTIVE_TRADING_DAYS,
    MIN_DIRECTIONAL_OUTCOMES,
    MIN_MATURED_OBSERVATIONS,
    _prediction_payload_sha,
    _wait_band,
    evaluate_forward_contract,
)


def test_forward_contracts_are_small_frozen_and_xau_only() -> None:
    assert [item.contract_id for item in FROZEN_CONTRACTS] == [
        "XAU15_SILVER_LR_V1",
        "XAU30_FX_HGB_V1",
    ]
    assert FROZEN_CONTRACTS[0].context_group == "SILVER"
    assert FROZEN_CONTRACTS[0].model_family == "LOGISTIC_REGRESSION"
    assert FROZEN_CONTRACTS[1].context_group == "FX_USD"
    assert FROZEN_CONTRACTS[1].model_family == "HIST_GRADIENT_BOOSTING"


def test_wait_band_uses_only_decision_time_spread_and_atr() -> None:
    contract = FROZEN_CONTRACTS[0].target
    row = {
        "spread_relative_to_price": 0.00010,
        "atr_14": 10.0,
        "bar_close": 4_000.0,
    }
    expected = max(1.5 * 0.00010, 0.10 * 10.0 / 4_000.0)
    assert _wait_band(row, contract) == expected


def test_prediction_hash_is_independent_of_record_write_time() -> None:
    first = {
        "contract_id": "XAU15_SILVER_LR_V1",
        "decision_broker_timestamp": "2026-09-02T12:00:00+00:00",
        "recorded_at_utc": "2026-09-02T09:00:01+00:00",
        "context_prob_down": 0.4,
        "context_prob_up": 0.6,
        "prediction_payload_sha256": None,
    }
    second = dict(first)
    second["recorded_at_utc"] = "2026-09-02T09:00:10+00:00"
    assert _prediction_payload_sha(first) == _prediction_payload_sha(second)


def _forward_frame(rows: int, days: int, strong_context: bool) -> pd.DataFrame:
    # Spread observations across the requested number of active UTC dates.
    timestamps: list[pd.Timestamp] = []
    per_day = int(np.ceil(rows / days))
    start = pd.Timestamp("2026-09-03T00:05:00Z")
    for day in range(days):
        base = start + pd.Timedelta(days=day)
        for index in range(per_day):
            if len(timestamps) >= rows:
                break
            timestamps.append(base + pd.Timedelta(minutes=5 * index))
    y = np.arange(rows) % 2
    if strong_context:
        context_up = np.where(y == 1, 0.70, 0.30)
    else:
        context_up = np.full(rows, 0.51)
    # Weak paired baseline predicts DOWN with low confidence on every row.
    baseline_up = np.full(rows, 0.49)
    raw = np.where(y == 1, 0.0010, -0.0010)
    return pd.DataFrame({
        "outcome_status": "MATURED",
        "directional_eligible": True,
        "normalized_decision_utc": timestamps,
        "directional_target": y,
        "context_prob_down": 1.0 - context_up,
        "context_prob_up": context_up,
        "baseline_prob_down": 1.0 - baseline_up,
        "baseline_prob_up": baseline_up,
        "raw_future_return": raw,
        "decision_wait_band": 0.00010,
    })


def test_forward_evidence_waits_for_frozen_minimum_sample() -> None:
    frame = _forward_frame(MIN_MATURED_OBSERVATIONS - 1, MIN_ACTIVE_TRADING_DAYS, True)
    result = evaluate_forward_contract(frame, frozen_prior={"down": 0.5, "up": 0.5})
    assert result["minimum_sample_ready"] is False
    assert result["decision"] == "COLLECTING"


def test_forward_evidence_can_pass_only_after_full_frozen_sample() -> None:
    rows = max(MIN_MATURED_OBSERVATIONS, MIN_DIRECTIONAL_OUTCOMES, 520)
    frame = _forward_frame(rows, MIN_ACTIVE_TRADING_DAYS, True)
    result = evaluate_forward_contract(frame, frozen_prior={"down": 0.5, "up": 0.5})
    assert result["minimum_sample_ready"] is True
    assert result["decision"] == "FORWARD_EVIDENCE_PASS"
    assert result["failed_gates"] == []
    assert result["blocks_beating_paired_baseline"] == 4


def test_forward_gate_rejects_context_that_does_not_add_directional_value() -> None:
    rows = max(MIN_MATURED_OBSERVATIONS, MIN_DIRECTIONAL_OUTCOMES, 520)
    frame = _forward_frame(rows, MIN_ACTIVE_TRADING_DAYS, False)
    result = evaluate_forward_contract(frame, frozen_prior={"down": 0.5, "up": 0.5})
    assert result["minimum_sample_ready"] is True
    assert result["decision"] == "FORWARD_EVIDENCE_FAIL"
    assert "balanced_accuracy_delta" in result["failed_gates"]
