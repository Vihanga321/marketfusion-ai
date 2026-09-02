from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.v10b7_xauusd_crossmarket_relationships import (
    FX_RELATIONSHIP_COLUMNS,
    RELATIONSHIP_COLUMNS,
    SILVER_RELATIONSHIP_COLUMNS,
    SPECS,
    add_relationship_features,
    evidence_status,
)


def test_b7_specs_are_frozen_and_narrow() -> None:
    observed = {
        (
            item.horizon_minutes,
            item.target_contract_name,
            item.baseline,
            item.context_group,
            item.model_family,
            item.relationship_group,
        )
        for item in SPECS
    }
    assert observed == {
        (15, "V10B_REFERENCE", "PRICE_CORE_NO_CLOCK", "SILVER", "LOGISTIC_REGRESSION", "XAU_XAG_RELATIONSHIPS"),
        (30, "SELECTIVE_MEDIUM", "PRICE_STRUCTURE_VOLATILITY_NO_CLOCK", "FX_USD", "HIST_GRADIENT_BOOSTING", "XAU_USD_RELATIONSHIPS"),
    }


def test_relationship_feature_names_do_not_encode_future_or_target_fields() -> None:
    for columns in RELATIONSHIP_COLUMNS.values():
        for name in columns:
            lowered = name.lower()
            assert "future" not in lowered
            assert "target" not in lowered
            assert "outcome" not in lowered


def test_silver_relationships_are_deterministic_current_past_combinations() -> None:
    frame = pd.DataFrame(
        {
            "m5_return_5m": [0.002],
            "m5_return_15m": [0.006],
            "m5_return_60m": [0.012],
            "m5_volatility_60m": [0.004],
            "ctx_silver_ret_5m": [0.001],
            "ctx_silver_ret_15m": [0.003],
            "ctx_silver_ret_60m": [0.006],
            "ctx_silver_vol_60m": [0.002],
        }
    )
    out = add_relationship_features(frame, "XAU_XAG_RELATIONSHIPS")
    assert set(SILVER_RELATIONSHIP_COLUMNS).issubset(out.columns)
    assert np.isclose(out.loc[0, "rel_xau_xag_divergence_5m"], 0.001)
    assert np.isclose(out.loc[0, "rel_xau_xag_divergence_15m"], 0.003)
    assert np.isclose(out.loc[0, "rel_xau_xag_divergence_60m"], 0.006)
    assert out.loc[0, "rel_xau_xag_direction_agreement_5m"] == 1
    assert np.isclose(out.loc[0, "rel_xau_xag_vol_ratio_60m"], 2.0)


def test_fx_relationships_build_usd_composite_without_future_data() -> None:
    frame = pd.DataFrame(
        {
            "m5_return_5m": [-0.001],
            "m5_return_15m": [-0.003],
            "m5_return_60m": [-0.009],
            "m5_volatility_60m": [0.003],
            "ctx_usdjpy_ret_5m": [0.002],
            "ctx_usdjpy_ret_15m": [0.006],
            "ctx_usdjpy_ret_30m": [0.010],
            "ctx_usdjpy_ret_60m": [0.016],
            "ctx_usdjpy_vol_60m": [0.002],
            "ctx_eurusd_ret_5m": [-0.002],
            "ctx_eurusd_ret_15m": [-0.004],
            "ctx_eurusd_ret_30m": [-0.008],
            "ctx_eurusd_ret_60m": [-0.012],
            "ctx_eurusd_vol_60m": [0.004],
        }
    )
    out = add_relationship_features(frame, "XAU_USD_RELATIONSHIPS")
    assert set(FX_RELATIONSHIP_COLUMNS).issubset(out.columns)
    assert np.isclose(out.loc[0, "rel_usd_impulse_5m"], 0.002)
    assert np.isclose(out.loc[0, "rel_usd_impulse_15m"], 0.005)
    assert out.loc[0, "rel_fx_leg_agreement_5m"] == 1
    assert out.loc[0, "rel_xau_usd_inverse_alignment_5m"] == 1
    assert np.isclose(out.loc[0, "rel_xau_usd_vol_ratio_60m"], 1.0)


def _passing_row() -> dict[str, float | int | str]:
    return {
        "status": "EVALUATED",
        "balanced_accuracy_mean": 0.522,
        "recent_fold_balanced_accuracy": 0.518,
        "folds_beating_raw_context": 4,
        "balanced_accuracy_delta": 0.006,
        "log_loss_mean": 0.689,
        "raw_context_log_loss_mean": 0.692,
        "class_prior_log_loss_mean": 0.691,
        "cost_aware_mean": -0.00010,
        "raw_context_cost_aware_mean": -0.00012,
    }


def test_b7_gate_accepts_only_full_relationship_pass() -> None:
    status, failed = evidence_status(_passing_row())
    assert status == "CROSSMARKET_RELATIONSHIP_EVIDENCE_CANDIDATE"
    assert failed == []

    row = _passing_row()
    row["balanced_accuracy_delta"] = 0.0029
    status, failed = evidence_status(row)
    assert status == "NO_RELATIONSHIP_EVIDENCE"
    assert "relationship_balanced_accuracy_delta" in failed

    row = _passing_row()
    row["log_loss_mean"] = 0.691
    status, failed = evidence_status(row)
    assert status == "NO_RELATIONSHIP_EVIDENCE"
    assert "relationship_log_loss" in failed
