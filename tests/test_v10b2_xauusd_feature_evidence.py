from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.v10b2_xauusd_feature_evidence import (
    FEATURE_SET_GROUPS,
    MODEL_FAMILIES,
    TARGET_SPECS,
    _evidence_status,
    feature_names,
    quarantine_split,
)


class V10B2XAUUSDFeatureEvidenceTests(unittest.TestCase):
    def test_target_specs_are_predeclared_and_xauusd_focused(self) -> None:
        values = [(h, c.name) for h, c in TARGET_SPECS]
        self.assertEqual(values, [(15, "V10B_REFERENCE"), (30, "SELECTIVE_MEDIUM")])

    def test_feature_sets_are_causal_and_include_all_existing_groups(self) -> None:
        self.assertIn("MARKET_CORE", FEATURE_SET_GROUPS)
        self.assertIn("ALL_EXISTING_PRICE_CONTEXT", FEATURE_SET_GROUPS)
        self.assertEqual(MODEL_FAMILIES, ("LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING"))
        for set_name in FEATURE_SET_GROUPS:
            names = feature_names(set_name)
            self.assertGreater(len(names), 0)
            self.assertFalse(any(name.startswith(("target_", "outcome_", "future_")) for name in names))

    def test_quarantine_never_enters_feature_evaluation(self) -> None:
        rows = 6_000
        timestamps = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="5min")
        frame = pd.DataFrame({
            "decision_timestamp_utc": timestamps,
            "future_timestamp_utc": timestamps + pd.Timedelta(minutes=30),
            "directional_target": np.arange(rows) % 2,
        })
        research, info = quarantine_split(frame, 30)
        quarantine_start = pd.Timestamp(info["quarantine_start"])
        self.assertFalse(info["quarantine_evaluated"])
        self.assertTrue(info["purge_ok"])
        self.assertTrue((research["future_timestamp_utc"] < quarantine_start).all())
        self.assertLess(len(research), len(frame))

    def test_coin_flip_evidence_is_rejected(self) -> None:
        baseline = {
            "balanced_accuracy_mean": 0.502,
        }
        row = {
            "status": "EVALUATED",
            "feature_set": "MARKET_CORE_PLUS_STRUCTURE",
            "balanced_accuracy_mean": 0.505,
            "recent_fold_balanced_accuracy": 0.502,
            "folds_beating_class_prior": 2,
            "log_loss_mean": 0.6930,
            "class_prior_log_loss_mean": 0.6931,
            "cost_aware_mean": -0.0002,
            "previous_direction_cost_aware_mean": -0.0001,
        }
        status, failed = _evidence_status(row, baseline)
        self.assertEqual(status, "NO_FEATURE_EVIDENCE")
        self.assertIn("mean_balanced_accuracy", failed)
        self.assertIn("log_loss", failed)
        self.assertIn("cost_aware", failed)

    def test_material_feature_evidence_can_pass_research_gate(self) -> None:
        baseline = {"balanced_accuracy_mean": 0.505}
        row = {
            "status": "EVALUATED",
            "feature_set": "MARKET_CORE_PLUS_STRUCTURE",
            "balanced_accuracy_mean": 0.525,
            "recent_fold_balanced_accuracy": 0.522,
            "folds_beating_class_prior": 4,
            "log_loss_mean": 0.675,
            "class_prior_log_loss_mean": 0.693,
            "cost_aware_mean": 0.0002,
            "previous_direction_cost_aware_mean": 0.0001,
        }
        status, failed = _evidence_status(row, baseline)
        self.assertEqual(status, "FEATURE_EVIDENCE_CANDIDATE")
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
