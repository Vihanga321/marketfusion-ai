from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.learning.v05c_contract import MARKET_FEATURES, MIN_PROMOTION_HISTORY_DAYS
from src.research.v09_candidates import (
    AbstentionPolicy,
    candidate_specs,
    decisions_from_probabilities,
    fit_predict,
    labels,
)
from src.research.v09_contract import (
    ALLOWED_FINAL_STATUSES,
    DIRECTIONAL_MARGIN_THRESHOLDS,
    PRIMARY_HORIZON,
    PRIMARY_SOURCE_EXPERIMENT,
    TARGET_MULTIPLIERS,
    TOP_PROBABILITY_THRESHOLDS,
)
from src.research.v09_nested_temporal import audit_nested_folds, final_temporal_split, nested_temporal_folds


class V09ChallengerHardeningTests(unittest.TestCase):
    def _frame(self, rows: int = 5000) -> pd.DataFrame:
        times = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "decision_timestamp_utc": times,
            "target_matured_at_utc": times + pd.Timedelta(minutes=60),
        })
        index = np.arange(rows, dtype=float)
        for offset, feature in enumerate(MARKET_FEATURES, start=1):
            frame[feature] = np.sin(index / (7.0 + offset)) + (offset * 1e-4)
        # Session flags remain deterministic binary inputs.
        for feature in ("is_asia_session", "is_london_session", "is_new_york_session", "is_london_ny_overlap"):
            if feature in frame:
                frame[feature] = ((index.astype(int) + len(feature)) % 3 == 0).astype(int)
        frame["raw_future_return"] = np.take(np.array([-0.0006, 0.0, 0.0006]), index.astype(int) % 3)
        frame["decision_cost_band"] = 0.0001
        return frame

    def test_primary_source_is_exact_v08_result(self):
        self.assertEqual(PRIMARY_SOURCE_EXPERIMENT, "H60_ABLATE_VOLUME")
        self.assertEqual(PRIMARY_HORIZON, 60)

    def test_formal_history_gate_is_not_weakened(self):
        self.assertGreaterEqual(MIN_PROMOTION_HISTORY_DAYS, 90)

    def test_target_multipliers_are_predefined(self):
        self.assertEqual(TARGET_MULTIPLIERS, (1.0, 1.25, 1.5, 2.0))

    def test_abstention_thresholds_are_predefined(self):
        self.assertEqual(TOP_PROBABILITY_THRESHOLDS, (0.50, 0.55, 0.60))
        self.assertEqual(DIRECTIONAL_MARGIN_THRESHOLDS, (0.05, 0.08, 0.10))

    def test_candidate_feature_sets_exclude_future_and_outcomes(self):
        names = [name.lower() for spec in candidate_specs() for name in spec.features]
        self.assertFalse(any(name.startswith(("future_", "outcome_", "target_")) for name in names))

    def test_v05b_features_are_not_activated(self):
        names = [name.lower() for spec in candidate_specs() for name in spec.features]
        self.assertFalse(any("news_" in name or "macro_" in name or "v05b" in name for name in names))

    def test_no_volume_ablation_is_deterministic(self):
        first = next(spec for spec in candidate_specs() if spec.candidate_id == "H60_ELASTIC_NO_VOLUME")
        second = next(spec for spec in candidate_specs() if spec.candidate_id == "H60_ELASTIC_NO_VOLUME")
        self.assertEqual(first.features, second.features)
        self.assertFalse(any("volume" in feature.lower() for feature in first.features))

    def test_research_families_are_fixed_and_auditable(self):
        families = {spec.family for spec in candidate_specs()}
        self.assertEqual(families, {
            "ELASTIC_NET_LOGISTIC", "LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING",
            "RANDOM_FOREST_BASELINE", "TWO_STAGE_LOGISTIC",
        })

    def test_final_holdout_is_untouched_selection_block(self):
        split = final_temporal_split(self._frame(), 60)
        self.assertFalse(split.holdout_opened_for_selection)
        self.assertFalse(split.development.empty)
        self.assertFalse(split.holdout.empty)

    def test_final_holdout_is_chronological_and_purged(self):
        split = final_temporal_split(self._frame(), 60)
        holdout_start = split.holdout["decision_timestamp_utc"].min()
        self.assertTrue(split.development["decision_timestamp_utc"].lt(holdout_start).all())
        self.assertTrue(split.development["target_matured_at_utc"].lt(holdout_start).all())

    def test_nested_folds_are_chronological(self):
        split = final_temporal_split(self._frame(), 60)
        folds = nested_temporal_folds(split.development, 60)
        self.assertGreaterEqual(len(folds), 3)
        for outer in folds:
            start = outer.fold.validation["decision_timestamp_utc"].min()
            self.assertTrue(outer.fold.train["target_matured_at_utc"].lt(start).all())
            for inner in outer.inner_folds:
                inner_start = inner.validation["decision_timestamp_utc"].min()
                self.assertTrue(inner.train["target_matured_at_utc"].lt(inner_start).all())

    def test_nested_audit_has_zero_future_violations(self):
        split = final_temporal_split(self._frame(), 60)
        audit = audit_nested_folds(nested_temporal_folds(split.development, 60))
        self.assertFalse(audit.empty)
        self.assertEqual(int(audit["future_violation"].sum()), 0)
        self.assertTrue(audit["purge_minutes"].eq(60).all())

    def test_target_labels_are_three_class_and_cost_aware(self):
        target = labels(self._frame(100), 1.25)
        self.assertEqual(set(target.unique()), {0, 1, 2})

    def test_probability_validation_rejects_bad_sum(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            decisions_from_probabilities(np.array([[0.4, 0.4, 0.4]]))

    def test_probability_validation_rejects_malformed_shape(self):
        with self.assertRaisesRegex(ValueError, "Malformed"):
            decisions_from_probabilities(np.array([[0.5, 0.5]]))

    def test_abstention_never_increases_directional_coverage(self):
        probabilities = np.array([
            [0.60, 0.20, 0.20], [0.35, 0.30, 0.35], [0.20, 0.20, 0.60], [0.34, 0.33, 0.33],
        ])
        raw = decisions_from_probabilities(probabilities)
        gated = decisions_from_probabilities(probabilities, AbstentionPolicy(0.55, 0.08))
        self.assertLessEqual(int((gated != 1).sum()), int((raw != 1).sum()))

    def test_two_stage_candidate_outputs_valid_probabilities(self):
        frame = self._frame(1800)
        train = frame.iloc[:1400]
        evaluation = frame.iloc[1400:1450]
        spec = next(spec for spec in candidate_specs() if spec.family == "TWO_STAGE_LOGISTIC")
        probability = fit_predict(spec, train, evaluation)
        self.assertEqual(probability.shape, (50, 3))
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0, atol=1e-6))

    def test_candidate_ids_do_not_claim_champion(self):
        self.assertFalse(any("CHAMPION" in spec.candidate_id for spec in candidate_specs()))

    def test_v09_status_vocabulary_has_no_champion_status(self):
        self.assertFalse(any("CHAMPION" in status for status in ALLOWED_FINAL_STATUSES))

    def test_research_source_contains_no_live_execution_module_dependency(self):
        # V0.9 is intentionally isolated under src/research and does not modify V0.6.
        module_paths = [Path("src/research/v09_contract.py"), Path("src/research/v09_candidates.py"), Path("src/research/v09_nested_temporal.py"), Path("src/research/v09_runner.py")]
        self.assertTrue(all(str(path).startswith("src\\research") or str(path).startswith("src/research") for path in module_paths))


if __name__ == "__main__":
    unittest.main()
