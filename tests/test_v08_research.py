from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.evaluation.v08_contract import PREDEFINED_COST_MULTIPLIERS
from src.research.v08_experiments import chronological_split, experiment_specs, research_decision
from src.research.v08_feature_ablation import FEATURE_GROUPS, ablated_features, validate_groups
from src.research.v08_scorecard import SCORECARD_COLUMNS, write_scorecard


class ResearchTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        times = pd.date_range("2025-01-01", periods=1000, freq="5min", tz="UTC")
        return pd.DataFrame({"decision_timestamp_utc": times, "target_matured_at_utc": times + pd.Timedelta(minutes=15)})

    def test_chronological_split_and_purge(self):
        split = chronological_split(self._frame(), 15)
        self.assertLess(split.train["decision_timestamp_utc"].max(), split.validation["decision_timestamp_utc"].min())
        self.assertLess(split.validation["decision_timestamp_utc"].max(), split.holdout["decision_timestamp_utc"].min())
        self.assertTrue(split.train["target_matured_at_utc"].lt(split.validation["decision_timestamp_utc"].min()).all())

    def test_final_holdout_is_marked_unopened_during_selection(self):
        self.assertFalse(chronological_split(self._frame(), 15).holdout_opened)

    def test_no_random_shuffle_configuration(self):
        ordered = chronological_split(self._frame(), 15)
        self.assertTrue(ordered.train["decision_timestamp_utc"].is_monotonic_increasing)

    def test_target_thresholds_are_predefined(self):
        values = sorted({float(item["target_multiplier"]) for item in experiment_specs() if str(item["id"]).startswith("COST_BAND")})
        self.assertEqual(values, list(PREDEFINED_COST_MULTIPLIERS))

    def test_feature_ablation_groups_are_deterministic(self):
        validate_groups()
        self.assertEqual(tuple(ablated_features("VOLUME")), tuple(ablated_features("VOLUME")))
        self.assertEqual(set(FEATURE_GROUPS), {"PRICE_RETURN", "VOLATILITY", "SPREAD_LIQUIDITY", "VOLUME", "SESSION_TIME"})

    def test_v05b_is_not_activated(self):
        self.assertFalse(any("V05B" in str(item) for item in experiment_specs()))

    def test_research_features_exclude_future_and_outcome_fields(self):
        features = {str(feature).lower() for spec in experiment_specs() for feature in spec["features"]}
        self.assertFalse(any(name.startswith(("future_", "outcome_")) for name in features))

    def test_two_stage_is_isolated_research(self):
        spec = next(item for item in experiment_specs() if item["id"] == "TWO_STAGE_NO_TRADE")
        self.assertEqual(spec["family"], "TWO_STAGE_LOGISTIC")

    def test_research_cannot_directly_become_champion(self):
        row = {name: None for name in SCORECARD_COLUMNS}; row["decision"] = "CHAMPION"
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "cannot directly become CHAMPION"):
                write_scorecard(pd.DataFrame([row]), Path(temp) / "score.csv")

    def test_negative_holdout_cost_cannot_be_promising(self):
        self.assertEqual(research_decision({"ba": .40}, {"ba": .39, "cost": -0.00001}), "REJECT")


if __name__ == "__main__":
    unittest.main()
