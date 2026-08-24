from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.inference.v06a_contract import REQUIRED_FEATURES
from src.inference.v06a_engine import (
    infer_horizon,
    latest_market_snapshot,
    probability_decision,
    select_latest_champion,
    target_event_guard_from_frame,
)


NOW = pd.Timestamp("2026-08-24T12:00:00Z")


class FakeModel:
    feature_names = tuple(REQUIRED_FEATURES)

    def __init__(self, probabilities=(0.20, 0.20, 0.60)):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, x):
        return np.tile(self.probabilities, (len(x), 1))


def market_frame(decision: pd.Timestamp | None = None) -> pd.DataFrame:
    decision = decision or (NOW - pd.Timedelta(minutes=5))
    row = {name: 0.0 for name in REQUIRED_FEATURES}
    row.update({
        "decision_timestamp_utc": decision,
        "m1_available_from_utc": decision,
        "m5_available_from_utc": decision,
        "m15_available_from_utc": decision,
        "h1_available_from_utc": decision,
    })
    return pd.DataFrame([row])


class V06AShadowInferenceTests(unittest.TestCase):
    def test_latest_market_snapshot_is_causal(self):
        row, age = latest_market_snapshot(market_frame(), NOW)
        self.assertEqual(pd.Timestamp(row["decision_timestamp_utc"]), NOW - pd.Timedelta(minutes=5))
        self.assertAlmostEqual(age, 5.0)

    def test_future_availability_is_rejected(self):
        frame = market_frame()
        frame.loc[0, "m1_available_from_utc"] = NOW + pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            latest_market_snapshot(frame, NOW)

    def test_future_decision_timestamp_is_rejected(self):
        with self.assertRaises(RuntimeError):
            latest_market_snapshot(market_frame(NOW + pd.Timedelta(minutes=1)), NOW)

    def test_latest_champion_is_selected_per_horizon(self):
        records = [
            {"horizon_minutes": 15, "promotion_status": "CHAMPION", "created_at_utc": "2026-08-20T00:00:00Z", "model_id": "old"},
            {"horizon_minutes": 15, "promotion_status": "CHALLENGER", "created_at_utc": "2026-08-24T00:00:00Z", "model_id": "not-approved"},
            {"horizon_minutes": 15, "promotion_status": "CHAMPION", "created_at_utc": "2026-08-23T00:00:00Z", "model_id": "new"},
        ]
        self.assertEqual(select_latest_champion(records, 15)["model_id"], "new")
        self.assertIsNone(select_latest_champion(records, 60))

    def test_probability_gate_allows_only_strong_direction(self):
        result = probability_decision(np.array([0.20, 0.20, 0.60]))
        self.assertEqual(result["shadow_direction"], "UP")
        self.assertEqual(result["decision_gate"], "SHADOW_DIRECTION_ELIGIBLE")

    def test_probability_gate_waits_on_neutral(self):
        result = probability_decision(np.array([0.20, 0.60, 0.20]))
        self.assertEqual(result["shadow_direction"], "WAIT")
        self.assertEqual(result["decision_gate"], "WAIT_NEUTRAL")

    def test_probability_gate_waits_on_low_margin(self):
        result = probability_decision(np.array([0.44, 0.10, 0.46]))
        self.assertEqual(result["shadow_direction"], "WAIT")
        self.assertEqual(result["decision_gate"], "WAIT_LOW_CONFIDENCE")

    def test_invalid_probabilities_fail_closed(self):
        with self.assertRaises(RuntimeError):
            probability_decision(np.array([0.5, np.nan, 0.5]))
        with self.assertRaises(RuntimeError):
            probability_decision(np.array([0.5, 0.5]))

    def test_infer_horizon_uses_registered_feature_order(self):
        row = market_frame().iloc[0]
        result = infer_horizon(FakeModel(), row)
        self.assertEqual(result["shadow_direction"], "UP")

    def test_target_event_guard_blocks_pre_release_window(self):
        frame = pd.DataFrame([{
            "event_name": "Nonfarm Payrolls",
            "event_timestamp_utc": NOW + pd.Timedelta(minutes=20),
        }])
        result = target_event_guard_from_frame(frame, NOW)
        self.assertTrue(result.blocked)
        self.assertEqual(result.status, "AUDITED_TARGET_EVENT_RISK_WINDOW")

    def test_target_event_guard_does_not_block_far_event(self):
        frame = pd.DataFrame([{
            "event_name": "Nonfarm Payrolls",
            "event_timestamp_utc": NOW + pd.Timedelta(hours=4),
        }])
        result = target_event_guard_from_frame(frame, NOW)
        self.assertFalse(result.blocked)
        self.assertEqual(result.status, "VERIFIED_TARGET_EVENT_NOT_IN_BLOCK_WINDOW")


if __name__ == "__main__":
    unittest.main()
