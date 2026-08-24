from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.evaluation.v08_drift import distribution_drift, reference_hash


class DriftTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.reference = pd.DataFrame({"x": rng.normal(0, 1, 200)})

    def test_identical_distribution_is_normal(self):
        result = distribution_drift(self.reference, self.reference.iloc[:100].copy(), ["x"])
        self.assertEqual(result.iloc[0]["status"], "NORMAL")

    def test_large_shift_is_degraded(self):
        recent = pd.DataFrame({"x": self.reference.iloc[:100]["x"].to_numpy() + 5})
        self.assertEqual(distribution_drift(self.reference, recent, ["x"]).iloc[0]["status"], "DEGRADED")

    def test_insufficient_sample(self):
        self.assertEqual(distribution_drift(self.reference.iloc[:10], self.reference.iloc[:10], ["x"]).iloc[0]["status"], "INSUFFICIENT_DATA")

    def test_missing_feature_handled(self):
        self.assertEqual(distribution_drift(self.reference, pd.DataFrame({"y": range(100)}), ["x"]).iloc[0]["status"], "INSUFFICIENT_DATA")

    def test_training_reference_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference mismatch"):
            distribution_drift(self.reference, self.reference, ["x"], expected_reference_hash="bad")

    def test_future_feature_rejected(self):
        reference = self.reference.copy(); recent = self.reference.iloc[:100].copy()
        reference["decision_timestamp_utc"] = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
        recent["decision_timestamp_utc"] = pd.date_range("2027-01-01", periods=100, freq="h", tz="UTC")
        with self.assertRaisesRegex(ValueError, "Future feature"):
            distribution_drift(reference, recent, ["x"], recent_as_of_utc="2026-12-31T00:00:00Z")

    def test_reference_hash_deterministic(self):
        self.assertEqual(reference_hash(self.reference, ["x"]), reference_hash(self.reference.copy(), ["x"]))


if __name__ == "__main__":
    unittest.main()
