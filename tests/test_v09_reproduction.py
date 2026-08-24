from __future__ import annotations

import unittest

import pandas as pd

from src.research.v08_experiments import chronological_split
from src.research.v09_reproduction import infer_frozen_source_rows


class V09ReproductionTests(unittest.TestCase):
    def test_infers_original_prefix_when_purged_rows_are_not_in_scorecard_counts(self):
        rows = 8881
        times = pd.date_range("2026-01-01", periods=rows + 200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "decision_timestamp_utc": times,
            "target_matured_at_utc": times + pd.Timedelta(minutes=60),
        })
        frozen = frame.iloc[:rows].copy()
        split = chronological_split(frozen, 60)
        development_rows = len(split.train) + len(split.validation)
        holdout_rows = len(split.holdout)
        self.assertLess(development_rows + holdout_rows, rows)
        inferred = infer_frozen_source_rows(frame, development_rows, holdout_rows)
        self.assertEqual(inferred, rows)


if __name__ == "__main__":
    unittest.main()
