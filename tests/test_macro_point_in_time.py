from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_features_mt5_macro import released_today_asof  # noqa: E402


class MacroPointInTimeTests(unittest.TestCase):
    def test_release_day_flag_is_false_before_release(self) -> None:
        release = pd.Timestamp("2025-01-15T13:30:00Z")
        decisions = pd.Series(
            pd.to_datetime(
                [
                    "2025-01-15T12:00:00Z",
                    "2025-01-15T13:30:00Z",
                    "2025-01-15T18:00:00Z",
                    "2025-01-16T01:00:00Z",
                ],
                utc=True,
            )
        )
        actual = released_today_asof([release], decisions)
        self.assertEqual(actual.tolist(), [False, True, True, False])

    def test_empty_release_stream_is_all_false(self) -> None:
        decisions = pd.Series(pd.to_datetime(["2025-01-15T12:00:00Z"], utc=True))
        self.assertEqual(released_today_asof([], decisions).tolist(), [False])


if __name__ == "__main__":
    unittest.main()
