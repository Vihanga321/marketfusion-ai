from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.engines.engine_layer import engine_status
from src.engines.contract import ENGINE_CONTRACT_VERSION
from src.marketdata.market_calendar import forex_session_state


class V09AEngineTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        times = pd.date_range("2026-08-28T23:00:00Z", periods=12, freq="5min")
        return pd.DataFrame({
            "decision_timestamp_utc": times,
            "m5_return_5m": 0.0001,
            "m5_return_15m": 0.0002,
            "m5_return_60m": 0.0003,
            "m5_return_240m": 0.0004,
            "m5_volatility_60m": 0.0002,
            "m5_spread_points": 1.0,
        })

    def test_contract_and_engine_statuses_are_explicit(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "features.parquet"
            self.frame().to_parquet(path)
            result = engine_status("2026-08-29T00:00:00Z", path)
        self.assertEqual(result["contract_version"], ENGINE_CONTRACT_VERSION)
        self.assertTrue(result["observational_only"])
        self.assertFalse(result["v06_integration"])
        self.assertEqual(result["engines"]["technical"]["status"], "AVAILABLE")
        self.assertEqual(result["external_engines"]["news"]["status"], "UNAVAILABLE")

    def test_future_row_does_not_change_earlier_engine_output(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "features.parquet"
            frame = self.frame()
            frame.to_parquet(path)
            before = engine_status("2026-08-28T23:55:00Z", path)
            frame.loc[len(frame)] = [pd.Timestamp("2026-08-29T00:00:00Z"), -0.02, -0.02, -0.02, -0.02, 0.001, 4.0]
            frame.to_parquet(path)
            after = engine_status("2026-08-28T23:55:00Z", path)
        self.assertEqual(before["decision_timestamp_utc"], after["decision_timestamp_utc"])
        self.assertEqual(before["engines"]["technical"]["direction_score"], after["engines"]["technical"]["direction_score"])

    def test_session_engine_reuses_canonical_weekend_state(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "features.parquet"
            self.frame().to_parquet(path)
            result = engine_status("2026-08-29T12:00:00Z", path)
        self.assertEqual(result["engines"]["session"]["regime"], "WEEKEND")
        self.assertEqual(forex_session_state("2026-08-29T12:00:00Z")["current_session"], "WEEKEND")

    def test_missing_feature_store_is_unavailable(self):
        result = engine_status("2026-08-29T12:00:00Z", Path("does-not-exist.parquet"))
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertTrue(all(item["status"] == "UNAVAILABLE" for item in result["engines"].values()))


if __name__ == "__main__":
    unittest.main()
