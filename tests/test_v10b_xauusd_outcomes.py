from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.v08_contract import SOURCE_LABEL
from src.evaluation.v10b_xauusd_outcomes import matured_outcome_rows


class V10BXAUUSDOutcomeTests(unittest.TestCase):
    def _predictions(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "prediction_id": "gold-prediction-1",
            "prediction_payload_sha256": "abc123",
            "source_label": SOURCE_LABEL,
            "symbol": "XAUUSD",
            "decision_timestamp_utc": "2026-08-31T06:00:00Z",
            "market_mid": 4400.00,
            "h15_status": "APPROVED_CHAMPION",
            "h60_status": "NO_APPROVED_MODEL",
            "h240_status": "NO_APPROVED_MODEL",
        }])

    def _targets(self) -> pd.DataFrame:
        row: dict[str, object] = {
            "bar_close_utc": "2026-08-31T06:00:00Z",
            "close": 4400.00,
            "spread_points": 21.0,
        }
        for horizon in (15, 60, 240):
            row[f"outcome_future_timestamp_{horizon}m"] = pd.Timestamp("2026-08-31T06:00:00Z") + pd.Timedelta(minutes=horizon)
            row[f"outcome_future_return_{horizon}m"] = 0.001 if horizon == 15 else 0.002
            row[f"target_neutral_band_{horizon}m"] = 0.0002
            row[f"target_class_{horizon}m"] = 2
        return pd.DataFrame([row])

    def test_exact_gold_target_matures_without_forex_pip_conversion(self) -> None:
        outcomes, stats = matured_outcome_rows(
            self._predictions(), self._targets(), "2026-08-31T06:16:00Z"
        )
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["not_applicable"], 2)
        self.assertEqual(len(outcomes), 1)
        item = outcomes.iloc[0]
        self.assertEqual(item["horizon_minutes"], 15)
        self.assertEqual(item["target_class"], 2)
        self.assertAlmostEqual(item["cost_band"], 0.0002)
        self.assertAlmostEqual(item["future_price"], 4404.4)
        self.assertTrue(pd.isna(item["move_pips"]))
        self.assertEqual(item["evaluation_reason"], "EXACT_XAUUSD_V10A_TARGET_EVENT_MATURED")

    def test_outcome_does_not_mature_early(self) -> None:
        outcomes, stats = matured_outcome_rows(
            self._predictions(), self._targets(), "2026-08-31T06:14:59Z"
        )
        self.assertEqual(stats["waiting"], 1)
        self.assertEqual(stats["added"], 0)
        self.assertTrue(outcomes.empty)

    def test_target_timestamp_mismatch_is_invalid(self) -> None:
        targets = self._targets()
        targets.loc[0, "outcome_future_timestamp_15m"] = pd.Timestamp("2026-08-31T06:20:00Z")
        outcomes, stats = matured_outcome_rows(
            self._predictions(), targets, "2026-08-31T06:21:00Z"
        )
        self.assertEqual(stats["contract_mismatch"], 1)
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(outcomes.iloc[0]["outcome_status"], "INVALID")

    def test_live_quote_cannot_replace_completed_decision_reference(self) -> None:
        predictions = self._predictions()
        predictions.loc[0, "market_mid"] = 4401.25
        outcomes, stats = matured_outcome_rows(
            predictions, self._targets(), "2026-08-31T06:16:00Z"
        )
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(outcomes.iloc[0]["evaluation_reason"], "INVALID_OR_MISMATCHED_XAUUSD_REFERENCE_DATA")

    def test_non_xau_prediction_is_ignored(self) -> None:
        predictions = self._predictions()
        predictions.loc[0, "symbol"] = "EURUSD"
        outcomes, stats = matured_outcome_rows(
            predictions, self._targets(), "2026-08-31T06:16:00Z"
        )
        self.assertEqual(stats["added"], 0)
        self.assertTrue(outcomes.empty)


if __name__ == "__main__":
    unittest.main()
