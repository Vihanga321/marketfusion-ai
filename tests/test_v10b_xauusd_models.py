from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.research.v10b_xauusd_models import (
    ASSET_ID,
    FEATURE_GROUPS,
    HorizonData,
    _promotion_decision,
    build_horizon_data,
    build_research_features,
    feature_set,
    final_holdout_split,
    fit_temporal_model,
    infer_latest,
    validate_feature_columns,
    validate_probability_matrix,
)


class V10BXAUUSDModelTests(unittest.TestCase):
    def _raw_source(self, rows: int = 700) -> pd.DataFrame:
        timestamps = pd.date_range("2026-01-05T00:00:00Z", periods=rows, freq="5min")
        trend = np.linspace(0.0, 8.0, rows)
        wave = np.sin(np.arange(rows) / 11.0) * 0.8
        close = pd.Series(2700.0 + trend + wave)
        open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
        high = pd.concat([open_, close], axis=1).max(axis=1) + 0.35
        low = pd.concat([open_, close], axis=1).min(axis=1) - 0.35
        ret = close.pct_change()
        frame = pd.DataFrame({
            "decision_timestamp_utc": timestamps,
            "bar_open": open_.to_numpy(),
            "bar_high": high.to_numpy(),
            "bar_low": low.to_numpy(),
            "bar_close": close.to_numpy(),
            "m5_return_5m": ret.to_numpy(),
            "m5_return_15m": close.pct_change(3).to_numpy(),
            "m5_return_60m": close.pct_change(12).to_numpy(),
            "m5_return_240m": close.pct_change(48).to_numpy(),
            "m5_volatility_60m": ret.rolling(12).std().to_numpy(),
            "m5_volatility_240m": ret.rolling(48).std().to_numpy(),
            "m5_spread_points": np.full(rows, 22.0),
            "spread_relative_to_price": np.full(rows, 22.0 * 0.01) / close.to_numpy(),
            "spread_relative_to_atr": np.full(rows, 0.22),
            "atr_14": np.full(rows, 1.0),
            "utc_hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
        })
        for horizon in (15, 60, 240):
            classes = np.arange(rows) % 3
            frame[f"target_class_{horizon}m"] = classes
            frame[f"outcome_future_timestamp_{horizon}m"] = timestamps + pd.Timedelta(minutes=horizon)
            frame[f"outcome_future_return_{horizon}m"] = np.where(classes == 0, -0.001, np.where(classes == 2, 0.001, 0.0))
            frame[f"target_neutral_band_{horizon}m"] = 0.0002
        return frame

    def test_feature_groups_are_causal_and_complete(self) -> None:
        source = build_research_features(self._raw_source())
        all_features = feature_set("ALL_V10B")
        validate_feature_columns(all_features)
        self.assertGreater(len(all_features), len(FEATURE_GROUPS["MARKET_CORE"]))
        self.assertFalse(any(name.startswith(("target_", "outcome_", "future_")) for name in all_features))
        self.assertIn("pattern_bullish_engulfing", source.columns)
        self.assertIn("is_london_session", source.columns)
        self.assertIn("sweep_high_20", source.columns)

    def test_invalid_ohlc_is_rejected(self) -> None:
        source = self._raw_source(100)
        source.loc[50, "bar_low"] = source.loc[50, "bar_high"] + 1.0
        with self.assertRaisesRegex(ValueError, "OHLC"):
            build_research_features(source)

    def test_exact_future_target_guard(self) -> None:
        source = build_research_features(self._raw_source())
        data = build_horizon_data(source, 15, feature_set("BASELINE"))
        self.assertGreater(len(data.frame), 0)
        broken = source.copy()
        broken.loc[broken.index[-1], "outcome_future_timestamp_15m"] += pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "exact completed-bar horizon"):
            build_horizon_data(broken, 15, feature_set("BASELINE"))

    def test_final_holdout_is_chronological_and_purged(self) -> None:
        rows = 3_200
        timestamps = pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="5min")
        frame = pd.DataFrame({
            "decision_timestamp_utc": timestamps,
            "m5_return_5m": np.sin(np.arange(rows) / 10) / 10_000,
            "target_class": np.arange(rows) % 3,
            "raw_future_return": np.where(np.arange(rows) % 3 == 0, -0.001, 0.001),
            "decision_cost_band": 0.0002,
            "outcome_future_timestamp_15m": timestamps + pd.Timedelta(minutes=15),
        })
        data = HorizonData(15, frame, ("m5_return_5m",))
        split = final_holdout_split(data)
        research_times = frame.iloc[split.research_indices]["decision_timestamp_utc"]
        self.assertTrue(((research_times + pd.Timedelta(minutes=15)) < split.holdout_start).all())
        self.assertGreaterEqual(len(split.holdout_indices), 500)

    def test_probability_contract_is_fail_closed(self) -> None:
        validate_probability_matrix(np.array([[0.2, 0.3, 0.5]]))
        with self.assertRaises(ValueError):
            validate_probability_matrix(np.array([[0.2, 0.3, 0.6]]))
        with self.assertRaises(ValueError):
            validate_probability_matrix(np.array([[np.nan, 0.5, 0.5]]))

    def test_temporal_calibration_uses_three_class_training_only(self) -> None:
        rows = 1_700
        timestamps = pd.Series(pd.date_range("2025-01-01T00:00:00Z", periods=rows, freq="5min"))
        x = pd.DataFrame({
            "m5_return_5m": np.sin(np.arange(rows) / 7.0),
            "m5_return_15m": np.cos(np.arange(rows) / 9.0),
        })
        y = pd.Series(np.arange(rows) % 3)
        bundle = fit_temporal_model("LOGISTIC_REGRESSION", x, y, timestamps, 15)
        self.assertEqual(bundle.asset_id, ASSET_ID)
        self.assertEqual(bundle.calibration_status, "SIGMOID_TEMPORAL_HOLDOUT")
        probability = bundle.predict_proba(x.tail(5))
        self.assertEqual(probability.shape, (5, 3))
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))

    def test_weak_candidate_cannot_be_promoted(self) -> None:
        candidate = {"balanced_accuracy": 0.34, "macro_f1": 0.33, "log_loss": 1.08, "brier": 0.66}
        majority = {"balanced_accuracy": 0.333, "macro_f1": 0.30, "log_loss": 1.10, "brier": 0.66, "cost_aware_metric": -0.001}
        prior = {**majority, "log_loss": 1.09}
        previous = {**majority, "cost_aware_metric": -0.0001}
        promoted, reason = _promotion_decision(
            candidate, candidate, majority, prior, previous,
            "PASS", "SIGMOID_TEMPORAL_HOLDOUT", 365, 5,
        )
        self.assertFalse(promoted)
        self.assertIn("failed gates", reason)

    @patch("src.research.v10b_xauusd_models.approved_champions", return_value={})
    def test_no_champion_publishes_no_probabilities(self, _champions: object) -> None:
        source = pd.DataFrame({"decision_timestamp_utc": [pd.Timestamp("2026-08-31T00:00:00Z")]})
        result = infer_latest(source)
        for horizon in ("15", "60", "240"):
            item = result["horizons"][horizon]
            self.assertEqual(item["model_status"], "NO_APPROVED_MODEL")
            self.assertIsNone(item["prob_down"])
            self.assertIsNone(item["prob_neutral"])
            self.assertIsNone(item["prob_up"])
        self.assertFalse(result["trading_enabled"])

    def test_future_fields_cannot_enter_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "Target/future"):
            validate_feature_columns(("m5_return_5m", "outcome_future_return_15m"))


if __name__ == "__main__":
    unittest.main()
