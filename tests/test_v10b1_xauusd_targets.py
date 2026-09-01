from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.research.v10b_xauusd_models import build_research_features
from src.research.v10b1_xauusd_targets import (
    SCREEN_FEATURES,
    TARGET_CONTRACTS,
    apply_target_contract,
    evidence_gate,
    exact_future_frame,
    target_semantics,
)


class V10B1XAUUSDTargetTests(unittest.TestCase):
    def _source(self, rows: int = 1_000) -> pd.DataFrame:
        timestamps = pd.date_range("2026-01-05T00:00:00Z", periods=rows, freq="5min")
        wave = np.sin(np.arange(rows) / 17.0) * 2.0
        drift = np.linspace(0.0, 15.0, rows)
        close = pd.Series(2700.0 + drift + wave)
        open_ = close.shift(1).fillna(close.iloc[0] - 0.2)
        high = pd.concat([open_, close], axis=1).max(axis=1) + 0.4
        low = pd.concat([open_, close], axis=1).min(axis=1) - 0.4
        ret = close.pct_change()
        raw = pd.DataFrame({
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
            "m5_spread_points": np.full(rows, 20.0),
            "spread_relative_to_price": np.full(rows, 0.20) / close.to_numpy(),
            "spread_relative_to_atr": np.full(rows, 0.20),
            "atr_14": np.full(rows, 1.0),
            "utc_hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
        })
        for horizon in (15, 60, 240):
            raw[f"target_class_{horizon}m"] = np.arange(rows) % 3
            raw[f"outcome_future_timestamp_{horizon}m"] = timestamps + pd.Timedelta(minutes=horizon)
            raw[f"outcome_future_return_{horizon}m"] = 0.0
            raw[f"target_neutral_band_{horizon}m"] = 0.0002
        return build_research_features(raw)

    def test_exact_future_join_uses_timestamp_not_row_shift(self) -> None:
        source = self._source(300)
        missing_time = source.loc[120, "decision_timestamp_utc"]
        source = source.drop(index=120).reset_index(drop=True)
        result = exact_future_frame(source, 30)
        decision = missing_time - pd.Timedelta(minutes=30)
        row = result.loc[result["decision_timestamp_utc"].eq(decision)].iloc[0]
        self.assertTrue(pd.isna(row["future_close"]))
        self.assertTrue(pd.isna(row["raw_future_return"]))

    def test_wider_wait_zone_reduces_directional_coverage(self) -> None:
        source = exact_future_frame(self._source(), 60)
        reference = target_semantics(apply_target_contract(source, TARGET_CONTRACTS[0]))
        strict = target_semantics(apply_target_contract(source, TARGET_CONTRACTS[-1]))
        self.assertGreaterEqual(reference["directional_coverage"], strict["directional_coverage"])
        self.assertLessEqual(reference["wait_fraction"], strict["wait_fraction"])

    def test_new_horizons_are_supported_with_exact_completed_bars(self) -> None:
        source = self._source(500)
        for horizon in (30, 120):
            result = exact_future_frame(source, horizon)
            mature = result.dropna(subset=["future_close"])
            self.assertGreater(len(mature), 0)
            self.assertTrue(
                (mature["future_timestamp_utc"] == mature["decision_timestamp_utc"] + pd.Timedelta(minutes=horizon)).all()
            )

    def test_screen_features_exclude_targets_and_future_fields(self) -> None:
        self.assertGreater(len(SCREEN_FEATURES), 10)
        self.assertFalse(any(
            name.startswith(("target_", "outcome_", "future_")) or "future" in name
            for name in SCREEN_FEATURES
        ))

    def test_evidence_gate_never_means_model_approval(self) -> None:
        semantics = {"directional_coverage": 0.4}
        screen = {
            "holdout_rows": 2_000,
            "fold_count": 4,
            "folds_beating_class_prior": 4,
            "walk_forward_balanced_accuracy_mean": 0.54,
            "final_holdout": {
                "balanced_accuracy": 0.55,
                "log_loss": 0.65,
                "cost_aware_metric": 0.0002,
            },
            "class_prior_final_holdout": {"log_loss": 0.69},
            "previous_direction_final_holdout": {"cost_aware_metric": 0.0},
        }
        status, failed = evidence_gate(semantics, screen)
        self.assertEqual(status, "RESEARCH_CANDIDATE")
        self.assertEqual(failed, [])
        self.assertNotIn("APPROVED", status)
        self.assertNotIn("CHAMPION", status)


if __name__ == "__main__":
    unittest.main()
