from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.backfill.v10a_backfill import audit_bar_overlap, audit_dataset_overlap, dataframe_sha256, eligible_market_core_summary
from src.backfill.v10a_contract import FETCH_LIMITS, SOURCE_LABEL, TARGET_PROMOTION_HISTORY_DAYS
from src.learning.v05c_contract import MARKET_FEATURES, MIN_PROMOTION_HISTORY_DAYS
from src.learning.v05c_dataset import build_horizon_dataset, feature_group_eligibility
from src.marketdata.v05a_contract import FEATURE_COLUMNS, TIMEFRAMES


def market_source(rows: int = 300) -> pd.DataFrame:
    decision = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "decision_timestamp_utc": decision,
        "m5_close": 1.1 + index * 1e-7,
        "m1_available_from_utc": decision,
        "m5_available_from_utc": decision,
        "m15_available_from_utc": decision.floor("15min"),
        "h1_available_from_utc": decision.floor("h"),
        "feature_complete": np.ones(rows, dtype=bool),
    }
    for offset, feature in enumerate(MARKET_FEATURES):
        if feature not in data:
            data[feature] = 0.1 + np.sin(index / (offset + 3))
    data["m5_spread_points"] = np.full(rows, 2.0)
    for horizon in (15, 60, 240):
        future = pd.Series(decision + pd.Timedelta(minutes=horizon))
        mature = future <= decision[-1]
        data[f"outcome_future_timestamp_{horizon}m"] = future.where(mature)
        data[f"outcome_matured_at_utc_{horizon}m"] = future.where(mature)
        data[f"outcome_future_return_{horizon}m"] = pd.Series(np.sin(index / 11) * 1e-4).where(mature)
        data[f"outcome_direction_{horizon}m"] = pd.Series(np.sign(np.sin(index / 11))).where(mature)
    return pd.DataFrame(data)


def bar(open_time: str, close: float = 1.1, observed: str = "2026-01-02T00:00:00Z") -> pd.DataFrame:
    opened = pd.Timestamp(open_time)
    return pd.DataFrame([{
        "symbol": "EURUSD", "timeframe": "M5", "bar_open_utc": opened,
        "bar_close_utc": opened + pd.Timedelta(minutes=5),
        "first_observed_utc": pd.Timestamp(observed), "source": "TEST",
        "open": 1.0, "high": 1.2, "low": 0.9, "close": close,
        "tick_volume": 10, "spread_points": 2, "real_volume": 0,
    }])


class V10AAuditedBackfillTests(unittest.TestCase):
    def test_contract_covers_all_v05a_timeframes(self):
        self.assertEqual(set(FETCH_LIMITS), set(TIMEFRAMES))
        self.assertGreaterEqual(FETCH_LIMITS["M1"], 100_000)

    def test_formal_v05c_history_gate_is_imported_not_weakened(self):
        self.assertEqual(TARGET_PROMOTION_HISTORY_DAYS, MIN_PROMOTION_HISTORY_DAYS)
        self.assertEqual(TARGET_PROMOTION_HISTORY_DAYS, 90)

    def test_backfill_is_not_true_forward_shadow(self):
        self.assertEqual(SOURCE_LABEL, "MT5_HISTORICAL_BACKFILL")
        self.assertNotEqual(SOURCE_LABEL, "TRUE_FORWARD_SHADOW")

    def test_dataframe_hash_is_deterministic(self):
        frame = market_source(100)
        self.assertEqual(dataframe_sha256(frame), dataframe_sha256(frame.copy()))

    def test_identical_overlap_is_accepted_even_with_different_observation_metadata(self):
        stored = bar("2026-01-01T00:00:00Z", observed="2026-01-01T00:06:00Z")
        fetched = stored.copy()
        fetched["first_observed_utc"] = pd.Timestamp("2026-08-29T00:00:00Z")
        fetched["source"] = SOURCE_LABEL
        audit, conflicts = audit_bar_overlap(stored, fetched, "M5")
        self.assertEqual(int(audit["mismatch_rows"].sum()), 0)
        self.assertTrue(conflicts.empty)

    def test_immutable_price_overlap_mutation_is_rejected(self):
        stored = bar("2026-01-01T00:00:00Z", close=1.1)
        fetched = bar("2026-01-01T00:00:00Z", close=1.1002)
        audit, conflicts = audit_bar_overlap(stored, fetched, "M5")
        self.assertGreater(int(audit["mismatch_rows"].sum()), 0)
        self.assertEqual(len(conflicts), 1)

    def test_market_core_summary_excludes_incomplete_and_immature_rows(self):
        frame = market_source(300)
        frame.loc[:9, "feature_complete"] = False
        summary = eligible_market_core_summary(frame)
        self.assertEqual(summary["eligible_rows"], 300 - 48 - 10)
        self.assertEqual(summary["promotion_history_gate"], "NOT_MET")

    def test_v05c_training_dataset_excludes_feature_incomplete_rows(self):
        frame = market_source(300)
        frame.loc[:19, "feature_complete"] = False
        dataset = build_horizon_dataset(frame, 15)
        self.assertEqual(len(dataset.frame), 300 - 3 - 20)
        self.assertGreaterEqual(dataset.frame["decision_timestamp_utc"].min(), frame.loc[20, "decision_timestamp_utc"])

    def test_legacy_source_without_feature_complete_derives_exact_completeness(self):
        frame = market_source(300).drop(columns=["feature_complete"])
        frame.loc[0, MARKET_FEATURES[0]] = np.nan
        dataset = build_horizon_dataset(frame, 15)
        self.assertNotIn(frame.loc[0, "decision_timestamp_utc"], set(dataset.frame["decision_timestamp_utc"]))

    def test_feature_group_eligibility_counts_only_complete_fully_matured_rows(self):
        frame = market_source(300)
        frame.loc[:9, "feature_complete"] = False
        groups = feature_group_eligibility(frame)
        market = groups.loc[groups["group"].eq("MARKET_CORE")].iloc[0]
        self.assertEqual(int(market["rows"]), 300 - 48 - 10)

    def test_dataset_overlap_allows_null_to_be_filled_but_rejects_non_null_mutation(self):
        stored = market_source(100)
        candidate = stored.copy()
        stored.loc[0, FEATURE_COLUMNS[0]] = np.nan
        first = audit_dataset_overlap(stored, candidate)
        self.assertEqual(first["mismatches"], 0)
        self.assertGreater(first["filled_previous_nulls"], 0)
        candidate.loc[1, FEATURE_COLUMNS[0]] = float(candidate.loc[1, FEATURE_COLUMNS[0]]) + 1.0
        second = audit_dataset_overlap(stored, candidate)
        self.assertGreater(second["mismatches"], 0)

    def test_backfill_package_contains_no_trading_execution(self):
        root = Path(__file__).resolve().parents[1] / "src" / "backfill"
        text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        forbidden = ("order" + "_send", "submit" + "Order", "C" + "Trade")
        self.assertFalse(any(token in text for token in forbidden))


if __name__ == "__main__":
    unittest.main()
