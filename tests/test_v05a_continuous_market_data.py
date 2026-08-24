from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from src.marketdata.mt5_continuous_store import merge_immutable_bars, quote_to_frame, rates_to_completed_frame
from src.marketdata.v05a_contract import FEATURE_COLUMNS, TIMEFRAMES
from src.marketdata.v05a_daily_history import build_daily_history, finalized_daily_history
from src.marketdata.v05a_dataset import build_dataset_from_frames, model_feature_view
from src.marketdata.v05a_quality import build_quality_report


def synthetic_bars(start: str, periods: int, minutes: int) -> pd.DataFrame:
    opens = pd.date_range(start, periods=periods, freq=f"{minutes}min", tz="UTC")
    base = pd.Series(range(periods), dtype="float64") * 0.00001 + 1.1000
    closes = base + 0.00002
    return pd.DataFrame({
        "bar_open_utc": opens,
        "bar_close_utc": opens + pd.Timedelta(minutes=minutes),
        "open": base,
        "high": closes + 0.00003,
        "low": base - 0.00003,
        "close": closes,
        "tick_volume": pd.Series(range(periods), dtype="int64") + 100,
        "spread_points": 10,
        "real_volume": 0,
        "first_observed_utc": opens + pd.Timedelta(minutes=minutes, seconds=10),
        "source": "MT5",
    })


class V05AContinuousMarketDataTests(unittest.TestCase):
    def test_incomplete_bar_is_excluded(self):
        spec = TIMEFRAMES["M5"]
        capture = pd.Timestamp("2026-08-24T10:07:00Z")
        raw = pd.DataFrame({
            "time": [
                int(pd.Timestamp("2026-08-24T09:55:00Z").timestamp()),
                int(pd.Timestamp("2026-08-24T10:00:00Z").timestamp()),
                int(pd.Timestamp("2026-08-24T10:05:00Z").timestamp()),
            ],
            "open": [1.1, 1.2, 1.3],
            "high": [1.2, 1.3, 1.4],
            "low": [1.0, 1.1, 1.2],
            "close": [1.15, 1.25, 1.35],
            "tick_volume": [10, 11, 12],
            "spread": [8, 8, 8],
            "real_volume": [0, 0, 0],
        })
        frame = rates_to_completed_frame(raw, spec, capture)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["bar_open_utc"].iloc[-1], pd.Timestamp("2026-08-24T10:00:00Z"))

    def test_existing_bar_is_immutable_and_conflict_is_quarantinable(self):
        spec = TIMEFRAMES["M5"]
        existing = synthetic_bars("2026-08-24T10:00:00Z", 2, 5)
        incoming = existing.copy()
        incoming.loc[0, "close"] += 0.001
        merged, conflicts = merge_immutable_bars(existing, incoming, spec)
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertAlmostEqual(float(merged.loc[0, "close"]), float(existing.loc[0, "close"]))

    def test_quote_requires_nonnegative_spread(self):
        tick = SimpleNamespace(bid=1.10, ask=1.1002, time=1_777_000_000, time_msc=1_777_000_000_000)
        frame = quote_to_frame(tick, 0.00001, pd.Timestamp("2026-08-24T10:00:00Z"))
        self.assertAlmostEqual(float(frame["spread_points"].iloc[0]), 20.0)
        with self.assertRaises(ValueError):
            quote_to_frame(SimpleNamespace(bid=1.2, ask=1.1, time=1_777_000_000), 0.00001, pd.Timestamp("2026-08-24T10:00:00Z"))

    def _frames(self):
        return {
            "M1": synthetic_bars("2026-08-20T00:00:00Z", 5 * 24 * 60, 1),
            "M5": synthetic_bars("2026-08-20T00:00:00Z", 5 * 24 * 12, 5),
            "M15": synthetic_bars("2026-08-20T00:00:00Z", 5 * 24 * 4, 15),
            "H1": synthetic_bars("2026-08-20T00:00:00Z", 5 * 24, 60),
        }

    def test_features_never_use_future_timeframe_availability(self):
        dataset = build_dataset_from_frames(self._frames())
        decision = pd.to_datetime(dataset["decision_timestamp_utc"], utc=True)
        for label in TIMEFRAMES:
            available = pd.to_datetime(dataset[f"{label.lower()}_available_from_utc"], utc=True)
            self.assertFalse((available.notna() & (available > decision)).any(), label)

    def test_labels_mature_only_on_exact_future_m5_timestamp(self):
        dataset = build_dataset_from_frames(self._frames())
        for minutes in (15, 60, 240):
            labeled = dataset[f"outcome_future_return_{minutes}m"].notna()
            expected = pd.to_datetime(dataset.loc[labeled, "decision_timestamp_utc"], utc=True) + pd.Timedelta(minutes=minutes)
            actual = pd.to_datetime(dataset.loc[labeled, f"outcome_matured_at_utc_{minutes}m"], utc=True)
            self.assertTrue(actual.reset_index(drop=True).equals(expected.reset_index(drop=True)))
            self.assertFalse(labeled.iloc[-1])

    def test_model_feature_view_physically_excludes_outcomes(self):
        dataset = build_dataset_from_frames(self._frames())
        view = model_feature_view(dataset)
        self.assertEqual(set(FEATURE_COLUMNS).issubset(view.columns), True)
        self.assertFalse(any(column.startswith("outcome_") for column in view.columns))

    def test_daily_history_finalizes_prior_days_and_keeps_today_provisional(self):
        m5 = synthetic_bars("2026-08-20T00:00:00Z", 36 * 12, 5)
        now = pd.Timestamp(m5["bar_close_utc"].iloc[-1]) + pd.Timedelta(minutes=1)
        daily = build_daily_history(m5, now)
        self.assertEqual(daily["day_status"].iloc[-1], "PROVISIONAL_CURRENT_UTC_DAY")
        finalized = finalized_daily_history(daily)
        self.assertTrue((finalized["day_status"] == "FINALIZED_UTC_DAY").all())
        self.assertLess(len(finalized), len(daily))

    def test_quality_gate_passes_causal_synthetic_data(self):
        frames = self._frames()
        dataset = build_dataset_from_frames(frames)
        now = pd.Timestamp(frames["M1"]["bar_close_utc"].iloc[-1]) + pd.Timedelta(minutes=1)
        result = build_quality_report(frames, dataset, now)
        self.assertTrue(result.passed, result.report)


if __name__ == "__main__":
    unittest.main()
