from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.research.v10b5_xauusd_intraday_intermarket_evidence import (
    CLOCK_FEATURES,
    PRICE_BASELINES,
    _evidence_status,
    _rates_to_context_features,
    _ready_symbols,
    collect_context_history,
    exact_context_join,
    quarantine_split,
)


def _rates(start: str, rows: int, step_minutes: int = 5):
    base = pd.Timestamp(start, tz="UTC")
    result = []
    for index in range(rows):
        stamp = base + pd.Timedelta(minutes=step_minutes * index)
        close = 100.0 + index * 0.01
        result.append({
            "time": int(stamp.timestamp()),
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "tick_volume": 10,
            "spread": 2,
            "real_volume": 0,
        })
    return result


class _FakeMt5:
    TIMEFRAME_M5 = 5

    def __init__(self):
        self.calls = []

    def symbol_select(self, symbol: str, enabled: bool):
        return enabled

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        self.calls.append((symbol, timeframe, start_pos, count))
        return _rates("2026-01-01T00:00:00", max(count, 2_100))

    def last_error(self):
        return (0, "OK")


class V10B5IntradayEvidenceTests(unittest.TestCase):
    def test_price_baselines_exclude_wall_clock_fields(self):
        for columns in PRICE_BASELINES.values():
            self.assertFalse(CLOCK_FEATURES.intersection(columns))

    def test_context_features_are_keyed_to_completed_bar_close(self):
        frame = _rates_to_context_features(_rates("2026-01-01T00:00:00", 20), "SILVER")
        self.assertEqual(
            frame["decision_broker_timestamp"].iloc[0],
            pd.Timestamp("2026-01-01T00:05:00Z"),
        )
        self.assertIn("ctx_silver_ret_60m", frame.columns)

    def test_exact_context_join_never_asof_or_forward_fills(self):
        research = pd.DataFrame({
            "decision_timestamp_utc": pd.to_datetime([
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:15:00Z",
            ]),
            "future_timestamp_utc": pd.to_datetime([
                "2026-01-01T00:20:00Z",
                "2026-01-01T00:25:00Z",
                "2026-01-01T00:30:00Z",
            ]),
        })
        silver = pd.DataFrame({
            "decision_broker_timestamp": pd.to_datetime([
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:15:00Z",
            ]),
            "ctx_silver_ret_5m": [0.01, 0.02],
            "ctx_silver_ret_15m": [0.01, 0.02],
            "ctx_silver_ret_30m": [0.01, 0.02],
            "ctx_silver_ret_60m": [0.01, 0.02],
            "ctx_silver_vol_60m": [0.01, 0.02],
            "ctx_silver_range_fraction": [0.01, 0.02],
        })
        joined = exact_context_join(research, {"SILVER": silver}, "SILVER")
        self.assertEqual(len(joined), 2)
        self.assertNotIn(
            pd.Timestamp("2026-01-01T00:10:00Z"),
            set(joined["decision_broker_timestamp"]),
        )

    def test_collection_excludes_forming_bar_with_position_one(self):
        mt5 = _FakeMt5()
        with tempfile.TemporaryDirectory() as tmp:
            frames, manifest = collect_context_history(
                mt5,
                {"SILVER": "XAGUSD"},
                rows=2_100,
                root=Path(tmp),
                persist=False,
            )
        self.assertIn("SILVER", frames)
        self.assertEqual(manifest["SILVER"]["status"], "READY")
        self.assertTrue(mt5.calls)
        self.assertTrue(all(call[2] == 1 for call in mt5.calls))

    def test_ready_symbols_requires_clock_calibration(self):
        audit = {
            "decision": "INTRADAY_CONTEXT_READY_FOR_RESEARCH",
            "broker_clock_calibration": {"status": "PASS"},
            "families": [
                {"family": "SILVER", "status": "READY", "broker_symbol": "XAGUSD"},
                {"family": "USD_INDEX", "status": "UNAVAILABLE"},
            ],
        }
        self.assertEqual(_ready_symbols(audit), {"SILVER": "XAGUSD"})
        audit["broker_clock_calibration"] = {"status": "FAILED"}
        with self.assertRaises(RuntimeError):
            _ready_symbols(audit)

    def test_quarantine_never_evaluates_final_tail(self):
        rows = 5_000
        times = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "decision_timestamp_utc": times,
            "future_timestamp_utc": times + pd.Timedelta(minutes=15),
        })
        research, info = quarantine_split(frame, 15)
        quarantine_start = pd.Timestamp(info["quarantine_start_broker_epoch"])
        self.assertTrue((research["future_timestamp_utc"] < quarantine_start).all())
        self.assertFalse(info["quarantine_evaluated"])

    def test_evidence_gate_is_research_only(self):
        row = {
            "status": "EVALUATED",
            "balanced_accuracy_mean": 0.53,
            "recent_fold_balanced_accuracy": 0.52,
            "folds_beating_paired_baseline": 4,
            "balanced_accuracy_delta": 0.01,
            "log_loss_mean": 0.65,
            "baseline_log_loss_mean": 0.67,
            "class_prior_log_loss_mean": 0.69,
            "cost_aware_mean": 0.001,
            "baseline_cost_aware_mean": 0.0,
        }
        status, failed = _evidence_status(row)
        self.assertEqual(status, "INTRADAY_CONTEXT_EVIDENCE_CANDIDATE")
        self.assertEqual(failed, [])
        self.assertNotIn("APPROVED", status)
        self.assertNotIn("CHAMPION", status)


if __name__ == "__main__":
    unittest.main()
