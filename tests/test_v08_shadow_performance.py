from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.v08_calibration import calibration_monitor
from src.evaluation.v08_analysis import rolling_performance
from src.evaluation.v08_contract import HORIZONS, PREDICTION_COLUMNS, SOURCE_LABEL
from src.evaluation.v08_ledger import append_prediction, build_prediction_record, ingest_state, prediction_sha
from src.evaluation.v08_metrics import (
    brier_score, bootstrap_mean_interval, evaluate_horizon, expected_calibration_error,
    grouped_performance, multiclass_log_loss, safe_rate, wait_analysis, wilson_interval,
)
from src.evaluation.v08_outcomes import matured_outcome_rows
from src.evaluation.v08_runner import _acquire_lock, _pid_running, _release_lock
from src.evaluation.v08_windows import evaluate_window

DECISION = pd.Timestamp("2026-08-24T12:00:00Z")


def state() -> dict[str, object]:
    horizons = {str(h): {"model_id": None, "model_status": "NO_APPROVED_MODEL", "prob_down": None, "prob_neutral": None, "prob_up": None, "shadow_direction": "WAIT"} for h in HORIZONS}
    return {
        "contract_version": "v0.6c-unified-runtime-v1",
        "system": {"symbol": "EURUSD", "generated_time": {"utc": "2026-08-24T12:01:00Z"}, "registry": {"champion_count": 0}},
        "market": {"close": 1.16, "session": "LONDON", "regime": {"trend_regime": "UP", "volatility_regime": "NORMAL"}, "spread": {"current_points": 1.0, "status": "NORMAL"}},
        "predictions": {"horizons": horizons, "fusion": {"status": "NO_APPROVED_MODEL", "probabilities": None}},
        "decision": {"action": "WAIT", "confidence": "VERY_LOW", "gate": "WAIT_NO_MODEL", "decision_time": {"utc": DECISION.isoformat()}, "next_reassessment": {"utc": "2026-08-24T12:05:00Z"}},
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None},
        "event": {"status": "NONE", "minutes_to_event": 100},
        "health": {"sources": {"v05b_intelligence": "PASS_DEGRADED", "v05d_event": "PASS", "v06a_inference": "PASS_FAIL_CLOSED_NO_CHAMPION"}},
        "reasons": [{"code": "NO_APPROVED_MODEL", "blocking": True}],
    }


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame([build_prediction_record(state(), "2026-08-24T12:02:00Z", "test")])


def market_frame(include: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    row: dict[str, object] = {"decision_timestamp_utc": DECISION, "m5_close": 1.16, "m5_spread_points": 1.0}
    for horizon in HORIZONS:
        available = horizon in include
        row[f"outcome_future_timestamp_{horizon}m"] = DECISION + pd.Timedelta(minutes=horizon) if available else pd.NaT
        row[f"outcome_matured_at_utc_{horizon}m"] = DECISION + pd.Timedelta(minutes=horizon) if available else pd.NaT
        row[f"outcome_future_return_{horizon}m"] = 0.001 if available else np.nan
    return pd.DataFrame([row])


class LedgerTests(unittest.TestCase):
    def test_prediction_sha_is_deterministic(self):
        one = build_prediction_record(state(), "2026-08-24T12:02:00Z", "test")
        two = build_prediction_record(state(), "2026-08-24T12:03:00Z", "test")
        self.assertEqual(prediction_sha(one), prediction_sha(two))

    def test_same_prediction_duplicate_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "predictions.parquet"
            one = build_prediction_record(state(), "2026-08-24T12:02:00Z", "test")
            self.assertEqual(append_prediction(one, path, root / "conflicts")["status"], "APPENDED")
            two = build_prediction_record(state(), "2026-08-24T12:03:00Z", "test")
            self.assertEqual(append_prediction(two, path, root / "conflicts")["status"], "DEDUPLICATED")
            self.assertEqual(len(pd.read_parquet(path)), 1)

    def test_changed_duplicate_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "predictions.parquet"; conflicts = root / "conflicts"
            original = build_prediction_record(state(), "2026-08-24T12:02:00Z", "test")
            append_prediction(original, path, conflicts)
            changed_state = state(); changed_state["decision"]["confidence"] = "LOW"  # type: ignore[index]
            changed = build_prediction_record(changed_state, "2026-08-24T12:03:00Z", "test")
            self.assertEqual(append_prediction(changed, path, conflicts)["status"], "PREDICTION_MUTATION_CONFLICT")
            self.assertEqual(len(list(conflicts.glob("*.parquet"))), 1)

    def test_prediction_payload_contains_no_outcome(self):
        self.assertFalse(any("outcome" in name for name in PREDICTION_COLUMNS))
        self.assertFalse(any("outcome" in name for name in build_prediction_record(state(), "2026-08-24T12:02:00Z", "test")))

    def test_newer_model_cannot_replace_old_model_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "predictions.parquet"
            original = state(); original["predictions"]["horizons"]["15"]["model_id"] = "old"  # type: ignore[index]
            first = build_prediction_record(original, "2026-08-24T12:02:00Z", "test")
            append_prediction(first, path, root / "conflicts")
            newer = state(); newer["predictions"]["horizons"]["15"]["model_id"] = "new"  # type: ignore[index]
            second = build_prediction_record(newer, "2026-08-24T12:03:00Z", "test")
            self.assertEqual(append_prediction(second, path, root / "conflicts")["status"], "PREDICTION_MUTATION_CONFLICT")
            self.assertEqual(pd.read_parquet(path).iloc[0]["h15_model_id"], "old")

    def test_retroactive_shadow_generation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RETROACTIVE_SHADOW_REJECTED"):
            build_prediction_record(state(), "2026-08-24T12:11:00Z", "test")

    def test_late_repeat_of_already_forward_recorded_prediction_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "predictions.parquet"
            self.assertEqual(ingest_state(state(), "2026-08-24T12:02:00Z", path, root / "conflicts", "test")["status"], "APPENDED")
            self.assertEqual(ingest_state(state(), "2026-08-24T13:00:00Z", path, root / "conflicts", "test")["status"], "DEDUPLICATED")

    def test_forward_and_backtest_labels_are_distinct(self):
        self.assertEqual(build_prediction_record(state(), "2026-08-24T12:02:00Z", "test")["source_label"], SOURCE_LABEL)
        self.assertNotEqual(SOURCE_LABEL, "BACKTEST_RESEARCH")


class MonitorLockTests(unittest.TestCase):
    def test_nonexistent_pid_is_not_running(self):
        self.assertFalse(_pid_running(2_147_483_647))

    def test_stale_monitor_lock_is_replaced_and_released(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "v08_monitor.lock"
            lock.write_text("2147483647", encoding="ascii")
            with patch("src.evaluation.v08_runner.MONITOR_LOCK_FILE", lock):
                _acquire_lock()
                self.assertEqual(lock.read_text(encoding="ascii"), str(os.getpid()))
                _release_lock()
                self.assertFalse(lock.exists())

    def test_live_monitor_lock_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "v08_monitor.lock"
            lock.write_text(str(os.getpid()), encoding="ascii")
            with patch("src.evaluation.v08_runner.MONITOR_LOCK_FILE", lock):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    _acquire_lock()


class OutcomeTests(unittest.TestCase):
    def test_future_outcome_cannot_attach_early(self):
        result, stats = matured_outcome_rows(prediction_frame(), market_frame(), "2026-08-24T12:14:59Z")
        self.assertTrue(result.empty); self.assertEqual(stats["waiting"], 3)

    def test_exact_15m_timestamp(self):
        result, _ = matured_outcome_rows(prediction_frame(), market_frame(), "2026-08-24T12:15:00Z")
        self.assertEqual(pd.Timestamp(result.iloc[0]["future_timestamp_utc"]), DECISION + pd.Timedelta(minutes=15))

    def test_exact_60m_timestamp(self):
        result, _ = matured_outcome_rows(prediction_frame(), market_frame(), "2026-08-24T13:00:00Z")
        row = result.loc[result["horizon_minutes"].eq(60)].iloc[0]
        self.assertEqual(pd.Timestamp(row["future_timestamp_utc"]), DECISION + pd.Timedelta(minutes=60))

    def test_exact_240m_timestamp(self):
        result, _ = matured_outcome_rows(prediction_frame(), market_frame(), "2026-08-24T16:00:00Z")
        row = result.loc[result["horizon_minutes"].eq(240)].iloc[0]
        self.assertEqual(pd.Timestamp(row["future_timestamp_utc"]), DECISION + pd.Timedelta(minutes=240))

    def test_missing_future_data_stays_missing_without_interpolation(self):
        result, stats = matured_outcome_rows(prediction_frame(), market_frame((15,)), "2026-08-24T16:00:00Z")
        self.assertEqual(len(result), 1); self.assertEqual(stats["missing"], 2)

    def test_wrong_exact_timestamp_is_contract_mismatch(self):
        market = market_frame(); market.loc[0, "outcome_future_timestamp_15m"] = DECISION + pd.Timedelta(minutes=16)
        result, stats = matured_outcome_rows(prediction_frame(), market, "2026-08-24T12:15:00Z")
        self.assertTrue(result.empty); self.assertEqual(stats["contract_mismatch"], 1)

    def test_existing_outcome_deduplicates(self):
        first, _ = matured_outcome_rows(prediction_frame(), market_frame(), "2026-08-24T12:15:00Z")
        second, stats = matured_outcome_rows(prediction_frame(), market_frame(), "2026-08-24T12:15:00Z", first)
        self.assertEqual(len(second), 1); self.assertEqual(stats["added"], 0)


def joined_metrics_frame() -> pd.DataFrame:
    rows = []
    for index, (action, target) in enumerate((("WAIT", 1), ("BUY_BIAS", 2), ("SELL_BIAS", 0), ("WAIT", 2))):
        rows.append({"horizon_minutes": 15, "advisory": action, "target_class": target, "raw_return": [0.0, .001, -.001, .002][index], "cost_band": .0001, "move_pips": [0, 10, -10, 20][index], "blockers": json.dumps(["NO_APPROVED_MODEL"]) if action == "WAIT" else "[]", "session": "LONDON", "volatility_regime": "NORMAL", "event_risk": "NONE", "h15_p_down": None, "h15_p_neutral": None, "h15_p_up": None})
    return pd.DataFrame(rows)


class MetricTests(unittest.TestCase):
    def test_wait_calculation(self):
        result = wait_analysis(joined_metrics_frame())
        self.assertEqual(result["wait_rate"], .5); self.assertEqual(result["no_model_waits"], 2)

    def test_directional_accuracy_and_coverage(self):
        result = evaluate_horizon(joined_metrics_frame(), 15)
        self.assertEqual(result["directional_accuracy"], 1.0); self.assertEqual(result["advisory_coverage"], .5)

    def test_brier(self):
        self.assertAlmostEqual(brier_score([2], np.array([[.1, .2, .7]])), .14)

    def test_log_loss(self):
        self.assertAlmostEqual(multiclass_log_loss([2], np.array([[.1, .2, .7]])), -np.log(.7))

    def test_ece_and_mce(self):
        ece, mce = expected_calibration_error([2, 0], np.array([[.1, .2, .7], [.6, .3, .1]]))
        self.assertIsNotNone(ece); self.assertGreaterEqual(mce, ece)

    def test_empty_metrics_do_not_pretend_success(self):
        result = evaluate_horizon(pd.DataFrame(columns=["horizon_minutes"]), 15)
        self.assertEqual(result["sample_status"], "INSUFFICIENT_DATA"); self.assertIsNone(result["directional_accuracy"])
        self.assertIsNone(safe_rate(0, 0))

    def test_session_and_regime_aggregation_are_sample_gated(self):
        grouped = grouped_performance(joined_metrics_frame(), "session", 30)
        self.assertEqual(grouped.iloc[0]["status"], "INSUFFICIENT_SAMPLE")
        self.assertEqual(grouped_performance(joined_metrics_frame(), "volatility_regime", 30).iloc[0]["status"], "INSUFFICIENT_SAMPLE")
        self.assertEqual(grouped_performance(joined_metrics_frame(), "event_risk", 30).iloc[0]["status"], "INSUFFICIENT_SAMPLE")

    def test_rolling_metrics_stay_gated(self):
        frame = joined_metrics_frame()
        frame["decision_timestamp_utc"] = pd.date_range("2026-08-24T10:00:00Z", periods=len(frame), freq="5min")
        result = rolling_performance(frame, "2026-08-24T12:00:00Z")
        self.assertEqual(result["last_20"]["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["last_5_days"]["rows"], 4)

    def test_shadow_window_uses_only_completed_exact_m1_data(self):
        prediction = pd.Series({
            "prediction_id": "p1", "advisory": "BUY_BIAS", "market_mid": 1.1000,
            "suggested_start_utc": "2026-08-24T12:00:00Z", "suggested_end_utc": "2026-08-24T12:03:00Z",
        })
        bars = pd.DataFrame({
            "bar_open_utc": pd.date_range("2026-08-24T12:00:00Z", periods=3, freq="1min"),
            "high": [1.1002, 1.1005, 1.1003], "low": [1.0999, 1.0998, 1.1000], "close": [1.1001, 1.1002, 1.1003],
        })
        self.assertIsNone(evaluate_window(prediction, bars, "2026-08-24T12:02:59Z"))
        result = evaluate_window(prediction, bars, "2026-08-24T12:03:00Z")
        self.assertAlmostEqual(result["mfe_pips"], 5.0)  # type: ignore[index]
        self.assertAlmostEqual(result["mae_pips"], -2.0)  # type: ignore[index]

    def test_confidence_intervals(self):
        low, high = wilson_interval(8, 10)
        self.assertLess(low, .8); self.assertGreater(high, .8)
        self.assertEqual(bootstrap_mean_interval(range(30)), bootstrap_mean_interval(range(30)))

    def test_calibration_no_model_status(self):
        monitor = calibration_monitor(joined_metrics_frame())
        self.assertTrue(monitor["status"].eq("NO_APPROVED_MODEL_DATA").all())


if __name__ == "__main__":
    unittest.main()
