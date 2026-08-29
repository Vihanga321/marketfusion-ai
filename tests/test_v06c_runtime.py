from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.runtime.v06c_engine import assemble_state, dual_time, persist_state
from src.runtime.v06c_runner import _acquire_runtime_lock, _release_runtime_lock


NOW = "2026-08-24T12:00:00+00:00"


def advisory(gate="WAIT_NO_MODEL", status="PASS_FAIL_CLOSED_NO_CHAMPION"):
    return {
        "status": status, "symbol": "EURUSD", "generated_at_utc": NOW,
        "decision_timestamp_utc": "2026-08-24T11:55:00+00:00", "action": "WAIT",
        "direction": "WAIT", "confidence": "VERY_LOW", "decision_gate": gate,
        "next_reassessment_utc": "2026-08-24T12:05:00+00:00", "market": {"freshness": {"status": "FRESH"}, "session": "OVERLAP"},
        "model": {"fusion": {"direction": "WAIT"}}, "event": {"data_status": "FRESH"},
        "intelligence": {"status": "FRESH"}, "trade_window": {"status": "NOT_APPLICABLE_WAIT"}, "reasons": [],
    }


def state():
    return assemble_state(advisory(), {"status": "PASS_FAIL_CLOSED_NO_CHAMPION", "horizons": {}}, {"status": "PASS_RUNNING"}, {"status": "PASS_RUNNING"}, {"status": "AVAILABLE_EMPTY", "champion_count": 0})


class V06CRuntimeTests(unittest.TestCase):
    def test_01_dual_time_preserves_utc(self):
        self.assertEqual(dual_time(NOW)["utc"], NOW)

    def test_02_dual_time_uses_colombo_offset(self):
        self.assertTrue(dual_time(NOW)["asia_colombo"].endswith("+05:30"))

    def test_03_dashboard_top_level_contract(self):
        self.assertEqual(set(state()), {"contract_version", "system", "market", "predictions", "decision", "trade_window", "event", "intelligence", "health", "reasons"})

    def test_04_no_champion_overall_status(self):
        self.assertEqual(state()["system"]["status"], "PASS_FAIL_CLOSED_NO_CHAMPION")

    def test_05_wait_decision_has_no_execution(self):
        self.assertFalse(state()["decision"]["trading_enabled"])

    def test_06_manual_confirmation_is_required(self):
        self.assertTrue(state()["decision"]["manual_confirmation_required"])

    def test_07_degraded_intelligence_health(self):
        item = advisory(); item["intelligence"]["status"] = "DEGRADED"
        result = assemble_state(item, {"status": "PASS", "horizons": {}}, {}, {}, {"status": "AVAILABLE_EMPTY"})
        self.assertIn("v05b_intelligence", result["health"]["degraded_sources"])

    def test_08_missing_market_fails_closed(self):
        item = advisory("WAIT_STALE_MARKET_DATA", "PASS_WAIT"); item["market"]["freshness"]["status"] = "STALE"
        self.assertEqual(assemble_state(item, {}, {}, {}, {"status": "AVAILABLE_EMPTY"})["system"]["status"], "FAIL_CLOSED")

    def test_09_weekend_session_is_propagated_without_asia(self):
        item = advisory()
        item["market"]["session"] = "WEEKEND"
        result = assemble_state(item, {}, {}, {}, {"status": "AVAILABLE_EMPTY"})
        self.assertEqual(result["market"]["session"], "WEEKEND")
        self.assertNotEqual(result["market"]["session"], "ASIA")

    def test_10_history_appends_one_unique_decision(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("src.runtime.v06c_engine.LATEST_STATE_FILE", root / "latest.json"), patch("src.runtime.v06c_engine.STATE_HISTORY_FILE", root / "history.parquet"):
                self.assertTrue(persist_state(state()))
                self.assertFalse(persist_state(state()))

    def test_10_latest_state_is_valid_json(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "latest.json"
            with patch("src.runtime.v06c_engine.LATEST_STATE_FILE", path), patch("src.runtime.v06c_engine.STATE_HISTORY_FILE", Path(temp) / "history.parquet"):
                persist_state(state())
                self.assertEqual(json.loads(path.read_text())["decision"]["action"], "WAIT")

    def test_11_history_contains_no_outcome_fields(self):
        with TemporaryDirectory() as temp:
            history = Path(temp) / "history.parquet"
            with patch("src.runtime.v06c_engine.LATEST_STATE_FILE", Path(temp) / "latest.json"), patch("src.runtime.v06c_engine.STATE_HISTORY_FILE", history):
                persist_state(state())
                self.assertFalse(any("outcome" in name for name in pd.read_parquet(history).columns))

    def test_12_registry_reload_metadata_is_visible(self):
        item = state()
        self.assertIn("registry", item["system"])

    def test_13_runtime_lock_rejects_duplicate(self):
        with TemporaryDirectory() as temp:
            lock = Path(temp) / "runtime.lock"
            with patch("src.runtime.v06c_runner.RUNTIME_LOCK_FILE", lock):
                self.assertTrue(_acquire_runtime_lock())
                self.assertFalse(_acquire_runtime_lock())
                _release_runtime_lock()

    def test_14_runtime_lock_releases_cleanly(self):
        with TemporaryDirectory() as temp:
            lock = Path(temp) / "runtime.lock"
            with patch("src.runtime.v06c_runner.RUNTIME_LOCK_FILE", lock):
                self.assertTrue(_acquire_runtime_lock())
                _release_runtime_lock()
                self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
