from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from src.dashboard.v07_api import app, load_state, trading_route_count
from src.dashboard.v07_contract import DASHBOARD_PORT, VITE_ORIGINS
from src.dashboard.v07_market import load_candles


NOW = pd.Timestamp.now(tz="UTC")


def valid_state(generated: object = NOW) -> dict[str, object]:
    return {
        "contract_version": "v0.6c-unified-runtime-v1", "system": {"status": "PASS_FAIL_CLOSED_NO_CHAMPION", "generated_time": {"utc": pd.Timestamp(generated).isoformat()}},
        "market": {}, "predictions": {"horizons": {"15": {"model_status": "NO_APPROVED_MODEL", "prob_down": None, "prob_neutral": None, "prob_up": None}}},
        "decision": {"action": "WAIT", "confidence": "VERY_LOW", "trading_enabled": False},
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None},
        "health": {}, "reasons": [],
    }


class V07DashboardApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _state_file(self, payload: object):
        temp = TemporaryDirectory(); path = Path(temp.name) / "state.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return temp, path

    def test_01_valid_state_returned_without_mutation(self):
        temp, path = self._state_file(valid_state())
        with temp, patch("src.dashboard.v07_api.V06_STATE_FILE", path):
            response = self.client.get("/api/state")
            self.assertEqual(response.json(), valid_state())

    def test_02_missing_state_returns_degraded_wait(self):
        with TemporaryDirectory() as temp, patch("src.dashboard.v07_api.V06_STATE_FILE", Path(temp) / "missing.json"):
            payload = self.client.get("/api/state").json()
            self.assertEqual((payload["system"]["status"], payload["decision"]["action"]), ("SYSTEM_DATA_INVALID", "WAIT"))

    def test_03_malformed_state_returns_degraded_wait(self):
        temp, path = self._state_file({"broken": True})
        with temp, patch("src.dashboard.v07_api.V06_STATE_FILE", path):
            self.assertEqual(self.client.get("/api/state").json()["decision"]["action"], "WAIT")

    def test_04_account_fields_are_rejected(self):
        item = valid_state(); item["account"] = "123"
        temp, path = self._state_file(item)
        with temp:
            self.assertEqual(load_state(path)[0]["system"]["status"], "SYSTEM_DATA_INVALID")

    def test_05_balance_is_rejected(self):
        item = valid_state(); item["market"]["balance"] = 1
        temp, path = self._state_file(item)
        with temp: self.assertEqual(load_state(path)[1], "INVALID")

    def test_06_equity_is_rejected(self):
        item = valid_state(); item["system"]["equity"] = 1
        temp, path = self._state_file(item)
        with temp: self.assertEqual(load_state(path)[1], "INVALID")

    def test_07_orders_are_rejected(self):
        item = valid_state(); item["orders"] = []
        temp, path = self._state_file(item)
        with temp: self.assertEqual(load_state(path)[1], "INVALID")

    def test_08_positions_are_rejected(self):
        item = valid_state(); item["positions"] = []
        temp, path = self._state_file(item)
        with temp: self.assertEqual(load_state(path)[1], "INVALID")

    def test_09_candles_allow_only_contract_timeframes(self):
        with self.assertRaises(ValueError): load_candles("../M5", 10)

    def test_10_candle_limit_is_bounded_by_api(self):
        self.assertEqual(self.client.get("/api/market/candles?timeframe=M5&limit=1001").status_code, 422)

    def test_11_path_traversal_is_impossible(self):
        self.assertIn(self.client.get("/api/market/candles?timeframe=../../.env&limit=5").status_code, {400, 422, 500})

    def test_12_stale_state_is_marked_in_header(self):
        temp, path = self._state_file(valid_state(NOW - pd.Timedelta(minutes=5)))
        with temp, patch("src.dashboard.v07_api.V06_STATE_FILE", path):
            self.assertEqual(self.client.get("/api/state").headers["x-marketfusion-state-freshness"], "STALE")

    def test_13_no_trading_endpoints_exist(self):
        self.assertEqual(trading_route_count(), 0)
        for path in ("/trade", "/api/order", "/api/buy", "/api/sell", "/api/close-position"):
            self.assertEqual(self.client.post(path).status_code, 404)

    def test_14_null_probabilities_remain_null(self):
        temp, path = self._state_file(valid_state())
        with temp, patch("src.dashboard.v07_api.V06_STATE_FILE", path):
            horizon = self.client.get("/api/state").json()["predictions"]["horizons"]["15"]
            self.assertIsNone(horizon["prob_down"]); self.assertIsNone(horizon["prob_neutral"]); self.assertIsNone(horizon["prob_up"])

    def test_15_utc_timestamp_is_preserved(self):
        item = valid_state(); expected = item["system"]["generated_time"]["utc"]
        temp, path = self._state_file(item)
        with temp, patch("src.dashboard.v07_api.V06_STATE_FILE", path):
            self.assertEqual(self.client.get("/api/state").json()["system"]["generated_time"]["utc"], expected)

    def test_16_state_endpoint_is_get_only(self):
        self.assertEqual(self.client.post("/api/state").status_code, 405)

    def test_17_version_confirms_read_only_mode(self):
        payload = self.client.get("/api/system/version").json()
        self.assertFalse(payload["trading_enabled"])

    def test_18_v08_monitor_endpoint_is_read_only_and_fail_closed(self):
        payload = self.client.get("/api/v08/status").json()
        self.assertFalse(payload["trading_enabled"])
        self.assertTrue(payload["manual_execution_only"])
        self.assertIn(payload["status"], {"INSUFFICIENT_DATA", "PASS_MONITORING_NO_CHAMPION", "PASS_SHADOW_EVALUATION", "PASS_SHADOW_EVALUATION_DEGRADED_PROVIDERS"})

    def test_19_dashboard_port_and_cors_use_safe_local_default(self):
        self.assertEqual(DASHBOARD_PORT, 4173)
        self.assertEqual(VITE_ORIGINS, ("http://127.0.0.1:4173", "http://localhost:4173"))
        response = self.client.get("/api/health", headers={"Origin": "http://127.0.0.1:4173"})
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:4173")

    def test_20_old_dashboard_origin_is_not_allowed(self):
        response = self.client.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"})
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_21_operator_status_is_read_only_and_separates_health(self):
        temp, path = self._state_file(valid_state())
        with temp, TemporaryDirectory() as data_dir, \
                patch("src.dashboard.v07_api.V06_STATE_FILE", path), \
                patch("src.dashboard.v07_api.V05A_STATUS_FILE", Path(data_dir) / "missing.json"):
            payload = self.client.get("/api/operator/status").json()
            self.assertEqual(payload["contract_version"], "v0.7-operator-status-v1")
            self.assertFalse(payload["trading_window"]["trading_enabled"])
            self.assertEqual(payload["trading_window"]["execution"], "DISABLED")
            self.assertIn(payload["system_health"]["status"], {"PASS", "DEGRADED", "FAIL"})
            self.assertIn(payload["trading_state"], {"WAIT", "BLOCKED", "MARKET_CLOSED"})


if __name__ == "__main__":
    unittest.main()
