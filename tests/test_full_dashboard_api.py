from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.dashboard.full_api import app


class FullDashboardApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @staticmethod
    def payload() -> dict[str, object]:
        return {
            "contract_version": "v1.0c-xauusd-frozen-forward-validation-v1",
            "asset_id": "XAUUSD",
            "decision": "COLLECTING_FORWARD_DATA",
            "forward_start_utc": "2026-09-07T00:00:00+00:00",
            "contracts": {
                "XAU15_SILVER_LR_V1": {
                    "active_trading_days": 1,
                    "matured_observations": 10,
                    "directional_outcomes": 8,
                    "minimum_sample_ready": False,
                    "decision": "COLLECTING",
                }
            },
            "automatic_execution": "DISABLED",
            "runtime": "SHADOW_ADVISORY_ONLY",
            "manual_confirmation": "REQUIRED",
        }

    def test_forward_status_returns_file_without_mutation(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "latest_status.json"
            expected = self.payload()
            path.write_text(json.dumps(expected), encoding="utf-8")
            with patch("src.dashboard.full_api._forward_status_candidates", return_value=[path]):
                response = self.client.get("/api/evaluation/forward/status?symbol=XAUUSD")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), expected)

    def test_forward_status_is_get_only(self) -> None:
        response = self.client.post("/api/evaluation/forward/status?symbol=XAUUSD")
        self.assertEqual(response.status_code, 405)

    def test_forward_status_never_fabricates_missing_data(self) -> None:
        with TemporaryDirectory() as temp, patch("src.dashboard.full_api._forward_status_candidates", return_value=[Path(temp) / "missing.json"]):
            response = self.client.get("/api/evaluation/forward/status?symbol=XAUUSD")
        self.assertEqual(response.status_code, 404)

    def test_forward_status_rejects_invalid_contract(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "latest_status.json"
            path.write_text(json.dumps({"decision": "COLLECTING_FORWARD_DATA"}), encoding="utf-8")
            with patch("src.dashboard.full_api._forward_status_candidates", return_value=[path]):
                response = self.client.get("/api/evaluation/forward/status?symbol=XAUUSD")
        self.assertEqual(response.status_code, 503)

    def test_forward_status_not_configured_for_eurusd(self) -> None:
        response = self.client.get("/api/evaluation/forward/status?symbol=EURUSD")
        self.assertEqual(response.status_code, 404)

    def test_reliability_endpoint_is_fail_closed_before_start(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            health = root / "collector_health.json"
            outages = root / "outages.jsonl"
            with patch("src.dashboard.full_api.RELIABILITY_HEALTH_FILE", health), patch("src.dashboard.full_api.RELIABILITY_OUTAGE_FILE", outages):
                response = self.client.get("/api/system/reliability?symbol=XAUUSD")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "NOT_STARTED")
        self.assertEqual(payload["safety"]["automatic_execution"], "DISABLED")
        self.assertEqual(payload["safety"]["backfill"], "PROHIBITED")

    def test_reliability_endpoint_returns_real_heartbeat_and_outages(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            health = root / "collector_health.json"
            outages = root / "outages.jsonl"
            health.write_text(json.dumps({
                "contract_version": "v1.0c1-xauusd-reliability-supervisor-v1",
                "collector": {"state": "RUNNING", "reconnects": 2},
                "machine": {"disk_free_gb": 100.0},
                "safety": {"automatic_execution": "DISABLED", "backfill": "PROHIBITED"},
            }), encoding="utf-8")
            outages.write_text(json.dumps({"error": "MT5 IPC", "recovered": True}) + "\n", encoding="utf-8")
            with patch("src.dashboard.full_api.RELIABILITY_HEALTH_FILE", health), patch("src.dashboard.full_api.RELIABILITY_OUTAGE_FILE", outages):
                response = self.client.get("/api/system/reliability?symbol=XAUUSD")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["collector"]["reconnects"], 2)
        self.assertEqual(len(payload["recent_outages"]), 1)

    def test_reliability_endpoint_is_get_only(self) -> None:
        response = self.client.post("/api/system/reliability?symbol=XAUUSD")
        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()
