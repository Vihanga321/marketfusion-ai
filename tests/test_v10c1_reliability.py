from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.runtime import v10c1_reliability as reliability


class V10C1ReliabilityTests(unittest.TestCase):
    def test_machine_snapshot_reports_capacity_without_execution_features(self):
        with TemporaryDirectory() as temp:
            result = reliability.machine_snapshot(Path(temp))
        self.assertIn(result["disk_free_status"], {"PASS", "FAIL_LOW_DISK"})
        self.assertGreater(result["disk_total_gb"], 0)
        self.assertNotIn("trading_enabled", result)

    def test_single_instance_lock_rejects_second_holder(self):
        with TemporaryDirectory() as temp:
            lock_path = Path(temp) / "collector.lock"
            with reliability.SingleInstanceLock(lock_path):
                with self.assertRaises(RuntimeError):
                    with reliability.SingleInstanceLock(lock_path):
                        pass

    def test_safety_manifest_is_required_and_validated(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = reliability._frozen_paths(root)["manifest"]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({
                "automatic_execution": "DISABLED",
                "runtime": "SHADOW_ADVISORY_ONLY",
                "manual_confirmation": "REQUIRED",
                "retraining_during_forward_window": False,
                "models": {},
            }), encoding="utf-8")
            manifest = reliability._assert_initialized_safely(root)
            self.assertEqual(manifest["automatic_execution"], "DISABLED")

    def test_safety_manifest_rejects_execution_change(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = reliability._frozen_paths(root)["manifest"]
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({
                "automatic_execution": "ENABLED",
                "runtime": "SHADOW_ADVISORY_ONLY",
                "manual_confirmation": "REQUIRED",
                "retraining_during_forward_window": False,
                "models": {},
            }), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                reliability._assert_initialized_safely(root)

    def test_status_is_fail_closed_before_supervisor_starts(self):
        with TemporaryDirectory() as temp:
            payload = reliability.status(Path(temp))
        self.assertEqual(payload["collector"]["state"], "NOT_STARTED")
        self.assertEqual(payload["safety"]["automatic_execution"], "DISABLED")
        self.assertEqual(payload["safety"]["backfill"], "PROHIBITED")


if __name__ == "__main__":
    unittest.main()
