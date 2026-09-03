from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.dashboard.realtime import load_quote_snapshot, prepare_stream_payload


class DashboardRealtimeFreshnessTests(unittest.TestCase):
    @staticmethod
    def snapshot(received: datetime) -> dict[str, object]:
        return {
            "contract_version": "realtime-quote-v1",
            "symbol": "XAUUSD",
            "connection": "LIVE",
            "freshness": "LIVE",
            "received_at_utc": received.isoformat(),
            "normalized_tick_utc": received.isoformat(),
            "bid": 4441.0,
            "ask": 4442.0,
            "mid": 4441.5,
            "spread_points": 100.0,
            "sequence": 20,
            "trading_enabled": False,
        }

    def test_recent_snapshot_remains_live(self) -> None:
        now = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)
        payload = prepare_stream_payload(self.snapshot(now - timedelta(seconds=1)), now)
        self.assertEqual(payload["connection"], "LIVE")
        self.assertEqual(payload["freshness"], "LIVE")
        self.assertLess(float(payload["snapshot_age_ms"]), 3000)

    def test_stopped_snapshot_is_downgraded_to_stale(self) -> None:
        now = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)
        payload = prepare_stream_payload(self.snapshot(now - timedelta(seconds=5)), now)
        self.assertEqual(payload["connection"], "STALE")
        self.assertEqual(payload["freshness"], "STALE")

    def test_long_stopped_snapshot_is_disconnected(self) -> None:
        now = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)
        payload = prepare_stream_payload(self.snapshot(now - timedelta(seconds=20)), now)
        self.assertEqual(payload["connection"], "DISCONNECTED")
        self.assertEqual(payload["freshness"], "STALE")

    def test_disk_snapshot_is_aged_at_read_time(self) -> None:
        now = datetime(2026, 9, 3, 8, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "XAUUSD_quote.json"
            path.write_text(json.dumps(self.snapshot(now - timedelta(seconds=20))), encoding="utf-8")
            payload = load_quote_snapshot(path=path, symbol="XAUUSD", now=now)
        self.assertEqual(payload["connection"], "DISCONNECTED")
        self.assertEqual(payload["freshness"], "STALE")


if __name__ == "__main__":
    unittest.main()
