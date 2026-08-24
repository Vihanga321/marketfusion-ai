import unittest

import pandas as pd

from src.events.mt5_live_surprise_engine import build_live_surprises


class V04DLiveMt5SurpriseTests(unittest.TestCase):
    def _snapshot(self, captured: str, actual, event_id=840030015):
        return {
            "event_id": event_id,
            "event_time_server": "2026-09-04 15:30:00",
            "event_name": "Unemployment Rate" if event_id == 840030015 else "Nonfarm Payrolls",
            "event_code": "unemployment-rate" if event_id == 840030015 else "nonfarm-payrolls",
            "country_code": "US",
            "currency": "USD",
            "unit": "CALENDAR_UNIT_PERCENT" if event_id == 840030015 else "CALENDAR_UNIT_JOB",
            "multiplier": "CALENDAR_MULTIPLIER_NONE" if event_id == 840030015 else "CALENDAR_MULTIPLIER_THOUSANDS",
            "actual_value": actual,
            "exported_at_server": pd.Timestamp(captured).tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=3),
            "captured_at_gmt": captured,
        }

    def _consensus(self, event_id=840030015):
        return pd.DataFrame([{
            "event_id": event_id,
            "event_timestamp_utc": "2026-09-04T12:30:00Z",
            "event_name": "Unemployment Rate" if event_id == 840030015 else "Nonfarm Payrolls",
            "event_code": "unemployment-rate" if event_id == 840030015 else "nonfarm-payrolls",
            "forecast_value": 4.1 if event_id == 840030015 else 41.0,
            "captured_at_gmt": "2026-09-04T12:25:00Z",
            "consensus_status": "VERIFIED_PRE_RELEASE_CONSENSUS",
        }])

    def test_verified_post_release_actual_builds_raw_surprise(self):
        snapshots = pd.DataFrame([self._snapshot("2026-09-04T12:31:00Z", 4.3)])
        _, surprises = build_live_surprises(snapshots, self._consensus())
        self.assertEqual(len(surprises), 1)
        self.assertAlmostEqual(float(surprises.iloc[0]["raw_surprise"]), 0.2)
        self.assertEqual(surprises.iloc[0]["surprise_status"], "VERIFIED_LIVE_RAW_SURPRISE")

    def test_pre_release_actual_is_not_accepted(self):
        snapshots = pd.DataFrame([self._snapshot("2026-09-04T12:29:00Z", 4.3)])
        _, surprises = build_live_surprises(snapshots, self._consensus())
        self.assertTrue(surprises.empty)

    def test_earliest_post_release_actual_wins(self):
        snapshots = pd.DataFrame([
            self._snapshot("2026-09-04T12:31:00Z", 4.2),
            self._snapshot("2026-09-04T13:00:00Z", 4.4),
        ])
        _, surprises = build_live_surprises(snapshots, self._consensus())
        self.assertEqual(len(surprises), 1)
        self.assertAlmostEqual(float(surprises.iloc[0]["actual_value"]), 4.2)

    def test_missing_actual_yields_no_surprise(self):
        snapshots = pd.DataFrame([self._snapshot("2026-09-04T12:31:00Z", None)])
        _, surprises = build_live_surprises(snapshots, self._consensus())
        self.assertTrue(surprises.empty)


if __name__ == "__main__":
    unittest.main()
