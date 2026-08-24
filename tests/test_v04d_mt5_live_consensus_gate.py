from __future__ import annotations

import unittest

import pandas as pd

from src.events.mt5_live_consensus_gate import build_gate


def _row(event_id: int = 840030007, captured: str = "2026-09-11T12:25:00Z", event_server: str = "2026-09-11 15:30:00") -> dict[str, object]:
    return {
        "value_id": 1,
        "event_id": event_id,
        "event_time_server": event_server,
        "event_name": "CPI y/y",
        "event_code": "consumer-price-index-yy",
        "country_code": "US",
        "currency": "USD",
        "forecast_value": 2.9,
        "exported_at_server": "2026-09-11 15:25:00",
        "captured_at_gmt": captured,
    }


class V04DLiveConsensusGateTests(unittest.TestCase):
    def test_exact_target_pre_release_snapshot_becomes_eligible(self):
        gated, canonical = build_gate(pd.DataFrame([_row()]))
        item = gated.iloc[0]
        self.assertTrue(bool(item["server_offset_verified"]))
        self.assertAlmostEqual(float(item["server_offset_seconds"]), 10800.0)
        self.assertTrue(bool(item["event_identity_verified"]))
        self.assertTrue(bool(item["strictly_pre_release"]))
        self.assertTrue(bool(item["model_eligible_consensus"]))
        self.assertEqual(len(canonical), 1)

    def test_capture_exactly_at_release_is_rejected(self):
        row = _row(captured="2026-09-11T12:30:00Z")
        row["exported_at_server"] = "2026-09-11 15:30:00"
        gated, canonical = build_gate(pd.DataFrame([row]))
        self.assertFalse(bool(gated.iloc[0]["strictly_pre_release"]))
        self.assertFalse(bool(gated.iloc[0]["model_eligible_consensus"]))
        self.assertEqual(len(canonical), 0)

    def test_wrong_variant_id_is_not_eligible(self):
        row = _row(event_id=840030024)
        row["event_name"] = "U6 Unemployment Rate"
        row["event_code"] = "u6-unemployment-rate"
        gated, _ = build_gate(pd.DataFrame([row]))
        self.assertFalse(bool(gated.iloc[0]["event_identity_verified"]))

    def test_latest_verified_snapshot_is_canonical(self):
        early = _row(captured="2026-09-11T12:00:00Z")
        early["exported_at_server"] = "2026-09-11 15:00:00"
        early["forecast_value"] = 2.8
        late = _row(captured="2026-09-11T12:25:00Z")
        late["exported_at_server"] = "2026-09-11 15:25:00"
        late["value_id"] = 2
        late["forecast_value"] = 2.9
        _, canonical = build_gate(pd.DataFrame([early, late]))
        self.assertEqual(len(canonical), 1)
        self.assertAlmostEqual(float(canonical.iloc[0]["forecast_value"]), 2.9)

    def test_missing_forecast_stays_ineligible(self):
        row = _row()
        row["forecast_value"] = float("nan")
        gated, _ = build_gate(pd.DataFrame([row]))
        self.assertFalse(bool(gated.iloc[0]["model_eligible_consensus"]))
        self.assertEqual(gated.iloc[0]["consensus_status"], "FORECAST_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
