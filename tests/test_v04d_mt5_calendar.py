from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.events.mt5_calendar_importer import (
    build_historical_audit,
    build_snapshot_audit,
    classify_candidate_measure,
    historical_consensus_status,
    read_mt5_calendar_history_tsv,
    strictly_pre_release,
)


HEADER = [
    "value_id", "event_id", "event_time_server", "reference_period_server", "revision",
    "actual_value", "forecast_value", "prev_value", "revised_prev_value", "impact_type",
    "event_name", "event_code", "country_code", "currency", "unit", "importance",
    "multiplier", "digits", "time_mode", "sector", "frequency", "source_url",
    "exported_at_server",
]


def _base_row() -> dict[str, object]:
    return {
        "value_id": 1,
        "event_id": 2,
        "event_time_server": pd.Timestamp("2024-01-11 15:30:00"),
        "reference_period_server": pd.Timestamp("2023-12-01 00:00:00"),
        "revision": 0,
        "actual_value": 3.4,
        "forecast_value": 3.2,
        "prev_value": 3.1,
        "revised_prev_value": float("nan"),
        "impact_type": "CALENDAR_IMPACT_NEGATIVE",
        "event_name": "CPI y/y",
        "event_code": "cpi-yoy",
        "country_code": "US",
        "currency": "USD",
        "unit": "CALENDAR_UNIT_PERCENT",
        "importance": "CALENDAR_IMPORTANCE_HIGH",
        "multiplier": "CALENDAR_MULTIPLIER_NONE",
        "digits": 1,
        "time_mode": "CALENDAR_TIMEMODE_DATETIME",
        "sector": "CALENDAR_SECTOR_PRICES",
        "frequency": "CALENDAR_FREQUENCY_MONTH",
        "source_url": "https://example.invalid",
        "exported_at_server": pd.Timestamp("2026-08-24 10:00:00"),
    }


class V04DFreeMt5CalendarTests(unittest.TestCase):
    def test_candidate_mapping_is_not_approval(self):
        measure, status = classify_candidate_measure("Core CPI y/y", "core-cpi-yoy")
        self.assertEqual(measure, "core_cpi_yoy_sa")
        self.assertEqual(status, "CANDIDATE_NAME_CODE_MATCH")

    def test_historical_forecast_presence_remains_unverified(self):
        self.assertEqual(
            historical_consensus_status(2.9),
            "HISTORICAL_FORECAST_PRESENT_UNVERIFIED_SNAPSHOT_TIME",
        )
        audit = build_historical_audit(pd.DataFrame([_base_row()]))
        self.assertFalse(bool(audit.loc[0, "point_in_time_verified"]))
        self.assertFalse(bool(audit.loc[0, "model_eligible_consensus"]))
        self.assertTrue(pd.isna(audit.loc[0, "consensus_available_from_utc"]))

    def test_missing_historical_forecast_stays_unavailable(self):
        self.assertEqual(historical_consensus_status(float("nan")), "UNAVAILABLE")

    def test_snapshot_proves_capture_time_not_event_identity(self):
        row = _base_row()
        row["captured_at_gmt"] = pd.Timestamp("2024-01-11T13:00:00Z")
        audit = build_snapshot_audit(pd.DataFrame([row]))
        self.assertTrue(bool(audit.loc[0, "snapshot_capture_timestamp_verified"]))
        self.assertFalse(bool(audit.loc[0, "event_identity_verified"]))
        self.assertFalse(bool(audit.loc[0, "model_eligible_consensus"]))

    def test_pre_release_gate_is_strict(self):
        self.assertTrue(
            strictly_pre_release("2024-01-11T13:29:59Z", "2024-01-11T13:30:00Z")
        )
        self.assertFalse(
            strictly_pre_release("2024-01-11T13:30:00Z", "2024-01-11T13:30:00Z")
        )
        self.assertFalse(
            strictly_pre_release("2024-01-11T13:30:01Z", "2024-01-11T13:30:00Z")
        )

    def test_reader_rejects_non_usd_rows(self):
        row = [
            "1", "2", "2024.01.11 15:30:00", "2023.12.01 00:00:00", "0",
            "3.4", "3.2", "3.1", "", "CALENDAR_IMPACT_NEGATIVE",
            "CPI y/y", "cpi-yoy", "US", "EUR", "CALENDAR_UNIT_PERCENT",
            "CALENDAR_IMPORTANCE_HIGH", "CALENDAR_MULTIPLIER_NONE", "1",
            "CALENDAR_TIMEMODE_DATETIME", "CALENDAR_SECTOR_PRICES",
            "CALENDAR_FREQUENCY_MONTH", "https://example.invalid", "2026.08.24 10:00:00",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calendar.tsv"
            path.write_text("\t".join(HEADER) + "\n" + "\t".join(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-USD"):
                read_mt5_calendar_history_tsv(path)


if __name__ == "__main__":
    unittest.main()
