from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.events.mt5_calendar_identity_audit import (
    REJECTED_VARIANT_EVENT_IDS,
    TARGET_EVENT_IDS,
    _month_key,
    _read,
    audit_identity,
)


def _row(event_id: int, code: str, name: str, unit: str = "CALENDAR_UNIT_PERCENT") -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_code": code,
        "event_name": name,
        "event_time_server": pd.Timestamp("2024-01-11 15:30:00"),
        "reference_period_server": pd.Timestamp("2023-12-01 00:00:00"),
        "actual_value": 3.4,
        "forecast_value": 3.2,
        "prev_value": 3.1,
        "revised_prev_value": float("nan"),
        "country_code": "US",
        "currency": "USD",
        "unit": unit,
        "importance": "CALENDAR_IMPORTANCE_HIGH",
        "multiplier": "CALENDAR_MULTIPLIER_NONE",
        "digits": 1,
        "time_mode": "CALENDAR_TIMEMODE_DATETIME",
        "frequency": "CALENDAR_FREQUENCY_MONTH",
    }


class V04DMt5IdentityTests(unittest.TestCase):
    def test_exact_target_ids_are_the_four_intended_measures(self):
        self.assertEqual(set(TARGET_EVENT_IDS), {840030007, 840030008, 840030015, 840030016})
        self.assertEqual(
            {item["measure_id"] for item in TARGET_EVENT_IDS.values()},
            {"headline_cpi_yoy_sa", "core_cpi_yoy_sa", "unemployment_rate", "nonfarm_payroll_change"},
        )

    def test_similar_variants_are_explicitly_rejected(self):
        self.assertIn(840030024, REJECTED_VARIANT_EVENT_IDS)
        self.assertIn(840030023, REJECTED_VARIANT_EVENT_IDS)

    def test_exact_identity_candidate_still_not_model_eligible(self):
        expected = TARGET_EVENT_IDS[840030007]
        identity, _ = audit_identity(pd.DataFrame([
            _row(840030007, expected["event_code"], expected["event_name"]),
        ]))
        row = identity.loc[identity["event_id"].eq(840030007)].iloc[0]
        self.assertEqual(row["identity_status"], "EXACT_EVENT_ID_CANDIDATE")
        self.assertFalse(bool(row["historical_pit_verified"]))
        self.assertFalse(bool(row["model_eligible_consensus"]))

    def test_name_or_code_mutation_requires_review(self):
        expected = TARGET_EVENT_IDS[840030015]
        identity, _ = audit_identity(pd.DataFrame([
            _row(840030015, expected["event_code"], "U6 Unemployment Rate"),
        ]))
        row = identity.loc[identity["event_id"].eq(840030015)].iloc[0]
        self.assertEqual(row["identity_status"], "IDENTITY_REVIEW_REQUIRED")

    def test_non_usd_identity_requires_review(self):
        expected = TARGET_EVENT_IDS[840030016]
        row = _row(840030016, expected["event_code"], expected["event_name"])
        row["currency"] = "EUR"
        identity, _ = audit_identity(pd.DataFrame([row]))
        item = identity.loc[identity["event_id"].eq(840030016)].iloc[0]
        self.assertEqual(item["identity_status"], "IDENTITY_REVIEW_REQUIRED")

    def test_month_key_accepts_mixed_date_and_timestamp_strings(self):
        result = _month_key(pd.Series([
            "2024-01-01",
            "2024-02-01 00:00:00",
            "2024-03-01T00:00:00+00:00",
        ]))
        self.assertEqual(result.tolist(), ["2024-01", "2024-02", "2024-03"])

    def test_reader_accepts_mixed_mt5_datetime_formats(self):
        rows = [
            _row(840030007, "consumer-price-index-yy", "CPI y/y"),
            _row(840030007, "consumer-price-index-yy", "CPI y/y"),
        ]
        rows[0]["event_time_server"] = "2024-01-11"
        rows[0]["reference_period_server"] = "2023-12-01"
        rows[1]["event_time_server"] = "2024-02-13 15:30:00"
        rows[1]["reference_period_server"] = "2024-01-01 00:00:00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            frame = _read(path)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["event_time_server"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["reference_period_server"]))
        self.assertEqual(frame["event_time_server"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-11", "2024-02-13"])


if __name__ == "__main__":
    unittest.main()
