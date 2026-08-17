from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from events import validate_event_table as validator  # noqa: E402
from events.reaction_semantics import (  # noqa: E402
    assert_market_reaction_training_gate,
    pre_event_bar_open,
    reaction_bar_open,
    requested_times_only,
)


def valid_events() -> pd.DataFrame:
    rows = []
    eastern = ZoneInfo("America/New_York")
    for month in range(1, 9):
        reference = pd.Timestamp(year=2024, month=month, day=1, tz="UTC")
        release_date = (reference + pd.offsets.MonthEnd(0) + pd.Timedelta(days=10)).date()
        for event_type in ("us_cpi_release", "us_employment_situation"):
            local = datetime(
                release_date.year, release_date.month, release_date.day, 8, 30, tzinfo=eastern
            )
            utc = pd.Timestamp(local).tz_convert("UTC")
            rows.append(
                {
                    "event_id": f"bls:{event_type}:{utc.strftime('%Y%m%dT%H%MZ')}",
                    "event_timestamp_utc": utc,
                    "event_timestamp_local": local.isoformat(),
                    "local_timezone": "America/New_York",
                    "country": "US",
                    "currency": "USD",
                    "event_type": event_type,
                    "event_name": event_type,
                    "reference_period": reference,
                    "actual": np.nan,
                    "forecast": np.nan,
                    "previous": np.nan,
                    "revised_previous": np.nan,
                    "source": "official test fixture",
                    "source_url": "https://example.invalid",
                    "timestamp_source": validator.RECONSTRUCTED_TIMESTAMP_SOURCE,
                    "timestamp_precision": "minute_reconstructed",
                    "stated_timezone_abbreviation": local.tzname(),
                    "model_eligible": True,
                    "quarantine_reason": "",
                    "provenance_tier": "dual_official_reconstruction",
                    "alfred_crosscheck_series": "crosscheck",
                }
            )
    return pd.DataFrame(rows)


class EventTimestampValidationTests(unittest.TestCase):
    def validate(self, frame: pd.DataFrame):
        with tempfile.TemporaryDirectory() as temp:
            placeholder = Path(temp) / "events.parquet"
            placeholder.write_bytes(b"fixture")
            with patch.object(validator, "BLS_FILE", placeholder), patch.object(
                validator.pd, "read_parquet", return_value=frame.copy()
            ):
                return validator.validate(write_report=False)

    def test_valid_dst_aware_minute_timestamps_pass(self) -> None:
        result = self.validate(valid_events())
        self.assertTrue(result.passed, result.failures)

    def test_subminute_timestamp_is_rejected(self) -> None:
        frame = valid_events()
        frame.loc[0, "event_timestamp_utc"] += pd.Timedelta(seconds=30)
        result = self.validate(frame)
        self.assertFalse(result.passed)
        self.assertTrue(any("minute boundaries" in failure for failure in result.failures))

    def test_inconsistent_local_dst_timestamp_is_rejected(self) -> None:
        frame = valid_events()
        frame.loc[0, "event_timestamp_local"] = "2025-02-10T08:30:00-04:00"
        result = self.validate(frame)
        self.assertFalse(result.passed)
        self.assertTrue(any("local/UTC/DST" in failure for failure in result.failures))

    def test_missing_event_type_is_rejected(self) -> None:
        frame = valid_events()
        frame.loc[0, "event_type"] = ""
        result = self.validate(frame)
        self.assertFalse(result.passed)
        self.assertTrue(any("missing event_type" in failure for failure in result.failures))


class ReactionSemanticsTests(unittest.TestCase):
    def test_horizon_indexing(self) -> None:
        event = pd.Timestamp("2025-01-10T13:30:00Z")
        self.assertEqual(pre_event_bar_open(event), pd.Timestamp("2025-01-10T13:29:00Z"))
        self.assertEqual(reaction_bar_open(event, 1), pd.Timestamp("2025-01-10T13:30:00Z"))
        self.assertEqual(reaction_bar_open(event, 5), pd.Timestamp("2025-01-10T13:34:00Z"))
        self.assertEqual(reaction_bar_open(event, 240), pd.Timestamp("2025-01-10T17:29:00Z"))

    def test_training_gate_rejects_non_pass(self) -> None:
        frame = pd.DataFrame(
            {"status": ["PASS", "INCOMPLETE"], "model_eligible_market_reaction": [True, False]}
        )
        with self.assertRaisesRegex(ValueError, "non-PASS rows"):
            assert_market_reaction_training_gate(frame)

    def test_training_gate_requires_explicit_eligibility(self) -> None:
        frame = pd.DataFrame(
            {"status": ["PASS", "PASS"], "model_eligible_market_reaction": [True, "false"]}
        )
        with self.assertRaisesRegex(ValueError, "lack model_eligible"):
            assert_market_reaction_training_gate(frame)

    def test_out_of_range_mt5_cache_bars_are_quarantined(self) -> None:
        times = pd.Series(
            pd.to_datetime(
                ["2025-01-10T13:20:00Z", "2025-01-10T13:30:00Z", "2026-05-11T22:33:00Z"],
                utc=True,
            )
        )
        usable, quarantined = requested_times_only(
            times,
            pd.Timestamp("2025-01-10T13:20:00Z"),
            pd.Timestamp("2025-01-10T17:40:00Z"),
        )
        self.assertEqual(len(usable), 2)
        self.assertEqual(quarantined, 1)


if __name__ == "__main__":
    unittest.main()
