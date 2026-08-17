from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "events" / "export_events_for_jforex.py"
SPEC = importlib.util.spec_from_file_location("export_events_for_jforex", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def event_frame() -> pd.DataFrame:
    rows = []
    for year in (2024, 2025):
        rows.extend(
            [
                {
                    "event_id": f"employment-{year}-01",
                    "event_type": "us_employment_situation",
                    "reference_period": f"{year - 1}-12-01T00:00:00Z",
                    "event_timestamp_utc": f"{year}-01-05T13:30:00Z",
                    "timestamp_precision": "minute_reconstructed",
                    "timestamp_source": "test",
                    "model_eligible": True,
                },
                {
                    "event_id": f"employment-{year}-02",
                    "event_type": "us_employment_situation",
                    "reference_period": f"{year}-01-01T00:00:00Z",
                    "event_timestamp_utc": f"{year}-02-02T13:30:00Z",
                    "timestamp_precision": "minute_reconstructed",
                    "timestamp_source": "test",
                    "model_eligible": True,
                },
                {
                    "event_id": f"cpi-{year}-01",
                    "event_type": "us_cpi",
                    "reference_period": f"{year - 1}-12-01T00:00:00Z",
                    "event_timestamp_utc": f"{year}-01-11T13:30:00Z",
                    "timestamp_precision": "minute_reconstructed",
                    "timestamp_source": "test",
                    "model_eligible": True,
                },
                {
                    "event_id": f"cpi-{year}-02",
                    "event_type": "us_cpi",
                    "reference_period": f"{year}-01-01T00:00:00Z",
                    "event_timestamp_utc": f"{year}-02-13T13:30:00Z",
                    "timestamp_precision": "minute_reconstructed",
                    "timestamp_source": "test",
                    "model_eligible": True,
                },
            ]
        )
    return pd.DataFrame(rows)


class ExportEventsForJForexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        fake_event_file = Path(self.temp.name) / "events.parquet"
        fake_event_file.write_bytes(b"test fixture placeholder")
        self.original_event_file = module.EVENT_FILE
        module.EVENT_FILE = fake_event_file

    def tearDown(self) -> None:
        module.EVENT_FILE = self.original_event_file
        self.temp.cleanup()

    def build(self, frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with patch.object(module.pd, "read_parquet", return_value=frame.copy()):
            return module.build_export(**kwargs)

    def test_sample_per_year_covers_every_event_type(self) -> None:
        result = self.build(event_frame(), sample_per_year=1)
        self.assertEqual(len(result), 4)
        grouped = (
            pd.to_datetime(result["event_timestamp_utc"], utc=True)
            .dt.year.to_frame("year")
            .assign(event_type=result["event_type"].to_numpy())
            .value_counts()
        )
        self.assertEqual(set(grouped.index), {
            (2024, "us_cpi"),
            (2024, "us_employment_situation"),
            (2025, "us_cpi"),
            (2025, "us_employment_situation"),
        })
        self.assertTrue((grouped == 1).all())

    def test_sample_and_limit_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            self.build(event_frame(), sample_per_year=1, limit=2)

    def test_sample_and_event_id_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            self.build(event_frame(), sample_per_year=1, event_id="cpi-2024-01")

    def test_duplicate_event_id_is_rejected(self) -> None:
        frame = event_frame()
        frame.loc[1, "event_id"] = frame.loc[0, "event_id"]
        with self.assertRaisesRegex(RuntimeError, "duplicate event_id"):
            self.build(frame)

    def test_missing_event_type_is_rejected(self) -> None:
        frame = event_frame()
        frame.loc[0, "event_type"] = ""
        with self.assertRaisesRegex(RuntimeError, "missing event_type"):
            self.build(frame)


if __name__ == "__main__":
    unittest.main()
