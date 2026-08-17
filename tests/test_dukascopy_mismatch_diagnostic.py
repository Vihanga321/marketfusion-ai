from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "events" / "prepare_dukascopy_mismatch_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("prepare_mismatch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class MismatchDiagnosticPreparationTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path]:
        rows = []
        for event_id in diagnostic.TARGET_EVENT_IDS:
            rows.append(
                {
                    "event_id": event_id,
                    "event_type": "test",
                    "reference_period": "2022-01-01T00:00:00Z",
                    "event_timestamp_utc": "2022-01-01T12:30:00Z",
                }
            )
        rows.append(
            {
                "event_id": "unrelated-pass",
                "event_type": "test",
                "reference_period": "2022-01-01T00:00:00Z",
                "event_timestamp_utc": "2022-01-02T12:30:00Z",
            }
        )
        events = root / "events.tsv"
        summary = root / "summary.tsv"
        pd.DataFrame(rows).to_csv(events, sep="\t", index=False)
        pd.DataFrame(
            [{"event_id": row["event_id"], "status": "MISMATCH"} for row in rows[:-1]]
            + [{"event_id": "unrelated-pass", "status": "PASS"}]
        ).to_csv(summary, sep="\t", index=False)
        return events, summary

    def test_selects_only_exact_four_in_fixed_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            events, summary = self._write_inputs(Path(temp))
            selected = diagnostic.select_target_events(events, summary)
            self.assertEqual(selected["event_id"].tolist(), list(diagnostic.TARGET_EVENT_IDS))

    def test_rejects_any_changed_production_mismatch_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            events, summary = self._write_inputs(Path(temp))
            frame = pd.read_csv(summary, sep="\t", dtype=str)
            frame.loc[len(frame)] = {"event_id": "unexpected-mismatch", "status": "MISMATCH"}
            frame.to_csv(summary, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "does not equal"):
                diagnostic.select_target_events(events, summary)

    def test_rejects_missing_target_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            events, summary = self._write_inputs(Path(temp))
            frame = pd.read_csv(events, sep="\t", dtype=str)
            frame = frame.iloc[1:]
            frame.to_csv(events, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "lacks target"):
                diagnostic.select_target_events(events, summary)


if __name__ == "__main__":
    unittest.main()
