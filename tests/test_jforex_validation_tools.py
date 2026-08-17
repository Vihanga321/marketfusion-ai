from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


batching = load_module(
    "prepare_jforex_validation_batches",
    "src/events/prepare_jforex_validation_batches.py",
)
resume = load_module(
    "resume_jforex_validation",
    "src/events/resume_jforex_validation.py",
)
sys.modules["resume_jforex_validation"] = resume
aggregator = load_module(
    "aggregate_jforex_validation",
    "src/events/aggregate_jforex_validation.py",
)
safety = load_module("repo_safety_check", "scripts/repo_safety_check.py")


EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "reference_period",
    "event_timestamp_utc",
    "timestamp_precision",
    "timestamp_source",
]


def write_events(path: Path, count: int = 5) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "event_id": f"event-{index:02d}",
                "event_type": "us_cpi" if index % 2 == 0 else "us_employment_situation",
                "reference_period": "2024-01-01T00:00:00Z",
                "event_timestamp_utc": f"2025-01-{index + 1:02d}T13:30:00Z",
                "timestamp_precision": "minute_reconstructed",
                "timestamp_source": "test",
            }
        )
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    return frame


def write_batch_summary(path: Path, events: pd.DataFrame, status: str = "PASS") -> None:
    rows = []
    for _, event in events.iterrows():
        is_pass = status == "PASS"
        is_incomplete = status == "INCOMPLETE"
        is_mismatch = status == "MISMATCH"
        rows.append(
            {
                "event_id": event["event_id"],
                "event_timestamp_utc": event["event_timestamp_utc"],
                "expected_minutes": 261,
                "native_bid_bars": 261 if status != "ERROR" else 0,
                "historical_ticks": 1000,
                "rebuilt_minutes": 221 if is_incomplete else (0 if status == "ERROR" else 261),
                "matched_minutes": 221 if is_incomplete else (260 if is_mismatch else (0 if status == "ERROR" else 261)),
                "mismatched_minutes": 1 if is_mismatch else 0,
                "missing_minutes": 40 if is_incomplete else (261 if status == "ERROR" else 0),
                "invalid_spread_ticks": 0,
                "missing_chunks": 1 if is_incomplete else 0,
                "max_abs_ohlc_diff": 0.0001 if is_mismatch else 0.0,
                "status": status,
                "error": "" if is_pass else ("provider gap" if is_incomplete else "synthetic diagnostic"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    chunk_rows = []
    for _, event in events.iterrows():
        missing = status == "INCOMPLETE"
        chunk_rows.append({
            "event_id": event["event_id"],
            "data_kind": "ticks",
            "chunk_from_utc": event["event_timestamp_utc"],
            "chunk_to_utc": event["event_timestamp_utc"],
            "attempts": 3,
            "rows": 0 if missing else 1000,
            "status": "MISSING" if missing else "OK",
            "error": "no ticks returned" if missing else "",
        })
    pd.DataFrame(chunk_rows).to_csv(path.with_name(aggregator.CHUNK_FILENAME), sep="\t", index=False)


class BatchPreparationTests(unittest.TestCase):
    def test_batches_are_complete_deterministic_and_stale_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events_file = root / "events.tsv"
            write_events(events_file, count=5)
            out = root / "batches"

            stale = out / "batch_999"
            stale.mkdir(parents=True)
            (stale / "events.tsv").write_text("stale", encoding="utf-8")

            manifest = batching.prepare_batches(events_file, out, batch_size=2)
            self.assertEqual(manifest["event_count"].tolist(), [2, 2, 1])
            self.assertEqual(manifest["batch_id"].tolist(), ["batch_001", "batch_002", "batch_003"])
            self.assertFalse(stale.exists())

            loaded_ids = []
            for batch_id in manifest["batch_id"]:
                frame = pd.read_csv(out / batch_id / "events.tsv", sep="\t", dtype=str)
                loaded_ids.extend(frame["event_id"].tolist())
            self.assertEqual(loaded_ids, [f"event-{index:02d}" for index in range(5)])

    def test_duplicate_event_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events_file = root / "events.tsv"
            frame = write_events(events_file, count=3)
            frame.loc[1, "event_id"] = frame.loc[0, "event_id"]
            frame.to_csv(events_file, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "duplicate event_id"):
                batching.prepare_batches(events_file, root / "out", batch_size=2)


class AggregationTests(unittest.TestCase):
    def test_complete_pass_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_file = root / "events.tsv"
            expected = write_events(expected_file, count=4)
            batch_root = root / "batches"
            write_batch_summary(
                batch_root / "batch_001" / "result" / aggregator.SUMMARY_FILENAME,
                expected.iloc[:2],
            )
            write_batch_summary(
                batch_root / "batch_002" / "result" / aggregator.SUMMARY_FILENAME,
                expected.iloc[2:],
            )

            result = aggregator.aggregate(
                expected_file,
                batch_root,
                root / "summary.tsv",
                root / "report.txt",
                require_metadata=False,
            )
            self.assertEqual(len(result), 4)
            self.assertTrue(result["status"].eq("PASS").all())
            self.assertIn("FULL_VALIDATION_STATUS: PASS", (root / "report.txt").read_text())

    def test_missing_expected_event_is_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_file = root / "events.tsv"
            expected = write_events(expected_file, count=3)
            batch_root = root / "batches"
            write_batch_summary(
                batch_root / "batch_001" / "result" / aggregator.SUMMARY_FILENAME,
                expected.iloc[:2],
            )
            with self.assertRaisesRegex(RuntimeError, "event-set mismatch"):
                aggregator.aggregate(
                    expected_file, batch_root, root / "out.tsv", root / "report.txt",
                    require_metadata=False,
                )

    def test_false_pass_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_file = root / "events.tsv"
            expected = write_events(expected_file, count=1)
            summary_path = root / "batches" / "batch_001" / "result" / aggregator.SUMMARY_FILENAME
            write_batch_summary(summary_path, expected)
            summary = pd.read_csv(summary_path, sep="\t")
            summary.loc[0, "missing_minutes"] = 1
            summary.to_csv(summary_path, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "PASS row"):
                aggregator.aggregate(
                    expected_file,
                    root / "batches",
                    root / "out.tsv",
                    root / "report.txt",
                    require_metadata=False,
                )

    def test_incomplete_is_preserved_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_file = root / "events.tsv"
            expected = write_events(expected_file, count=1)
            write_batch_summary(
                root / "batches" / "batch_001" / "result" / aggregator.SUMMARY_FILENAME,
                expected,
                status="INCOMPLETE",
            )
            result = aggregator.aggregate(
                expected_file,
                root / "batches",
                root / "out.tsv",
                root / "report.txt",
                require_metadata=False,
            )
            self.assertEqual(result.loc[0, "status"], "INCOMPLETE")
            self.assertIn("FULL_VALIDATION_STATUS: INCOMPLETE", (root / "report.txt").read_text())
            self.assertFalse(bool(result.loc[0, "model_eligible_market_reaction"]))
            gaps = pd.read_csv(root / "dukascopy_incomplete_events.tsv", sep="\t")
            self.assertEqual(gaps.loc[0, "missing_source"], "tick_history")

    def test_duplicate_and_unexpected_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_file = root / "events.tsv"
            expected = write_events(expected_file, count=2)
            summary_path = root / "batches" / "batch_001" / "result" / aggregator.SUMMARY_FILENAME
            write_batch_summary(summary_path, expected)
            duplicate = pd.read_csv(summary_path, sep="\t")
            duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
            duplicate.to_csv(summary_path, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "duplicate events"):
                aggregator.aggregate(expected_file, root / "batches", root / "out.tsv", root / "report.txt", require_metadata=False)

            write_batch_summary(summary_path, expected)
            unexpected = pd.read_csv(summary_path, sep="\t")
            unexpected.loc[0, "event_id"] = "unknown-event"
            unexpected.to_csv(summary_path, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "event-set mismatch"):
                aggregator.aggregate(expected_file, root / "batches", root / "out.tsv", root / "report.txt", require_metadata=False)

    def test_unknown_status_and_invalid_numeric_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected_file = root / "events.tsv"
            expected = write_events(expected_file, count=1)
            summary_path = root / "batches" / "batch_001" / "result" / aggregator.SUMMARY_FILENAME
            write_batch_summary(summary_path, expected)
            frame = pd.read_csv(summary_path, sep="\t")
            frame.loc[0, "status"] = "MAYBE"
            frame.to_csv(summary_path, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "Unknown validation status"):
                aggregator.aggregate(expected_file, root / "batches", root / "out.tsv", root / "report.txt", require_metadata=False)

            write_batch_summary(summary_path, expected)
            frame = pd.read_csv(summary_path, sep="\t")
            frame["matched_minutes"] = frame["matched_minutes"].astype(float)
            frame.loc[0, "matched_minutes"] = 260.5
            frame.to_csv(summary_path, sep="\t", index=False)
            with self.assertRaisesRegex(RuntimeError, "integer field"):
                aggregator.aggregate(expected_file, root / "batches", root / "out.tsv", root / "report.txt", require_metadata=False)

    def test_mismatch_and_error_are_preserved_and_ineligible(self) -> None:
        for status in ("MISMATCH", "ERROR"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                expected_file = root / "events.tsv"
                expected = write_events(expected_file, count=1)
                write_batch_summary(
                    root / "batches" / "batch_001" / "result" / aggregator.SUMMARY_FILENAME,
                    expected,
                    status=status,
                )
                result = aggregator.aggregate(
                    expected_file, root / "batches", root / "out.tsv", root / "report.txt",
                    require_metadata=False,
                )
                self.assertEqual(result.loc[0, "status"], status)
                self.assertFalse(bool(result.loc[0, "model_eligible_market_reaction"]))


class ResumeTests(unittest.TestCase):
    def write_metadata(self, input_file: Path, result_dir: Path) -> None:
        metadata = resume.fingerprint(input_file)
        resume.atomic_text(
            result_dir / resume.METADATA_FILENAME,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )

    def test_only_incomplete_events_are_planned_and_pass_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events_file = root / "events.tsv"
            events = write_events(events_file, count=2)
            result_dir = root / "result"
            write_batch_summary(result_dir / resume.SUMMARY_FILENAME, events.iloc[[0]], status="PASS")
            passing = pd.read_csv(result_dir / resume.SUMMARY_FILENAME, sep="\t")
            write_batch_summary(root / "incomplete" / resume.SUMMARY_FILENAME, events.iloc[[1]], status="INCOMPLETE")
            incomplete = pd.read_csv(root / "incomplete" / resume.SUMMARY_FILENAME, sep="\t")
            pd.concat([passing, incomplete], ignore_index=True).to_csv(
                result_dir / resume.SUMMARY_FILENAME, sep="\t", index=False
            )
            pass_chunks = pd.read_csv(result_dir / resume.CHUNK_FILENAME, sep="\t")
            incomplete_chunks = pd.read_csv(root / "incomplete" / resume.CHUNK_FILENAME, sep="\t")
            pd.concat([pass_chunks, incomplete_chunks], ignore_index=True).to_csv(
                result_dir / resume.CHUNK_FILENAME, sep="\t", index=False
            )
            self.write_metadata(events_file, result_dir)

            attempt_input = root / "attempt.tsv"
            plan_file = root / "plan.json"
            plan = resume.plan(events_file, result_dir, attempt_input, plan_file)
            self.assertEqual(plan["mode"], "RETRY_INCOMPLETE")
            self.assertEqual(pd.read_csv(attempt_input, sep="\t")["event_id"].tolist(), ["event-01"])

            attempt_result = root / "attempt_result"
            write_batch_summary(attempt_result / resume.SUMMARY_FILENAME, events.iloc[[1]], status="PASS")
            merged = resume.merge(events_file, result_dir, attempt_result, plan_file)
            self.assertTrue(merged["status"].eq("PASS").all())
            self.assertEqual(set(merged["event_id"]), {"event-00", "event-01"})

    def test_changed_input_invalidates_cached_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events_file = root / "events.tsv"
            events = write_events(events_file, count=1)
            result_dir = root / "result"
            write_batch_summary(result_dir / resume.SUMMARY_FILENAME, events, status="PASS")
            self.write_metadata(events_file, result_dir)
            events.loc[0, "event_type"] = "changed_type"
            events.to_csv(events_file, sep="\t", index=False)
            plan = resume.plan(events_file, result_dir, root / "attempt.tsv", root / "plan.json")
            self.assertEqual(plan["mode"], "RETRY_ALL")
            self.assertFalse(bool(plan["reuse_compatible_result"]))

    def test_missing_chunk_diagnostics_invalidates_cached_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events_file = root / "events.tsv"
            events = write_events(events_file, count=1)
            result_dir = root / "result"
            write_batch_summary(result_dir / resume.SUMMARY_FILENAME, events, status="PASS")
            self.write_metadata(events_file, result_dir)
            (result_dir / resume.CHUNK_FILENAME).unlink()

            plan = resume.plan(events_file, result_dir, root / "attempt.tsv", root / "plan.json")
            self.assertEqual(plan["mode"], "RETRY_ALL")
            self.assertFalse(bool(plan["reuse_compatible_result"]))


class SafetyPolicyTests(unittest.TestCase):
    def test_documentation_iengine_comment_is_not_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "jforex-event-exporter" / "src" / "main" / "java" / "Probe.java"
            source.parent.mkdir(parents=True)
            source.write_text("// This validator never obtains IEngine.\nclass Probe {}\n", encoding="utf-8")
            old_root = safety.ROOT
            safety.ROOT = root
            try:
                errors: list[str] = []
                safety.check_read_only_policy([source], errors)
                self.assertEqual(errors, [])
            finally:
                safety.ROOT = old_root

    def test_real_iengine_import_is_a_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "jforex-event-exporter" / "src" / "main" / "java" / "Bad.java"
            source.parent.mkdir(parents=True)
            source.write_text("import com.dukascopy.api.IEngine;\nclass Bad {}\n", encoding="utf-8")
            old_root = safety.ROOT
            safety.ROOT = root
            try:
                errors: list[str] = []
                safety.check_read_only_policy([source], errors)
                self.assertEqual(len(errors), 1)
            finally:
                safety.ROOT = old_root

    def test_order_submission_and_direct_bi5_are_violations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "src" / "Bad.java"
            source.parent.mkdir(parents=True)
            source.write_text(
                'class Bad { void x() { submitOrder(); } String u = "https://example.test/file.bi5"; }\n',
                encoding="utf-8",
            )
            old_root = safety.ROOT
            safety.ROOT = root
            try:
                errors: list[str] = []
                safety.check_read_only_policy([source], errors)
                self.assertEqual(len(errors), 2)
            finally:
                safety.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
