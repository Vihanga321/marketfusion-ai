from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load("v04a_contract", "src/events/v04a_contract.py")
import sys
sys.modules["v04a_contract"] = contract
adjudication = load("build_v04a_adjudication", "src/events/build_v04a_adjudication.py")


class V04AContractTests(unittest.TestCase):
    def test_frozen_evidence_detects_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = contract.FROZEN_EVIDENCE_SHA256
            try:
                contract.FROZEN_EVIDENCE_SHA256 = {"evidence.txt": "00" * 32}
                (root / "evidence.txt").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "frozen evidence changed"):
                    contract.verify_frozen_evidence(root)
            finally:
                contract.FROZEN_EVIDENCE_SHA256 = old

    def test_horizon_contract_is_exact(self) -> None:
        self.assertEqual(contract.HORIZON_BAR_OFFSETS, {1: 0, 5: 4, 15: 14, 60: 59, 240: 239})
        self.assertEqual(contract.PIP_SIZE, 0.0001)
        self.assertEqual(contract.FLAT_TOLERANCE_PIPS, 0.0)


class EligibleUniverseTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path, Path]:
        mismatch_ids = list(contract.MISMATCH_EVENT_IDS)
        rows = []
        events = []
        diagnostic = []
        statuses = ["PASS"] * 242 + ["INCOMPLETE"] * 30 + ["MISMATCH"] * 4
        for index, status in enumerate(statuses):
            event_id = mismatch_ids[index - 272] if status == "MISMATCH" else f"event-{index:03d}"
            timestamp = pd.Timestamp("2015-01-01T13:30:00Z") + pd.Timedelta(days=index)
            event_type = "us_cpi_release" if index % 2 == 0 else "us_employment_situation"
            rows.append(
                {
                    "event_id": event_id, "event_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "status": status, "failure_reason": "HISTORY_EMPTY" if status == "INCOMPLETE" else (
                        "DATA_MISMATCH" if status == "MISMATCH" else ""
                    ), "validator_version": contract.CONTRACT_VERSION, "event_type": event_type,
                }
            )
            events.append(
                {
                    "event_id": event_id, "event_type": event_type,
                    "reference_period": "2014-12-01T00:00:00Z",
                    "event_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                    "timestamp_precision": "minute_reconstructed", "timestamp_source": "official fixture",
                }
            )
            if status == "MISMATCH":
                diagnostic.append(
                    {
                        "event_id": event_id, "minute_utc": timestamp.isoformat().replace("+00:00", "Z"),
                        "classification": "PROVIDER_NATIVE_TICK_DISAGREEMENT", "max_diff": "0.00001",
                        "evidence": "fixture tick replay differs from native BID",
                    }
                )
        summary = root / "summary.tsv"
        event_path = root / "events.tsv"
        mismatch_path = root / "mismatch.tsv"
        pd.DataFrame(rows).to_csv(summary, sep="\t", index=False)
        pd.DataFrame(events).to_csv(event_path, sep="\t", index=False)
        pd.DataFrame(diagnostic).to_csv(mismatch_path, sep="\t", index=False)
        return summary, event_path, mismatch_path

    def test_universe_is_strict_and_quarantines_34(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = self._fixtures(Path(temp))
            frame, eligible, _ = adjudication.build_adjudication(*paths)
            self.assertEqual(len(frame), 276)
            self.assertEqual(len(eligible), 242)
            self.assertTrue(eligible["production_status"].eq("PASS").all())
            self.assertTrue(eligible["model_eligible_market_reaction"].all())
            quarantine = set(frame.loc[~frame["model_eligible_market_reaction"], "event_id"])
            self.assertEqual(len(quarantine), 34)
            self.assertFalse(bool(quarantine & set(eligible["event_id"])))
            self.assertFalse(bool(contract.MISMATCH_EVENT_IDS & set(eligible["event_id"])))
            self.assertFalse(eligible["event_id"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
