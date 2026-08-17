from __future__ import annotations

import importlib.util
import sys
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


contract = load("v04a_contract_reactions", "src/events/v04a_contract.py")
sys.modules["v04a_contract"] = contract
reactions = load("build_v04a_event_reactions", "src/events/build_v04a_event_reactions.py")


def synthetic_window(event_time: pd.Timestamp, missing: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    for index, minute in enumerate(
        pd.date_range(event_time - pd.Timedelta(minutes=10), event_time + pd.Timedelta(minutes=250), freq="min")
    ):
        if missing is not None and minute == missing:
            continue
        bid = 1.1000 + index * 0.00001
        ask = bid + 0.0002
        mid = (bid + ask) / 2.0
        rows.append(
            {
                "event_id": "event", "event_timestamp_utc": event_time, "minute_utc": minute,
                "bid_open": bid, "bid_high": bid + 0.00002, "bid_low": bid - 0.00002, "bid_close": bid,
                "ask_open": ask, "ask_high": ask + 0.00002, "ask_low": ask - 0.00002, "ask_close": ask,
                "mid_open": mid, "mid_high": mid + 0.00003, "mid_low": mid - 0.00003, "mid_close": mid,
                "spread_open": 0.0002, "spread_high": 0.00025, "spread_low": 0.00015,
                "spread_close": 0.0002, "tick_count": 2,
            }
        )
    return pd.DataFrame(rows)


def event_row(event_time: pd.Timestamp) -> pd.Series:
    return pd.Series(
        {
            "event_id": "event", "event_type": "us_cpi_release", "event_timestamp_utc": event_time,
            "reference_period": pd.Timestamp("2023-12-01", tz="UTC"),
            "timestamp_precision": "minute_reconstructed", "timestamp_source": "official",
            "market_data_source": "Dukascopy JForex historical ticks", "production_status": "PASS",
            "adjudication_class": "STRICT_PASS", "model_eligible_market_reaction": True,
            "reaction_contract_version": contract.REACTION_CONTRACT_VERSION,
        }
    )


class ReactionSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event_time = pd.Timestamp("2024-01-12T13:30:00Z")

    def test_exact_horizon_indexing(self) -> None:
        self.assertEqual(reactions.pre_event_bar_start(self.event_time), pd.Timestamp("2024-01-12T13:29:00Z"))
        expected = {1: 0, 5: 4, 15: 14, 60: 59, 240: 239}
        for horizon, offset in expected.items():
            self.assertEqual(
                reactions.reaction_bar_start(self.event_time, horizon),
                self.event_time + pd.Timedelta(minutes=offset),
            )

    def test_pip_direction_and_continuation_logic(self) -> None:
        self.assertAlmostEqual(reactions.pips(0.0001), 1.0)
        self.assertEqual(reactions.direction(0.1), "UP")
        self.assertEqual(reactions.direction(-0.1), "DOWN")
        self.assertEqual(reactions.direction(0.0), "FLAT")
        self.assertEqual(reactions.continuation_reversal("UP", "UP"), (True, False))
        self.assertEqual(reactions.continuation_reversal("UP", "DOWN"), (False, True))
        self.assertEqual(reactions.continuation_reversal("FLAT", "DOWN"), (False, False))

    def test_missing_minute_is_rejected_without_interpolation(self) -> None:
        missing = self.event_time + pd.Timedelta(minutes=14)
        with self.assertRaisesRegex(ValueError, "minute count"):
            reactions.extract_event(synthetic_window(self.event_time, missing), event_row(self.event_time))

    def test_negative_spread_is_rejected(self) -> None:
        window = synthetic_window(self.event_time)
        window.loc[10, "spread_low"] = -0.00001
        with self.assertRaisesRegex(ValueError, "negative reconstructed spread"):
            reactions.extract_event(window, event_row(self.event_time))

    def test_inconsistent_midpoint_is_rejected(self) -> None:
        window = synthetic_window(self.event_time)
        window.loc[10, "mid_close"] += 0.00001
        with self.assertRaisesRegex(ValueError, "MID open/close"):
            reactions.extract_event(window, event_row(self.event_time))

    def test_reactions_excursions_spreads_and_horizons(self) -> None:
        row, normalized = reactions.extract_event(synthetic_window(self.event_time), event_row(self.event_time))
        self.assertEqual(len(normalized), 5)
        self.assertEqual(row["outcome_bar_timestamp_1m_utc"], self.event_time)
        self.assertEqual(row["outcome_bar_timestamp_5m_utc"], self.event_time + pd.Timedelta(minutes=4))
        self.assertAlmostEqual(row["context_pre_event_spread_pips"], 2.0)
        self.assertAlmostEqual(row["outcome_max_spread_5m_pips"], 2.5)
        self.assertAlmostEqual(row["outcome_max_up_5m_pips"], 0.8)
        self.assertAlmostEqual(row["outcome_max_down_5m_pips"], -0.2)
        self.assertAlmostEqual(row["outcome_range_5m_pips"], 1.0)

    def test_outcome_leakage_gate(self) -> None:
        reactions.assert_no_post_event_outcomes(["context_pre_mid_close", "event_type"])
        with self.assertRaisesRegex(ValueError, "cannot enter"):
            reactions.assert_no_post_event_outcomes(["context_pre_mid_close", "outcome_reaction_1m_pips"])

    def test_untrusted_event_values_are_schema_only_not_dataset_columns(self) -> None:
        wide, _ = reactions._empty_reactions()
        for untrusted in ("actual", "previous", "consensus", "surprise"):
            self.assertNotIn(untrusted, wide.columns)
        schema = reactions.historical_memory_schema()
        future = schema.loc[schema["population_status"].eq("future_stage_not_populated"), "field_or_pattern"]
        self.assertIn("consensus", set(future))

    def test_cached_window_requires_matching_fingerprint_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            window = root / "data" / "dukascopy" / "v04a_reaction" / "windows" / "event.tsv"
            window.parent.mkdir(parents=True)
            window.write_text("fingerprinted", encoding="utf-8")
            status = pd.Series(
                {
                    "reaction_contract_version": contract.REACTION_CONTRACT_VERSION,
                    "source": contract.REACTION_SOURCE_ID,
                    "event_timestamp_utc": self.event_time,
                    "window_start_utc": self.event_time - pd.Timedelta(minutes=10),
                    "window_end_utc": self.event_time + pd.Timedelta(minutes=250),
                    "minute_count": "261",
                    "window_file": "data/dukascopy/v04a_reaction/windows/event.tsv",
                    "data_sha256": contract.sha256_file(window),
                }
            )
            self.assertEqual(reactions.resolve_validated_window(status, event_row(self.event_time), root), window)
            status["data_sha256"] = "00" * 32
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                reactions.resolve_validated_window(status, event_row(self.event_time), root)


if __name__ == "__main__":
    unittest.main()
