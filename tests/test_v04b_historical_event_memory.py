from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.memory import analogue_retrieval as retrieval
from src.memory import build_historical_event_memory as builder
from src.memory.retrieval_feature_registry import RetrievalFeature, validate_feature
from src.memory import v04b_contract


def feature(name: str, required: bool = True) -> RetrievalFeature:
    return RetrievalFeature(
        name=name,
        data_type="float64",
        category="test",
        availability_rule="known by event_timestamp_utc",
        normalization_method="candidate_history_robust_scale",
        weight=1.0,
        required=required,
        source="synthetic unit test",
        point_in_time_safe=True,
    )


TEST_FEATURES = (feature("context_x"), feature("context_optional", required=False))


def synthetic_memory() -> pd.DataFrame:
    rows = []
    for index in range(8):
        timestamp = pd.Timestamp("2020-01-01T13:30:00Z") + pd.DateOffset(months=index)
        row = {
            "event_id": f"cpi-{index}",
            "event_type": "us_cpi_release" if index != 3 else "us_employment_situation",
            "event_timestamp_utc": timestamp,
        }
        for registered in retrieval.FEATURES:
            row[registered.name] = float(index) if registered.name == "context_return_5m_pips" else 0.0
            if not registered.required:
                row[registered.name] = np.nan if index in (1, 6) else float(index % 2)
        for horizon in retrieval.HORIZONS:
            row[f"outcome_reaction_{horizon}_pips"] = float(index - 2)
            row[f"outcome_direction_{horizon}"] = "UP" if index > 2 else "DOWN"
        rows.append(row)
    return pd.DataFrame(rows)


def eligible_rows(count: int = 242) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(count)],
            "production_status": ["PASS"] * count,
            "adjudication_class": ["STRICT_PASS"] * count,
            "reaction_extraction_status": ["COMPLETE"] * count,
            "model_eligible_market_reaction": [True] * count,
        }
    )


class MemoryEligibilityTests(unittest.TestCase):
    def test_changed_reaction_source_hard_fails_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "v04a_event_reactions.parquet"
            changed.write_bytes(b"changed source")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                v04b_contract.verify_reaction_dataset(changed)

    def test_memory_contains_exactly_strict_eligible_events(self) -> None:
        builder.validate_eligible_identity(eligible_rows())
        with self.assertRaisesRegex(RuntimeError, "exactly 242"):
            builder.validate_eligible_identity(eligible_rows(241))

    def test_quarantined_events_cannot_enter_memory(self) -> None:
        frame = eligible_rows()
        frame.loc[17, "adjudication_class"] = "PROVIDER_HISTORY_INCOMPLETE"
        with self.assertRaisesRegex(RuntimeError, "Quarantined"):
            builder.validate_eligible_identity(frame)

    def test_outcome_columns_cannot_enter_registry(self) -> None:
        unsafe = feature("context_safe")
        validate_feature(unsafe)
        with self.assertRaisesRegex(ValueError, "explicit context"):
            validate_feature(
                RetrievalFeature(
                    name="outcome_reaction_1m_pips", data_type="float64", category="outcome",
                    availability_rule="known at event_timestamp_utc", normalization_method="candidate_history_robust_scale",
                    weight=1.0, required=False, source="future", point_in_time_safe=True,
                )
            )

    def test_context_availability_must_be_at_or_before_event(self) -> None:
        timestamp = pd.Timestamp("2024-01-01T13:30:00Z")
        frame = pd.DataFrame(
            {
                "event_timestamp_utc": [timestamp],
                "context_price_available_through_utc": [timestamp - pd.Timedelta(milliseconds=1)],
                "context_macro_x__available_from_utc": [timestamp],
                "context_macro_x__vintage_date": [timestamp - pd.Timedelta(days=1)],
            }
        )
        builder.validate_context_availability(frame)
        frame.loc[0, "context_macro_x__available_from_utc"] = timestamp + pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(RuntimeError, "availability after"):
            builder.validate_context_availability(frame)

    def test_macro_vintage_must_not_be_after_event(self) -> None:
        timestamp = pd.Timestamp("2024-01-01T13:30:00Z")
        frame = pd.DataFrame(
            {
                "event_timestamp_utc": [timestamp],
                "context_price_available_through_utc": [timestamp - pd.Timedelta(minutes=1)],
                "context_macro_x__available_from_utc": [timestamp],
                "context_macro_x__vintage_date": [timestamp + pd.Timedelta(days=1)],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "Macro vintage"):
            builder.validate_context_availability(frame)

    def test_volatility_regime_thresholds_use_only_prior_same_type_events(self) -> None:
        frame = pd.DataFrame(
            {
                "event_type": ["us_cpi_release"] * 12,
                "event_timestamp_utc": pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC"),
                "context_realized_vol_10m": list(range(1, 13)),
            }
        )
        regimes = builder.add_causal_volatility_regime(frame, minimum_history=10)
        self.assertTrue(regimes.iloc[:10]["context_volatility_regime_code"].isna().all())
        self.assertAlmostEqual(regimes.iloc[10]["context_volatility_regime_upper"], np.quantile(range(1, 11), 2 / 3))
        self.assertEqual(regimes.iloc[10]["context_volatility_regime"], "HIGH")


class CausalRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = synthetic_memory()
        self.query = self.memory.iloc[-1].copy()

    def retrieve(self, memory: pd.DataFrame | None = None, query: pd.Series | None = None):
        return retrieval.retrieve_analogues(
            self.memory if memory is None else memory,
            self.query if query is None else query,
            k=3,
            same_event_type=True,
        )

    def test_query_cannot_retrieve_itself(self) -> None:
        response = self.retrieve()
        self.assertNotIn(self.query["event_id"], [item["event_id"] for item in response["analogues"]])

    def test_future_events_cannot_be_candidates(self) -> None:
        query = self.memory.iloc[4].copy()
        response = self.retrieve(query=query)
        self.assertTrue(all(pd.Timestamp(item["timestamp"]) < query["event_timestamp_utc"] for item in response["analogues"]))

    def test_same_event_type_filter(self) -> None:
        candidates = retrieval.eligible_candidates(self.memory, self.query, same_event_type=True)
        self.assertTrue(candidates["event_type"].eq(self.query["event_type"]).all())
        all_types = retrieval.eligible_candidates(self.memory, self.query, same_event_type=False)
        self.assertIn("us_employment_situation", set(all_types["event_type"]))

    def test_normalization_uses_candidate_history_only(self) -> None:
        candidates = pd.DataFrame({"context_x": [0.0, 1.0, 2.0], "context_optional": [0.0, 1.0, np.nan]})
        scales = retrieval.candidate_history_scales(candidates, TEST_FEATURES)
        self.assertEqual(scales["context_x"].center, 1.0)
        self.assertEqual(scales["context_x"].denominator, 1.0)

    def test_future_extreme_event_does_not_change_ranking(self) -> None:
        before = [item["event_id"] for item in self.retrieve()["analogues"]]
        future = self.query.copy()
        future["event_id"] = "future-extreme"
        future["event_timestamp_utc"] += pd.Timedelta(days=30)
        future["context_return_5m_pips"] = 1e12
        extended = pd.concat([self.memory, future.to_frame().T], ignore_index=True)
        after = [item["event_id"] for item in self.retrieve(memory=extended)["analogues"]]
        self.assertEqual(before, after)

    def test_ranking_is_deterministic(self) -> None:
        first = [(item["event_id"], item["distance"]) for item in self.retrieve()["analogues"]]
        second = [(item["event_id"], item["distance"]) for item in self.retrieve()["analogues"]]
        self.assertEqual(first, second)

    def test_missing_optional_context_is_supported(self) -> None:
        query = self.query.copy()
        query["context_volatility_regime_code"] = np.nan
        response = self.retrieve(query=query)
        self.assertEqual(len(response["analogues"]), 3)

    def test_missing_required_context_fails(self) -> None:
        query = self.query.copy()
        query["context_return_5m_pips"] = np.nan
        with self.assertRaisesRegex(ValueError, "Required query"):
            self.retrieve(query=query)

    def test_distances_are_finite(self) -> None:
        response = self.retrieve()
        self.assertTrue(all(np.isfinite(item["distance"]) for item in response["analogues"]))

    def test_historical_outcomes_are_attached_after_ranking(self) -> None:
        response = self.retrieve()
        first = response["analogues"][0]
        source = self.memory.set_index("event_id").loc[first["event_id"]]
        self.assertEqual(first["historical_outcomes"]["reaction_1m_pips"], source["outcome_reaction_1m_pips"])
        self.assertFalse(any(key.startswith("outcome_") for key in first["matched_context"]))

    def test_retrieval_engine_rejects_rows_containing_outcomes(self) -> None:
        candidates = self.memory.iloc[:3]
        with self.assertRaisesRegex(ValueError, "engine boundary"):
            retrieval._context_only_ranking(self.query, candidates)
        isolated = retrieval.retrieval_input_view(self.memory)
        self.assertFalse(any(column.startswith("outcome_") for column in isolated))

    def test_changing_query_outcome_does_not_change_ranking(self) -> None:
        baseline = [(item["event_id"], item["distance"]) for item in self.retrieve()["analogues"]]
        mutated = copy.deepcopy(self.query)
        for column in mutated.index:
            if column.startswith("outcome_"):
                mutated[column] = 999999.0 if "reaction" in column else "MUTATED"
        changed = [(item["event_id"], item["distance"]) for item in self.retrieve(query=mutated)["analogues"]]
        self.assertEqual(baseline, changed)


if __name__ == "__main__":
    unittest.main()
