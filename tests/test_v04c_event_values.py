from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.events import build_v04c_event_values as values
from src.events.decision_mode_registry import (
    DecisionMode, assert_mode_allows, project_mode, validate_decision_time,
)
from src.events.event_value_source_registry import source_for
from src.events.v04c_contract import verify_v04b_memory


EVENT_TIME = pd.Timestamp("2024-02-13T13:30:00Z")


def valid_value_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "event_id": "bls:us_cpi_release:20240213T1330Z",
                "event_timestamp_utc": EVENT_TIME,
                "event_type": "us_cpi_release",
                "measure_id": "headline_cpi_yoy_sa",
                "measure_name": "Headline CPI year-over-year, seasonally adjusted",
                "unit": "percent_change_year_ago",
                "raw_source_unit": "percent_change_year_ago",
                "reference_period": pd.Timestamp("2024-01-01T00:00:00Z"),
                "actual_value": 3.1,
                "raw_source_value": 3.1,
                "previous_value_pre_release": np.nan,
                "previous_value_revised_at_release": np.nan,
                "consensus_value": np.nan,
                "actual_available_from_utc": EVENT_TIME,
                "previous_available_from_utc": pd.NaT,
                "revision_available_from_utc": pd.NaT,
                "prior_period_newly_released_at_release": np.nan,
                "prior_period_new_release_available_from_utc": pd.NaT,
                "consensus_available_from_utc": pd.NaT,
                "source_vintage": pd.Timestamp("2024-02-13T00:00:00Z"),
                "point_in_time_verified": True,
                "event_revision_raw": np.nan,
                "event_revision_abs": np.nan,
                "event_revision_direction": None,
            }
        ]
    )
    for column in (
        "event_timestamp_utc", "reference_period", "actual_available_from_utc",
        "previous_available_from_utc", "revision_available_from_utc",
        "prior_period_new_release_available_from_utc", "consensus_available_from_utc", "source_vintage",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return values.add_causal_surprise_normalization(values.calculate_surprise_fields(frame))


def raw_surprise_history(count: int, future_offset: int = 0) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "event_id": f"event-{future_offset + index}",
                "event_timestamp_utc": pd.Timestamp("2020-01-01T13:30:00Z") + pd.DateOffset(months=future_offset + index),
                "measure_id": "nonfarm_payroll_change",
                "actual_value": float(future_offset + index),
                "consensus_value": 0.0,
            }
        )
    return values.calculate_surprise_fields(pd.DataFrame(rows))


class EventValueContractTests(unittest.TestCase):
    def test_v04b_fingerprint_hard_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "v04b_historical_event_memory.parquet"
            changed.write_bytes(b"changed memory")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 changed"):
                verify_v04b_memory(changed)

    def test_unique_normalized_measure_identity(self) -> None:
        duplicated = pd.concat([valid_value_frame(), valid_value_frame()], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, "identity is not unique"):
            values.validate_event_values(duplicated, expected_rows=None)

    def test_official_event_mapping(self) -> None:
        source = source_for("us_cpi_release", "headline_cpi_yoy_sa")
        self.assertEqual(source.authority, "U.S. Bureau of Labor Statistics")
        frame = valid_value_frame()
        frame.loc[0, "measure_id"] = "invented_measure"
        with self.assertRaisesRegex(RuntimeError, "Official event mapping absent"):
            values.validate_event_values(frame, expected_rows=None)

    def test_actual_availability_at_t(self) -> None:
        frame = valid_value_frame()
        values.validate_event_values(frame, expected_rows=None)
        frame.loc[0, "actual_available_from_utc"] = EVENT_TIME + pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(RuntimeError, "availability must equal"):
            values.validate_event_values(frame, expected_rows=None)

    def test_previous_value_must_be_available_before_t(self) -> None:
        frame = valid_value_frame()
        frame.loc[0, "previous_value_pre_release"] = 3.0
        frame.loc[0, "previous_available_from_utc"] = EVENT_TIME
        with self.assertRaisesRegex(RuntimeError, "strictly before"):
            values.validate_event_values(frame, expected_rows=None)

    def test_later_revision_is_rejected(self) -> None:
        frame = valid_value_frame()
        frame.loc[0, "previous_value_pre_release"] = 3.0
        frame.loc[0, "previous_available_from_utc"] = EVENT_TIME - pd.Timedelta(days=30)
        frame.loc[0, "previous_value_revised_at_release"] = 2.9
        frame.loc[0, "event_revision_raw"] = -0.1
        frame.loc[0, "revision_available_from_utc"] = EVENT_TIME + pd.Timedelta(days=1)
        with self.assertRaisesRegex(RuntimeError, "Later or mistimed"):
            values.validate_event_values(frame, expected_rows=None)

    def test_release_time_revision_is_preserved_separately(self) -> None:
        frame = valid_value_frame()
        frame.loc[0, "previous_value_pre_release"] = 3.0
        frame.loc[0, "previous_available_from_utc"] = EVENT_TIME - pd.Timedelta(days=30)
        frame.loc[0, "previous_value_revised_at_release"] = 2.9
        frame.loc[0, "event_revision_raw"] = -0.1
        frame.loc[0, "event_revision_abs"] = 0.1
        frame.loc[0, "event_revision_direction"] = "DOWN"
        frame.loc[0, "revision_available_from_utc"] = EVENT_TIME
        values.validate_event_values(frame, expected_rows=None)
        self.assertEqual(frame.loc[0, "previous_value_pre_release"], 3.0)
        self.assertEqual(frame.loc[0, "previous_value_revised_at_release"], 2.9)

    def test_prior_period_first_published_at_t_is_not_a_revision(self) -> None:
        frame = valid_value_frame()
        frame.loc[0, "prior_period_newly_released_at_release"] = 3.0
        frame.loc[0, "prior_period_new_release_available_from_utc"] = EVENT_TIME
        values.validate_event_values(frame, expected_rows=None)
        self.assertTrue(pd.isna(frame.loc[0, "previous_value_pre_release"]))
        self.assertTrue(pd.isna(frame.loc[0, "previous_value_revised_at_release"]))

    def test_consensus_after_t_is_rejected(self) -> None:
        frame = valid_value_frame()
        frame.loc[0, "consensus_value"] = 3.0
        frame.loc[0, "consensus_available_from_utc"] = EVENT_TIME
        frame = values.add_causal_surprise_normalization(values.calculate_surprise_fields(frame.drop(columns=[
            "surprise_signed_raw", "surprise_abs_raw", "surprise_status",
            "surprise_strength_direction", "surprise_inflation_direction",
            "surprise_z_robust", "normalization_history_count", "normalization_method",
            "normalization_timestamp_cutoff", "surprise_bucket_causal",
        ])))
        with self.assertRaisesRegex(RuntimeError, "strictly before"):
            values.validate_event_values(frame, expected_rows=None)

    def test_missing_consensus_produces_null_surprise(self) -> None:
        frame = valid_value_frame()
        self.assertTrue(pd.isna(frame.loc[0, "surprise_signed_raw"]))
        self.assertEqual(frame.loc[0, "surprise_status"], "CONSENSUS_UNAVAILABLE")

    def test_raw_surprise_arithmetic(self) -> None:
        frame = pd.DataFrame([{"measure_id": "nonfarm_payroll_change", "actual_value": 225.0, "consensus_value": 190.0}])
        result = values.calculate_surprise_fields(frame)
        self.assertEqual(result.loc[0, "surprise_signed_raw"], 35.0)

    def test_unit_preservation(self) -> None:
        frame = valid_value_frame()
        frame.loc[0, "unit"] = "thousands_of_jobs_change"
        with self.assertRaisesRegex(RuntimeError, "Unit mismatch"):
            values.validate_event_values(frame, expected_rows=None)

    def test_unemployment_direction_is_separate_from_raw_arithmetic(self) -> None:
        frame = pd.DataFrame([{"measure_id": "unemployment_rate", "actual_value": 4.2, "consensus_value": 4.0}])
        result = values.calculate_surprise_fields(frame)
        self.assertAlmostEqual(result.loc[0, "surprise_signed_raw"], 0.2)
        self.assertEqual(result.loc[0, "surprise_strength_direction"], "WEAKER_LABOR")


class CausalSurpriseTests(unittest.TestCase):
    def test_prior_only_surprise_normalization(self) -> None:
        result = values.add_causal_surprise_normalization(raw_surprise_history(11))
        row = result.iloc[10]
        expected = (10.0 - np.median(np.arange(10))) / np.subtract(*np.quantile(np.arange(10), [0.75, 0.25]))
        self.assertAlmostEqual(row["surprise_z_robust"], expected)

    def test_query_event_excluded_from_normalization_history(self) -> None:
        result = values.add_causal_surprise_normalization(raw_surprise_history(11))
        self.assertEqual(result.iloc[10]["normalization_history_count"], 10)
        self.assertEqual(result.iloc[10]["normalization_timestamp_cutoff"], result.iloc[9]["event_timestamp_utc"])

    def test_future_event_excluded_from_normalization_history(self) -> None:
        base = values.add_causal_surprise_normalization(raw_surprise_history(11))
        extended_raw = pd.concat(
            [raw_surprise_history(11), raw_surprise_history(1, future_offset=50).assign(actual_value=1e9)],
            ignore_index=True,
        )
        extended = values.add_causal_surprise_normalization(values.calculate_surprise_fields(
            extended_raw.drop(columns=["surprise_signed_raw", "surprise_abs_raw", "surprise_status", "surprise_strength_direction", "surprise_inflation_direction"])
        ))
        self.assertAlmostEqual(base.iloc[10]["surprise_z_robust"], extended[extended.event_id.eq("event-10")].iloc[0]["surprise_z_robust"])

    def test_changing_future_event_values_does_not_alter_past_rows(self) -> None:
        raw = raw_surprise_history(12)
        first = values.add_causal_surprise_normalization(raw)
        raw.loc[11, "actual_value"] = 999999.0
        raw = values.calculate_surprise_fields(raw.drop(columns=["surprise_signed_raw", "surprise_abs_raw", "surprise_status", "surprise_strength_direction", "surprise_inflation_direction"]))
        second = values.add_causal_surprise_normalization(raw)
        pd.testing.assert_series_equal(first.iloc[:11]["surprise_z_robust"], second.iloc[:11]["surprise_z_robust"])

    def test_changing_future_reaction_does_not_alter_surprise(self) -> None:
        frame = pd.DataFrame([{
            "measure_id": "nonfarm_payroll_change", "actual_value": 225.0,
            "consensus_value": 190.0, "outcome_reaction_5m_pips": 10.0,
        }])
        first = values.calculate_surprise_fields(frame).loc[0, "surprise_signed_raw"]
        frame.loc[0, "outcome_reaction_5m_pips"] = -9999.0
        second = values.calculate_surprise_fields(frame).loc[0, "surprise_signed_raw"]
        self.assertEqual(first, second)


class DecisionModeTests(unittest.TestCase):
    def test_pre_release_rejects_actual(self) -> None:
        with self.assertRaises(ValueError):
            assert_mode_allows(["release_payload_event_payroll_actual_value"], DecisionMode.PRE_RELEASE)

    def test_pre_release_rejects_surprise(self) -> None:
        with self.assertRaises(ValueError):
            assert_mode_allows(["release_payload_event_payroll_surprise_signed_raw"], DecisionMode.PRE_RELEASE)

    def test_post_release_permits_audited_release_payload(self) -> None:
        assert_mode_allows(["release_payload_event_payroll_actual_value"], DecisionMode.POST_RELEASE_IMMEDIATE)
        validate_decision_time(DecisionMode.POST_RELEASE_IMMEDIATE, EVENT_TIME, EVENT_TIME)

    def test_decision_timestamps_enforce_mode_boundary(self) -> None:
        validate_decision_time(DecisionMode.PRE_RELEASE, EVENT_TIME - pd.Timedelta(microseconds=1), EVENT_TIME)
        with self.assertRaises(ValueError):
            validate_decision_time(DecisionMode.PRE_RELEASE, EVENT_TIME, EVENT_TIME)
        with self.assertRaises(ValueError):
            validate_decision_time(
                DecisionMode.POST_RELEASE_IMMEDIATE,
                EVENT_TIME - pd.Timedelta(microseconds=1), EVENT_TIME,
            )

    def test_both_modes_reject_future_price_outcomes(self) -> None:
        for mode in DecisionMode:
            with self.assertRaisesRegex(ValueError, "forbidden"):
                assert_mode_allows(["outcome_reaction_5m_pips"], mode)

    def test_mode_projection_never_contains_outcomes(self) -> None:
        frame = pd.DataFrame([{
            "event_id": "e", "event_type": "t", "event_timestamp_utc": EVENT_TIME,
            "context_x": 1.0, "pre_release_context_x": 2.0,
            "release_payload_x": 3.0, "outcome_reaction_1m_pips": 4.0,
        }])
        for mode in DecisionMode:
            projected = project_mode(frame, mode)
            self.assertFalse(any(column.startswith("outcome_") for column in projected))

    def test_no_outcomes_enter_release_value_construction(self) -> None:
        frame = valid_value_frame()
        frame["outcome_reaction_1m_pips"] = 10.0
        with self.assertRaisesRegex(RuntimeError, "outcome entered"):
            values.validate_event_values(frame, expected_rows=None)


if __name__ == "__main__":
    unittest.main()
