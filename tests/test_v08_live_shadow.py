from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.dashboard.v07_api import app
from src.evaluation.v08_contract import HORIZONS
from src.evaluation.v08_ledger import (
    append_prediction, build_prediction_record, ingest_state, recording_eligibility,
)
from src.evaluation.v08_observations import observations_frame, shadow_summary
from src.evaluation.v08_outcomes import attach_matured_outcomes, matured_outcome_rows
from src.learning.v05c_dataset import decision_cost_band, target_class

DECISION = pd.Timestamp("2026-08-24T12:00:00Z")


def runtime_state(*, fresh: bool = True, approved: bool = True, weekend: bool = False) -> dict[str, object]:
    horizons: dict[str, object] = {}
    for horizon in HORIZONS:
        if approved and horizon == 15:
            horizons[str(horizon)] = {
                "model_id": "champion-15", "model_status": "APPROVED_CHAMPION",
                "prob_down": 0.7, "prob_neutral": 0.2, "prob_up": 0.1,
                "shadow_direction": "DOWN", "decision_gate": "PASS_SHADOW_INFERENCE",
            }
        else:
            horizons[str(horizon)] = {
                "model_id": None, "model_status": "NO_APPROVED_MODEL",
                "prob_down": None, "prob_neutral": None, "prob_up": None,
                "shadow_direction": "WAIT", "decision_gate": "WAIT_NO_APPROVED_MODEL",
            }
    freshness = "FRESH" if fresh else "STALE"
    return {
        "contract_version": "v0.6c-unified-runtime-v1",
        "system": {"symbol": "EURUSD", "status": "PASS", "generated_time": {"utc": "2026-08-24T12:01:00Z"}, "registry": {"champion_count": int(approved)}},
        "market": {"close": 1.16, "session": "WEEKEND" if weekend else "LONDON", "freshness": {"status": freshness, "observed_at_utc": DECISION.isoformat(), "age_minutes": 1}, "regime": {"trend_regime": "DOWN", "volatility_regime": "NORMAL"}, "spread": {"current_points": 1.0, "status": "NORMAL"}},
        "predictions": {"horizons": horizons, "fusion": {"status": "PARTIAL", "probabilities": {"down": .7, "neutral": .2, "up": .1}}},
        "decision": {"action": "WAIT", "confidence": "LOW", "gate": "WAIT_EVENT_DATA_INCOMPLETE", "decision_time": {"utc": DECISION.isoformat()}, "next_reassessment": {"utc": "2026-08-24T12:05:00Z"}},
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None},
        "event": {"status": "EVENT_DATA_INCOMPLETE", "minutes_to_event": None},
        "health": {"sources": {"v05b_intelligence": "PASS", "v05d_event": "PASS", "v06a_inference": "PASS"}},
        "reasons": [{"code": "EVENT_DATA_INCOMPLETE", "blocking": True}],
    }


def market_outcome(return_value: float = -0.001, valid: bool = True) -> pd.DataFrame:
    row: dict[str, object] = {"decision_timestamp_utc": DECISION, "m5_close": 1.16, "m5_spread_points": 1.0}
    for horizon in HORIZONS:
        due = DECISION + pd.Timedelta(minutes=horizon)
        row[f"outcome_future_timestamp_{horizon}m"] = due if valid else due + pd.Timedelta(minutes=1)
        row[f"outcome_matured_at_utc_{horizon}m"] = due
        row[f"outcome_future_return_{horizon}m"] = return_value
    return pd.DataFrame([row])


class LiveShadowRecorderTests(unittest.TestCase):
    def test_fresh_approved_prediction_creates_single_durable_decision(self):
        self.assertTrue(recording_eligibility(runtime_state())["eligible"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.parquet"
            result = ingest_state(runtime_state(), "2026-08-24T12:02:00Z", path, Path(temp) / "conflicts", "test")
            self.assertEqual(result["status"], "APPENDED")
            self.assertEqual(len(pd.read_parquet(path)), 1)

    def test_no_approved_model_keeps_probabilities_null(self):
        record = build_prediction_record(runtime_state(approved=False), "2026-08-24T12:02:00Z", "test")
        frame = observations_frame(pd.DataFrame([record]), pd.DataFrame(), "2026-08-24T16:00:00Z")
        self.assertTrue(frame["evaluation_status"].eq("NO_APPROVED_MODEL").all())
        self.assertTrue(frame[["prob_down", "prob_neutral", "prob_up"]].isna().all(axis=None))

    def test_same_decision_and_restart_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / "predictions.parquet"
            first = ingest_state(runtime_state(), "2026-08-24T12:02:00Z", path, root / "conflicts", "test")
            second = ingest_state(runtime_state(), "2026-08-24T13:00:00Z", path, root / "conflicts", "test")
            self.assertEqual((first["status"], second["status"]), ("APPENDED", "DEDUPLICATED"))
            self.assertEqual(len(pd.read_parquet(path)), 1)

    def test_weekend_and_stale_rows_are_not_recording_eligible(self):
        self.assertEqual(recording_eligibility(runtime_state(weekend=True))["reason"], "MARKET_CLOSED_WEEKEND")
        self.assertEqual(recording_eligibility(runtime_state(fresh=False))["status"], "NO_NEW_FRESH_DECISION_EVENT")

    def test_incomplete_features_and_malformed_probabilities_are_rejected(self):
        incomplete = runtime_state(); incomplete["reasons"].append({"code": "INSUFFICIENT_FEATURES", "blocking": True})  # type: ignore[union-attr]
        self.assertEqual(recording_eligibility(incomplete)["reason"], "FEATURE_ROW_INCOMPLETE")
        malformed = runtime_state(); malformed["predictions"]["horizons"]["15"]["prob_up"] = 0.2  # type: ignore[index]
        self.assertEqual(recording_eligibility(malformed)["status"], "STATE_CONTRACT_REJECTED")

    def test_only_approved_horizon_matures(self):
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        outcomes, stats = matured_outcome_rows(prediction, market_outcome(), "2026-08-24T16:00:00Z")
        self.assertEqual(outcomes["horizon_minutes"].tolist(), [15])
        self.assertEqual(stats["not_applicable"], 2)

    def test_target_class_exactly_reuses_v05c_logic(self):
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        market = market_outcome()
        outcomes, _ = matured_outcome_rows(prediction, market, "2026-08-24T12:15:00Z")
        expected = int(target_class(pd.Series([-0.001]), decision_cost_band(market)).iloc[0])
        self.assertEqual(int(outcomes.iloc[0]["target_class"]), expected)

    def test_invalid_target_contract_is_retained_not_scored(self):
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        outcomes, _ = matured_outcome_rows(prediction, market_outcome(valid=False), "2026-08-24T12:15:00Z")
        frame = observations_frame(prediction, outcomes, "2026-08-24T12:15:00Z")
        row = frame.loc[frame["horizon_minutes"].eq(15)].iloc[0]
        self.assertEqual(row["evaluation_status"], "INVALID")
        self.assertIsNone(row["direction_correct"])

    def test_wait_is_separate_from_underlying_direction_correctness(self):
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        outcomes, _ = matured_outcome_rows(prediction, market_outcome(), "2026-08-24T12:15:00Z")
        row = observations_frame(prediction, outcomes, "2026-08-24T12:15:00Z").iloc[0]
        self.assertTrue(row["advisory_wait"])
        self.assertEqual(row["predicted_class"], "DOWN")
        self.assertTrue(row["underlying_direction_correct"])

    def test_pending_survives_restart_and_evaluates_later(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); predictions = root / "predictions.parquet"; market = root / "market.parquet"; outcomes = root / "outcomes.parquet"
            append_prediction(build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test"), predictions, root / "conflicts")
            market_outcome().to_parquet(market, index=False)
            first = attach_matured_outcomes(predictions, market, outcomes, "2026-08-24T12:14:59Z")
            second = attach_matured_outcomes(predictions, market, outcomes, "2026-08-24T12:15:00Z")
            self.assertEqual(first["added"], 0)
            self.assertEqual(second["added"], 1)
            self.assertEqual(len(pd.read_parquet(outcomes)), 1)

    def test_metrics_are_sample_gated(self):
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        outcomes, _ = matured_outcome_rows(prediction, market_outcome(), "2026-08-24T12:15:00Z")
        summary = shadow_summary(observations_frame(prediction, outcomes, "2026-08-24T12:15:00Z"), "2026-08-24T12:15:00Z")
        self.assertEqual(summary["horizons"]["15"]["sample_count"], 1)
        self.assertIsNone(summary["horizons"]["15"]["accuracy"])

    def test_missing_matured_target_remains_pending_data(self):
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        market = market_outcome(); market["outcome_future_return_15m"] = np.nan
        outcomes, _ = matured_outcome_rows(prediction, market, "2026-08-24T12:15:00Z")
        row = observations_frame(prediction, outcomes, "2026-08-24T12:15:00Z").iloc[0]
        self.assertEqual(row["evaluation_status"], "PENDING_DATA")


class LiveShadowApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        prediction = pd.DataFrame([build_prediction_record(runtime_state(), "2026-08-24T12:02:00Z", "test")])
        outcomes, _ = matured_outcome_rows(prediction, market_outcome(), "2026-08-24T12:15:00Z")
        self.frame = observations_frame(prediction, outcomes, "2026-08-24T12:15:00Z")

    def test_summary_recent_detail_and_limits(self):
        with patch("src.dashboard.v07_api.observations_frame", return_value=self.frame), patch("src.dashboard.v07_api.shadow_summary", return_value=shadow_summary(self.frame)):
            self.assertEqual(self.client.get("/api/evaluation/shadow/summary").status_code, 200)
            recent = self.client.get("/api/evaluation/shadow/recent?limit=2")
            self.assertEqual((recent.status_code, recent.json()["count"]), (200, 2))
            identity = str(self.frame.iloc[0]["observation_id"])
            self.assertEqual(self.client.get(f"/api/evaluation/shadow/observations/{identity}").status_code, 200)
            self.assertEqual(self.client.get("/api/evaluation/shadow/recent?limit=201").status_code, 422)


if __name__ == "__main__":
    unittest.main()
