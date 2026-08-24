from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.learning.v05c_contract import MARKET_FEATURES, MODEL_REGISTRY_COLUMNS
from src.learning.v05c_dataset import (
    build_horizon_dataset,
    decision_cost_band,
    feature_group_eligibility,
    validate_feature_registry,
)
from src.learning.v05c_models import (
    baseline_probabilities,
    build_estimator,
    evaluate_probabilities,
    fit_calibrated_model,
)
from src.learning.v05c_registry import (
    append_registry,
    dump_immutable_model,
    file_sha256,
    validate_registry,
    write_immutable_bytes,
)
from src.learning.v05c_runner import promotion_decision
from src.learning.v05c_walk_forward import PurgedFold, purged_walk_forward, validate_fold


def synthetic_source(rows: int = 800) -> pd.DataFrame:
    decision = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    index = np.arange(rows, dtype=float)
    data: dict[str, object] = {
        "decision_timestamp_utc": decision,
        "m5_close": 1.10 + np.sin(index / 30) * 0.001,
        "m1_available_from_utc": decision,
        "m5_available_from_utc": decision,
        "m15_available_from_utc": decision.floor("15min"),
        "h1_available_from_utc": decision.floor("h"),
    }
    for offset, feature in enumerate(MARKET_FEATURES):
        if feature not in data:
            data[feature] = np.sin(index / (offset + 3)) + offset / 100
    data["m5_spread_points"] = np.where((index.astype(int) % 3) == 0, 0, 2)
    for horizon in (15, 60, 240):
        future = pd.Series(decision + pd.Timedelta(minutes=horizon))
        raw = pd.Series(np.sin(index / (7 + horizon / 15)) * 0.0003)
        mature = future <= decision[-1]
        data[f"outcome_future_timestamp_{horizon}m"] = future.where(mature)
        data[f"outcome_matured_at_utc_{horizon}m"] = future.where(mature)
        data[f"outcome_future_return_{horizon}m"] = raw.where(mature)
    return pd.DataFrame(data)


def registry_record(path: Path, digest: str, model_id: str = "model-1") -> dict[str, object]:
    record = {column: "x" for column in MODEL_REGISTRY_COLUMNS}
    record.update({
        "model_id": model_id, "horizon_minutes": 15, "feature_count": len(MARKET_FEATURES),
        "training_rows": 100, "walk_forward_folds": 5, "balanced_accuracy": 0.4,
        "macro_f1": 0.4, "log_loss": 1.0, "brier": 0.6, "coverage": 0.5,
        "cost_aware_metric": 0.0, "artifact_path": str(path), "artifact_sha256": digest,
    })
    return record


class V05CControlledLearningTests(unittest.TestCase):
    def test_outcome_fields_cannot_enter_features(self):
        with self.assertRaises(ValueError):
            validate_feature_registry(["m5_return_5m", "outcome_future_return_15m"])

    def test_future_timestamp_rejected(self):
        frame = synthetic_source()
        frame.loc[10, "m1_available_from_utc"] = frame.loc[10, "decision_timestamp_utc"] + pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "Future feature availability"):
            build_horizon_dataset(frame, 15)

    def test_immature_labels_are_excluded(self):
        frame = synthetic_source()
        dataset = build_horizon_dataset(frame, 240)
        self.assertEqual(len(dataset.frame), len(frame) - 48)
        self.assertTrue(dataset.frame["target_matured_at_utc"].notna().all())

    def test_wrong_horizon_label_timing_is_rejected(self):
        frame = synthetic_source()
        frame.loc[0, "outcome_future_timestamp_15m"] += pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "exact horizon"):
            build_horizon_dataset(frame, 15)

    def test_chronological_split_only(self):
        data = build_horizon_dataset(synthetic_source(1_200), 15)
        folds = purged_walk_forward(data.frame["decision_timestamp_utc"], 15, min_validation_rows=50)
        self.assertTrue(all(fold.train_end < fold.validation_start for fold in folds))

    def test_purge_prevents_overlapping_label_leakage(self):
        data = build_horizon_dataset(synthetic_source(1_200), 240)
        folds = purged_walk_forward(data.frame["decision_timestamp_utc"], 240, min_validation_rows=50)
        for fold in folds:
            self.assertLess(fold.train_end + pd.Timedelta(minutes=240), fold.validation_start)

    def test_fold_with_future_training_row_is_rejected(self):
        times = pd.Series(pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC"))
        fold = PurgedFold(1, np.array([0, 15]), np.array([10, 11]), times.iloc[0], times.iloc[15], times.iloc[10], times.iloc[11], 15)
        with self.assertRaisesRegex(ValueError, "violations"):
            validate_fold(fold, times, 15)

    def test_scaler_fitted_only_on_training(self):
        x = pd.DataFrame({"a": [1.0, 2.0, 100.0], "b": [1.0, np.nan, 3.0]})
        model = build_estimator("LOGISTIC_REGRESSION")
        model.fit(x, pd.Series([0, 1, 2]))
        self.assertAlmostEqual(float(model.named_steps["scaler"].center_[0]), 2.0)

    def test_imputer_fitted_only_on_training(self):
        x = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
        model = build_estimator("HIST_GRADIENT_BOOSTING")
        model.fit(x, pd.Series([0, 1, 2]))
        self.assertAlmostEqual(float(model.named_steps["imputer"].statistics_[0]), 2.0)

    def test_validation_data_cannot_influence_preprocessing(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        first = build_estimator("LOGISTIC_REGRESSION").fit(train, pd.Series([0, 1, 2]))
        validation = pd.DataFrame({"a": [1e12], "b": [-1e12]})
        first.predict(validation)
        second = build_estimator("LOGISTIC_REGRESSION").fit(train, pd.Series([0, 1, 2]))
        np.testing.assert_array_equal(first.named_steps["imputer"].statistics_, second.named_steps["imputer"].statistics_)

    def test_future_news_cannot_enter_training_row(self):
        market = synthetic_source(600)
        future = pd.DataFrame({"decision_timestamp_utc": [market["decision_timestamp_utc"].max() + pd.Timedelta(days=1)]})
        groups = feature_group_eligibility(market, future)
        row = groups.loc[groups["group"].eq("INTELLIGENCE_V05B")].iloc[0]
        self.assertEqual(int(row["rows"]), 0)

    def test_v05b_disabled_when_insufficient_history(self):
        groups = feature_group_eligibility(synthetic_source(600), pd.DataFrame({"decision_timestamp_utc": pd.date_range("2026-01-01", periods=10, tz="UTC")}))
        self.assertFalse(bool(groups.loc[groups["group"].eq("INTELLIGENCE_V05B"), "eligible"].iloc[0]))

    def test_v05b_requires_chronological_validation_even_with_history(self):
        market = synthetic_source(30_000)
        intelligence = pd.DataFrame({"decision_timestamp_utc": market["decision_timestamp_utc"].iloc[:26_000]})
        groups = feature_group_eligibility(market, intelligence, intelligence_validation_passed=False)
        self.assertFalse(bool(groups.loc[groups["group"].eq("INTELLIGENCE_V05B"), "eligible"].iloc[0]))

    def test_no_retrospective_v05b_backfill(self):
        self.assertFalse(any(feature.startswith("news_") or feature.startswith("macro_") for feature in MARKET_FEATURES))

    def test_live_surprise_disabled_with_zero_rows(self):
        groups = feature_group_eligibility(synthetic_source(600), live_surprises=pd.DataFrame())
        self.assertFalse(bool(groups.loc[groups["group"].eq("LIVE_SURPRISE"), "eligible"].iloc[0]))

    def test_deterministic_dataset_fingerprint(self):
        frame = synthetic_source()
        self.assertEqual(build_horizon_dataset(frame, 15).fingerprint, build_horizon_dataset(frame.copy(), 15).fingerprint)

    def test_mutated_data_changes_fingerprint(self):
        frame = synthetic_source()
        original = build_horizon_dataset(frame, 15).fingerprint
        frame.loc[0, "m5_return_5m"] += 0.1
        self.assertNotEqual(original, build_horizon_dataset(frame, 15).fingerprint)

    def test_artifact_sha_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "model.bin"
            digest = write_immutable_bytes(artifact, b"first")
            artifact.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                validate_registry([registry_record(artifact, digest)])

    def test_registered_model_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "model.joblib"
            dump_immutable_model(artifact, {"version": 1})
            with self.assertRaisesRegex(RuntimeError, "silently overwritten"):
                dump_immutable_model(artifact, {"version": 2})

    def test_registry_record_cannot_mutate(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "model.bin"
            digest = write_immutable_bytes(artifact, b"model")
            registry = Path(temp) / "registry.json"
            first = registry_record(artifact, digest)
            append_registry(registry, [first])
            changed = dict(first, macro_f1=0.9)
            with self.assertRaisesRegex(RuntimeError, "mutation rejected"):
                append_registry(registry, [changed])

    def test_baseline_comparison_uses_training_only_priors(self):
        validation = pd.DataFrame({"m5_return_5m": [1.0, -1.0], "m5_return_15m": [1.0, -1.0]})
        first = baseline_probabilities("MAJORITY_CLASS", pd.Series([0, 0, 1]), validation)
        second = baseline_probabilities("MAJORITY_CLASS", pd.Series([2, 2, 1]), validation)
        self.assertFalse(np.array_equal(first, second))

    def test_no_promotion_when_challenger_fails_gate(self):
        candidate, majority, momentum = self._promotion_rows()
        promoted, reason = promotion_decision(candidate, majority, momentum, 30, 5, None)
        self.assertFalse(promoted)
        self.assertIn("history_days", reason)

    def test_promotion_only_when_every_gate_passes(self):
        candidate, majority, momentum = self._promotion_rows()
        promoted, reason = promotion_decision(candidate, majority, momentum, 120, 5, None)
        self.assertTrue(promoted, reason)

    def _promotion_rows(self):
        candidate = pd.Series({
            "balanced_accuracy": 0.50, "macro_f1": 0.50, "log_loss": 0.80,
            "calibration_status": "SIGMOID_TEMPORAL_HOLDOUT", "stability_status": "PASS",
            "cost_aware_metric": 0.001,
        })
        majority = pd.Series({"balanced_accuracy": 0.33, "macro_f1": 0.20, "log_loss": 1.0})
        momentum = pd.Series({"cost_aware_metric": 0.0})
        return candidate, majority, momentum

    def test_calibration_uses_temporal_training_holdout_only(self):
        rows = 1_700
        x = pd.DataFrame({"a": np.sin(np.arange(rows) / 5), "b": np.cos(np.arange(rows) / 7)})
        y = pd.Series(np.arange(rows) % 3)
        times = pd.Series(pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"))
        model = fit_calibrated_model("LOGISTIC_REGRESSION", x, y, times, 15)
        self.assertEqual(model.calibration_status, "SIGMOID_TEMPORAL_HOLDOUT")
        self.assertLess(pd.Timestamp(model.fit_audit["base_train_end"]), pd.Timestamp(model.fit_audit["calibration_start"]))
        self.assertEqual(model.fit_audit["final_evaluation_rows_used_for_fit"], 0)

    def test_feature_ordering_persisted(self):
        rows = 1_700
        x = pd.DataFrame({"b": np.sin(np.arange(rows)), "a": np.cos(np.arange(rows))})
        y = pd.Series(np.arange(rows) % 3)
        times = pd.Series(pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"))
        model = fit_calibrated_model("LOGISTIC_REGRESSION", x, y, times, 15)
        self.assertEqual(model.feature_names, ("b", "a"))

    def test_target_cost_band_uses_decision_spread(self):
        frame = synthetic_source(100)
        band = decision_cost_band(frame)
        self.assertTrue((band > 0).all())
        changed = frame.copy()
        changed["m5_spread_points"] = 10
        self.assertTrue((decision_cost_band(changed) > band).all())

    def test_missing_commission_remains_unknown(self):
        probability = np.tile([0.2, 0.3, 0.5], (3, 1))
        metrics = evaluate_probabilities([0, 1, 2], probability, [0.1, 0.0, 0.2], [0.01, 0.01, 0.01])
        self.assertEqual(metrics["commission_status"], "UNKNOWN")

    def test_model_registry_schema_validation(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            validate_registry([{"model_id": "incomplete"}], verify_artifacts=False)

    def test_horizons_remain_independent(self):
        frame = synthetic_source()
        fifteen = build_horizon_dataset(frame, 15)
        sixty = build_horizon_dataset(frame, 60)
        self.assertNotEqual(fifteen.fingerprint, sixty.fingerprint)
        self.assertNotEqual(len(fifteen.frame), len(sixty.frame))

    def test_research_modules_expose_no_execution_methods(self):
        forbidden = ("order" + "_send", "submit" + "Order", "C" + "Trade")
        learning_root = Path(__file__).resolve().parents[1] / "src" / "learning"
        text = "\n".join(path.read_text(encoding="utf-8") for path in learning_root.glob("*.py"))
        self.assertFalse(any(token in text for token in forbidden))


if __name__ == "__main__":
    unittest.main()
