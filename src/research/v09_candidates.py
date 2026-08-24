"""Deterministic V0.9 candidate families, abstention policies, and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline

from src.learning.v05c_contract import CLASS_LABELS, COST_BAND_MULTIPLIER, MARKET_FEATURES, RANDOM_STATE
from src.learning.v05c_dataset import target_class
from src.learning.v05c_models import baseline_probabilities, build_estimator, multiclass_brier
from src.research.v08_abstention import elastic_net_estimator, two_stage_probabilities
from src.research.v08_feature_ablation import FEATURE_GROUPS, ablated_features


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    features: tuple[str, ...]
    target_multiplier: float = COST_BAND_MULTIPLIER


@dataclass(frozen=True)
class AbstentionPolicy:
    top_probability: float
    directional_margin: float

    @property
    def policy_id(self) -> str:
        return f"P{self.top_probability:.2f}_M{self.directional_margin:.2f}"


def candidate_specs() -> tuple[CandidateSpec, ...]:
    no_volume = ablated_features("VOLUME")
    specs = [
        CandidateSpec("H60_ELASTIC_ALL", "ELASTIC_NET_LOGISTIC", tuple(MARKET_FEATURES)),
        CandidateSpec("H60_ELASTIC_NO_VOLUME", "ELASTIC_NET_LOGISTIC", no_volume),
        CandidateSpec("H60_LOGISTIC_NO_VOLUME", "LOGISTIC_REGRESSION", no_volume),
        CandidateSpec("H60_HISTGB_NO_VOLUME", "HIST_GRADIENT_BOOSTING", no_volume),
        CandidateSpec("H60_RF_NO_VOLUME", "RANDOM_FOREST_BASELINE", no_volume),
        CandidateSpec("H60_TWO_STAGE_NO_VOLUME", "TWO_STAGE_LOGISTIC", no_volume),
    ]
    for group in ("SESSION_TIME", "SPREAD_LIQUIDITY", "VOLATILITY", "PRICE_RETURN"):
        specs.append(CandidateSpec(f"H60_ELASTIC_NO_{group}", "ELASTIC_NET_LOGISTIC", ablated_features(group)))
    return tuple(specs)


def labels(frame: pd.DataFrame, multiplier: float) -> pd.Series:
    adjusted = pd.to_numeric(frame["decision_cost_band"], errors="raise") * float(multiplier) / COST_BAND_MULTIPLIER
    return target_class(pd.to_numeric(frame["raw_future_return"], errors="raise"), adjusted).astype(int)


def _aligned(model: Any, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame), dtype=float)
    result = np.full((len(frame), 3), 1e-12, dtype=float)
    for index, value in enumerate(model.classes_):
        result[:, CLASS_LABELS.index(int(value))] = raw[:, index]
    result /= result.sum(axis=1, keepdims=True)
    return result


def _estimator(family: str) -> Any:
    if family == "ELASTIC_NET_LOGISTIC":
        return elastic_net_estimator()
    if family in {"LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING"}:
        return build_estimator(family)
    if family == "RANDOM_FOREST_BASELINE":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=120, max_depth=8, min_samples_leaf=10,
                class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unsupported V0.9 family: {family}")


def fit_predict(spec: CandidateSpec, train: pd.DataFrame, evaluation: pd.DataFrame, multiplier: float | None = None) -> np.ndarray:
    multiplier = spec.target_multiplier if multiplier is None else float(multiplier)
    train_y = labels(train, multiplier)
    x_train = train.loc[:, list(spec.features)]
    x_eval = evaluation.loc[:, list(spec.features)]
    if spec.family == "TWO_STAGE_LOGISTIC":
        return two_stage_probabilities(x_train, train_y, x_eval)
    model = _estimator(spec.family)
    model.fit(x_train, train_y)
    return _aligned(model, x_eval)


def temporal_calibrated_predict(
    spec: CandidateSpec,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    horizon_minutes: int,
    multiplier: float,
) -> tuple[np.ndarray, str]:
    """Sigmoid calibration fitted only on the final 20% of historical training data."""
    ordered = train.sort_values("decision_timestamp_utc").reset_index(drop=True)
    if len(ordered) < 1500:
        return fit_predict(spec, ordered, evaluation, multiplier), "INSUFFICIENT_SAMPLE_UNCALIBRATED"
    calibration_start_index = int(len(ordered) * 0.80)
    calibration = ordered.iloc[calibration_start_index:].copy()
    calibration_start = pd.Timestamp(calibration["decision_timestamp_utc"].min())
    base_pool = ordered.iloc[:calibration_start_index].copy()
    base = base_pool.loc[pd.to_datetime(base_pool["target_matured_at_utc"], utc=True).lt(calibration_start)].copy()
    base_y = labels(base, multiplier)
    calibration_y = labels(calibration, multiplier)
    if len(base) < 1000 or len(calibration) < 250 or base_y.nunique() < 3 or calibration_y.nunique() < 3:
        return fit_predict(spec, ordered, evaluation, multiplier), "INSUFFICIENT_SAMPLE_UNCALIBRATED"
    base_prob = fit_predict(spec, base, calibration, multiplier)
    calibrator = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    calibrator.fit(np.log(np.clip(base_prob, 1e-12, 1.0)), calibration_y)
    eval_prob = fit_predict(spec, base, evaluation, multiplier)
    calibrated = np.asarray(calibrator.predict_proba(np.log(np.clip(eval_prob, 1e-12, 1.0))), dtype=float)
    result = np.full((len(evaluation), 3), 1e-12, dtype=float)
    for index, value in enumerate(calibrator.classes_):
        result[:, int(value)] = calibrated[:, index]
    result /= result.sum(axis=1, keepdims=True)
    return result, "SIGMOID_TEMPORAL_HOLDOUT"


def decisions_from_probabilities(probabilities: np.ndarray, policy: AbstentionPolicy | None = None) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not np.isfinite(probabilities).all():
        raise ValueError("Malformed probability matrix")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to one")
    decisions = probabilities.argmax(axis=1).astype(int)
    if policy is None:
        return decisions
    ordered = np.sort(probabilities, axis=1)
    top = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    abstain = (top < policy.top_probability) | (margin < policy.directional_margin)
    decisions[abstain] = 1
    return decisions


def metrics(
    actual: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    policy: AbstentionPolicy | None = None,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=int)
    predicted = decisions_from_probabilities(probabilities, policy)
    acted = predicted != 1
    direction = predicted - 1
    raw = pd.to_numeric(frame["raw_future_return"], errors="raise").to_numpy(dtype=float)
    cost = pd.to_numeric(frame["decision_cost_band"], errors="raise").to_numpy(dtype=float)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual_array, predicted)),
        "macro_f1": float(f1_score(actual_array, predicted, average="macro", zero_division=0)),
        "brier": float(multiclass_brier(actual_array, probabilities)),
        "log_loss": float(log_loss(actual_array, probabilities, labels=[0, 1, 2])),
        "coverage": float(acted.mean()),
        "wait_rate": float((~acted).mean()),
        "directional_accuracy": float((predicted[acted] == actual_array[acted]).mean()) if acted.any() else float("nan"),
        "cost_aware_metric": float(np.mean(direction[acted] * raw[acted] - cost[acted])) if acted.any() else float("nan"),
    }


def baseline_metrics(train: pd.DataFrame, validation: pd.DataFrame, multiplier: float) -> dict[str, dict[str, float]]:
    train_y = labels(train, multiplier)
    validation_y = labels(validation, multiplier)
    result: dict[str, dict[str, float]] = {}
    for name in ("MAJORITY_CLASS", "MOMENTUM", "MEAN_REVERSION", "RECENT_DIRECTION"):
        probability = baseline_probabilities(name, train_y, validation)
        result[name] = metrics(validation_y, probability, validation)
    return result


def feature_group_membership() -> dict[str, tuple[str, ...]]:
    return {key: tuple(value) for key, value in FEATURE_GROUPS.items()}
