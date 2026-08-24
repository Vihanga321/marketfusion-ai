"""Auditable V0.5C model families, calibration, baselines, and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.learning.v05c_contract import (
    CLASS_LABELS,
    MAX_RECENT_FOLD_DEGRADATION,
    MAX_WORST_FOLD_DEFICIT,
    MIN_BASE_TRAINING_ROWS,
    MIN_CALIBRATION_ROWS,
    RANDOM_STATE,
)


def available_families() -> tuple[list[str], list[str]]:
    families = ["LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING"]
    unavailable: list[str] = []
    try:
        import xgboost  # noqa: F401
        families.append("XGBOOST")
    except (ImportError, OSError) as exc:
        unavailable.append(f"XGBOOST: {type(exc).__name__}: unavailable/incompatible")
    return families, unavailable


def build_estimator(family: str) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if family == "LOGISTIC_REGRESSION":
        return Pipeline([
            ("imputer", imputer),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(max_iter=3_000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    if family == "HIST_GRADIENT_BOOSTING":
        return Pipeline([
            ("imputer", imputer),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=0.1, random_state=RANDOM_STATE,
            )),
        ])
    if family == "XGBOOST":
        from xgboost import XGBClassifier
        return Pipeline([
            ("imputer", imputer),
            ("classifier", XGBClassifier(
                objective="multi:softprob", num_class=3, n_estimators=60,
                max_depth=3, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.9, eval_metric="mlogloss", n_jobs=1,
                random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unsupported candidate family: {family}")


def _aligned_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    aligned = np.full((len(x), len(CLASS_LABELS)), 1e-12, dtype=float)
    for source_index, label in enumerate(model.classes_):
        aligned[:, CLASS_LABELS.index(int(label))] = raw[:, source_index]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


@dataclass
class CalibratedModel:
    base_model: Any
    calibrator: Any | None
    feature_names: tuple[str, ...]
    calibration_status: str
    fit_audit: dict[str, object]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        ordered = x.loc[:, list(self.feature_names)]
        base_probability = _aligned_probabilities(self.base_model, ordered)
        if self.calibrator is None:
            return base_probability
        log_probability = np.log(np.clip(base_probability, 1e-12, 1.0))
        return _aligned_probabilities(self.calibrator, pd.DataFrame(log_probability))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(CLASS_LABELS)[self.predict_proba(x).argmax(axis=1)]


def fit_calibrated_model(
    family: str,
    x: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    horizon_minutes: int,
) -> CalibratedModel:
    times = pd.to_datetime(timestamps, utc=True, errors="raise").reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)
    calibration_rows = max(MIN_CALIBRATION_ROWS, int(len(x) * 0.20))
    calibration_start_index = len(x) - calibration_rows
    status = "INSUFFICIENT_SAMPLE"
    audit: dict[str, object] = {
        "base_train_rows": len(x), "calibration_rows": 0,
        "base_train_end": times.iloc[-1], "calibration_start": pd.NaT,
        "final_evaluation_rows_used_for_fit": 0,
    }
    if calibration_start_index > 0:
        calibration_start = times.iloc[calibration_start_index]
        base_mask = (times + pd.Timedelta(minutes=horizon_minutes)) < calibration_start
        base_indices = np.flatnonzero(base_mask.to_numpy())
        calibration_indices = np.arange(calibration_start_index, len(x))
        enough = (
            len(base_indices) >= MIN_BASE_TRAINING_ROWS
            and len(calibration_indices) >= MIN_CALIBRATION_ROWS
            and y.iloc[base_indices].nunique() == 3
            and y.iloc[calibration_indices].nunique() == 3
        )
        if enough:
            base = build_estimator(family)
            base.fit(x.iloc[base_indices], y.iloc[base_indices])
            calibration_probability = _aligned_probabilities(base, x.iloc[calibration_indices])
            calibrator = LogisticRegression(max_iter=3_000, random_state=RANDOM_STATE)
            calibrator.fit(np.log(np.clip(calibration_probability, 1e-12, 1.0)), y.iloc[calibration_indices])
            audit.update({
                "base_train_rows": len(base_indices), "calibration_rows": len(calibration_indices),
                "base_train_end": times.iloc[base_indices[-1]], "calibration_start": calibration_start,
            })
            return CalibratedModel(base, calibrator, tuple(x.columns), "SIGMOID_TEMPORAL_HOLDOUT", audit)
    base = build_estimator(family)
    base.fit(x, y)
    return CalibratedModel(base, None, tuple(x.columns), status, audit)


def multiclass_brier(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.zeros_like(probabilities)
    labels = np.asarray(y_true, dtype=int)
    truth[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    raw_returns: pd.Series | np.ndarray,
    cost_band: pd.Series | np.ndarray,
) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    predicted = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    raw = np.asarray(raw_returns, dtype=float)
    cost = np.asarray(cost_band, dtype=float)
    acted = predicted != 1
    direction = predicted - 1
    result: dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "precision": float(precision_score(y, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=list(CLASS_LABELS))),
        "brier": multiclass_brier(y, probabilities),
        "coverage": float(acted.mean()),
        "directional_accuracy_acted": float((predicted[acted] == y[acted]).mean()) if acted.any() else np.nan,
        "cost_aware_metric": float(np.mean(direction[acted] * raw[acted] - cost[acted])) if acted.any() else np.nan,
        "commission_status": "UNKNOWN",
    }
    try:
        result["roc_auc_ovr"] = float(roc_auc_score(y, probabilities, labels=list(CLASS_LABELS), multi_class="ovr", average="macro"))
    except ValueError:
        result["roc_auc_ovr"] = np.nan
    for label, name in ((0, "down"), (1, "neutral"), (2, "up")):
        mask = predicted == label
        result[f"mean_raw_return_pred_{name}"] = float(np.mean(raw[mask])) if mask.any() else np.nan
        result[f"median_raw_return_pred_{name}"] = float(np.median(raw[mask])) if mask.any() else np.nan
    return result


def baseline_probabilities(
    name: str,
    train_target: pd.Series,
    validation: pd.DataFrame,
) -> np.ndarray:
    count = train_target.value_counts().reindex(CLASS_LABELS, fill_value=0).astype(float) + 1.0
    priors = (count / count.sum()).to_numpy()
    if name == "MAJORITY_CLASS":
        return np.tile(priors, (len(validation), 1))
    if name == "RECENT_DIRECTION":
        signal = pd.to_numeric(validation["m5_return_5m"], errors="coerce").fillna(0).to_numpy()
    elif name == "MOMENTUM":
        signal = pd.to_numeric(validation["m5_return_15m"], errors="coerce").fillna(0).to_numpy()
    elif name == "MEAN_REVERSION":
        signal = -pd.to_numeric(validation["m5_return_15m"], errors="coerce").fillna(0).to_numpy()
    else:
        raise ValueError(f"Unknown baseline: {name}")
    predicted = np.where(signal > 0, 2, np.where(signal < 0, 0, 1))
    probability = np.full((len(validation), 3), 0.05)
    probability[np.arange(len(validation)), predicted] = 0.90
    return probability


def calibration_bins(
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    horizon_minutes: int,
    family: str,
    fold_id: int,
) -> list[dict[str, object]]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == np.asarray(y_true, dtype=int)
    bins = np.minimum((confidence * 10).astype(int), 9)
    rows: list[dict[str, object]] = []
    for bin_id in range(10):
        mask = bins == bin_id
        if not mask.any():
            continue
        rows.append({
            "horizon_minutes": horizon_minutes, "family": family, "fold_id": fold_id,
            "bin_id": bin_id, "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()), "observed_accuracy": float(correct[mask].mean()),
        })
    return rows


def stability_status(candidate_folds: pd.DataFrame, baseline_folds: pd.DataFrame) -> tuple[str, str]:
    if len(candidate_folds) < 3 or len(baseline_folds) != len(candidate_folds):
        return "INSUFFICIENT_DATA", "fewer than three comparable folds"
    candidate = candidate_folds.sort_values("fold_id")["balanced_accuracy"].astype(float)
    baseline = baseline_folds.sort_values("fold_id")["balanced_accuracy"].astype(float)
    worst_ok = candidate.min() >= baseline.mean() - MAX_WORST_FOLD_DEFICIT
    beating = int((candidate.to_numpy() > baseline.to_numpy()).sum())
    recent_ok = candidate.iloc[-1] >= candidate.iloc[:-1].mean() - MAX_RECENT_FOLD_DEGRADATION
    passed = worst_ok and recent_ok and beating >= ceil(len(candidate) / 2)
    reason = f"worst_ok={worst_ok}; recent_ok={recent_ok}; folds_beating_majority={beating}/{len(candidate)}"
    return ("PASS" if passed else "FAIL"), reason
