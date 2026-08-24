"""Predefined, non-live abstention and two-stage research helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.learning.v05c_contract import RANDOM_STATE


def elastic_net_estimator() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", RobustScaler()),
        ("classifier", SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5, max_iter=2_000,
            tol=1e-4, class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def _aligned(model: Pipeline, frame: pd.DataFrame, labels: tuple[int, ...]) -> np.ndarray:
    raw = model.predict_proba(frame)
    result = np.zeros((len(frame), len(labels)))
    for index, value in enumerate(model.classes_):
        result[:, labels.index(int(value))] = raw[:, index]
    return result


def two_stage_probabilities(
    train_x: pd.DataFrame, train_y: pd.Series, evaluation_x: pd.DataFrame,
) -> np.ndarray:
    """TRADEABLE/NO_TRADE followed by DOWN/UP; research output only."""
    tradeable = train_y.ne(1).astype(int)
    stage_one = elastic_net_estimator().fit(train_x, tradeable)
    trade_probability = _aligned(stage_one, evaluation_x, (0, 1))[:, 1]
    directional_mask = train_y.ne(1)
    if directional_mask.sum() < 2 or train_y.loc[directional_mask].nunique() < 2:
        raise ValueError("Insufficient directional classes for isolated two-stage research")
    stage_two = elastic_net_estimator().fit(train_x.loc[directional_mask], train_y.loc[directional_mask])
    direction_probability = _aligned(stage_two, evaluation_x, (0, 2))
    result = np.column_stack((trade_probability * direction_probability[:, 0], 1 - trade_probability, trade_probability * direction_probability[:, 1]))
    result /= result.sum(axis=1, keepdims=True)
    return result
