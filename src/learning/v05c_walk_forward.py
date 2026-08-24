"""Purged chronological expanding-window validation for V0.5C."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.learning.v05c_contract import MIN_VALIDATION_ROWS, N_WALK_FORWARD_FOLDS


@dataclass(frozen=True)
class PurgedFold:
    fold_id: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_minutes: int


def validate_fold(fold: PurgedFold, timestamps: pd.Series, horizon_minutes: int) -> int:
    train = pd.to_datetime(timestamps.iloc[fold.train_indices], utc=True)
    validation = pd.to_datetime(timestamps.iloc[fold.validation_indices], utc=True)
    violation = int((train >= validation.min()).sum())
    violation += int(((train + pd.Timedelta(minutes=horizon_minutes)) >= validation.min()).sum())
    if violation:
        raise ValueError(f"Fold {fold.fold_id} has {violation} future/overlap violations")
    return 0


def purged_walk_forward(
    timestamps: pd.Series,
    horizon_minutes: int,
    n_splits: int = N_WALK_FORWARD_FOLDS,
    min_validation_rows: int = MIN_VALIDATION_ROWS,
) -> list[PurgedFold]:
    times = pd.to_datetime(timestamps, utc=True, errors="raise").reset_index(drop=True)
    if not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("Walk-forward timestamps must be unique and chronological")
    initial = len(times) // 2
    validation_size = (len(times) - initial) // n_splits
    if validation_size < min_validation_rows:
        raise ValueError("INSUFFICIENT_DATA for requested walk-forward folds")
    folds: list[PurgedFold] = []
    for fold_id in range(1, n_splits + 1):
        validation_start_index = initial + (fold_id - 1) * validation_size
        validation_end_index = len(times) if fold_id == n_splits else validation_start_index + validation_size
        validation_indices = np.arange(validation_start_index, validation_end_index)
        validation_start = times.iloc[validation_start_index]
        train_mask = (times + pd.Timedelta(minutes=horizon_minutes)) < validation_start
        train_indices = np.flatnonzero(train_mask.to_numpy())
        if not len(train_indices):
            raise ValueError("Purge removed every training row")
        fold = PurgedFold(
            fold_id=fold_id, train_indices=train_indices, validation_indices=validation_indices,
            train_start=times.iloc[train_indices[0]], train_end=times.iloc[train_indices[-1]],
            validation_start=validation_start, validation_end=times.iloc[validation_indices[-1]],
            purge_minutes=horizon_minutes,
        )
        validate_fold(fold, times, horizon_minutes)
        folds.append(fold)
    return folds


def fold_audit_row(fold: PurgedFold, horizon_minutes: int) -> dict[str, object]:
    return {
        "horizon_minutes": horizon_minutes, "fold_id": fold.fold_id,
        "train_start": fold.train_start, "train_end": fold.train_end,
        "validation_start": fold.validation_start, "validation_end": fold.validation_end,
        "purge_minutes": fold.purge_minutes, "train_rows": len(fold.train_indices),
        "validation_rows": len(fold.validation_indices), "future_violation": 0,
    }
