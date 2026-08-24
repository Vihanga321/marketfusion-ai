"""Purged nested chronological split utilities for V0.9 research."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.v09_contract import (
    FINAL_HOLDOUT_FRACTION,
    MIN_FINAL_HOLDOUT_ROWS,
    MIN_INNER_TRAIN_ROWS,
    MIN_OUTER_TRAIN_ROWS,
    MIN_VALIDATION_ROWS,
    N_INNER_FOLDS,
    N_OUTER_FOLDS,
)


@dataclass(frozen=True)
class TemporalFold:
    level: str
    outer_fold: int
    inner_fold: int | None
    train: pd.DataFrame
    validation: pd.DataFrame
    purge_minutes: int

    def audit_row(self) -> dict[str, object]:
        return {
            "level": self.level,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "train_start": self.train["decision_timestamp_utc"].min(),
            "train_end": self.train["decision_timestamp_utc"].max(),
            "validation_start": self.validation["decision_timestamp_utc"].min(),
            "validation_end": self.validation["decision_timestamp_utc"].max(),
            "purge_minutes": self.purge_minutes,
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "future_violation": 0,
        }


@dataclass(frozen=True)
class OuterFold:
    fold: TemporalFold
    inner_folds: tuple[TemporalFold, ...]


@dataclass(frozen=True)
class FinalTemporalSplit:
    development: pd.DataFrame
    holdout: pd.DataFrame
    holdout_opened_for_selection: bool = False


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_timestamp_utc", "target_matured_at_utc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Temporal split missing columns: " + ", ".join(sorted(missing)))
    result = frame.copy()
    result["decision_timestamp_utc"] = pd.to_datetime(result["decision_timestamp_utc"], utc=True, errors="raise")
    result["target_matured_at_utc"] = pd.to_datetime(result["target_matured_at_utc"], utc=True, errors="raise")
    result = result.sort_values("decision_timestamp_utc").reset_index(drop=True)
    if not result["decision_timestamp_utc"].is_monotonic_increasing:
        raise ValueError("Decision timestamps are not monotonic")
    return result


def final_temporal_split(frame: pd.DataFrame, horizon_minutes: int) -> FinalTemporalSplit:
    ordered = _ordered(frame)
    if len(ordered) < MIN_OUTER_TRAIN_ROWS + MIN_FINAL_HOLDOUT_ROWS + MIN_VALIDATION_ROWS:
        return FinalTemporalSplit(ordered.iloc[:0], ordered.iloc[:0])
    holdout_rows = max(MIN_FINAL_HOLDOUT_ROWS, int(np.ceil(len(ordered) * FINAL_HOLDOUT_FRACTION)))
    holdout_start_index = len(ordered) - holdout_rows
    holdout = ordered.iloc[holdout_start_index:].copy()
    holdout_start = holdout["decision_timestamp_utc"].min()
    development = ordered.iloc[:holdout_start_index].loc[
        ordered.iloc[:holdout_start_index]["target_matured_at_utc"].lt(holdout_start)
    ].copy()
    if development.empty or holdout.empty:
        return FinalTemporalSplit(ordered.iloc[:0], ordered.iloc[:0])
    if not development["target_matured_at_utc"].lt(holdout_start).all():
        raise ValueError("Final holdout purge violation")
    actual_gap = holdout_start - development["decision_timestamp_utc"].max()
    if actual_gap < pd.Timedelta(minutes=horizon_minutes):
        raise ValueError("Final holdout horizon purge is too small")
    return FinalTemporalSplit(development.reset_index(drop=True), holdout.reset_index(drop=True))


def _expanding_folds(
    frame: pd.DataFrame,
    horizon_minutes: int,
    n_folds: int,
    min_train_rows: int,
    level: str,
    outer_fold: int,
) -> tuple[TemporalFold, ...]:
    ordered = _ordered(frame)
    remaining = len(ordered) - min_train_rows
    if remaining < MIN_VALIDATION_ROWS:
        return ()
    validation_size = max(MIN_VALIDATION_ROWS, remaining // n_folds)
    folds: list[TemporalFold] = []
    for fold_index in range(n_folds):
        validation_start_index = min_train_rows + fold_index * validation_size
        if validation_start_index >= len(ordered):
            break
        validation_end_index = len(ordered) if fold_index == n_folds - 1 else min(len(ordered), validation_start_index + validation_size)
        validation = ordered.iloc[validation_start_index:validation_end_index].copy()
        if len(validation) < MIN_VALIDATION_ROWS:
            continue
        validation_start = validation["decision_timestamp_utc"].min()
        train_pool = ordered.iloc[:validation_start_index]
        train = train_pool.loc[train_pool["target_matured_at_utc"].lt(validation_start)].copy()
        if len(train) < min_train_rows:
            continue
        if not train["target_matured_at_utc"].lt(validation_start).all():
            raise ValueError("Purged walk-forward leakage")
        if not train["decision_timestamp_utc"].lt(validation_start).all():
            raise ValueError("Future training timestamp detected")
        folds.append(TemporalFold(
            level=level,
            outer_fold=outer_fold,
            inner_fold=(fold_index + 1 if level == "INNER" else None),
            train=train.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            purge_minutes=horizon_minutes,
        ))
    return tuple(folds)


def nested_temporal_folds(frame: pd.DataFrame, horizon_minutes: int) -> tuple[OuterFold, ...]:
    outer = _expanding_folds(
        frame, horizon_minutes, N_OUTER_FOLDS, MIN_OUTER_TRAIN_ROWS,
        level="OUTER", outer_fold=0,
    )
    result: list[OuterFold] = []
    for outer_number, outer_fold in enumerate(outer, start=1):
        normalized_outer = TemporalFold(
            level="OUTER", outer_fold=outer_number, inner_fold=None,
            train=outer_fold.train, validation=outer_fold.validation,
            purge_minutes=horizon_minutes,
        )
        inner = _expanding_folds(
            normalized_outer.train, horizon_minutes, N_INNER_FOLDS,
            MIN_INNER_TRAIN_ROWS, level="INNER", outer_fold=outer_number,
        )
        if not inner:
            continue
        result.append(OuterFold(normalized_outer, inner))
    return tuple(result)


def audit_nested_folds(folds: tuple[OuterFold, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outer in folds:
        rows.append(outer.fold.audit_row())
        rows.extend(fold.audit_row() for fold in outer.inner_folds)
    return pd.DataFrame(rows)
