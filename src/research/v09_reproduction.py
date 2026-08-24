"""Exact-prefix reproducibility bridge for the frozen V0.8 H60 result.

V0.8's development row count excludes two horizon purges. Therefore the original
source length cannot be reconstructed by simply adding development + holdout rows.
This module finds the unique historical prefix whose V0.8 split reproduces both
recorded row counts before comparing metrics.
"""
from __future__ import annotations

import pandas as pd

from src.research.v08_experiments import (
    _fit_predict as v08_fit_predict,
    _labels as v08_labels,
    _metrics as v08_metrics,
    chronological_split as v08_chronological_split,
    experiment_specs as v08_experiment_specs,
)
from src.research.v09_contract import (
    PRIMARY_HORIZON,
    PRIMARY_SOURCE_EXPERIMENT,
    PRIMARY_V08_SPEC_ID,
    REPRO_TOLERANCE,
    V08_SCORECARD,
)


def _safe_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="raise").iloc[0])


def infer_frozen_source_rows(dataset: pd.DataFrame, development_rows: int, holdout_rows: int) -> int | None:
    """Find the original V0.8 source prefix from its purged split row counts."""
    minimum = development_rows + holdout_rows
    # Two 60m-vs-5m purge zones account for only a few dozen rows; the wider
    # bound deliberately avoids hard-coding that count while staying deterministic.
    maximum = min(len(dataset), minimum + 500)
    matches: list[int] = []
    for source_rows in range(minimum, maximum + 1):
        split = v08_chronological_split(dataset.iloc[:source_rows].copy(), PRIMARY_HORIZON)
        if len(split.train) + len(split.validation) == development_rows and len(split.holdout) == holdout_rows:
            matches.append(source_rows)
    return matches[0] if len(matches) == 1 else None


def reproduce_v08(dataset: pd.DataFrame) -> dict[str, object]:
    if not V08_SCORECARD.exists():
        raise FileNotFoundError(f"Missing frozen V0.8 scorecard: {V08_SCORECARD}")
    scorecard = pd.read_csv(V08_SCORECARD)
    selected = scorecard.loc[scorecard["experiment_id"].eq(PRIMARY_SOURCE_EXPERIMENT)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected exactly one {PRIMARY_SOURCE_EXPERIMENT} row, found {len(selected)}")
    expected = selected.iloc[0]
    development_rows = int(expected["development_rows"])
    holdout_rows = int(expected["holdout_rows"])
    source_rows = infer_frozen_source_rows(dataset, development_rows, holdout_rows)
    if source_rows is None:
        return {
            "status": "INSUFFICIENT_HISTORY" if len(dataset) < development_rows + holdout_rows else "RESEARCH_RESULT_NOT_REPRODUCIBLE",
            "development_rows": development_rows,
            "holdout_rows": holdout_rows,
            "available_rows": len(dataset),
            "reason": "unique frozen V0.8 source prefix could not be reconstructed",
        }
    frozen_prefix = dataset.iloc[:source_rows].copy().reset_index(drop=True)
    split = v08_chronological_split(frozen_prefix, PRIMARY_HORIZON)
    spec = next(item for item in v08_experiment_specs() if item["id"] == PRIMARY_V08_SPEC_ID)
    multiplier = float(spec["target_multiplier"])
    train_y = v08_labels(split.train, multiplier)
    validation_y = v08_labels(split.validation, multiplier)
    validation_probability = v08_fit_predict(str(spec["family"]), tuple(spec["features"]), split.train, train_y, split.validation)
    development = v08_metrics(validation_y, validation_probability, split.validation["raw_future_return"], split.validation["decision_cost_band"])
    combined = pd.concat([split.train, split.validation], ignore_index=True)
    combined_y = v08_labels(combined, multiplier)
    holdout_y = v08_labels(split.holdout, multiplier)
    holdout_probability = v08_fit_predict(str(spec["family"]), tuple(spec["features"]), combined, combined_y, split.holdout)
    holdout = v08_metrics(holdout_y, holdout_probability, split.holdout["raw_future_return"], split.holdout["decision_cost_band"])
    comparisons = {
        "development_BA": (development["ba"], _safe_float(expected["development_BA"])),
        "development_brier": (development["brier"], _safe_float(expected["development_brier"])),
        "holdout_BA": (holdout["ba"], _safe_float(expected["holdout_BA"])),
        "holdout_brier": (holdout["brier"], _safe_float(expected["holdout_brier"])),
        "coverage": (holdout["coverage"], _safe_float(expected["coverage"])),
        "wait_rate": (holdout["wait_rate"], _safe_float(expected["wait_rate"])),
        "cost_aware_metric": (holdout["cost"], _safe_float(expected["cost_aware_metric"])),
    }
    differences = {name: abs(float(actual) - float(recorded)) for name, (actual, recorded) in comparisons.items()}
    passed = all(value <= REPRO_TOLERANCE for value in differences.values())
    result: dict[str, object] = {
        "status": "PASS" if passed else "RESEARCH_RESULT_NOT_REPRODUCIBLE",
        "source_experiment": PRIMARY_SOURCE_EXPERIMENT,
        "frozen_source_rows": source_rows,
        "development_rows": development_rows,
        "holdout_rows": holdout_rows,
        "family": str(expected["model_family"]),
        "target_contract": str(expected["target_contract"]),
        "tolerance": REPRO_TOLERANCE,
    }
    for name, (actual, recorded) in comparisons.items():
        result[f"{name}_actual"] = float(actual)
        result[f"{name}_recorded"] = float(recorded)
        result[f"{name}_absolute_difference"] = differences[name]
    return result
