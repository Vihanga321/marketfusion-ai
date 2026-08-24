"""Schema and write gate for V0.8 research-only experiment scorecards."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.v08_contract import SCORECARD_REPORT

SCORECARD_COLUMNS = (
    "experiment_id", "horizon", "model_family", "target_contract", "feature_groups",
    "development_rows", "holdout_rows", "development_BA", "holdout_BA",
    "development_brier", "holdout_brier", "coverage", "wait_rate",
    "cost_aware_metric", "stability", "decision",
)
ALLOWED_DECISIONS = {"PROMISING_FOR_V05C_CHALLENGER", "REJECT", "INSUFFICIENT_DATA"}


def write_scorecard(frame: pd.DataFrame, path: Path = SCORECARD_REPORT) -> None:
    missing = sorted(set(SCORECARD_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Scorecard fields missing: " + ", ".join(missing))
    forbidden = set(frame["decision"].dropna().astype(str)) - ALLOWED_DECISIONS
    if forbidden:
        raise ValueError("V0.8 research cannot directly become CHAMPION: " + ", ".join(sorted(forbidden)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(SCORECARD_COLUMNS)].to_csv(path, index=False)
