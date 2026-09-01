"""MarketFusion V1.0B.1 XAUUSD target and selective-prediction research.

This phase does NOT promote production models.  It compares economically
meaningful WAIT-zone contracts and horizons, then runs a bounded causal
binary directional screen only on moves outside the WAIT zone.  The goal is
to decide which target/horizon contracts deserve a later controlled model
promotion phase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.assets.contracts import ROOT, asset_paths
from src.learning.v05c_walk_forward import purged_walk_forward
from src.research.v10b_xauusd_models import (
    ASSET_ID,
    FEATURE_GROUPS,
    load_xauusd_research_frame,
    validate_feature_columns,
)

CONTRACT_VERSION = "v1.0b1-xauusd-selective-target-research-v1"
HORIZONS = (15, 30, 60, 120, 240)
RANDOM_STATE = 1811
MIN_TOTAL_ROWS = 5_000
MIN_DIRECTIONAL_ROWS = 2_000
MIN_HOLDOUT_ROWS = 750
FINAL_HOLDOUT_FRACTION = 0.15
MIN_COVERAGE = 0.10
MAX_COVERAGE = 0.90
MIN_HOLDOUT_BALANCED_ACCURACY = 0.52
MIN_WALK_FORWARD_BALANCED_ACCURACY = 0.515
MIN_LOG_LOSS_IMPROVEMENT = 0.01


@dataclass(frozen=True)
class TargetContract:
    name: str
    spread_multiplier: float
    atr_multiplier: float
    description: str


TARGET_CONTRACTS = (
    TargetContract(
        "V10B_REFERENCE", 1.5, 0.10,
        "Original V1.0B cost-aware WAIT zone; included as the reference contract.",
    ),
    TargetContract(
        "SELECTIVE_MEDIUM", 2.0, 0.20,
        "Moderately wider WAIT zone intended to remove small/noisy gold moves.",
    ),
    TargetContract(
        "SELECTIVE_WIDE", 3.0, 0.35,
        "Wide WAIT zone for higher-conviction directional research only.",
    ),
    TargetContract(
        "SELECTIVE_STRICT", 4.0, 0.50,
        "Strict WAIT zone; lower coverage is accepted only if evidence improves materially.",
    ),
)


def screening_features() -> tuple[str, ...]:
    ordered: list[str] = []
    for group in ("MARKET_CORE", "STRUCTURE", "VOLATILITY"):
        for name in FEATURE_GROUPS[group]:
            if name not in ordered:
                ordered.append(name)
    values = tuple(ordered)
    validate_feature_columns(values)
    return values


SCREEN_FEATURES = screening_features()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def exact_future_frame(source: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    """Attach only an exact future completed-M5 close; gaps are never row-shifted."""
    if horizon_minutes not in HORIZONS or horizon_minutes % 5:
        raise ValueError("Unsupported XAUUSD target horizon")
    required = {
        "decision_timestamp_utc", "bar_close", "atr_14",
        "spread_relative_to_price", *SCREEN_FEATURES,
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("Missing V1.0B.1 XAUUSD columns: " + ", ".join(missing))
    work = source.copy()
    work["decision_timestamp_utc"] = _utc(work["decision_timestamp_utc"])
    if work["decision_timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate XAUUSD decision timestamps are forbidden")
    future = work.loc[:, ["decision_timestamp_utc", "bar_close"]].rename(columns={
        "decision_timestamp_utc": "future_timestamp_utc",
        "bar_close": "future_close",
    })
    work["future_timestamp_utc"] = work["decision_timestamp_utc"] + pd.Timedelta(minutes=horizon_minutes)
    merged = work.merge(future, on="future_timestamp_utc", how="left", validate="many_to_one")
    merged["raw_future_return"] = (
        pd.to_numeric(merged["future_close"], errors="coerce")
        / pd.to_numeric(merged["bar_close"], errors="coerce") - 1.0
    )
    return merged


def apply_target_contract(frame: pd.DataFrame, contract: TargetContract) -> pd.DataFrame:
    """Build DOWN/WAIT/UP labels from decision-time spread and ATR only."""
    work = frame.copy()
    close = pd.to_numeric(work["bar_close"], errors="coerce").replace(0, np.nan)
    spread_fraction = pd.to_numeric(work["spread_relative_to_price"], errors="coerce").abs()
    atr_fraction = pd.to_numeric(work["atr_14"], errors="coerce").abs() / close
    band = np.maximum(
        contract.spread_multiplier * spread_fraction,
        contract.atr_multiplier * atr_fraction,
    )
    work["decision_wait_band"] = band
    raw = pd.to_numeric(work["raw_future_return"], errors="coerce")
    work["target_class"] = np.where(
        raw < -band, 0,
        np.where(raw > band, 2, 1),
    )
    work.loc[raw.isna() | pd.isna(band), "target_class"] = np.nan
    work["directional_target"] = np.where(
        work["target_class"].eq(0), 0,
        np.where(work["target_class"].eq(2), 1, np.nan),
    )
    return work


def target_semantics(frame: pd.DataFrame) -> dict[str, Any]:
    matured = frame.loc[frame["target_class"].notna()].copy()
    total = len(matured)
    counts = matured["target_class"].astype(int).value_counts().reindex([0, 1, 2], fill_value=0)
    directional = int(counts[0] + counts[2])
    excess = (matured["raw_future_return"].abs() - matured["decision_wait_band"]).clip(lower=0)
    return {
        "matured_rows": total,
        "down_rows": int(counts[0]),
        "wait_rows": int(counts[1]),
        "up_rows": int(counts[2]),
        "down_fraction": float(counts[0] / total) if total else None,
        "wait_fraction": float(counts[1] / total) if total else None,
        "up_fraction": float(counts[2] / total) if total else None,
        "directional_coverage": float(directional / total) if total else None,
        "median_wait_band": float(matured["decision_wait_band"].median()) if total else None,
        "median_abs_future_return": float(matured["raw_future_return"].abs().median()) if total else None,
        "median_excess_move": float(excess.median()) if total else None,
    }


def _pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", RobustScaler()),
        ("classifier", LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def _probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(x), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("V1.0B.1 expects binary DOWN/UP probabilities")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid V1.0B.1 probability matrix")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("V1.0B.1 probabilities do not sum to one")
    return values


def _class_prior(train_y: pd.Series, rows: int) -> np.ndarray:
    counts = train_y.value_counts().reindex([0, 1], fill_value=0).astype(float)
    prior = ((counts + 1.0) / (counts.sum() + 2.0)).to_numpy(dtype=float)
    return np.tile(prior, (rows, 1))


def _previous_direction(validation: pd.DataFrame) -> np.ndarray:
    ret = pd.to_numeric(validation["m5_return_5m"], errors="coerce").fillna(0.0).to_numpy()
    predicted_up = ret >= 0
    result = np.empty((len(validation), 2), dtype=float)
    result[:, 0] = np.where(predicted_up, 0.10, 0.90)
    result[:, 1] = np.where(predicted_up, 0.90, 0.10)
    return result


def _binary_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = frame["directional_target"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    raw = pd.to_numeric(frame["raw_future_return"], errors="raise").to_numpy(dtype=float)
    band = pd.to_numeric(frame["decision_wait_band"], errors="raise").to_numpy(dtype=float)
    direction = np.where(predicted == 1, 1.0, -1.0)
    return {
        "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(np.mean((probabilities[:, 1] - y) ** 2)),
        "cost_aware_metric": float(np.mean(direction * raw - band)),
    }


def _directional_frame(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.loc[:, list(SCREEN_FEATURES)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    directional = frame["directional_target"].notna()
    work = frame.loc[complete & directional].copy()
    work["decision_timestamp_utc"] = _utc(work["decision_timestamp_utc"])
    work["future_timestamp_utc"] = _utc(work["future_timestamp_utc"])
    work["directional_target"] = work["directional_target"].astype(int)
    return work.sort_values("decision_timestamp_utc").reset_index(drop=True)


def _final_split(frame: pd.DataFrame, horizon_minutes: int) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    if len(frame) < MIN_DIRECTIONAL_ROWS + MIN_HOLDOUT_ROWS:
        raise ValueError("INSUFFICIENT_DIRECTIONAL_DATA")
    holdout_rows = max(MIN_HOLDOUT_ROWS, int(len(frame) * FINAL_HOLDOUT_FRACTION))
    start_index = len(frame) - holdout_rows
    holdout_start = pd.Timestamp(frame["decision_timestamp_utc"].iloc[start_index])
    timestamps = _utc(frame["decision_timestamp_utc"])
    research = np.flatnonzero(((timestamps + pd.Timedelta(minutes=horizon_minutes)) < holdout_start).to_numpy())
    holdout = np.arange(start_index, len(frame))
    if len(research) < MIN_DIRECTIONAL_ROWS or len(holdout) < MIN_HOLDOUT_ROWS:
        raise ValueError("INSUFFICIENT_DIRECTIONAL_DATA_AFTER_PURGE")
    return research, holdout, holdout_start


def directional_screen(frame: pd.DataFrame, horizon_minutes: int) -> dict[str, Any]:
    directional = _directional_frame(frame)
    research_idx, holdout_idx, holdout_start = _final_split(directional, horizon_minutes)
    research = directional.iloc[research_idx].reset_index(drop=True)
    holdout = directional.iloc[holdout_idx].reset_index(drop=True)
    folds = purged_walk_forward(
        research["decision_timestamp_utc"], horizon_minutes,
        n_splits=4, min_validation_rows=250,
    )
    fold_metrics: list[dict[str, float]] = []
    prior_fold_metrics: list[dict[str, float]] = []
    for fold in folds:
        train = research.iloc[fold.train_indices]
        validation = research.iloc[fold.validation_indices]
        model = _pipeline()
        model.fit(train.loc[:, list(SCREEN_FEATURES)], train["directional_target"])
        probability = _probabilities(model, validation.loc[:, list(SCREEN_FEATURES)])
        fold_metrics.append(_binary_metrics(validation, probability))
        prior_fold_metrics.append(_binary_metrics(
            validation, _class_prior(train["directional_target"], len(validation))
        ))
    walk_balanced = float(np.mean([row["balanced_accuracy"] for row in fold_metrics]))
    folds_beating_prior = int(sum(
        row["balanced_accuracy"] > baseline["balanced_accuracy"]
        for row, baseline in zip(fold_metrics, prior_fold_metrics)
    ))

    final_model = _pipeline()
    final_model.fit(research.loc[:, list(SCREEN_FEATURES)], research["directional_target"])
    holdout_probability = _probabilities(final_model, holdout.loc[:, list(SCREEN_FEATURES)])
    holdout_metrics = _binary_metrics(holdout, holdout_probability)
    prior_metrics = _binary_metrics(
        holdout, _class_prior(research["directional_target"], len(holdout))
    )
    previous_metrics = _binary_metrics(holdout, _previous_direction(holdout))
    return {
        "feature_count": len(SCREEN_FEATURES),
        "research_rows": len(research),
        "holdout_rows": len(holdout),
        "holdout_start": holdout_start.isoformat(),
        "fold_count": len(folds),
        "folds_beating_class_prior": folds_beating_prior,
        "walk_forward_balanced_accuracy_mean": walk_balanced,
        "walk_forward": fold_metrics,
        "final_holdout": holdout_metrics,
        "class_prior_final_holdout": prior_metrics,
        "previous_direction_final_holdout": previous_metrics,
        "calibration_status": "NOT_USED_TARGET_SCREEN_ONLY",
    }


def evidence_gate(semantics: dict[str, Any], screen: dict[str, Any]) -> tuple[str, list[str]]:
    coverage = float(semantics["directional_coverage"] or 0.0)
    holdout = screen["final_holdout"]
    prior = screen["class_prior_final_holdout"]
    previous = screen["previous_direction_final_holdout"]
    failed: list[str] = []
    if not (MIN_COVERAGE <= coverage <= MAX_COVERAGE):
        failed.append("coverage")
    if int(screen["holdout_rows"]) < MIN_HOLDOUT_ROWS:
        failed.append("holdout_rows")
    if float(screen["walk_forward_balanced_accuracy_mean"]) < MIN_WALK_FORWARD_BALANCED_ACCURACY:
        failed.append("walk_forward_balanced_accuracy")
    if int(screen["folds_beating_class_prior"]) < max(2, int(np.ceil(screen["fold_count"] / 2))):
        failed.append("walk_forward_baseline_stability")
    if float(holdout["balanced_accuracy"]) < MIN_HOLDOUT_BALANCED_ACCURACY:
        failed.append("holdout_balanced_accuracy")
    if float(holdout["log_loss"]) > float(prior["log_loss"]) - MIN_LOG_LOSS_IMPROVEMENT:
        failed.append("holdout_log_loss")
    if float(holdout["cost_aware_metric"]) < float(previous["cost_aware_metric"]):
        failed.append("holdout_cost_aware")
    return ("RESEARCH_CANDIDATE" if not failed else "REJECTED_RESEARCH_TARGET", failed)


def evaluate_contract(source: pd.DataFrame, horizon: int, contract: TargetContract) -> dict[str, Any]:
    future = exact_future_frame(source, horizon)
    targeted = apply_target_contract(future, contract)
    semantics = target_semantics(targeted)
    result: dict[str, Any] = {
        "asset_id": ASSET_ID,
        "horizon_minutes": horizon,
        "contract": asdict(contract),
        "semantics": semantics,
        "production_integration": False,
    }
    if int(semantics["matured_rows"]) < MIN_TOTAL_ROWS or (
        int(semantics["down_rows"]) + int(semantics["up_rows"])
    ) < MIN_DIRECTIONAL_ROWS:
        result.update({
            "status": "INSUFFICIENT_DATA",
            "failed_gates": ["minimum_directional_sample"],
            "screen": None,
        })
        return result
    try:
        screen = directional_screen(targeted, horizon)
    except ValueError as exc:
        result.update({"status": "INSUFFICIENT_DATA", "failed_gates": [str(exc)], "screen": None})
        return result
    status, failed = evidence_gate(semantics, screen)
    result.update({"status": status, "failed_gates": failed, "screen": screen})
    return result


def _rank_key(result: dict[str, Any]) -> tuple[int, float, float, float]:
    screen = result.get("screen") or {}
    holdout = screen.get("final_holdout") or {}
    return (
        1 if result.get("status") == "RESEARCH_CANDIDATE" else 0,
        float(holdout.get("balanced_accuracy") or -1.0),
        -float(holdout.get("log_loss") or 99.0),
        float((result.get("semantics") or {}).get("directional_coverage") or 0.0),
    )


def run_research(root: Path = ROOT, persist: bool = True) -> dict[str, Any]:
    source = load_xauusd_research_frame(root)
    results: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for contract in TARGET_CONTRACTS:
            results.append(evaluate_contract(source, horizon, contract))
    candidates = sorted(
        [row for row in results if row["status"] == "RESEARCH_CANDIDATE"],
        key=_rank_key, reverse=True,
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "asset_class": "PRECIOUS_METAL",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_rows": len(source),
        "source_start": pd.Timestamp(source["decision_timestamp_utc"].min()).isoformat(),
        "source_end": pd.Timestamp(source["decision_timestamp_utc"].max()).isoformat(),
        "horizons": list(HORIZONS),
        "target_contracts": [asdict(item) for item in TARGET_CONTRACTS],
        "screen_feature_count": len(SCREEN_FEATURES),
        "results": results,
        "research_candidate_count": len(candidates),
        "best_research_candidate": candidates[0] if candidates else None,
        "decision": "TARGET_CANDIDATE_FOUND" if candidates else "NO_TARGET_CANDIDATE",
        "next_phase": "V1.0B.2_CONTROLLED_MODEL_RESEARCH" if candidates else "REVISE_XAUUSD_FEATURE_OR_TARGET_DESIGN",
        "production_integration": False,
        "model_promotion_performed": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }
    if persist:
        reports = asset_paths(ASSET_ID, root=root).reports
        _atomic_json(reports / "v10b1_target_research.json", report)
        flat_rows: list[dict[str, Any]] = []
        for row in results:
            semantics = row["semantics"]
            screen = row.get("screen") or {}
            holdout = screen.get("final_holdout") or {}
            flat_rows.append({
                "asset_id": ASSET_ID,
                "horizon_minutes": row["horizon_minutes"],
                "contract_name": row["contract"]["name"],
                "spread_multiplier": row["contract"]["spread_multiplier"],
                "atr_multiplier": row["contract"]["atr_multiplier"],
                "status": row["status"],
                "failed_gates": ";".join(row["failed_gates"]),
                **semantics,
                "walk_forward_balanced_accuracy_mean": screen.get("walk_forward_balanced_accuracy_mean"),
                "holdout_balanced_accuracy": holdout.get("balanced_accuracy"),
                "holdout_macro_f1": holdout.get("macro_f1"),
                "holdout_log_loss": holdout.get("log_loss"),
                "holdout_brier": holdout.get("brier"),
                "holdout_cost_aware_metric": holdout.get("cost_aware_metric"),
            })
        _write_csv(reports / "v10b1_target_leaderboard.csv", flat_rows)
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()
    report = run_research(ROOT, persist=not args.no_persist)
    summary = {
        "contract_version": report["contract_version"],
        "asset_id": report["asset_id"],
        "decision": report["decision"],
        "research_candidate_count": report["research_candidate_count"],
        "best_research_candidate": report["best_research_candidate"],
        "next_phase": report["next_phase"],
        "production_integration": False,
        "automatic_execution": "DISABLED",
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
