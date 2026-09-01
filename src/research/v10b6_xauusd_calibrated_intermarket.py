"""MarketFusion V1.0B.6 leakage-safe calibrated XAUUSD intermarket research.

This phase is exploratory and cannot promote a model. It re-tests only a small
set of V1.0B.5 near-pass contracts using an inner chronological calibration
slice inside each outer purged walk-forward training fold. XAUUSD remains the
only prediction target; all other MT5 instruments remain read-only context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.assets.contracts import ROOT, asset_paths
from src.learning.v05c_walk_forward import purged_walk_forward
from src.research.v10b_xauusd_models import ASSET_ID, load_xauusd_research_frame
from src.research.v10b4_xauusd_intraday_context_audit import run_audit
from src.research.v10b5_xauusd_intraday_intermarket_evidence import (
    CONTEXT_COLLECTION_ROWS,
    CONTEXT_GROUPS,
    MIN_ALIGNED_RESEARCH_ROWS,
    MIN_BALANCED_ACCURACY_DELTA,
    MIN_FOLDS_BEATING_PAIRED_BASELINE,
    MIN_LOG_LOSS_IMPROVEMENT,
    MIN_MEAN_BALANCED_ACCURACY,
    MIN_RECENT_BALANCED_ACCURACY,
    MIN_VALIDATION_ROWS,
    N_SPLITS,
    PRICE_BASELINES,
    TARGET_SPECS,
    _class_prior,
    _directional_frame,
    _group_columns,
    _metrics,
    _pipeline,
    _previous_direction,
    _probabilities,
    _ready_symbols,
    collect_context_history,
    exact_context_join,
    quarantine_split,
)

CONTRACT_VERSION = "v1.0b6-xauusd-calibrated-intermarket-v1"
PREDICTION_TARGET = "XAUUSD"
RANDOM_STATE = 2601
CALIBRATION_FRACTION = 0.20
MIN_CALIBRATION_ROWS = 400
MIN_BASE_TRAIN_ROWS = 1_500
PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class CandidateSpec:
    horizon_minutes: int
    target_contract_name: str
    baseline: str
    context_group: str
    model_family: str
    selection_reason: str


# These are frozen from the real V1.0B.5 leaderboard. This phase is therefore
# exploratory only; even a positive result must be frozen and forward-validated
# before any later promotion research.
CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        30,
        "SELECTIVE_MEDIUM",
        "PRICE_STRUCTURE_VOLATILITY_NO_CLOCK",
        "FX_USD",
        "HIST_GRADIENT_BOOSTING",
        "V1.0B.5 passed every evidence gate except class-prior log loss and beat the paired baseline in 5/5 folds.",
    ),
    CandidateSpec(
        30,
        "SELECTIVE_MEDIUM",
        "PRICE_CORE_NO_CLOCK",
        "SILVER",
        "LOGISTIC_REGRESSION",
        "V1.0B.5 passed all gates except the paired log-loss improvement margin.",
    ),
    CandidateSpec(
        15,
        "V10B_REFERENCE",
        "PRICE_CORE_NO_CLOCK",
        "SILVER",
        "LOGISTIC_REGRESSION",
        "V1.0B.5 passed all gates except the paired log-loss improvement margin.",
    ),
    CandidateSpec(
        15,
        "V10B_REFERENCE",
        "PRICE_CORE_NO_CLOCK",
        "EQUITY_RISK",
        "LOGISTIC_REGRESSION",
        "V1.0B.5 passed all gates except the paired log-loss improvement margin.",
    ),
)

_TARGET_LOOKUP = {(horizon, contract.name): contract for horizon, contract in TARGET_SPECS}
for _candidate in CANDIDATES:
    if (_candidate.horizon_minutes, _candidate.target_contract_name) not in _TARGET_LOOKUP:
        raise RuntimeError("V1.0B.6 candidate references an unknown target contract")
    if _candidate.baseline not in PRICE_BASELINES:
        raise RuntimeError("V1.0B.6 candidate references an unknown price baseline")
    if _candidate.context_group not in CONTEXT_GROUPS:
        raise RuntimeError("V1.0B.6 candidate references an unknown context group")


def temporal_calibration_split(
    outer_train: pd.DataFrame,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split outer training data into base-train and trailing calibration data.

    The base-train section is purged so no base-training target outcome reaches
    into the calibration period. The calibration rows are themselves already
    inside the outer purged training fold, so their outcomes precede validation.
    """
    work = outer_train.sort_values("decision_broker_timestamp").reset_index(drop=True).copy()
    if len(work) < MIN_BASE_TRAIN_ROWS + MIN_CALIBRATION_ROWS:
        raise ValueError("INSUFFICIENT_OUTER_TRAIN_ROWS_FOR_TEMPORAL_CALIBRATION")

    calibration_rows = max(MIN_CALIBRATION_ROWS, int(len(work) * CALIBRATION_FRACTION))
    if len(work) - calibration_rows < MIN_BASE_TRAIN_ROWS:
        calibration_rows = len(work) - MIN_BASE_TRAIN_ROWS
    if calibration_rows < MIN_CALIBRATION_ROWS:
        raise ValueError("INSUFFICIENT_CALIBRATION_ROWS")

    calibration_start_index = len(work) - calibration_rows
    calibration = work.iloc[calibration_start_index:].copy().reset_index(drop=True)
    calibration_start = pd.Timestamp(calibration["decision_broker_timestamp"].iloc[0])

    future = pd.to_datetime(work["future_timestamp_utc"], utc=True, errors="raise")
    base_mask = future < calibration_start
    base_train = work.loc[base_mask & (work.index < calibration_start_index)].copy().reset_index(drop=True)

    if len(base_train) < MIN_BASE_TRAIN_ROWS:
        raise ValueError("CALIBRATION_PURGE_LEFT_TOO_FEW_BASE_TRAIN_ROWS")
    if base_train["directional_target"].nunique() < 2 or calibration["directional_target"].nunique() < 2:
        raise ValueError("INSUFFICIENT_CLASS_VARIATION_FOR_TEMPORAL_CALIBRATION")
    if not (pd.to_datetime(base_train["future_timestamp_utc"], utc=True) < calibration_start).all():
        raise ValueError("BASE_TRAIN_CALIBRATION_OVERLAP")

    return base_train, calibration, {
        "horizon_minutes": int(horizon_minutes),
        "base_train_rows": int(len(base_train)),
        "calibration_rows": int(len(calibration)),
        "calibration_start_broker_epoch": calibration_start.isoformat(),
        "base_train_future_overlap": 0,
        "calibration_fraction_target": CALIBRATION_FRACTION,
    }


def _logit_feature(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Platt calibration requires binary probability matrices")
    up = np.clip(values[:, 1], PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(up / (1.0 - up)).reshape(-1, 1)


def fit_platt_calibrator(probabilities: np.ndarray, target: pd.Series | np.ndarray) -> LogisticRegression:
    y = np.asarray(target, dtype=int)
    if np.unique(y).size < 2:
        raise ValueError("Platt calibration requires both directional classes")
    calibrator = LogisticRegression(
        C=1_000.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(_logit_feature(probabilities), y)
    return calibrator


def apply_platt_calibrator(calibrator: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    calibrated = calibrator.predict_proba(_logit_feature(probabilities))
    classes = list(calibrator.classes_)
    if classes != [0, 1]:
        raise ValueError("Unexpected Platt calibrator class order")
    values = np.asarray(calibrated, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid calibrated probability matrix")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Calibrated probabilities do not sum to one")
    return values


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def evaluate_candidate(
    aligned: pd.DataFrame,
    candidate: CandidateSpec,
) -> dict[str, Any]:
    baseline_columns = PRICE_BASELINES[candidate.baseline]
    context_columns = _group_columns(candidate.context_group)
    all_columns = baseline_columns + context_columns
    missing = sorted(set(all_columns) - set(aligned.columns))
    if missing:
        return {"status": "MISSING_COLUMNS", "missing_columns": missing}

    complete = aligned.loc[:, list(all_columns)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    data = aligned.loc[complete].reset_index(drop=True)
    if len(data) < MIN_ALIGNED_RESEARCH_ROWS:
        return {"status": "INSUFFICIENT_PREQUARANTINE_OVERLAP", "rows": int(len(data))}

    folds = purged_walk_forward(
        data["decision_broker_timestamp"],
        candidate.horizon_minutes,
        n_splits=N_SPLITS,
        min_validation_rows=MIN_VALIDATION_ROWS,
    )

    context_rows: list[dict[str, float]] = []
    baseline_rows: list[dict[str, float]] = []
    raw_context_rows: list[dict[str, float]] = []
    raw_baseline_rows: list[dict[str, float]] = []
    prior_rows: list[dict[str, float]] = []
    previous_rows: list[dict[str, float]] = []
    split_audit: list[dict[str, Any]] = []

    for fold in folds:
        outer_train = data.iloc[fold.train_indices].copy()
        validation = data.iloc[fold.validation_indices].copy()
        if outer_train["directional_target"].nunique() < 2 or validation["directional_target"].nunique() < 2:
            return {"status": "INSUFFICIENT_CLASS_VARIATION", "rows": int(len(data))}

        try:
            base_train, calibration, split_info = temporal_calibration_split(
                outer_train,
                candidate.horizon_minutes,
            )
        except ValueError as exc:
            return {"status": str(exc), "rows": int(len(data))}
        split_info["fold_id"] = int(fold.fold_id)
        split_audit.append(split_info)

        baseline_model = _pipeline(candidate.model_family)
        baseline_model.fit(base_train.loc[:, list(baseline_columns)], base_train["directional_target"])
        baseline_cal_raw = _probabilities(baseline_model, calibration.loc[:, list(baseline_columns)])
        baseline_calibrator = fit_platt_calibrator(baseline_cal_raw, calibration["directional_target"])
        baseline_validation_raw = _probabilities(baseline_model, validation.loc[:, list(baseline_columns)])
        baseline_validation = apply_platt_calibrator(baseline_calibrator, baseline_validation_raw)

        context_model = _pipeline(candidate.model_family)
        context_model.fit(base_train.loc[:, list(all_columns)], base_train["directional_target"])
        context_cal_raw = _probabilities(context_model, calibration.loc[:, list(all_columns)])
        context_calibrator = fit_platt_calibrator(context_cal_raw, calibration["directional_target"])
        context_validation_raw = _probabilities(context_model, validation.loc[:, list(all_columns)])
        context_validation = apply_platt_calibrator(context_calibrator, context_validation_raw)

        raw_baseline_rows.append(_metrics(validation, baseline_validation_raw))
        raw_context_rows.append(_metrics(validation, context_validation_raw))
        baseline_rows.append(_metrics(validation, baseline_validation))
        context_rows.append(_metrics(validation, context_validation))
        prior_rows.append(_metrics(validation, _class_prior(outer_train["directional_target"], len(validation))))
        previous_rows.append(_metrics(validation, _previous_direction(validation)))

    beats = sum(
        context["balanced_accuracy"] > baseline["balanced_accuracy"]
        for context, baseline in zip(context_rows, baseline_rows)
    )

    calibrated_log_loss = _mean(context_rows, "log_loss")
    raw_log_loss = _mean(raw_context_rows, "log_loss")
    return {
        "status": "EVALUATED",
        "rows": int(len(data)),
        "fold_count": int(len(folds)),
        "baseline": candidate.baseline,
        "context_group": candidate.context_group,
        "context_families": list(CONTEXT_GROUPS[candidate.context_group]),
        "family": candidate.model_family,
        "balanced_accuracy_mean": _mean(context_rows, "balanced_accuracy"),
        "baseline_balanced_accuracy_mean": _mean(baseline_rows, "balanced_accuracy"),
        "balanced_accuracy_delta": _mean(context_rows, "balanced_accuracy") - _mean(baseline_rows, "balanced_accuracy"),
        "recent_fold_balanced_accuracy": float(context_rows[-1]["balanced_accuracy"]),
        "folds_beating_paired_baseline": int(beats),
        "macro_f1_mean": _mean(context_rows, "macro_f1"),
        "baseline_macro_f1_mean": _mean(baseline_rows, "macro_f1"),
        "log_loss_mean": calibrated_log_loss,
        "baseline_log_loss_mean": _mean(baseline_rows, "log_loss"),
        "class_prior_log_loss_mean": _mean(prior_rows, "log_loss"),
        "raw_context_log_loss_mean": raw_log_loss,
        "raw_baseline_log_loss_mean": _mean(raw_baseline_rows, "log_loss"),
        "calibration_log_loss_gain": raw_log_loss - calibrated_log_loss,
        "brier_mean": _mean(context_rows, "brier"),
        "baseline_brier_mean": _mean(baseline_rows, "brier"),
        "cost_aware_mean": _mean(context_rows, "cost_aware_metric"),
        "baseline_cost_aware_mean": _mean(baseline_rows, "cost_aware_metric"),
        "previous_direction_cost_aware_mean": _mean(previous_rows, "cost_aware_metric"),
        "context_folds": context_rows,
        "baseline_folds": baseline_rows,
        "raw_context_folds": raw_context_rows,
        "raw_baseline_folds": raw_baseline_rows,
        "calibration_split_audit": split_audit,
    }


def evidence_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("status") != "EVALUATED":
        return "NO_CALIBRATED_INTRADAY_EVIDENCE", [str(row.get("status"))]
    failed: list[str] = []
    if row["balanced_accuracy_mean"] < MIN_MEAN_BALANCED_ACCURACY:
        failed.append("mean_balanced_accuracy")
    if row["recent_fold_balanced_accuracy"] < MIN_RECENT_BALANCED_ACCURACY:
        failed.append("recent_fold_balanced_accuracy")
    if row["folds_beating_paired_baseline"] < MIN_FOLDS_BEATING_PAIRED_BASELINE:
        failed.append("paired_fold_stability")
    if row["balanced_accuracy_delta"] < MIN_BALANCED_ACCURACY_DELTA:
        failed.append("balanced_accuracy_delta")
    if row["log_loss_mean"] > row["baseline_log_loss_mean"] - MIN_LOG_LOSS_IMPROVEMENT:
        failed.append("paired_log_loss")
    if row["log_loss_mean"] >= row["class_prior_log_loss_mean"]:
        failed.append("class_prior_log_loss")
    if row["cost_aware_mean"] < row["baseline_cost_aware_mean"]:
        failed.append("paired_cost_aware")
    if row["calibration_log_loss_gain"] < 0:
        failed.append("calibration_log_loss_non_improving")
    return (
        "CALIBRATED_INTRADAY_EVIDENCE_CANDIDATE" if not failed else "NO_CALIBRATED_INTRADAY_EVIDENCE",
        failed,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: list[dict[str, Any]] = []
    excluded = {
        "context_folds", "baseline_folds", "raw_context_folds", "raw_baseline_folds", "calibration_split_audit"
    }
    for row in rows:
        item = {key: value for key, value in row.items() if key not in excluded}
        if isinstance(item.get("failed_gates"), list):
            item["failed_gates"] = ";".join(item["failed_gates"])
        if isinstance(item.get("context_families"), list):
            item["context_families"] = ";".join(item["context_families"])
        flat.append(item)
    pd.DataFrame(flat).to_csv(path, index=False, lineterminator="\n")


def run_research(
    mt5: Any,
    *,
    root: Path = ROOT,
    persist: bool = True,
    context_rows: int = CONTEXT_COLLECTION_ROWS,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    captured = captured_at or datetime.now(timezone.utc)
    captured = captured.replace(tzinfo=timezone.utc) if captured.tzinfo is None else captured.astimezone(timezone.utc)

    audit = run_audit(mt5, root=root, persist=persist, captured_at=captured)
    ready_symbols = _ready_symbols(audit)
    context_frames, context_manifest = collect_context_history(
        mt5,
        ready_symbols,
        rows=context_rows,
        root=root,
        persist=persist,
    )
    usable = {
        family for family, item in context_manifest.items()
        if item.get("status") == "READY" and family in context_frames
    }

    source = load_xauusd_research_frame(root)
    results: list[dict[str, Any]] = []
    quarantine: dict[str, Any] = {}
    research_cache: dict[tuple[int, str], pd.DataFrame] = {}

    for candidate in CANDIDATES:
        contract = _TARGET_LOOKUP[(candidate.horizon_minutes, candidate.target_contract_name)]
        target_key = (candidate.horizon_minutes, candidate.target_contract_name)
        label = f"{candidate.horizon_minutes}m:{candidate.target_contract_name}"
        if target_key not in research_cache:
            directional = _directional_frame(source, candidate.horizon_minutes, contract)
            research, quarantine_info = quarantine_split(directional, candidate.horizon_minutes)
            research_cache[target_key] = research
            quarantine[label] = quarantine_info
        research = research_cache[target_key]

        missing_families = [family for family in CONTEXT_GROUPS[candidate.context_group] if family not in usable]
        if missing_families:
            row: dict[str, Any] = {
                "status": "CONTEXT_GROUP_UNAVAILABLE",
                "missing_context_families": missing_families,
            }
        else:
            aligned = exact_context_join(research, context_frames, candidate.context_group)
            row = evaluate_candidate(aligned, candidate)

        status, failed = evidence_status(row)
        row.update({
            "asset_id": ASSET_ID,
            "target_label": label,
            "horizon_minutes": candidate.horizon_minutes,
            "target_contract": asdict(contract),
            "baseline": candidate.baseline,
            "context_group": candidate.context_group,
            "family": candidate.model_family,
            "selection_reason": candidate.selection_reason,
            "evidence_status": status,
            "failed_gates": failed,
            "quarantine_evaluated": False,
        })
        results.append(row)

    candidates = [row for row in results if row.get("evidence_status") == "CALIBRATED_INTRADAY_EVIDENCE_CANDIDATE"]
    candidates.sort(
        key=lambda row: (
            float(row.get("balanced_accuracy_delta", -999.0)),
            float(row.get("balanced_accuracy_mean", -999.0)),
            -float(row.get("log_loss_mean", 999.0)),
        ),
        reverse=True,
    )
    evaluated = [row for row in results if row.get("status") == "EVALUATED"]
    evaluated.sort(
        key=lambda row: (
            -len(row.get("failed_gates", [])),
            float(row.get("balanced_accuracy_mean", -999.0)),
            float(row.get("balanced_accuracy_delta", -999.0)),
        ),
        reverse=True,
    )

    decision = "CALIBRATED_INTRADAY_EVIDENCE_FOUND" if candidates else "CALIBRATED_INTRADAY_EVIDENCE_WEAK"
    if not evaluated:
        decision = "INSUFFICIENT_CALIBRATED_INTERMARKET_HISTORY"
    next_phase = {
        "CALIBRATED_INTRADAY_EVIDENCE_FOUND": "V1.0B.7_FREEZE_AND_FORWARD_VALIDATE_INTERMARKET_CONTRACT",
        "CALIBRATED_INTRADAY_EVIDENCE_WEAK": "ADD_CAUSAL_CROSS_MARKET_RELATIONSHIPS_OR_COLLECT_NEW_FORWARD_DATA",
        "INSUFFICIENT_CALIBRATED_INTERMARKET_HISTORY": "COLLECT_MORE_POINT_IN_TIME_CONTEXT_BEFORE_RESEARCH",
    }[decision]

    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "prediction_target": PREDICTION_TARGET,
        "decision": decision,
        "calibrated_candidate_count": int(len(candidates)),
        "best_calibrated_candidate": candidates[0] if candidates else None,
        "best_screen_result": evaluated[0] if evaluated else None,
        "next_phase": next_phase,
        "candidate_selection_source": "REAL_V1.0B.5_LEADERBOARD",
        "exploratory_selection": True,
        "promotion_eligible": False,
        "context_read_only": True,
        "verified_ready_symbols": ready_symbols,
        "context_manifest": context_manifest,
        "broker_clock_calibration": audit.get("broker_clock_calibration"),
        "historical_join_clock": "RAW_MT5_BROKER_EPOCH_KEY",
        "calibration_method": "PLATT_LOGISTIC_ON_INNER_TRAILING_TRAIN_SLICE",
        "calibration_fraction": CALIBRATION_FRACTION,
        "quarantine": quarantine,
        "quarantine_evaluated": False,
        "model_promotion_performed": False,
        "production_integration": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }

    if persist:
        reports = asset_paths(PREDICTION_TARGET, root).reports
        _atomic_json(reports / "v10b6_calibrated_intermarket_evidence.json", report)
        _write_csv(reports / "v10b6_calibrated_intermarket_leaderboard.csv", results)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--context-rows", type=int, default=CONTEXT_COLLECTION_ROWS)
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"MetaTrader5 unavailable: {exc}")
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    try:
        report = run_research(mt5, persist=not args.no_persist, context_rows=args.context_rows)
        compact = {
            "contract_version": report["contract_version"],
            "asset_id": report["asset_id"],
            "decision": report["decision"],
            "calibrated_candidate_count": report["calibrated_candidate_count"],
            "best_calibrated_candidate": report["best_calibrated_candidate"],
            "best_screen_result": report["best_screen_result"],
            "next_phase": report["next_phase"],
            "promotion_eligible": report["promotion_eligible"],
            "production_integration": report["production_integration"],
            "automatic_execution": report["automatic_execution"],
        }
        print(json.dumps(compact, indent=2, default=str))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
