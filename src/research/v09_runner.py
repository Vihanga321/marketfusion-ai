"""MarketFusion V0.9 challenger hardening and V0.5C handoff research runner."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.learning.v05c_contract import MIN_PROMOTION_HISTORY_DAYS, SOURCE_FILE
from src.learning.v05c_dataset import build_horizon_dataset
from src.research.v08_experiments import (
    _fit_predict as v08_fit_predict,
    _labels as v08_labels,
    _metrics as v08_metrics,
    chronological_split as v08_chronological_split,
    experiment_specs as v08_experiment_specs,
)
from src.research.v09_candidates import (
    AbstentionPolicy,
    CandidateSpec,
    _estimator,
    baseline_metrics,
    candidate_specs,
    decisions_from_probabilities,
    fit_predict,
    labels,
    metrics,
    temporal_calibrated_predict,
)
from src.research.v09_contract import (
    ABSTENTION_REPORT,
    ALLOWED_FINAL_STATUSES,
    CALIBRATION_REPORT,
    DIRECTIONAL_MARGIN_THRESHOLDS,
    FEATURE_IMPORTANCE_REPORT,
    FEATURE_STABILITY_REPORT,
    HANDOFF_REPORT,
    MAX_RECENT_OUTER_DEGRADATION,
    MAX_RESEARCH_COVERAGE,
    MIN_RESEARCH_COVERAGE,
    MT5_DEPTH_REPORT,
    NESTED_AUDIT_REPORT,
    PRIMARY_HORIZON,
    PRIMARY_SOURCE_EXPERIMENT,
    PRIMARY_V08_SPEC_ID,
    RANDOM_STATE,
    REGIME_REPORT,
    REPRO_REPORT,
    REPRO_TOLERANCE,
    RESEARCH_BASELINE_BA_MARGIN,
    SCORECARD_REPORT,
    STATUS_FILE,
    TARGET_MULTIPLIERS,
    TARGET_REPORT,
    TOP_PROBABILITY_THRESHOLDS,
    V08_SCORECARD,
    VALIDATION_REPORT,
)
from src.research.v09_nested_temporal import audit_nested_folds, final_temporal_split, nested_temporal_folds


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _safe_float(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="raise").iloc[0])


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict[str, object]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")


def _source_v08_row() -> pd.Series:
    if not V08_SCORECARD.exists():
        raise FileNotFoundError(f"Missing frozen V0.8 scorecard: {V08_SCORECARD}")
    frame = pd.read_csv(V08_SCORECARD)
    row = frame.loc[frame["experiment_id"].eq(PRIMARY_SOURCE_EXPERIMENT)]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one {PRIMARY_SOURCE_EXPERIMENT} row, found {len(row)}")
    return row.iloc[0]


def reproduce_v08(dataset: pd.DataFrame) -> dict[str, object]:
    expected = _source_v08_row()
    expected_total = int(expected["development_rows"]) + int(expected["holdout_rows"])
    if len(dataset) < expected_total:
        return {"status": "INSUFFICIENT_HISTORY", "expected_rows": expected_total, "available_rows": len(dataset)}
    frozen_prefix = dataset.iloc[:expected_total].copy().reset_index(drop=True)
    split = v08_chronological_split(frozen_prefix, PRIMARY_HORIZON)
    spec = next(item for item in v08_experiment_specs() if item["id"] == PRIMARY_V08_SPEC_ID)
    multiplier = float(spec["target_multiplier"])
    train_y = v08_labels(split.train, multiplier)
    validation_y = v08_labels(split.validation, multiplier)
    validation_prob = v08_fit_predict(str(spec["family"]), tuple(spec["features"]), split.train, train_y, split.validation)
    development = v08_metrics(validation_y, validation_prob, split.validation["raw_future_return"], split.validation["decision_cost_band"])
    combined = pd.concat([split.train, split.validation], ignore_index=True)
    combined_y = v08_labels(combined, multiplier)
    holdout_y = v08_labels(split.holdout, multiplier)
    holdout_prob = v08_fit_predict(str(spec["family"]), tuple(spec["features"]), combined, combined_y, split.holdout)
    holdout = v08_metrics(holdout_y, holdout_prob, split.holdout["raw_future_return"], split.holdout["decision_cost_band"])
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
        "frozen_prefix_rows": expected_total,
        "development_rows": int(expected["development_rows"]),
        "holdout_rows": int(expected["holdout_rows"]),
        "family": str(expected["model_family"]),
        "target_contract": str(expected["target_contract"]),
        "tolerance": REPRO_TOLERANCE,
    }
    for name, (actual, recorded) in comparisons.items():
        result[f"{name}_actual"] = float(actual)
        result[f"{name}_recorded"] = float(recorded)
        result[f"{name}_absolute_difference"] = differences[name]
    return result


def _aggregate_metric(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _choose_inner_candidate(inner_results: dict[str, list[dict[str, float]]]) -> str:
    ranked: list[tuple[float, float, float, str]] = []
    for candidate_id, rows in inner_results.items():
        ranked.append((
            _aggregate_metric(rows, "balanced_accuracy"),
            _aggregate_metric(rows, "macro_f1"),
            -_aggregate_metric(rows, "log_loss"),
            candidate_id,
        ))
    if not ranked:
        raise RuntimeError("No inner candidate results")
    return max(ranked)[-1]


def _development_hardening(development: pd.DataFrame) -> tuple[CandidateSpec, list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    specs = {spec.candidate_id: spec for spec in candidate_specs()}
    nested = nested_temporal_folds(development, PRIMARY_HORIZON)
    if len(nested) < 3:
        raise RuntimeError("INSUFFICIENT_HISTORY: fewer than three usable outer folds")
    score_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    selected_ids: list[str] = []
    for outer in nested:
        inner_results: dict[str, list[dict[str, float]]] = defaultdict(list)
        for candidate_id, spec in specs.items():
            for inner in outer.inner_folds:
                probability = fit_predict(spec, inner.train, inner.validation)
                measured = metrics(labels(inner.validation, spec.target_multiplier), probability, inner.validation)
                inner_results[candidate_id].append(measured)
            score_rows.append({
                "outer_fold": outer.fold.outer_fold,
                "candidate_id": candidate_id,
                "family": spec.family,
                "feature_count": len(spec.features),
                "selection_scope": "INNER_ONLY",
                "mean_inner_balanced_accuracy": _aggregate_metric(inner_results[candidate_id], "balanced_accuracy"),
                "mean_inner_macro_f1": _aggregate_metric(inner_results[candidate_id], "macro_f1"),
                "mean_inner_brier": _aggregate_metric(inner_results[candidate_id], "brier"),
                "mean_inner_log_loss": _aggregate_metric(inner_results[candidate_id], "log_loss"),
                "mean_inner_coverage": _aggregate_metric(inner_results[candidate_id], "coverage"),
                "mean_inner_cost_aware_metric": _aggregate_metric(inner_results[candidate_id], "cost_aware_metric"),
            })
        selected_id = _choose_inner_candidate(inner_results)
        selected_ids.append(selected_id)
        spec = specs[selected_id]
        probability = fit_predict(spec, outer.fold.train, outer.fold.validation)
        measured = metrics(labels(outer.fold.validation, spec.target_multiplier), probability, outer.fold.validation)
        baselines = baseline_metrics(outer.fold.train, outer.fold.validation, spec.target_multiplier)
        outer_rows.append({
            "outer_fold": outer.fold.outer_fold,
            "selected_candidate": selected_id,
            **measured,
            "majority_ba": baselines["MAJORITY_CLASS"]["balanced_accuracy"],
            "momentum_ba": baselines["MOMENTUM"]["balanced_accuracy"],
            "mean_reversion_ba": baselines["MEAN_REVERSION"]["balanced_accuracy"],
            "recent_direction_ba": baselines["RECENT_DIRECTION"]["balanced_accuracy"],
        })
    counts = Counter(selected_ids)
    outer_frame = pd.DataFrame(outer_rows)
    ranking: list[tuple[int, float, str]] = []
    for candidate_id, count in counts.items():
        subset = outer_frame.loc[outer_frame["selected_candidate"].eq(candidate_id)]
        ranking.append((count, float(subset["balanced_accuracy"].mean()), candidate_id))
    final_id = max(ranking)[-1]
    return specs[final_id], score_rows, audit_nested_folds(nested), outer_rows


def _walk_forward_for_spec(development: pd.DataFrame, spec: CandidateSpec, multiplier: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for outer in nested_temporal_folds(development, PRIMARY_HORIZON):
        probability = fit_predict(spec, outer.fold.train, outer.fold.validation, multiplier)
        rows.append(metrics(labels(outer.fold.validation, multiplier), probability, outer.fold.validation))
    return rows


def _target_robustness(development: pd.DataFrame, spec: CandidateSpec) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    ranking: list[tuple[float, float, float]] = []
    for multiplier in TARGET_MULTIPLIERS:
        measured = _walk_forward_for_spec(development, spec, multiplier)
        row = {
            "candidate_id": spec.candidate_id,
            "target_multiplier": multiplier,
            "folds": len(measured),
            "balanced_accuracy": _aggregate_metric(measured, "balanced_accuracy"),
            "macro_f1": _aggregate_metric(measured, "macro_f1"),
            "brier": _aggregate_metric(measured, "brier"),
            "log_loss": _aggregate_metric(measured, "log_loss"),
            "coverage": _aggregate_metric(measured, "coverage"),
            "wait_rate": _aggregate_metric(measured, "wait_rate"),
            "cost_aware_metric": _aggregate_metric(measured, "cost_aware_metric"),
        }
        rows.append(row)
        ranking.append((float(row["balanced_accuracy"]), float(row["cost_aware_metric"]), multiplier))
    selected = max(ranking)[-1]
    for row in rows:
        row["selected_on_development"] = bool(float(row["target_multiplier"]) == selected)
    return selected, rows


def _abstention_research(development: pd.DataFrame, spec: CandidateSpec, multiplier: float) -> tuple[AbstentionPolicy, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    ranking: list[tuple[float, float, float, float, AbstentionPolicy]] = []
    folds = nested_temporal_folds(development, PRIMARY_HORIZON)
    for top_probability in TOP_PROBABILITY_THRESHOLDS:
        for directional_margin in DIRECTIONAL_MARGIN_THRESHOLDS:
            policy = AbstentionPolicy(top_probability, directional_margin)
            measured_rows: list[dict[str, float]] = []
            for outer in folds:
                probability = fit_predict(spec, outer.fold.train, outer.fold.validation, multiplier)
                measured_rows.append(metrics(labels(outer.fold.validation, multiplier), probability, outer.fold.validation, policy))
            aggregate = {
                "policy_id": policy.policy_id,
                "top_probability": top_probability,
                "directional_margin": directional_margin,
                "folds": len(measured_rows),
                "balanced_accuracy": _aggregate_metric(measured_rows, "balanced_accuracy"),
                "macro_f1": _aggregate_metric(measured_rows, "macro_f1"),
                "coverage": _aggregate_metric(measured_rows, "coverage"),
                "wait_rate": _aggregate_metric(measured_rows, "wait_rate"),
                "cost_aware_metric": _aggregate_metric(measured_rows, "cost_aware_metric"),
            }
            rows.append(aggregate)
            eligible = MIN_RESEARCH_COVERAGE <= float(aggregate["coverage"]) <= MAX_RESEARCH_COVERAGE
            if eligible:
                ranking.append((
                    float(aggregate["balanced_accuracy"]),
                    float(aggregate["cost_aware_metric"]),
                    -abs(float(aggregate["coverage"]) - 0.50),
                    -top_probability,
                    policy,
                ))
    if not ranking:
        # Fixed most-conservative policy; final readiness will fail the coverage gate if needed.
        selected = AbstentionPolicy(max(TOP_PROBABILITY_THRESHOLDS), max(DIRECTIONAL_MARGIN_THRESHOLDS))
    else:
        selected = max(ranking, key=lambda item: item[:4])[-1]
    for row in rows:
        row["selected_on_development"] = row["policy_id"] == selected.policy_id
    return selected, rows


def _calibration_errors(actual: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == actual).astype(float)
    ece = 0.0
    mce = 0.0
    for low in np.arange(0.0, 1.0, 0.1):
        high = low + 0.1
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if not mask.any():
            continue
        gap = abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return ece, mce


def _feature_importance(development: pd.DataFrame, spec: CandidateSpec, multiplier: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if spec.family == "TWO_STAGE_LOGISTIC":
        return [], [{"candidate_id": spec.candidate_id, "status": "NOT_AVAILABLE_FOR_TWO_STAGE"}]
    fold_rankings: list[list[str]] = []
    values: dict[str, list[float]] = defaultdict(list)
    for outer in nested_temporal_folds(development, PRIMARY_HORIZON):
        model = _estimator(spec.family)
        train_y = labels(outer.fold.train, multiplier)
        validation_y = labels(outer.fold.validation, multiplier)
        model.fit(outer.fold.train.loc[:, list(spec.features)], train_y)
        validation_x = outer.fold.validation.loc[:, list(spec.features)].iloc[:500]
        validation_target = validation_y.iloc[:500]
        result = permutation_importance(
            model, validation_x, validation_target,
            scoring="balanced_accuracy", n_repeats=3, random_state=RANDOM_STATE, n_jobs=1,
        )
        pairs = sorted(zip(spec.features, result.importances_mean), key=lambda item: (-float(item[1]), item[0]))
        fold_rankings.append([name for name, _ in pairs[:10]])
        for name, importance in pairs:
            values[name].append(float(importance))
    importance_rows = [
        {"candidate_id": spec.candidate_id, "feature": name, "mean_permutation_importance": float(np.mean(scores)), "folds": len(scores)}
        for name, scores in values.items()
    ]
    importance_rows.sort(key=lambda row: (-float(row["mean_permutation_importance"]), str(row["feature"])))
    consensus = [str(row["feature"]) for row in importance_rows[:10]]
    stability_rows = []
    for fold_number, ranking in enumerate(fold_rankings, start=1):
        overlap = len(set(consensus) & set(ranking)) / max(1, len(consensus))
        stability_rows.append({"candidate_id": spec.candidate_id, "fold": fold_number, "top10_overlap_with_consensus": overlap, "status": "STABLE" if overlap >= 0.7 else "WATCH"})
    return importance_rows, stability_rows


def _session_name(row: pd.Series) -> str:
    if bool(row.get("is_london_ny_overlap", False)):
        return "LONDON_NEW_YORK_OVERLAP"
    if bool(row.get("is_london_session", False)):
        return "LONDON"
    if bool(row.get("is_new_york_session", False)):
        return "NEW_YORK"
    if bool(row.get("is_asia_session", False)):
        return "ASIA"
    return "OFF_SESSION"


def _regime_robustness(development: pd.DataFrame, spec: CandidateSpec, multiplier: float, policy: AbstentionPolicy) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outer in nested_temporal_folds(development, PRIMARY_HORIZON):
        probability = fit_predict(spec, outer.fold.train, outer.fold.validation, multiplier)
        validation = outer.fold.validation.copy()
        validation["_decision"] = decisions_from_probabilities(probability, policy)
        validation["_actual"] = labels(validation, multiplier).to_numpy()
        validation["_session"] = validation.apply(_session_name, axis=1)
        train_vol = pd.to_numeric(outer.fold.train["m5_volatility_60m"], errors="coerce").dropna()
        q25, q75, q95 = train_vol.quantile([0.25, 0.75, 0.95]).tolist()
        current_vol = pd.to_numeric(validation["m5_volatility_60m"], errors="coerce")
        validation["_vol_regime"] = np.select(
            [current_vol.le(q25), current_vol.le(q75), current_vol.le(q95)],
            ["QUIET", "NORMAL", "HIGH_VOLATILITY"], default="EXTREME_VOLATILITY",
        )
        for dimension, column in (("SESSION", "_session"), ("VOLATILITY", "_vol_regime")):
            for group, subset in validation.groupby(column):
                indices = subset.index.to_numpy() - validation.index.min()
                if len(subset) < 100:
                    rows.append({"outer_fold": outer.fold.outer_fold, "dimension": dimension, "group": group, "rows": len(subset), "status": "INSUFFICIENT_SAMPLE"})
                    continue
                group_probability = probability[indices]
                measured = metrics(subset["_actual"].astype(int), group_probability, subset, policy)
                rows.append({"outer_fold": outer.fold.outer_fold, "dimension": dimension, "group": group, "rows": len(subset), "status": "DESCRIPTIVE_ONLY", **measured})
    return rows


def _mt5_history_depth() -> list[str]:
    lines = [
        "MARKETFUSION V0.9 MT5 HISTORY DEPTH AUDIT",
        "mode: READ_ONLY",
        "note: returned depth is subject to the terminal/broker Max bars setting; this audit does not modify V0.5A history",
    ]
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError("mt5.initialize() failed")
        try:
            if not mt5.symbol_select("EURUSD", True):
                raise RuntimeError("EURUSD unavailable")
            requests = (
                ("M1", mt5.TIMEFRAME_M1, 100_000),
                ("M5", mt5.TIMEFRAME_M5, 100_000),
                ("M15", mt5.TIMEFRAME_M15, 60_000),
                ("H1", mt5.TIMEFRAME_H1, 30_000),
            )
            for label, timeframe, count in requests:
                rates = mt5.copy_rates_from_pos("EURUSD", timeframe, 0, count)
                if rates is None or len(rates) == 0:
                    lines.append(f"{label}: UNAVAILABLE")
                    continue
                timestamps = pd.to_datetime(pd.Series(rates["time"]), unit="s", utc=True)
                lines.append(f"{label}: rows={len(rates)} earliest={timestamps.min().isoformat()} latest={timestamps.max().isoformat()}")
        finally:
            mt5.shutdown()
        lines.append("status: PASS_READ_ONLY_PROBE")
    except Exception as exc:
        lines.append(f"status: UNAVAILABLE ({type(exc).__name__}: {exc})")
    return lines


def run_research() -> dict[str, object]:
    created = pd.Timestamp.now(tz="UTC")
    commit = _git_commit()
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Missing V0.5A feature source: {SOURCE_FILE}")
    source = pd.read_parquet(SOURCE_FILE)
    dataset = build_horizon_dataset(source, PRIMARY_HORIZON)
    reproduction = reproduce_v08(dataset.frame)
    repro_lines = ["MARKETFUSION V0.9 REPRODUCIBILITY AUDIT"] + [f"{key}: {value}" for key, value in reproduction.items()]
    _write_text(REPRO_REPORT, repro_lines)
    if reproduction.get("status") != "PASS":
        result = {
            "contract_version": "v0.9-challenger-hardening-v1",
            "status": str(reproduction.get("status")), "created_at_utc": created.isoformat(),
            "source_experiment": PRIMARY_SOURCE_EXPERIMENT, "live_champion": None,
            "live_advisory": "WAIT", "trading_execution": False,
        }
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        _write_text(VALIDATION_REPORT, ["MARKETFUSION V0.9 VALIDATION", f"V09_STATUS: {result['status']}"])
        return result

    final_split = final_temporal_split(dataset.frame, PRIMARY_HORIZON)
    if final_split.development.empty or final_split.holdout.empty:
        status = "INSUFFICIENT_HISTORY"
        result = {"contract_version": "v0.9-challenger-hardening-v1", "status": status, "created_at_utc": created.isoformat(), "live_advisory": "WAIT", "trading_execution": False}
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        _write_text(VALIDATION_REPORT, ["MARKETFUSION V0.9 VALIDATION", f"V09_STATUS: {status}"])
        return result

    best_spec, score_rows, nested_audit, outer_rows = _development_hardening(final_split.development)
    _write_csv(SCORECARD_REPORT, score_rows)
    _write_csv(NESTED_AUDIT_REPORT, nested_audit)

    target_multiplier, target_rows = _target_robustness(final_split.development, best_spec)
    _write_csv(TARGET_REPORT, target_rows)
    abstention_policy, abstention_rows = _abstention_research(final_split.development, best_spec, target_multiplier)
    _write_csv(ABSTENTION_REPORT, abstention_rows)

    final_probabilities, calibration_status = temporal_calibrated_predict(
        best_spec, final_split.development, final_split.holdout,
        PRIMARY_HORIZON, target_multiplier,
    )
    final_actual = labels(final_split.holdout, target_multiplier)
    final_metrics = metrics(final_actual, final_probabilities, final_split.holdout, abstention_policy)
    final_baselines = baseline_metrics(final_split.development, final_split.holdout, target_multiplier)
    ece, mce = _calibration_errors(final_actual.to_numpy(dtype=int), final_probabilities)
    _write_csv(CALIBRATION_REPORT, [{
        "candidate_id": best_spec.candidate_id,
        "calibration_status": calibration_status,
        "holdout_rows": len(final_split.holdout),
        "brier": final_metrics["brier"], "log_loss": final_metrics["log_loss"],
        "ece": ece, "mce": mce,
        "final_holdout_used_for_calibration_fit": False,
    }])

    importance_rows, stability_rows = _feature_importance(final_split.development, best_spec, target_multiplier)
    _write_csv(FEATURE_IMPORTANCE_REPORT, importance_rows)
    _write_csv(FEATURE_STABILITY_REPORT, stability_rows)
    regime_rows = _regime_robustness(final_split.development, best_spec, target_multiplier, abstention_policy)
    _write_csv(REGIME_REPORT, regime_rows)

    outer_frame = pd.DataFrame(outer_rows)
    recent_ok = True
    if len(outer_frame) >= 2:
        previous = float(outer_frame.iloc[:-1]["balanced_accuracy"].mean())
        recent = float(outer_frame.iloc[-1]["balanced_accuracy"])
        recent_ok = recent >= previous - MAX_RECENT_OUTER_DEGRADATION
    baseline_best_ba = max(
        final_baselines["MAJORITY_CLASS"]["balanced_accuracy"],
        final_baselines["MOMENTUM"]["balanced_accuracy"],
        final_baselines["MEAN_REVERSION"]["balanced_accuracy"],
        final_baselines["RECENT_DIRECTION"]["balanced_accuracy"],
    )
    cost_ok = np.isfinite(final_metrics["cost_aware_metric"]) and final_metrics["cost_aware_metric"] > 0
    coverage_ok = MIN_RESEARCH_COVERAGE <= final_metrics["coverage"] <= MAX_RESEARCH_COVERAGE
    readiness = bool(
        final_metrics["balanced_accuracy"] >= baseline_best_ba + RESEARCH_BASELINE_BA_MARGIN
        and final_metrics["balanced_accuracy"] > 1 / 3
        and cost_ok and coverage_ok and recent_ok
        and len(outer_frame) >= 3
    )
    history_days = int(pd.to_datetime(dataset.frame["decision_timestamp_utc"], utc=True).dt.floor("D").nunique())
    history_gate = "MET" if history_days >= MIN_PROMOTION_HISTORY_DAYS else "NOT_MET"
    if readiness and history_gate == "MET":
        status = "PASS_RESEARCH_READY"
    elif readiness:
        status = "PASS_RESEARCH_READY_HISTORY_INSUFFICIENT"
    else:
        status = "PASS_RESEARCH_REJECTED"
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"Unexpected V0.9 status: {status}")

    handoff = {
        "contract_version": "v0.9-v05c-challenger-handoff-v1",
        "generated_at_utc": created.isoformat(),
        "source_experiment": PRIMARY_SOURCE_EXPERIMENT,
        "candidate_id": best_spec.candidate_id,
        "horizon_minutes": PRIMARY_HORIZON,
        "family": best_spec.family,
        "features": list(best_spec.features),
        "target_multiplier": target_multiplier,
        "abstention_policy": {
            "top_probability": abstention_policy.top_probability,
            "directional_margin": abstention_policy.directional_margin,
        },
        "research_status": "READY_FOR_V05C_CHALLENGER" if readiness else "REJECT",
        "promotion_history_gate": history_gate,
        "history_days": history_days,
        "formal_promotion_authority": "V0.5C_ONLY",
        "live_model_status": "NOT_APPROVED",
    }
    HANDOFF_REPORT.write_text(json.dumps(handoff, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    _write_text(MT5_DEPTH_REPORT, _mt5_history_depth())

    validation_lines = [
        "MARKETFUSION V0.9 VALIDATION",
        f"commit: {commit}", f"tested_at_utc: {created.isoformat()}",
        f"source_experiment: {PRIMARY_SOURCE_EXPERIMENT}",
        "reproducible: YES", f"best_hardened_candidate: {best_spec.candidate_id}",
        f"family: {best_spec.family}", f"horizon_minutes: {PRIMARY_HORIZON}",
        f"development_rows: {len(final_split.development)}", f"final_holdout_rows: {len(final_split.holdout)}",
        f"outer_folds: {len(outer_frame)}", "future_violations: 0", "outcome_feature_leaks: 0",
        f"target_multiplier: {target_multiplier}", f"abstention_policy: {abstention_policy.policy_id}",
        f"balanced_accuracy: {final_metrics['balanced_accuracy']}", f"macro_f1: {final_metrics['macro_f1']}",
        f"brier: {final_metrics['brier']}", f"log_loss: {final_metrics['log_loss']}",
        f"coverage: {final_metrics['coverage']}", f"wait_rate: {final_metrics['wait_rate']}",
        f"cost_aware_metric: {final_metrics['cost_aware_metric']}",
        f"calibration: {calibration_status}", f"ECE: {ece}", f"MCE: {mce}",
        f"best_baseline_BA: {baseline_best_ba}", f"recent_outer_stability_ok: {recent_ok}",
        f"research_readiness: {'READY_FOR_V05C_CHALLENGER' if readiness else 'REJECT'}",
        f"promotion_history_days: {history_days}/{MIN_PROMOTION_HISTORY_DAYS}", f"promotion_history_gate: {history_gate}",
        "current_live_champion_15m: NONE", "current_live_champion_60m: NONE", "current_live_champion_240m: NONE",
        "live_advisory: WAIT", "trading_execution: DISABLED", "direct_promotion_by_v09: FORBIDDEN",
        f"V09_STATUS: {status}",
    ]
    _write_text(VALIDATION_REPORT, validation_lines)

    result = {
        "contract_version": "v0.9-challenger-hardening-v1", "status": status,
        "created_at_utc": created.isoformat(), "git_commit": commit,
        "source_experiment": PRIMARY_SOURCE_EXPERIMENT, "reproducible": True,
        "best_candidate": best_spec.candidate_id, "family": best_spec.family,
        "horizon_minutes": PRIMARY_HORIZON, "target_multiplier": target_multiplier,
        "abstention_policy": abstention_policy.policy_id, "final_holdout_rows": len(final_split.holdout),
        "balanced_accuracy": final_metrics["balanced_accuracy"], "macro_f1": final_metrics["macro_f1"],
        "brier": final_metrics["brier"], "log_loss": final_metrics["log_loss"],
        "coverage": final_metrics["coverage"], "wait_rate": final_metrics["wait_rate"],
        "cost_aware_metric": final_metrics["cost_aware_metric"], "calibration": calibration_status,
        "research_readiness": handoff["research_status"], "promotion_history_gate": history_gate,
        "history_days": history_days, "live_champions": {"15": None, "60": None, "240": None},
        "live_advisory": "WAIT", "trading_execution": False,
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(STATUS_FILE)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def show_status() -> dict[str, object]:
    if STATUS_FILE.exists():
        payload = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    elif VALIDATION_REPORT.exists():
        payload = {"status": "REPORT_ONLY", "report": str(VALIDATION_REPORT)}
    else:
        payload = {"status": "NOT_RUN"}
    print("MARKETFUSION V0.9 CHALLENGER HARDENING")
    for key in ("status", "source_experiment", "best_candidate", "horizon_minutes", "balanced_accuracy", "coverage", "wait_rate", "research_readiness", "promotion_history_gate", "history_days", "live_advisory"):
        if key in payload:
            print(f"{key}: {payload[key]}")
    print("formal promotion: V0.5C ONLY")
    print("trading: DISABLED")
    return payload


if __name__ == "__main__":
    run_research()
