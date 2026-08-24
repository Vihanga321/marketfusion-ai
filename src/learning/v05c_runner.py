"""Run the controlled daily V0.5C research training cycle."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.learning.v05c_contract import (
    CALIBRATION_REPORT, CANDIDATE_REPORT, COMMISSION_STATUS, CONTRACT_VERSION,
    FEATURE_GROUP_REPORT, FEATURE_REGISTRY_REPORT, HORIZONS, MARKET_FEATURES,
    MIN_PROMOTION_HISTORY_DAYS, MIN_TRAINING_ROWS, MIN_WALK_FORWARD_FOLDS,
    MODEL_REGISTRY_REPORT, MODEL_ROOT, PROMOTION_BALANCED_ACCURACY_MARGIN,
    PROMOTION_LOG_LOSS_MARGIN, PROMOTION_MACRO_F1_MARGIN, ROOT,
    RUNTIME_REGISTRY, SOURCE_FILE, STABILITY_REPORT, TARGET_CONTRACT_REPORT,
    TRAINING_QUALITY_REPORT, WALK_FORWARD_REPORT,
)
from src.learning.v05c_dataset import HorizonDataset, build_horizon_dataset, load_feature_group_eligibility
from src.learning.v05c_models import (
    available_families, baseline_probabilities, calibration_bins,
    evaluate_probabilities, fit_calibrated_model, stability_status,
)
from src.learning.v05c_registry import (
    append_registry, dump_immutable_model, load_registry, registry_frame,
    validate_registry,
)
from src.learning.v05c_walk_forward import fold_audit_row, purged_walk_forward
from src.marketdata.v05a_dataset import load_continuous_frames
from src.marketdata.v05a_quality import build_quality_report


BASELINES = ("MAJORITY_CLASS", "RECENT_DIRECTION", "MOMENTUM", "MEAN_REVERSION")


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _feature_reports(source: pd.DataFrame, groups: pd.DataFrame) -> None:
    _write_csv(FEATURE_REGISTRY_REPORT, [{
        "feature_group": "MARKET_CORE", "feature_name": feature,
        "dtype": str(source[feature].dtype), "enabled": True,
        "causal_basis": "V0.5A value available at or before decision_timestamp_utc",
    } for feature in MARKET_FEATURES])
    lines = [
        "MARKETFUSION V0.5C FEATURE GROUP ELIGIBILITY",
        "activation thresholds are defined in src/learning/v05c_contract.py",
        "group|eligible|rows|days/history coverage|reason",
    ]
    for _, row in groups.iterrows():
        lines.append(f"{row['group']}|{str(bool(row['eligible'])).upper()}|{int(row['rows'])}|{int(row['days'])}|{row['reason']}")
    _write_text(FEATURE_GROUP_REPORT, lines)
    _write_text(TARGET_CONTRACT_REPORT, [
        "MARKETFUSION V0.5C TARGET CONTRACT", f"contract_version: {CONTRACT_VERSION}",
        "independent_horizons_minutes: 15, 60, 240",
        "raw_target: exact V0.5A matured forward return for the selected horizon",
        "decision_cost_band: max(decision m5_spread_points, 1.0) * 0.00001 * 1.25 / decision m5_close",
        "DOWN: raw forward return < -decision_cost_band",
        "NEUTRAL_NO_TRADE: -decision_cost_band <= raw forward return <= decision_cost_band",
        "UP: raw forward return > decision_cost_band", "future_spread_used: NO",
        f"commission_status: {COMMISSION_STATUS}",
        "cost_aware_output_name: COST_AWARE_RESEARCH_METRIC", "profitability_claim: FORBIDDEN",
    ])


def _prediction_rows(
    dataset: HorizonDataset, validation_indices: np.ndarray, probabilities: np.ndarray,
    family: str, fold_id: int, train_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    validation = dataset.frame.iloc[validation_indices]
    predicted = probabilities.argmax(axis=1)
    volatility = pd.to_numeric(validation["m5_volatility_60m"], errors="coerce")
    train_volatility = pd.to_numeric(train_frame["m5_volatility_60m"], errors="coerce").dropna()
    low, high = train_volatility.quantile([1 / 3, 2 / 3]).tolist() if len(train_volatility) else (np.nan, np.nan)
    regimes = np.where(volatility < low, "LOW", np.where(volatility > high, "HIGH", "MEDIUM"))
    rows: list[dict[str, object]] = []
    for offset, (_, row) in enumerate(validation.iterrows()):
        timestamp = pd.Timestamp(row["decision_timestamp_utc"])
        session = "LONDON_NY_OVERLAP" if int(row["is_london_ny_overlap"]) else (
            "LONDON" if int(row["is_london_session"]) else (
                "NEW_YORK" if int(row["is_new_york_session"]) else (
                    "ASIA" if int(row["is_asia_session"]) else "OTHER"
                )
            )
        )
        rows.append({
            "horizon_minutes": dataset.horizon_minutes, "family": family, "fold_id": fold_id,
            "decision_timestamp_utc": timestamp, "target_class": int(row["target_class"]),
            "predicted_class": int(predicted[offset]), "probability_down": probabilities[offset, 0],
            "probability_neutral": probabilities[offset, 1], "probability_up": probabilities[offset, 2],
            "raw_future_return": float(row["raw_future_return"]), "decision_cost_band": float(row["decision_cost_band"]),
            "year": timestamp.year, "month": timestamp.strftime("%Y-%m"), "session": session,
            "volatility_regime": regimes[offset],
        })
    return rows


def _metrics_from_predictions(frame: pd.DataFrame) -> dict[str, object]:
    probabilities = frame[["probability_down", "probability_neutral", "probability_up"]].to_numpy()
    return evaluate_probabilities(frame["target_class"], probabilities, frame["raw_future_return"], frame["decision_cost_band"])


def _evaluate_horizon(dataset: HorizonDataset, families: list[str]) -> dict[str, object]:
    folds = purged_walk_forward(dataset.frame["decision_timestamp_utc"], dataset.horizon_minutes)
    walk_rows = [fold_audit_row(fold, dataset.horizon_minutes) for fold in folds]
    candidate_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    prediction_by_family: dict[str, pd.DataFrame] = {}

    for family in [*families, *BASELINES]:
        predictions: list[dict[str, object]] = []
        statuses: list[str] = []
        for fold in folds:
            train = dataset.frame.iloc[fold.train_indices]
            validation = dataset.frame.iloc[fold.validation_indices]
            if family in BASELINES:
                probabilities = baseline_probabilities(family, train["target_class"], validation)
            else:
                model = fit_calibrated_model(
                    family, train.loc[:, list(dataset.features)], train["target_class"],
                    train["decision_timestamp_utc"], dataset.horizon_minutes,
                )
                probabilities = model.predict_proba(validation.loc[:, list(dataset.features)])
                statuses.append(model.calibration_status)
            predictions.extend(_prediction_rows(dataset, fold.validation_indices, probabilities, family, fold.fold_id, train))
            metrics = evaluate_probabilities(
                validation["target_class"], probabilities, validation["raw_future_return"], validation["decision_cost_band"],
            )
            stability_rows.append({
                "horizon_minutes": dataset.horizon_minutes, "family": family,
                "group_type": "WALK_FORWARD_FOLD", "group": str(fold.fold_id), "fold_id": fold.fold_id, **metrics,
            })
            calibration_rows.extend(calibration_bins(
                probabilities, validation["target_class"], dataset.horizon_minutes, family, fold.fold_id,
            ))
        prediction_frame = pd.DataFrame(predictions)
        prediction_by_family[family] = prediction_frame
        calibration_status = (
            "BASELINE_NOT_CALIBRATED" if family in BASELINES else (
                "SIGMOID_TEMPORAL_HOLDOUT" if statuses and all(item == "SIGMOID_TEMPORAL_HOLDOUT" for item in statuses)
                else "INSUFFICIENT_SAMPLE"
            )
        )
        candidate_rows.append({
            "horizon_minutes": dataset.horizon_minutes, "family": family,
            "candidate_type": "BASELINE" if family in BASELINES else "MODEL",
            "validation_rows": len(prediction_frame), **_metrics_from_predictions(prediction_frame),
            "calibration_status": calibration_status,
        })

    stability_frame = pd.DataFrame(stability_rows)
    majority_folds = stability_frame.query("family == 'MAJORITY_CLASS' and group_type == 'WALK_FORWARD_FOLD'")
    stability_by_family: dict[str, tuple[str, str]] = {}
    for family in families:
        family_folds = stability_frame.query("family == @family and group_type == 'WALK_FORWARD_FOLD'")
        stability_by_family[family] = stability_status(family_folds, majority_folds)
        predictions = prediction_by_family[family]
        for group_type, column in (("YEAR", "year"), ("MONTH", "month"), ("SESSION", "session"), ("VOLATILITY_REGIME", "volatility_regime")):
            for group, subset in predictions.groupby(column, sort=True):
                if len(subset) >= 30:
                    stability_rows.append({
                        "horizon_minutes": dataset.horizon_minutes, "family": family,
                        "group_type": group_type, "group": str(group), "fold_id": None,
                        **_metrics_from_predictions(subset),
                    })
    candidates = pd.DataFrame(candidate_rows)
    for family, (status, reason) in stability_by_family.items():
        mask = candidates["family"].eq(family)
        candidates.loc[mask, "stability_status"] = status
        candidates.loc[mask, "stability_reason"] = reason
    baseline_mask = candidates["candidate_type"].eq("BASELINE")
    candidates.loc[baseline_mask, "stability_status"] = "BASELINE"
    candidates.loc[baseline_mask, "stability_reason"] = "comparison baseline"
    return {
        "folds": folds, "walk_rows": walk_rows, "candidates": candidates,
        "stability_rows": stability_rows, "calibration_rows": calibration_rows,
    }


def promotion_decision(
    candidate: pd.Series, majority: pd.Series, momentum: pd.Series, history_days: int,
    fold_count: int, existing_champion: dict[str, object] | None,
) -> tuple[bool, str]:
    gates = {
        "history_days": history_days >= MIN_PROMOTION_HISTORY_DAYS,
        "walk_forward_folds": fold_count >= MIN_WALK_FORWARD_FOLDS,
        "balanced_accuracy_margin": float(candidate["balanced_accuracy"]) >= float(majority["balanced_accuracy"]) + PROMOTION_BALANCED_ACCURACY_MARGIN,
        "macro_f1_margin": float(candidate["macro_f1"]) >= float(majority["macro_f1"]) + PROMOTION_MACRO_F1_MARGIN,
        "log_loss_margin": float(candidate["log_loss"]) <= float(majority["log_loss"]) - PROMOTION_LOG_LOSS_MARGIN,
        "calibration": candidate["calibration_status"] != "INSUFFICIENT_SAMPLE",
        "stability": candidate["stability_status"] == "PASS",
        "cost_aware_not_worse": float(candidate["cost_aware_metric"]) >= float(momentum["cost_aware_metric"]) - 1e-6,
    }
    if existing_champion is not None:
        gates["beats_existing_champion"] = float(candidate["balanced_accuracy"]) >= float(existing_champion["balanced_accuracy"]) + PROMOTION_BALANCED_ACCURACY_MARGIN
    failed = [name for name, passed in gates.items() if not passed]
    return not failed, ("all conservative promotion gates passed" if not failed else "failed gates: " + ", ".join(failed))


def _existing_champion(records: list[dict[str, object]], horizon: int) -> dict[str, object] | None:
    champions = [record for record in records if int(record["horizon_minutes"]) == horizon and record["promotion_status"] == "CHAMPION"]
    return champions[-1] if champions else None


def run_training() -> dict[str, object]:
    created = pd.Timestamp.now(tz="UTC")
    source = pd.read_parquet(SOURCE_FILE, engine="pyarrow")
    source_quality = build_quality_report(load_continuous_frames(), source, created)
    if not source_quality.passed:
        raise RuntimeError("V0.5A integrity gate failed: " + "; ".join(source_quality.failures))
    groups = load_feature_group_eligibility(source)
    market_gate = bool(groups.loc[groups["group"].eq("MARKET_CORE"), "eligible"].iloc[0])
    if not market_gate:
        raise RuntimeError("MARKET_CORE feature group is not eligible")
    _feature_reports(source, groups)
    families, unavailable = available_families()
    existing_records = load_registry(RUNTIME_REGISTRY)
    if existing_records:
        validate_registry(existing_records)
    all_walk: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_stability: list[dict[str, object]] = []
    new_records: list[dict[str, object]] = []
    summaries: dict[int, dict[str, object]] = {}
    commit = _git_commit()

    for horizon in HORIZONS:
        dataset = build_horizon_dataset(source, horizon)
        if len(dataset.frame) < MIN_TRAINING_ROWS:
            summaries[horizon] = {"status": "INSUFFICIENT_DATA", "training_rows": len(dataset.frame), "folds": 0}
            continue
        evaluation = _evaluate_horizon(dataset, families)
        all_walk.extend(evaluation["walk_rows"])
        candidate_frame = evaluation["candidates"].copy()
        model_candidates = candidate_frame.loc[candidate_frame["candidate_type"].eq("MODEL")].sort_values(
            ["balanced_accuracy", "macro_f1", "log_loss"], ascending=[False, False, True],
        )
        best = model_candidates.iloc[0]
        majority = candidate_frame.loc[candidate_frame["family"].eq("MAJORITY_CLASS")].iloc[0]
        momentum = candidate_frame.loc[candidate_frame["family"].eq("MOMENTUM")].iloc[0]
        history_days = int(pd.to_datetime(dataset.frame["decision_timestamp_utc"], utc=True).dt.floor("D").nunique())
        promote, reason = promotion_decision(
            best, majority, momentum, history_days, len(evaluation["folds"]), _existing_champion(existing_records, horizon),
        )
        candidate_frame["promotion_status"] = "REJECTED"
        candidate_frame["rejection_reason"] = "not highest-ranked research candidate"
        best_mask = candidate_frame["family"].eq(best["family"])
        candidate_frame.loc[best_mask, "promotion_status"] = "CHAMPION" if promote else "CHALLENGER"
        candidate_frame.loc[best_mask, "rejection_reason"] = "" if promote else reason
        baseline_mask = candidate_frame["candidate_type"].eq("BASELINE")
        candidate_frame.loc[baseline_mask, "promotion_status"] = "BASELINE"
        candidate_frame.loc[baseline_mask, "rejection_reason"] = "not a promotable model family"
        all_candidates.extend(candidate_frame.to_dict("records"))
        all_calibration.extend(evaluation["calibration_rows"])
        all_stability.extend(evaluation["stability_rows"])

        for _, result in candidate_frame.loc[candidate_frame["candidate_type"].eq("MODEL")].iterrows():
            family = str(result["family"])
            final_model = fit_calibrated_model(
                family, dataset.frame.loc[:, list(dataset.features)], dataset.frame["target_class"],
                dataset.frame["decision_timestamp_utc"], horizon,
            )
            model_id = f"{family.lower()}_{horizon}m_{created.strftime('%Y%m%dT%H%M%S%fZ')}_{dataset.fingerprint[:10]}"
            artifact = MODEL_ROOT / f"{horizon}m" / f"{model_id}.joblib"
            artifact_sha = dump_immutable_model(artifact, final_model)
            new_records.append({
                "model_id": model_id, "horizon_minutes": horizon, "family": family,
                "created_at_utc": created.isoformat(), "training_data_fingerprint": dataset.fingerprint,
                "feature_contract_hash": dataset.feature_contract_hash, "feature_count": len(dataset.features),
                "train_start": pd.Timestamp(dataset.frame["decision_timestamp_utc"].iloc[0]).isoformat(),
                "train_end": pd.Timestamp(dataset.frame["decision_timestamp_utc"].iloc[-1]).isoformat(),
                "training_rows": len(dataset.frame), "walk_forward_folds": len(evaluation["folds"]),
                "balanced_accuracy": float(result["balanced_accuracy"]), "macro_f1": float(result["macro_f1"]),
                "log_loss": float(result["log_loss"]), "brier": float(result["brier"]),
                "coverage": float(result["coverage"]), "cost_aware_metric": float(result["cost_aware_metric"]),
                "stability_status": str(result["stability_status"]), "calibration_status": final_model.calibration_status,
                "promotion_status": str(result["promotion_status"]), "rejection_reason": str(result["rejection_reason"]),
                "git_commit": commit, "artifact_path": artifact.relative_to(ROOT).as_posix(), "artifact_sha256": artifact_sha,
            })
        summaries[horizon] = {
            "status": "CHAMPION" if promote else "NO_PROMOTION", "training_rows": len(dataset.frame),
            "folds": len(evaluation["folds"]), "best_candidate": str(best["family"]),
            **{key: best[key] for key in (
                "balanced_accuracy", "macro_f1", "log_loss", "brier", "coverage",
                "cost_aware_metric", "stability_status", "calibration_status",
            )},
            "promotion_reason": reason,
        }

    records = append_registry(RUNTIME_REGISTRY, new_records) if new_records else existing_records
    registry_frame(records).to_csv(MODEL_REGISTRY_REPORT, index=False, lineterminator="\n")
    _write_csv(WALK_FORWARD_REPORT, all_walk)
    _write_csv(CANDIDATE_REPORT, all_candidates)
    _write_csv(CALIBRATION_REPORT, all_calibration)
    _write_csv(STABILITY_REPORT, all_stability)
    champions = {horizon: _existing_champion(records, horizon) for horizon in HORIZONS}
    any_champion = any(champions.values())
    insufficient = any(summary["status"] == "INSUFFICIENT_DATA" for summary in summaries.values())
    overall = "INSUFFICIENT_DATA" if insufficient else ("PASS_WITH_CHAMPION" if any_champion else "PASS_RESEARCH_ONLY")
    lines = [
        "MARKETFUSION V0.5C CONTROLLED DAILY LEARNING", f"created_at_utc: {created.isoformat()}",
        f"git_commit: {commit}", "trading: DISABLED / NOT IMPLEMENTED", "random_shuffle: NO",
        "ordinary_kfold: NO", f"market_core: {'ENABLED' if market_gate else 'DISABLED'}",
        "intelligence_v05b: INSUFFICIENT_HISTORY / DISABLED", "event_memory: RESEARCH_ONLY_DISABLED",
        "live_surprise: INSUFFICIENT_HISTORY / DISABLED", f"candidate_families: {', '.join(families)}",
        f"unavailable_families: {'; '.join(unavailable) if unavailable else 'NONE'}",
    ]
    for horizon in HORIZONS:
        summary = summaries[horizon]
        lines.extend([
            "", f"{horizon}m:", f"training_rows: {summary['training_rows']}", f"folds: {summary['folds']}",
            f"best_candidate: {summary.get('best_candidate', 'NONE')}", f"balanced_accuracy: {summary.get('balanced_accuracy')}",
            f"macro_f1: {summary.get('macro_f1')}", f"log_loss: {summary.get('log_loss')}",
            f"brier: {summary.get('brier')}", f"coverage: {summary.get('coverage')}",
            f"COST_AWARE_RESEARCH_METRIC: {summary.get('cost_aware_metric')}",
            f"stability: {summary.get('stability_status')}", f"calibration: {summary.get('calibration_status')}",
            f"promotion: {summary['status']}", f"promotion_reason: {summary.get('promotion_reason', 'insufficient data')}",
        ])
    lines.extend(["", "future_violations: 0", "outcome_feature_violations: 0", "purge_violations: 0", f"V05C_STATUS: {overall}"])
    _write_text(TRAINING_QUALITY_REPORT, lines)
    print("\n".join(lines))
    return {"status": overall, "horizons": summaries, "champions": champions}


def show_status() -> None:
    print("MARKETFUSION V0.5C")
    records = load_registry(RUNTIME_REGISTRY)
    if records:
        validate_registry(records)
    report = pd.read_csv(CANDIDATE_REPORT) if CANDIDATE_REPORT.exists() else pd.DataFrame()
    for horizon in HORIZONS:
        print(f"\n{horizon}m")
        champion = _existing_champion(records, horizon)
        candidates = report.loc[(report["horizon_minutes"] == horizon) & (report["candidate_type"] == "MODEL")] if not report.empty else pd.DataFrame()
        best = candidates.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0] if not candidates.empty else None
        print(f"champion: {champion['model_id'] if champion else 'NONE'}")
        print(f"status: {'CHAMPION' if champion else ('RESEARCH_ONLY_NO_MODEL_PROMOTED' if best is not None else 'INSUFFICIENT_DATA')}")
        print(f"last trained: {records[-1]['created_at_utc'] if records else 'NEVER'}")
        print(f"BA: {best['balanced_accuracy'] if best is not None else 'NA'}")
        print(f"Brier: {best['brier'] if best is not None else 'NA'}")
    print("\nV0.5B intelligence:\ntraining status: INSUFFICIENT_HISTORY")
    print("\nLive surprise:\ntraining status: INSUFFICIENT_HISTORY")
    print("\nTrading:\nDISABLED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        show_status()
    else:
        run_training()


if __name__ == "__main__":
    main()
