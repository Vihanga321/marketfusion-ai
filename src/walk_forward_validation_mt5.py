"""Strict expanding-window validation for clean MT5 H1-derived features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from build_features_mt5 import FEATURES_MT5, OUTPUT_FILE as FEATURE_FILE
from validate_mt5_history import validate_timeframe


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = ROOT / "reports"
RANDOM_SEED = 42


def safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probability)) if np.unique(y_true).size == 2 else np.nan


def calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignment == index
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - y_true[mask].mean())
    return float(error)


def calibration_rows(
    year: int, model: str, y_true: np.ndarray, probability: np.ndarray
) -> list[dict]:
    edges = np.linspace(0, 1, 11)
    assignment = np.clip(np.digitize(probability, edges[1:-1]), 0, 9)
    rows = []
    for index in range(10):
        mask = assignment == index
        if mask.any():
            rows.append(
                {
                    "test_year": year,
                    "model": model,
                    "bin": index + 1,
                    "bin_lower": edges[index],
                    "bin_upper": edges[index + 1],
                    "sample_count": int(mask.sum()),
                    "mean_predicted_probability": float(probability[mask].mean()),
                    "observed_positive_rate": float(y_true[mask].mean()),
                }
            )
    return rows


def metrics(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": safe_auc(y_true, probability),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "calibration_error": calibration_error(y_true, probability),
    }


def xgboost_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        importance_type="gain",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def validate_horizon(
    horizon: int,
    embargo_hours: int,
    first_test_year: int,
    last_test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon not in {1, 4}:
        raise ValueError("Horizon must be 1 or 4")
    if embargo_hours < horizon:
        raise ValueError("Embargo must be at least as long as the horizon")
    h1_quality = validate_timeframe("H1", write_report=True)
    if not h1_quality.passed:
        raise RuntimeError("MT5 H1 validation failed; model training blocked: " + "; ".join(h1_quality.failures))
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Feature file not found; run build_features_mt5.py first: {FEATURE_FILE}")

    target_column = f"target_{horizon}h"
    future_timestamp_column = f"future_timestamp_{horizon}h"
    data = pd.read_parquet(FEATURE_FILE, engine="pyarrow").sort_values("timestamp_utc").reset_index(drop=True)
    required = [*FEATURES_MT5, "timestamp_utc", "decision_timestamp", future_timestamp_column, target_column]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError("Feature file is missing columns: " + ", ".join(missing))
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    data["decision_timestamp"] = pd.to_datetime(data["decision_timestamp"], utc=True)
    data[future_timestamp_column] = pd.to_datetime(data[future_timestamp_column], utc=True)
    data = data.dropna(subset=required).copy()
    data["label_available_timestamp"] = data[future_timestamp_column] + pd.Timedelta(hours=1)

    yearly_rows: list[dict] = []
    calibration_output: list[dict] = []
    importance_output: list[dict] = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}

    print("=" * 78)
    print(f"MARKETFUSION AI - MT5 {horizon}H EXPANDING-WINDOW VALIDATION")
    print("=" * 78)
    print(f"Embargo: {embargo_hours} hours")

    for year in range(first_test_year, last_test_year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        next_year = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
        test = data[
            (data["decision_timestamp"] >= year_start)
            & (data["decision_timestamp"] < next_year)
            & (data["label_available_timestamp"] < next_year)
        ].copy()
        if test.empty:
            continue
        test_start = test["decision_timestamp"].min()
        cutoff = test_start - pd.Timedelta(hours=embargo_hours)
        train = data[
            (data["decision_timestamp"] < cutoff)
            & (data["label_available_timestamp"] < cutoff)
        ].copy()
        if len(train) < 1_000 or train[target_column].nunique() < 2:
            print(f"Skipping {year}: only {len(train):,} eligible training rows")
            continue

        X_train = train[FEATURES_MT5]
        y_train = train[target_column].astype(int)
        X_test = test[FEATURES_MT5]
        y_test = test[target_column].astype(int)
        y_true = y_test.to_numpy()
        rng = np.random.default_rng(RANDOM_SEED + year + horizon * 100)
        random_probability = rng.random(len(test))
        positive_rate = float(y_train.mean())
        majority_class = int(positive_rate >= 0.5)

        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED))
        logistic.fit(X_train, y_train)
        logistic_probability = logistic.predict_proba(X_test)[:, 1]
        xgb = xgboost_model()
        xgb.fit(X_train, y_train)
        xgb_probability = xgb.predict_proba(X_test)[:, 1]
        previous = (test["return_1h"].to_numpy() > 0).astype(int)

        outputs = {
            "random": (random_probability >= 0.5, random_probability),
            "majority": (np.full(len(test), majority_class), np.full(len(test), positive_rate)),
            "previous_direction": (previous, previous.astype(float)),
            "logistic_regression": (logistic_probability >= 0.5, logistic_probability),
            "xgboost": (xgb_probability >= 0.5, xgb_probability),
        }
        for model_name, (prediction, probability) in outputs.items():
            prediction = np.asarray(prediction, dtype=int)
            probability = np.asarray(probability, dtype=float)
            yearly_rows.append(
                {
                    "test_year": year,
                    "model": model_name,
                    "horizon_hours": horizon,
                    "embargo_hours": embargo_hours,
                    "train_start": train["decision_timestamp"].min().isoformat(),
                    "train_end": train["decision_timestamp"].max().isoformat(),
                    "test_start": test["decision_timestamp"].min().isoformat(),
                    "test_end": test["decision_timestamp"].max().isoformat(),
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "class_0_count": int((y_true == 0).sum()),
                    "class_1_count": int((y_true == 1).sum()),
                    "class_0_rate": float((y_true == 0).mean()),
                    "class_1_rate": float((y_true == 1).mean()),
                    **metrics(y_true, prediction, probability),
                }
            )
            calibration_output.extend(calibration_rows(year, model_name, y_true, probability))
            store = pooled.setdefault(model_name, {"truth": [], "prediction": [], "probability": []})
            store["truth"].append(y_true)
            store["prediction"].append(prediction)
            store["probability"].append(probability)
        for feature, importance in zip(FEATURES_MT5, xgb.feature_importances_):
            importance_output.append({"test_year": year, "feature": feature, "importance": float(importance)})
        print(
            f"{year}: train={len(train):,}, test={len(test):,}, "
            f"XGBoost accuracy={yearly_rows[-1]['accuracy']:.4f}, AUC={yearly_rows[-1]['roc_auc']:.4f}"
        )

    yearly = pd.DataFrame(yearly_rows)
    if yearly.empty:
        raise RuntimeError(f"No eligible {horizon}H folds were produced")

    summary_rows = []
    for model_name, group in yearly.groupby("model", sort=False):
        truth = np.concatenate(pooled[model_name]["truth"])
        prediction = np.concatenate(pooled[model_name]["prediction"])
        probability = np.concatenate(pooled[model_name]["probability"])
        summary_rows.append(
            {
                "model": model_name,
                "horizon_hours": horizon,
                "test_years": ",".join(str(value) for value in sorted(group["test_year"].unique())),
                "years_evaluated": group["test_year"].nunique(),
                "total_test_samples": int(group["test_samples"].sum()),
                **{f"average_{column}": group[column].mean() for column in (
                    "accuracy", "balanced_accuracy", "roc_auc", "precision", "recall", "f1",
                    "brier_score", "calibration_error",
                )},
                "pooled_accuracy": accuracy_score(truth, prediction),
                "pooled_balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "pooled_roc_auc": safe_auc(truth, probability),
                "pooled_precision": precision_score(truth, prediction, zero_division=0),
                "pooled_recall": recall_score(truth, prediction, zero_division=0),
                "pooled_f1": f1_score(truth, prediction, zero_division=0),
                "pooled_brier_score": brier_score_loss(truth, probability),
                "pooled_calibration_error": calibration_error(truth, probability),
                "pooled_class_1_rate": float(truth.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("average_balanced_accuracy", ascending=False)
    null_brier = summary.loc[summary["model"] == "majority", "pooled_brier_score"].iloc[0]
    summary["calibration_acceptable"] = (
        (summary["pooled_calibration_error"] <= 0.05)
        & (summary["pooled_brier_score"] <= null_brier)
    )

    pivot = yearly.pivot(index="test_year", columns="model", values="balanced_accuracy")
    baseline_columns = [column for column in pivot.columns if column != "xgboost"]
    xgb_yearly_wins = pivot["xgboost"] > pivot[baseline_columns].max(axis=1)
    consistent_wins = bool(xgb_yearly_wins.all())
    xgb_summary = summary[summary["model"] == "xgboost"].iloc[0]
    edge_supported = bool(
        consistent_wins
        and xgb_summary["pooled_roc_auc"] > 0.5
        and xgb_summary["pooled_brier_score"] < null_brier
    )
    summary["xgboost_beats_all_models_every_year"] = consistent_wins
    summary["predictive_edge_supported"] = edge_supported

    calibration = pd.DataFrame(calibration_output)
    importance_raw = pd.DataFrame(importance_output)
    importance = (
        importance_raw.groupby("feature")["importance"]
        .agg(mean_importance="mean", std_importance="std", folds="count")
        .sort_values("mean_importance", ascending=False)
        .reset_index()
    )
    importance.insert(0, "rank", range(1, len(importance) + 1))

    suffix = f"{horizon}h"
    summary_file = REPORT_DIRECTORY / f"mt5_validation_summary_{suffix}.csv"
    yearly_file = REPORT_DIRECTORY / f"mt5_yearly_results_{suffix}.csv"
    calibration_file = REPORT_DIRECTORY / f"mt5_calibration_{suffix}.csv"
    importance_file = REPORT_DIRECTORY / f"mt5_feature_importance_{suffix}.csv"
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    yearly.to_csv(yearly_file, index=False)
    calibration.to_csv(calibration_file, index=False)
    importance.to_csv(importance_file, index=False)

    print()
    print(summary[[
        "model", "average_accuracy", "average_balanced_accuracy", "average_roc_auc",
        "average_brier_score", "average_calibration_error", "calibration_acceptable",
    ]].to_string(index=False))
    print(f"XGBoost yearly wins over every comparator: {int(xgb_yearly_wins.sum())}/{len(xgb_yearly_wins)}")
    print(f"EDGE_VERDICT: {'CANDIDATE REQUIRING FURTHER TESTS' if edge_supported else 'NOT ESTABLISHED'}")
    print(f"Saved: {summary_file}")
    print(f"Saved: {yearly_file}")
    print(f"Saved: {calibration_file}")
    print(f"Saved: {importance_file}")
    return summary, yearly


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", choices=("1", "4", "both"), default="both")
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--first-test-year", type=int, default=2021)
    parser.add_argument("--last-test-year", type=int, default=pd.Timestamp.now(tz="UTC").year)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    horizons = (1, 4) if args.horizon == "both" else (int(args.horizon),)
    try:
        for horizon in horizons:
            validate_horizon(horizon, args.embargo_hours, args.first_test_year, args.last_test_year)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
