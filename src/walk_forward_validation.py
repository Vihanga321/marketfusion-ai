"""Expanding-window validation of the current model and four baselines."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_model import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "eurusd_features.csv"
MODEL_FILE = ROOT / "models" / "eurusd_xgboost_v01.pkl"
REPORT_DIR = ROOT / "reports"
YEARLY_FILE = REPORT_DIR / "yearly_results.csv"
SUMMARY_FILE = REPORT_DIR / "validation_summary.csv"
IMPORTANCE_FILE = REPORT_DIR / "feature_importance.csv"
TEST_YEARS = range(2021, 2027)
RANDOM_SEED = 42
SUSPECT_OHLC_FEATURES = {"range", "upper_wick", "lower_wick", "atr_14"}
ABLATION_FEATURES = [feature for feature in FEATURES if feature not in SUSPECT_OHLC_FEATURES]


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else np.nan


def metric_row(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "roc_auc": safe_auc(y_true, probability),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    df["target_date"] = df["date"].shift(-1)
    if df[FEATURES + ["target"]].isna().any().any():
        raise ValueError("NaN values found in model features or target")

    artifact = joblib.load(MODEL_FILE)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Expected the saved artifact to contain a 'model' entry")
    saved_features = list(artifact.get("features", []))
    if saved_features != list(FEATURES):
        raise ValueError("Saved-model feature order does not match src/train_model.py")
    xgb_template = artifact["model"]

    yearly_rows: list[dict] = []
    importance_rows: list[dict] = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}

    print("=" * 72)
    print("MARKETFUSION AI - EXPANDING-WINDOW VALIDATION")
    print("=" * 72)

    for year in TEST_YEARS:
        test = df[df["date"].dt.year == year].copy()
        if test.empty:
            continue
        test_start = test["date"].min()
        # A row's label belongs to target_date, so purge labels realized in the test period.
        train = df[(df["date"] < test_start) & (df["target_date"] < test_start)].copy()
        if train.empty:
            continue

        X_train, y_train = train[FEATURES], train["target"].astype(int)
        X_test, y_test = test[FEATURES], test["target"].astype(int)
        y_true = y_test.to_numpy()
        rng = np.random.default_rng(RANDOM_SEED + year)

        random_probability = rng.random(len(test))
        train_positive_rate = float(y_train.mean())
        majority_class = int(train_positive_rate >= 0.5)

        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED))
        logistic.fit(X_train, y_train)
        logistic_probability = logistic.predict_proba(X_test)[:, 1]

        xgb_model = clone(xgb_template)
        xgb_model.fit(X_train, y_train)
        xgb_probability = xgb_model.predict_proba(X_test)[:, 1]

        # Diagnostic only: quantify how much of XGBoost's score comes from the
        # Yahoo high/low geometry identified by validate_data.py.
        ablated_xgb_model = clone(xgb_template)
        ablated_xgb_model.fit(train[ABLATION_FEATURES], y_train)
        ablated_xgb_probability = ablated_xgb_model.predict_proba(test[ABLATION_FEATURES])[:, 1]

        outputs = {
            "random_prediction": (random_probability >= 0.5, random_probability),
            "majority_class": (np.full(len(test), majority_class), np.full(len(test), train_positive_rate)),
            "previous_day_direction": ((test["return_1d"].to_numpy() > 0).astype(int), (test["return_1d"].to_numpy() > 0).astype(float)),
            "logistic_regression": (logistic_probability >= 0.5, logistic_probability),
            "current_xgboost": (xgb_probability >= 0.5, xgb_probability),
            "xgboost_without_suspect_ohlc": (
                ablated_xgb_probability >= 0.5,
                ablated_xgb_probability,
            ),
        }

        for model_name, (prediction, probability) in outputs.items():
            prediction = np.asarray(prediction, dtype=int)
            probability = np.asarray(probability, dtype=float)
            row = {
                "test_year": year,
                "model": model_name,
                "train_start": train["date"].min().date().isoformat(),
                "train_end": train["date"].max().date().isoformat(),
                "test_start": test["date"].min().date().isoformat(),
                "test_end": test["date"].max().date().isoformat(),
                "train_samples": len(train),
                "test_samples": len(test),
                "class_0_count": int((y_true == 0).sum()),
                "class_1_count": int((y_true == 1).sum()),
                "class_0_rate": float((y_true == 0).mean()),
                "class_1_rate": float((y_true == 1).mean()),
                **metric_row(y_true, prediction, probability),
            }
            yearly_rows.append(row)
            store = pooled.setdefault(model_name, {"truth": [], "prediction": [], "probability": []})
            store["truth"].append(y_true)
            store["prediction"].append(prediction)
            store["probability"].append(probability)

        for feature, importance in zip(FEATURES, xgb_model.feature_importances_):
            importance_rows.append({"test_year": year, "feature": feature, "importance": float(importance)})

        current_xgb_row = next(
            row for row in reversed(yearly_rows)
            if row["test_year"] == year and row["model"] == "current_xgboost"
        )
        xgb_accuracy = current_xgb_row["accuracy"]
        print(
            f"{year}: train {train['date'].min().date()} to {train['date'].max().date()} "
            f"({len(train):,}), test {len(test):,}, XGBoost accuracy {xgb_accuracy:.4f}"
        )

    yearly = pd.DataFrame(yearly_rows)
    if yearly.empty:
        raise RuntimeError("No walk-forward folds were produced")

    summary_rows = []
    for model_name, group in yearly.groupby("model", sort=False):
        truth = np.concatenate(pooled[model_name]["truth"])
        prediction = np.concatenate(pooled[model_name]["prediction"])
        probability = np.concatenate(pooled[model_name]["probability"])
        summary_rows.append(
            {
                "model": model_name,
                "years_evaluated": group["test_year"].nunique(),
                "total_test_samples": int(group["test_samples"].sum()),
                "average_accuracy": group["accuracy"].mean(),
                "accuracy_std_across_years": group["accuracy"].std(ddof=0),
                "average_roc_auc": group["roc_auc"].mean(),
                "average_precision": group["precision"].mean(),
                "average_recall": group["recall"].mean(),
                "average_f1": group["f1"].mean(),
                "pooled_accuracy": accuracy_score(truth, prediction),
                "pooled_roc_auc": safe_auc(truth, probability),
                "pooled_precision": precision_score(truth, prediction, zero_division=0),
                "pooled_recall": recall_score(truth, prediction, zero_division=0),
                "pooled_f1": f1_score(truth, prediction, zero_division=0),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("average_accuracy", ascending=False)

    raw_importance = pd.DataFrame(importance_rows)
    importance = (
        raw_importance.groupby("feature")["importance"]
        .agg(mean_importance="mean", std_importance="std", min_importance="min", max_importance="max", folds="count")
        .sort_values("mean_importance", ascending=False)
        .reset_index()
    )
    importance.insert(0, "rank", range(1, len(importance) + 1))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(YEARLY_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    importance.to_csv(IMPORTANCE_FILE, index=False)

    print()
    print("Average results (unweighted mean across test years):")
    print(summary[["model", "average_accuracy", "average_roc_auc", "average_precision", "average_recall", "average_f1"]].to_string(index=False))
    print()
    print("Top 20 XGBoost features (mean gain importance across folds):")
    print(importance[["rank", "feature", "mean_importance", "std_importance"]].head(20).to_string(index=False))
    print()
    print(f"Saved: {SUMMARY_FILE}")
    print(f"Saved: {YEARLY_FILE}")
    print(f"Saved: {IMPORTANCE_FILE}")


if __name__ == "__main__":
    main()
