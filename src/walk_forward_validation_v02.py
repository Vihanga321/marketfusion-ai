"""Purged expanding-window validation for MarketFusion AI V0.2."""

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

from build_features_v02 import FEATURES_V02, OUTPUT_FILE as FEATURE_FILE
from validate_oanda_data import DATA_FILE, validate


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
RANDOM_SEED = 42


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else np.nan


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if mask.any():
            error += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(error)


def classification_metrics(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "roc_auc": safe_auc(y_true, probability),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "calibration_error": expected_calibration_error(y_true, probability),
    }


def calibration_rows(
    year: int, model_name: str, y_true: np.ndarray, probability: np.ndarray
) -> list[dict[str, float | int | str]]:
    edges = np.linspace(0, 1, 11)
    assignments = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, 9)
    rows = []
    for index in range(10):
        mask = assignments == index
        if not mask.any():
            continue
        rows.append(
            {
                "test_year": year,
                "model": model_name,
                "bin": index + 1,
                "bin_lower": edges[index],
                "bin_upper": edges[index + 1],
                "sample_count": int(mask.sum()),
                "mean_predicted_probability": float(probability[mask].mean()),
                "observed_positive_rate": float(y_true[mask].mean()),
            }
        )
    return rows


def trading_cost_metrics(
    prediction: np.ndarray,
    current_mid: np.ndarray,
    current_bid: np.ndarray,
    current_ask: np.ndarray,
    future_mid: np.ndarray,
    future_bid: np.ndarray,
    future_ask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    long_trade = prediction == 1
    gross_pips = np.where(long_trade, future_mid - current_mid, current_mid - future_mid) * 10_000
    net_pips = np.where(long_trade, future_bid - current_ask, current_bid - future_ask) * 10_000
    metrics = {
        "mean_gross_pips": float(np.mean(gross_pips)),
        "mean_net_pips": float(np.mean(net_pips)),
        "median_net_pips": float(np.median(net_pips)),
        "net_win_rate": float(np.mean(net_pips > 0)),
        "total_net_pips": float(np.sum(net_pips)),
        "mean_spread_cost_pips": float(np.mean(gross_pips - net_pips)),
    }
    return metrics, gross_pips, net_pips


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


def run(horizon: int, embargo_hours: int, first_test_year: int, last_test_year: int) -> None:
    quality = validate(DATA_FILE, write_report=True)
    if not quality.passed:
        raise RuntimeError(
            "OANDA validation failed; model training is blocked: " + "; ".join(quality.failures)
        )
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"V0.2 feature file not found; run build_features_v02.py first: {FEATURE_FILE}")
    if horizon not in {1, 4}:
        raise ValueError("Horizon must be 1 or 4 hours")
    if embargo_hours < horizon:
        raise ValueError("Embargo must be at least as long as the forecast horizon")

    data = pd.read_parquet(FEATURE_FILE, engine="pyarrow").sort_values("timestamp").reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    future_timestamp_column = f"future_timestamp_{horizon}h"
    target_column = f"target_{horizon}h"
    required = [
        *FEATURES_V02, "decision_timestamp", future_timestamp_column, target_column,
        "mid_close", "bid_close", "ask_close",
    ]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError("Feature file is missing columns: " + ", ".join(missing))
    data[future_timestamp_column] = pd.to_datetime(data[future_timestamp_column], utc=True)
    data["decision_timestamp"] = pd.to_datetime(data["decision_timestamp"], utc=True)
    data["label_available_timestamp"] = data[future_timestamp_column] + pd.Timedelta(hours=1)
    data = data.dropna(subset=required).copy()

    raw = pd.read_parquet(DATA_FILE, columns=["timestamp", "mid_close", "bid_close", "ask_close"], engine="pyarrow")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    future_quotes = raw.set_index("timestamp")[["mid_close", "bid_close", "ask_close"]]

    yearly_rows: list[dict] = []
    calibration_output: list[dict] = []
    importance_output: list[dict] = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}

    print("=" * 78)
    print(f"MARKETFUSION AI V0.2 - {horizon}H PURGED WALK-FORWARD VALIDATION")
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
        training_cutoff = test_start - pd.Timedelta(hours=embargo_hours)
        train = data[
            (data["decision_timestamp"] < training_cutoff)
            & (data["label_available_timestamp"] < training_cutoff)
        ].copy()
        if len(train) < 1_000 or train[target_column].nunique() < 2:
            print(f"Skipping {year}: only {len(train):,} eligible training rows")
            continue

        X_train = train[FEATURES_V02]
        y_train = train[target_column].astype(int)
        X_test = test[FEATURES_V02]
        y_test = test[target_column].astype(int)
        y_true = y_test.to_numpy()
        rng = np.random.default_rng(RANDOM_SEED + year + 100 * horizon)

        random_probability = rng.random(len(test))
        train_positive_rate = float(y_train.mean())
        majority_class = int(train_positive_rate >= 0.5)

        logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED))
        logistic.fit(X_train, y_train)
        logistic_probability = logistic.predict_proba(X_test)[:, 1]

        xgb = xgboost_model()
        xgb.fit(X_train, y_train)
        xgb_probability = xgb.predict_proba(X_test)[:, 1]

        previous_direction = (test["return_1h"].to_numpy() > 0).astype(int)
        outputs = {
            "random_prediction": (random_probability >= 0.5, random_probability),
            "majority_class": (np.full(len(test), majority_class), np.full(len(test), train_positive_rate)),
            "previous_direction": (previous_direction, previous_direction.astype(float)),
            "logistic_regression": (logistic_probability >= 0.5, logistic_probability),
            "xgboost": (xgb_probability >= 0.5, xgb_probability),
        }

        exits = future_quotes.reindex(pd.DatetimeIndex(test[future_timestamp_column]))
        if exits.isna().any().any():
            raise ValueError(f"Missing future bid/ask quotes in trading-cost evaluation for {year}")
        current_mid = test["mid_close"].to_numpy()
        current_bid = test["bid_close"].to_numpy()
        current_ask = test["ask_close"].to_numpy()
        exit_mid = exits["mid_close"].to_numpy()
        exit_bid = exits["bid_close"].to_numpy()
        exit_ask = exits["ask_close"].to_numpy()

        for model_name, (prediction, probability) in outputs.items():
            prediction = np.asarray(prediction, dtype=int)
            probability = np.asarray(probability, dtype=float)
            cost_metrics, gross_pips, net_pips = trading_cost_metrics(
                prediction, current_mid, current_bid, current_ask, exit_mid, exit_bid, exit_ask
            )
            row = {
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
                **classification_metrics(y_true, prediction, probability),
                **cost_metrics,
            }
            yearly_rows.append(row)
            calibration_output.extend(calibration_rows(year, model_name, y_true, probability))
            store = pooled.setdefault(
                model_name,
                {"truth": [], "prediction": [], "probability": [], "gross_pips": [], "net_pips": []},
            )
            store["truth"].append(y_true)
            store["prediction"].append(prediction)
            store["probability"].append(probability)
            store["gross_pips"].append(gross_pips)
            store["net_pips"].append(net_pips)

        for feature, importance in zip(FEATURES_V02, xgb.feature_importances_):
            importance_output.append(
                {"test_year": year, "feature": feature, "importance": float(importance)}
            )

        xgb_row = yearly_rows[-1]
        print(
            f"{year}: train={len(train):,}, test={len(test):,}, "
            f"XGBoost accuracy={xgb_row['accuracy']:.4f}, "
            f"net={xgb_row['mean_net_pips']:.3f} pips/signal"
        )

    yearly = pd.DataFrame(yearly_rows)
    if yearly.empty:
        raise RuntimeError("No eligible walk-forward folds were produced")

    summaries = []
    for model_name, group in yearly.groupby("model", sort=False):
        store = pooled[model_name]
        truth = np.concatenate(store["truth"])
        prediction = np.concatenate(store["prediction"])
        probability = np.concatenate(store["probability"])
        gross_pips = np.concatenate(store["gross_pips"])
        net_pips = np.concatenate(store["net_pips"])
        summaries.append(
            {
                "model": model_name,
                "horizon_hours": horizon,
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
                "pooled_calibration_error": expected_calibration_error(truth, probability),
                "pooled_mean_gross_pips": float(gross_pips.mean()),
                "pooled_mean_net_pips": float(net_pips.mean()),
                "pooled_median_net_pips": float(np.median(net_pips)),
                "pooled_net_win_rate": float((net_pips > 0).mean()),
                "pooled_total_net_pips": float(net_pips.sum()),
                "pooled_mean_spread_cost_pips": float((gross_pips - net_pips).mean()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("average_balanced_accuracy", ascending=False)
    pooled_positive_rate = float(np.concatenate(next(iter(pooled.values()))["truth"]).mean())
    null_brier = pooled_positive_rate * (1 - pooled_positive_rate)
    summary["pooled_null_brier_score"] = null_brier
    summary["calibration_acceptable"] = (
        (summary["pooled_calibration_error"] <= 0.05)
        & (summary["pooled_brier_score"] <= null_brier)
    )
    calibration = pd.DataFrame(calibration_output)
    importance_raw = pd.DataFrame(importance_output)
    importance = (
        importance_raw.groupby("feature")["importance"]
        .agg(mean_importance="mean", std_importance="std", min_importance="min", max_importance="max", folds="count")
        .sort_values("mean_importance", ascending=False)
        .reset_index()
    )
    importance.insert(0, "rank", range(1, len(importance) + 1))

    suffix = f"{horizon}h"
    summary_file = REPORT_DIR / f"v02_validation_summary_{suffix}.csv"
    yearly_file = REPORT_DIR / f"v02_yearly_results_{suffix}.csv"
    calibration_file = REPORT_DIR / f"v02_calibration_{suffix}.csv"
    importance_file = REPORT_DIR / f"v02_feature_importance_{suffix}.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    yearly.to_csv(yearly_file, index=False)
    calibration.to_csv(calibration_file, index=False)
    importance.to_csv(importance_file, index=False)

    print()
    print("Average results across years:")
    print(summary[[
        "model", "average_accuracy", "average_balanced_accuracy", "average_roc_auc",
        "average_brier_score", "average_calibration_error", "pooled_mean_net_pips",
    ]].to_string(index=False))
    print()
    print("Top XGBoost features:")
    print(importance[["rank", "feature", "mean_importance"]].head(20).to_string(index=False))
    print()
    print(f"Saved: {summary_file}")
    print(f"Saved: {yearly_file}")
    print(f"Saved: {calibration_file}")
    print(f"Saved: {importance_file}")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=(1, 4), default=1)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--first-test-year", type=int, default=2021)
    parser.add_argument("--last-test-year", type=int, default=pd.Timestamp.now(tz="UTC").year)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    try:
        run(args.horizon, args.embargo_hours, args.first_test_year, args.last_test_year)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
