"""Matched-fold comparison of price-only and point-in-time macro MT5 models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features_mt5 import FEATURES_MT5
from build_features_mt5_macro import MACRO_FEATURES, OUTPUT_FILE as FEATURE_FILE
from validate_mt5_history import validate_timeframe
from walk_forward_validation_mt5 import (
    RANDOM_SEED,
    calibration_error,
    calibration_rows,
    metrics,
    safe_auc,
    xgboost_model,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = ROOT / "reports"


def logistic_model(with_missing: bool):
    steps = []
    if with_missing:
        steps.append(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    steps.extend([StandardScaler(), LogisticRegression(max_iter=2_000, random_state=RANDOM_SEED)])
    return make_pipeline(*steps)


def regime_metrics(
    year: int,
    horizon: int,
    model: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    release_day: np.ndarray,
) -> list[dict]:
    rows = []
    for regime, mask in (("macro_release_day", release_day), ("non_release_day", ~release_day)):
        if not mask.any():
            continue
        rows.append(
            {
                "test_year": year,
                "horizon_hours": horizon,
                "model": model,
                "regime": regime,
                "samples": int(mask.sum()),
                "class_1_rate": float(y_true[mask].mean()),
                **metrics(y_true[mask], prediction[mask], probability[mask]),
            }
        )
    return rows


def validate_horizon(
    horizon: int,
    embargo_hours: int = 24,
    first_test_year: int = 2021,
    last_test_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon not in {1, 4}:
        raise ValueError("Horizon must be 1 or 4")
    if embargo_hours < horizon:
        raise ValueError("Embargo must be at least the forecast horizon")
    last_test_year = last_test_year or pd.Timestamp.now(tz="UTC").year
    quality = validate_timeframe("H1", write_report=True)
    if not quality.passed:
        raise RuntimeError("MT5 H1 validation failed: " + "; ".join(quality.failures))
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Run build_features_mt5_macro.py first: {FEATURE_FILE}")

    target = f"target_{horizon}h"
    future_timestamp = f"future_timestamp_{horizon}h"
    data = pd.read_parquet(FEATURE_FILE, engine="pyarrow").sort_values("decision_timestamp").reset_index(drop=True)
    required = FEATURES_MT5 + MACRO_FEATURES + ["decision_timestamp", future_timestamp, target]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError("Macro feature file missing columns: " + ", ".join(missing))
    data["decision_timestamp"] = pd.to_datetime(data["decision_timestamp"], utc=True)
    data[future_timestamp] = pd.to_datetime(data[future_timestamp], utc=True)
    # Never drop a row because macro information was unavailable. Missingness is
    # itself handled causally inside each training fold.
    data = data.dropna(subset=FEATURES_MT5 + ["decision_timestamp", future_timestamp, target]).copy()
    data["label_available_timestamp"] = data[future_timestamp] + pd.Timedelta(hours=1)

    yearly_rows: list[dict] = []
    calibration_output: list[dict] = []
    regime_output: list[dict] = []
    gain_output: list[dict] = []
    permutation_output: list[dict] = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    price_features = FEATURES_MT5
    combined_features = FEATURES_MT5 + MACRO_FEATURES

    print("=" * 78)
    print(f"MT5 PRICE VS POINT-IN-TIME MACRO - {horizon}H")
    print("=" * 78)
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
        cutoff = test["decision_timestamp"].min() - pd.Timedelta(hours=embargo_hours)
        train = data[
            (data["decision_timestamp"] < cutoff)
            & (data["label_available_timestamp"] < cutoff)
        ].copy()
        if len(train) < 1_000 or train[target].nunique() < 2:
            print(f"Skipping {year}: only {len(train):,} eligible training rows")
            continue

        y_train = train[target].astype(int)
        y_true = test[target].astype(int).to_numpy()
        price_lr = logistic_model(with_missing=False)
        macro_lr = logistic_model(with_missing=True)
        price_xgb = xgboost_model()
        macro_xgb = xgboost_model()
        price_lr.fit(train[price_features], y_train)
        price_xgb.fit(train[price_features], y_train)
        macro_lr.fit(train[combined_features], y_train)
        macro_xgb.fit(train[combined_features], y_train)

        probabilities = {
            "price_logistic": price_lr.predict_proba(test[price_features])[:, 1],
            "price_xgboost": price_xgb.predict_proba(test[price_features])[:, 1],
            "price_macro_logistic": macro_lr.predict_proba(test[combined_features])[:, 1],
            "price_macro_xgboost": macro_xgb.predict_proba(test[combined_features])[:, 1],
        }
        release_day = test["macro_release_day"].astype(bool).to_numpy()
        for model_name, probability in probabilities.items():
            probability = np.asarray(probability, dtype=float)
            prediction = (probability >= 0.5).astype(int)
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
                "class_1_rate": float(y_true.mean()),
                **metrics(y_true, prediction, probability),
            }
            yearly_rows.append(row)
            calibration_output.extend(calibration_rows(year, model_name, y_true, probability))
            regime_output.extend(
                regime_metrics(year, horizon, model_name, y_true, prediction, probability, release_day)
            )
            store = pooled.setdefault(model_name, {"truth": [], "prediction": [], "probability": []})
            store["truth"].append(y_true)
            store["prediction"].append(prediction)
            store["probability"].append(probability)

        for feature, importance in zip(combined_features, macro_xgb.feature_importances_):
            gain_output.append(
                {
                    "test_year": year,
                    "horizon_hours": horizon,
                    "feature": feature,
                    "is_macro": feature in MACRO_FEATURES,
                    "gain_importance": float(importance),
                }
            )
        baseline_auc = safe_auc(y_true, probabilities["price_macro_xgboost"])
        for feature_index, feature in enumerate(MACRO_FEATURES):
            shuffled = test[combined_features].copy()
            rng = np.random.default_rng(RANDOM_SEED + year * 1000 + horizon * 100 + feature_index)
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            shuffled_probability = macro_xgb.predict_proba(shuffled)[:, 1]
            permutation_output.append(
                {
                    "test_year": year,
                    "horizon_hours": horizon,
                    "feature": feature,
                    "auc_decrease": baseline_auc - safe_auc(y_true, shuffled_probability),
                }
            )
        year_rows = {row["model"]: row for row in yearly_rows if row["test_year"] == year}
        print(
            f"{year}: train={len(train):,}, test={len(test):,}, "
            f"price XGB BA={year_rows['price_xgboost']['balanced_accuracy']:.4f}, "
            f"macro XGB BA={year_rows['price_macro_xgboost']['balanced_accuracy']:.4f}"
        )

    yearly = pd.DataFrame(yearly_rows)
    if yearly.empty:
        raise RuntimeError(f"No eligible {horizon}H folds were produced")
    summary_rows = []
    metric_columns = [
        "accuracy", "balanced_accuracy", "roc_auc", "precision", "recall", "f1",
        "brier_score", "calibration_error",
    ]
    for model_name, group in yearly.groupby("model", sort=False):
        truth = np.concatenate(pooled[model_name]["truth"])
        prediction = np.concatenate(pooled[model_name]["prediction"])
        probability = np.concatenate(pooled[model_name]["probability"])
        summary_rows.append(
            {
                "model": model_name,
                "horizon_hours": horizon,
                "test_years": ",".join(str(year) for year in sorted(group["test_year"].unique())),
                "years_evaluated": group["test_year"].nunique(),
                "total_test_samples": int(group["test_samples"].sum()),
                **{f"average_{column}": group[column].mean() for column in metric_columns},
                "pooled_accuracy": float((truth == prediction).mean()),
                "pooled_balanced_accuracy": metrics(truth, prediction, probability)["balanced_accuracy"],
                "pooled_roc_auc": safe_auc(truth, probability),
                "pooled_brier_score": metrics(truth, prediction, probability)["brier_score"],
                "pooled_calibration_error": calibration_error(truth, probability),
            }
        )
    summary = pd.DataFrame(summary_rows)
    reference_truth = np.concatenate(next(iter(pooled.values()))["truth"])
    null_probability = float(reference_truth.mean())
    null_brier = float(np.mean((reference_truth - null_probability) ** 2))
    summary["null_brier_score"] = null_brier
    summary["calibration_acceptable"] = (
        (summary["pooled_calibration_error"] <= 0.05)
        & (summary["pooled_brier_score"] <= null_brier)
    )
    comparisons = (("logistic", "price_logistic", "price_macro_logistic"), ("xgboost", "price_xgboost", "price_macro_xgboost"))
    increments = []
    for algorithm, price_name, macro_name in comparisons:
        pair = yearly[yearly["model"].isin([price_name, macro_name])].pivot(
            index="test_year", columns="model", values=["accuracy", "balanced_accuracy", "roc_auc", "brier_score"]
        )
        for year in pair.index:
            increments.append(
                {
                    "test_year": year,
                    "horizon_hours": horizon,
                    "algorithm": algorithm,
                    "accuracy_improvement": pair.loc[year, ("accuracy", macro_name)] - pair.loc[year, ("accuracy", price_name)],
                    "balanced_accuracy_improvement": pair.loc[year, ("balanced_accuracy", macro_name)] - pair.loc[year, ("balanced_accuracy", price_name)],
                    "roc_auc_improvement": pair.loc[year, ("roc_auc", macro_name)] - pair.loc[year, ("roc_auc", price_name)],
                    "brier_score_improvement": pair.loc[year, ("brier_score", price_name)] - pair.loc[year, ("brier_score", macro_name)],
                }
            )
    incremental = pd.DataFrame(increments)
    for algorithm, group in incremental.groupby("algorithm"):
        wins = int((group["balanced_accuracy_improvement"] > 0).sum())
        summary.loc[summary["model"] == f"price_macro_{algorithm}", "macro_balanced_accuracy_yearly_wins"] = wins
        summary.loc[summary["model"] == f"price_macro_{algorithm}", "macro_improvement_consistent_every_year"] = bool(wins == len(group))

    gain = pd.DataFrame(gain_output)
    gain_summary = gain[gain["is_macro"]].groupby("feature")["gain_importance"].agg(
        mean_gain_importance="mean", std_gain_importance="std", folds="count"
    ).reset_index().sort_values("mean_gain_importance", ascending=False)
    permutation = pd.DataFrame(permutation_output)
    permutation_summary = permutation.groupby("feature")["auc_decrease"].agg(
        mean_auc_decrease="mean", std_auc_decrease="std", folds="count"
    ).reset_index().sort_values("mean_auc_decrease", ascending=False)

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    suffix = f"{horizon}h"
    outputs = {
        REPORT_DIRECTORY / f"mt5_macro_validation_summary_{suffix}.csv": summary,
        REPORT_DIRECTORY / f"mt5_macro_yearly_results_{suffix}.csv": yearly,
        REPORT_DIRECTORY / f"mt5_macro_incremental_results_{suffix}.csv": incremental,
        REPORT_DIRECTORY / f"mt5_macro_calibration_{suffix}.csv": pd.DataFrame(calibration_output),
        REPORT_DIRECTORY / f"mt5_macro_regime_results_{suffix}.csv": pd.DataFrame(regime_output),
        REPORT_DIRECTORY / f"mt5_macro_feature_importance_{suffix}.csv": gain_summary,
        REPORT_DIRECTORY / f"mt5_macro_permutation_importance_{suffix}.csv": permutation_summary,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
    display = summary[[
        "model", "average_accuracy", "average_balanced_accuracy", "average_roc_auc",
        "average_brier_score", "average_calibration_error", "calibration_acceptable",
    ]].sort_values("average_balanced_accuracy", ascending=False)
    print(display.to_string(index=False))
    xgb_increment = incremental[incremental["algorithm"] == "xgboost"]
    print(
        f"Macro XGBoost BA improvement: {xgb_increment['balanced_accuracy_improvement'].mean():+.6f}; "
        f"yearly wins={int((xgb_increment['balanced_accuracy_improvement'] > 0).sum())}/{len(xgb_increment)}"
    )
    print("EDGE_VERDICT: NOT ESTABLISHED unless improvement is positive in every year and other metrics agree")
    return summary, yearly


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", choices=("1", "4", "both"), default="both")
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--first-test-year", type=int, default=2021)
    parser.add_argument("--last-test-year", type=int, default=pd.Timestamp.now(tz="UTC").year)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    horizons = (1, 4) if args.horizon == "both" else (int(args.horizon),)
    try:
        for selected_horizon in horizons:
            validate_horizon(
                selected_horizon,
                embargo_hours=args.embargo_hours,
                first_test_year=args.first_test_year,
                last_test_year=args.last_test_year,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
