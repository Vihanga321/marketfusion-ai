"""Search the processed dataset and model feature list for leakage signals."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_model import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "eurusd_features.csv"
MODEL_FILE = ROOT / "models" / "eurusd_xgboost_v01.pkl"
OUTPUT_FILE = ROOT / "reports" / "leakage_analysis.csv"
SUSPICIOUS_TOKENS = ("future", "target", "next", "lead", "forward", "tomorrow", "t_plus")
TEST_YEARS = range(2021, 2027)


def safe_auc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else np.nan


def univariate_walk_forward(df: pd.DataFrame, feature: str) -> tuple[float, float, int]:
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    folds = 0
    for year in TEST_YEARS:
        test = df[df["date"].dt.year == year]
        if test.empty:
            continue
        test_start = test["date"].min()
        # Purge the final training row when its label is realized in the test period.
        train = df[(df["date"] < test_start) & (df["target_date"] < test_start)]
        train = train.dropna(subset=[feature, "target"])
        test = test.dropna(subset=[feature, "target"])
        if train.empty or test.empty or train["target"].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000, random_state=42))
        model.fit(train[[feature]], train["target"])
        probability = model.predict_proba(test[[feature]])[:, 1]
        truths.append(test["target"].to_numpy())
        probabilities.append(probability)
        predictions.append((probability >= 0.5).astype(int))
        folds += 1
    if not truths:
        return np.nan, np.nan, 0
    y = np.concatenate(truths)
    pred = np.concatenate(predictions)
    prob = np.concatenate(probabilities)
    return float(accuracy_score(y, pred)), safe_auc(pd.Series(y), prob), folds


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)
    df["target_date"] = df["date"].shift(-1)

    artifact = joblib.load(MODEL_FILE)
    artifact_features = list(artifact.get("features", [])) if isinstance(artifact, dict) else []
    code_features = list(FEATURES)
    model_features = artifact_features or code_features

    target = df["target"]
    numeric_columns = df.select_dtypes(include=np.number).columns
    target_correlations = df[numeric_columns].corr(numeric_only=True)["target"]
    future_correlations = (
        df[numeric_columns].corr(numeric_only=True)["future_return"]
        if "future_return" in numeric_columns
        else pd.Series(dtype=float)
    )

    rows = []
    for column in df.columns:
        suspicious_name = any(token in column.lower() for token in SUSPICIOUS_TOKENS)
        exact_target_match = False
        exact_target_inverse = False
        if column in numeric_columns and column != "target":
            valid = df[column].notna() & target.notna()
            values = df.loc[valid, column]
            if set(values.unique()).issubset({0, 1}):
                exact_target_match = bool((values.astype(int) == target[valid]).all())
                exact_target_inverse = bool((values.astype(int) == 1 - target[valid]).all())
        uni_accuracy = uni_auc = np.nan
        folds = 0
        if column in model_features:
            uni_accuracy, uni_auc, folds = univariate_walk_forward(df, column)
        rows.append(
            {
                "feature": column,
                "used_by_code": column in code_features,
                "used_by_saved_model": column in artifact_features,
                "suspicious_name": suspicious_name,
                "exact_target_match": exact_target_match,
                "exact_target_inverse": exact_target_inverse,
                "target_correlation": target_correlations.get(column, np.nan),
                "future_return_correlation": future_correlations.get(column, np.nan),
                "univariate_oof_accuracy": uni_accuracy,
                "univariate_oof_roc_auc": uni_auc,
                "univariate_folds": folds,
            }
        )

    results = pd.DataFrame(rows)
    results["absolute_target_correlation"] = results["target_correlation"].abs()
    results["absolute_future_return_correlation"] = results["future_return_correlation"].abs()
    results["suspicious_univariate_power"] = (
        results["univariate_oof_accuracy"].ge(0.65)
        | results["univariate_oof_roc_auc"].ge(0.70)
        | results["univariate_oof_roc_auc"].le(0.30)
    )
    results = results.sort_values(
        ["suspicious_univariate_power", "absolute_target_correlation"], ascending=[False, False]
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    forbidden = [feature for feature in model_features if any(token in feature.lower() for token in SUSPICIOUS_TOKENS)]
    missing_from_data = sorted(set(model_features).difference(df.columns))
    code_artifact_difference = sorted(set(code_features).symmetric_difference(artifact_features))
    print("=" * 72)
    print("MARKETFUSION AI - LEAKAGE CHECK")
    print("=" * 72)
    print(f"Code feature count: {len(code_features)}")
    print(f"Saved-model feature count: {len(artifact_features)}")
    print(f"Code/artifact feature differences: {code_artifact_difference or 'none'}")
    print(f"Model features missing from data: {missing_from_data or 'none'}")
    print(f"Forbidden future/target-named model features: {forbidden or 'none'}")
    print()
    print("Largest absolute target correlations:")
    display_columns = [
        "feature", "used_by_saved_model", "target_correlation",
        "future_return_correlation", "univariate_oof_accuracy", "univariate_oof_roc_auc",
    ]
    print(results.nlargest(15, "absolute_target_correlation")[display_columns].to_string(index=False))
    print()
    print("Strongest single-feature walk-forward models:")
    univariate = results[results["used_by_saved_model"]].sort_values("univariate_oof_roc_auc", ascending=False)
    print(univariate[display_columns].head(20).to_string(index=False))
    print()
    if forbidden:
        print("CRITICAL: Direct future/target columns are included in the model feature list.")
    else:
        print("PASS: future_close, future_return, and target are excluded from the explicit model feature list.")
    suspicious = univariate[univariate["suspicious_univariate_power"]]
    if not suspicious.empty:
        print("WARNING: Suspicious standalone predictive power found in: " + ", ".join(suspicious["feature"]))
    print("WARNING: Same-day OHLC features are only causal if the complete daily bar exists before the forecast decision.")
    print(f"Saved detailed leakage analysis: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
