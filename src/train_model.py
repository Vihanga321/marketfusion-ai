import pandas as pd
import joblib

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DATA_FILE = Path(
    "data/processed/eurusd_features.csv"
)

MODEL_FILE = Path(
    "models/eurusd_xgboost_v01.pkl"
)


FEATURES = [

    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",

    "range",
    "body",
    "upper_wick",
    "lower_wick",

    "price_vs_ma10",
    "price_vs_ma20",
    "ma10_vs_ma20",

    "rsi_14",

    "volatility_5",
    "volatility_10",
    "volatility_20",

    "atr_14",

    "momentum_3",
    "momentum_5",
    "momentum_10",

    "day_of_week",
    "month",
]


def train():

    print("=" * 60)
    print("MARKETFUSION AI - BRAIN V0.1")
    print("=" * 60)

    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)


    # IMPORTANT:
    # Time-based split.
    # Do NOT randomly shuffle forex data.

    split_index = int(
        len(df) * 0.80
    )

    train_df = df.iloc[
        :split_index
    ]

    test_df = df.iloc[
        split_index:
    ]


    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]


    print()
    print(
        "Training:"
        f" {train_df['date'].min().date()}"
        " -> "
        f"{train_df['date'].max().date()}"
    )

    print(
        "Testing:"
        f" {test_df['date'].min().date()}"
        " -> "
        f"{test_df['date'].max().date()}"
    )


    model = XGBClassifier(

        n_estimators=400,

        max_depth=4,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,
    )


    print()
    print("Training AI...")

    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )


    print()
    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"ROC AUC:  {auc:.4f}"
    )

    print("=" * 60)


    print()
    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_FILE,
    )


    print()
    print(
        f"Model saved: {MODEL_FILE}"
    )


    # Latest prediction

    latest = df.iloc[[-1]]

    latest_probability = (
        model.predict_proba(
            latest[FEATURES]
        )[0][1]
    )


    print()
    print("=" * 60)
    print("LATEST MODEL SIGNAL")
    print("=" * 60)

    print(
        f"Date: {latest['date'].iloc[0]}"
    )

    print(
        f"Close: {latest['close'].iloc[0]:.5f}"
    )

    print(
        f"UP probability:"
        f" {latest_probability * 100:.2f}%"
    )

    print(
        f"DOWN probability:"
        f" {(1-latest_probability) * 100:.2f}%"
    )

    if latest_probability >= 0.60:

        print(
            "Signal: BULLISH"
        )

    elif latest_probability <= 0.40:

        print(
            "Signal: BEARISH"
        )

    else:

        print(
            "Signal: NO TRADE"
        )


if __name__ == "__main__":
    train()