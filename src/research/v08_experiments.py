"""Manual controlled V0.8 model-improvement experiments with untouched holdout."""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from src.evaluation.v08_contract import PREDEFINED_COST_MULTIPLIERS, SCORECARD_REPORT
from src.learning.v05c_contract import COST_BAND_MULTIPLIER, HORIZONS, MARKET_FEATURES, RANDOM_STATE, SOURCE_FILE
from src.learning.v05c_dataset import build_horizon_dataset, target_class
from src.learning.v05c_models import multiclass_brier
from src.research.v08_abstention import elastic_net_estimator, two_stage_probabilities
from src.research.v08_feature_ablation import FEATURE_GROUPS, ablated_features, validate_groups
from src.research.v08_scorecard import SCORECARD_COLUMNS, write_scorecard


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame
    holdout_opened: bool = False


def chronological_split(frame: pd.DataFrame, horizon: int) -> TemporalSplit:
    ordered = frame.sort_values("decision_timestamp_utc").reset_index(drop=True)
    if len(ordered) < 600:
        return TemporalSplit(ordered.iloc[:0], ordered.iloc[:0], ordered.iloc[:0])
    validation_start = int(len(ordered) * 0.70)
    holdout_start = int(len(ordered) * 0.85)
    validation_time = pd.Timestamp(ordered.iloc[validation_start]["decision_timestamp_utc"])
    holdout_time = pd.Timestamp(ordered.iloc[holdout_start]["decision_timestamp_utc"])
    maturity = pd.to_datetime(ordered["target_matured_at_utc"], utc=True)
    train = ordered.loc[maturity.lt(validation_time)].copy()
    validation = ordered.iloc[validation_start:holdout_start].loc[maturity.iloc[validation_start:holdout_start].lt(holdout_time)].copy()
    holdout = ordered.iloc[holdout_start:].copy()
    if not train["decision_timestamp_utc"].lt(validation["decision_timestamp_utc"].min()).all():
        raise ValueError("Chronological research split violated")
    if not validation["decision_timestamp_utc"].lt(holdout["decision_timestamp_utc"].min()).all():
        raise ValueError("Final holdout overlaps selection data")
    return TemporalSplit(train, validation, holdout)


def experiment_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"id": "ELASTIC_NET_ALL", "family": "ELASTIC_NET_LOGISTIC", "features": tuple(MARKET_FEATURES), "target_multiplier": COST_BAND_MULTIPLIER},
        {"id": "TWO_STAGE_NO_TRADE", "family": "TWO_STAGE_LOGISTIC", "features": tuple(MARKET_FEATURES), "target_multiplier": COST_BAND_MULTIPLIER},
        {"id": "HIST_GB_FIXED", "family": "HIST_GRADIENT_BOOSTING", "features": tuple(MARKET_FEATURES), "target_multiplier": COST_BAND_MULTIPLIER},
        {"id": "RANDOM_FOREST_BASELINE", "family": "RANDOM_FOREST_BASELINE", "features": tuple(MARKET_FEATURES), "target_multiplier": COST_BAND_MULTIPLIER},
    ]
    specs.extend({"id": f"ABLATE_{group}", "family": "ELASTIC_NET_LOGISTIC", "features": ablated_features(group), "target_multiplier": COST_BAND_MULTIPLIER} for group in FEATURE_GROUPS)
    specs.extend({"id": f"COST_BAND_{str(value).replace('.', '_')}", "family": "ELASTIC_NET_LOGISTIC", "features": tuple(MARKET_FEATURES), "target_multiplier": value} for value in PREDEFINED_COST_MULTIPLIERS)
    return specs


def _labels(frame: pd.DataFrame, multiplier: float) -> pd.Series:
    adjusted = frame["decision_cost_band"] * float(multiplier) / COST_BAND_MULTIPLIER
    return target_class(frame["raw_future_return"], adjusted).astype(int)


def _fit_predict(family: str, features: tuple[str, ...], train: pd.DataFrame, train_y: pd.Series, evaluation: pd.DataFrame) -> np.ndarray:
    x_train, x_eval = train.loc[:, list(features)], evaluation.loc[:, list(features)]
    if family == "TWO_STAGE_LOGISTIC":
        return two_stage_probabilities(x_train, train_y, x_eval)
    if family == "ELASTIC_NET_LOGISTIC":
        model = elastic_net_estimator()
    elif family == "HIST_GRADIENT_BOOSTING":
        model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("classifier", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, random_state=RANDOM_STATE))])
    elif family == "RANDOM_FOREST_BASELINE":
        model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE))])
    else:
        raise ValueError(f"Unknown isolated research family: {family}")
    model.fit(x_train, train_y)
    raw = model.predict_proba(x_eval)
    aligned = np.full((len(evaluation), 3), 1e-12)
    for index, value in enumerate(model.classes_):
        aligned[:, int(value)] = raw[:, index]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _metrics(actual: pd.Series, probabilities: np.ndarray, raw: pd.Series, cost: pd.Series) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    acted = predicted != 1
    direction = predicted - 1
    return {
        "ba": float(balanced_accuracy_score(actual, predicted)),
        "brier": multiclass_brier(actual, probabilities),
        "coverage": float(acted.mean()), "wait_rate": float((~acted).mean()),
        "cost": float(np.mean(direction[acted] * raw.to_numpy()[acted] - cost.to_numpy()[acted])) if acted.any() else float("nan"),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
    }


def research_decision(development: dict[str, float], holdout: dict[str, float]) -> str:
    """A challenger cannot be promising when its untouched holdout cost metric is non-positive."""
    promising = (
        development["ba"] > 1 / 3
        and holdout["ba"] > 1 / 3
        and holdout["ba"] >= development["ba"] - 0.05
        and np.isfinite(holdout["cost"])
        and holdout["cost"] > 0
    )
    return "PROMISING_FOR_V05C_CHALLENGER" if promising else "REJECT"


def _linear_feature_analysis(train: pd.DataFrame, labels: pd.Series, features: tuple[str, ...]) -> dict[str, object]:
    """Development-training-only coefficient ranking with an expanding-fit stability check."""
    def ranking(frame: pd.DataFrame, target: pd.Series) -> list[str]:
        model = elastic_net_estimator().fit(frame.loc[:, list(features)], target)
        coefficients = np.abs(model.named_steps["classifier"].coef_[:, :len(features)]).mean(axis=0)
        order = np.argsort(-coefficients, kind="stable")
        return [features[index] for index in order[:min(10, len(features))]]

    full = ranking(train, labels)
    cutoff = max(100, int(len(train) * 0.7))
    early = ranking(train.iloc[:cutoff], labels.iloc[:cutoff])
    overlap = len(set(full) & set(early)) / max(1, len(full))
    return {
        "method": "ELASTIC_NET_ABS_COEFFICIENT_DEVELOPMENT_TRAINING_ONLY",
        "top_features": full, "expanding_fit_top10_overlap": overlap,
        "stability": "STABLE" if overlap >= 0.7 else "WATCH",
    }


def run_model_research() -> dict[str, object]:
    validate_groups()
    source = pd.read_parquet(SOURCE_FILE)
    score_rows: list[dict[str, object]] = []
    audit: dict[str, object] = {"selection_uses_holdout": False, "random_shuffle": False, "v05b_active": False, "horizons": {}, "feature_analysis": {}}
    for horizon in HORIZONS:
        dataset = build_horizon_dataset(source, horizon)
        split = chronological_split(dataset.frame, horizon)
        if split.train.empty or split.validation.empty or split.holdout.empty:
            for spec in experiment_specs():
                score_rows.append({"experiment_id": f"H{horizon}_{spec['id']}", "horizon": horizon, "model_family": spec["family"], "target_contract": f"PREDEFINED_COST_{spec['target_multiplier']}", "feature_groups": "MARKET_CORE", "development_rows": len(split.train) + len(split.validation), "holdout_rows": len(split.holdout), "decision": "INSUFFICIENT_DATA"})
            continue
        selection: list[tuple[float, dict[str, object], dict[str, float]]] = []
        for spec in experiment_specs():
            features = tuple(spec["features"])
            train_y = _labels(split.train, float(spec["target_multiplier"]))
            validation_y = _labels(split.validation, float(spec["target_multiplier"]))
            probabilities = _fit_predict(str(spec["family"]), features, split.train, train_y, split.validation)
            metrics = _metrics(validation_y, probabilities, split.validation["raw_future_return"], split.validation["decision_cost_band"])
            selection.append((metrics["ba"], spec, metrics))
        _, best_spec, _ = max(selection, key=lambda item: (item[0], str(item[1]["id"])))
        combined = pd.concat([split.train, split.validation], ignore_index=True)
        combined_y = _labels(combined, float(best_spec["target_multiplier"]))
        holdout_y = _labels(split.holdout, float(best_spec["target_multiplier"]))
        holdout_probs = _fit_predict(str(best_spec["family"]), tuple(best_spec["features"]), combined, combined_y, split.holdout)
        holdout_metrics = _metrics(holdout_y, holdout_probs, split.holdout["raw_future_return"], split.holdout["decision_cost_band"])
        audit["horizons"][str(horizon)] = {"selected_without_holdout": best_spec["id"], "train_end": str(split.train["decision_timestamp_utc"].max()), "validation_end": str(split.validation["decision_timestamp_utc"].max()), "holdout_start": str(split.holdout["decision_timestamp_utc"].min())}
        if best_spec["family"] == "ELASTIC_NET_LOGISTIC":
            audit["feature_analysis"][str(horizon)] = _linear_feature_analysis(split.train, _labels(split.train, float(best_spec["target_multiplier"])), tuple(best_spec["features"]))
        else:
            audit["feature_analysis"][str(horizon)] = {"method": "NOT_RUN_FOR_SELECTED_FAMILY", "stability": "INSUFFICIENT_DATA"}
        for _, spec, metrics in selection:
            selected = spec["id"] == best_spec["id"]
            decision = research_decision(metrics, holdout_metrics) if selected else "REJECT"
            score_rows.append({
                "experiment_id": f"H{horizon}_{spec['id']}", "horizon": horizon, "model_family": spec["family"],
                "target_contract": f"PREDEFINED_COST_{spec['target_multiplier']}", "feature_groups": "MARKET_CORE_ONLY:" + ("ALL" if len(spec["features"]) == len(MARKET_FEATURES) else f"{len(spec['features'])}_FEATURES"),
                "development_rows": len(split.train) + len(split.validation), "holdout_rows": len(split.holdout) if selected else 0,
                "development_BA": metrics["ba"], "holdout_BA": holdout_metrics["ba"] if selected else None,
                "development_brier": metrics["brier"], "holdout_brier": holdout_metrics["brier"] if selected else None,
                "coverage": holdout_metrics["coverage"] if selected else metrics["coverage"],
                "wait_rate": holdout_metrics["wait_rate"] if selected else metrics["wait_rate"],
                "cost_aware_metric": holdout_metrics["cost"] if selected else metrics["cost"],
                "stability": "UNTOUCHED_HOLDOUT_EVALUATED" if selected else "DEVELOPMENT_ONLY",
                "decision": decision,
            })
    scorecard = pd.DataFrame(score_rows, columns=SCORECARD_COLUMNS)
    write_scorecard(scorecard, SCORECARD_REPORT)
    audit_path = SCORECARD_REPORT.with_name("v08_model_research_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    promising = int(scorecard["decision"].eq("PROMISING_FOR_V05C_CHALLENGER").sum())
    evaluated = scorecard.loc[pd.to_numeric(scorecard["holdout_rows"], errors="coerce").fillna(0).gt(0)]
    best = None if evaluated.empty else str(evaluated.sort_values("holdout_BA", ascending=False).iloc[0]["experiment_id"])
    return {"status": "PASS_RESEARCH_ONLY", "experiments": len(scorecard), "promising": promising, "rejected": int(scorecard["decision"].eq("REJECT").sum()), "insufficient": int(scorecard["decision"].eq("INSUFFICIENT_DATA").sum()), "best_experiment": best, "promotion": "NOT_PERMITTED_V08"}


if __name__ == "__main__":
    print(json.dumps(run_model_research(), indent=2, default=str))
