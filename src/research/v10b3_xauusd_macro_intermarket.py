"""MarketFusion V1.0B.3 XAUUSD macro/intermarket evidence research.

Price-only evidence was weak in V1.0B.2.  This phase tests whether verified,
causally available Federal Reserve market context adds stable directional
evidence.  It is research-only: current-vintage FRED data can never promote a
production champion and the already exposed final tail remains quarantined.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.assets.contracts import ROOT, asset_paths
from src.intelligence.gold_macro_context import (
    CONTEXT_SERIES,
    context_feature_names,
    join_context_asof,
    load_gold_macro_context,
    refresh_gold_macro_context,
)
from src.learning.v05c_walk_forward import purged_walk_forward
from src.research.v10b_xauusd_models import ASSET_ID, FEATURE_GROUPS, load_xauusd_research_frame, validate_feature_columns
from src.research.v10b1_xauusd_targets import TargetContract
from src.research.v10b2_xauusd_feature_evidence import _directional_frame, quarantine_split

CONTRACT_VERSION = "v1.0b3-xauusd-macro-intermarket-evidence-v1"
RANDOM_STATE = 2003
N_SPLITS = 5
MIN_VALIDATION_ROWS = 300
MIN_RESEARCH_ROWS = 3_000
MIN_MEAN_BALANCED_ACCURACY = 0.515
MIN_RECENT_BALANCED_ACCURACY = 0.51
MIN_FOLDS_BEATING_PRIOR = 3
MIN_LOG_LOSS_IMPROVEMENT_VS_PRIOR = 0.002
MIN_DELTA_VS_PRICE_BASELINE = 0.003

TARGET_SPECS: tuple[tuple[int, TargetContract], ...] = (
    (15, TargetContract("V10B_REFERENCE", 1.5, 0.10, "Predeclared V1.0B.2 15m target.")),
    (30, TargetContract("SELECTIVE_MEDIUM", 2.0, 0.20, "Predeclared V1.0B.2 30m target.")),
)

PRICE_CORE = FEATURE_GROUPS["MARKET_CORE"]
ALL_PRICE_GROUPS = (
    "MARKET_CORE", "TECHNICAL", "PRICE_ACTION", "STRUCTURE", "LIQUIDITY",
    "SUPPORT_RESISTANCE", "VOLATILITY", "SESSION", "REGIME", "PATTERNS",
)


def _ordered(groups: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for group in groups:
        for name in FEATURE_GROUPS[group]:
            if name not in values:
                values.append(name)
    result = tuple(values)
    validate_feature_columns(result)
    return result


ALL_PRICE_FEATURES = _ordered(ALL_PRICE_GROUPS)
CONTEXT_FEATURES = context_feature_names()
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "PRICE_CORE": tuple(PRICE_CORE),
    "PRICE_CORE_PLUS_MACRO": tuple(dict.fromkeys((*PRICE_CORE, *CONTEXT_FEATURES))),
    "ALL_PRICE": ALL_PRICE_FEATURES,
    "ALL_PRICE_PLUS_MACRO": tuple(dict.fromkeys((*ALL_PRICE_FEATURES, *CONTEXT_FEATURES))),
}
MODEL_FAMILIES = ("LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING")


def _validate_features(names: tuple[str, ...]) -> None:
    for name in names:
        lower = name.lower()
        if lower.startswith(("target_", "outcome_", "future_")) or "future" in lower:
            raise ValueError(f"Future/target field cannot enter V1.0B.3: {name}")


def _pipeline(family: str) -> Pipeline:
    if family == "LOGISTIC_REGRESSION":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
    if family == "HIST_GRADIENT_BOOSTING":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
                l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unsupported V1.0B.3 family: {family}")


def _probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(x), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("V1.0B.3 requires binary DOWN/UP probabilities")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid V1.0B.3 probability matrix")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("V1.0B.3 probabilities do not sum to one")
    return values


def _class_prior(train_y: pd.Series, rows: int) -> np.ndarray:
    counts = train_y.value_counts().reindex([0, 1], fill_value=0).astype(float)
    prior = ((counts + 1.0) / (counts.sum() + 2.0)).to_numpy(dtype=float)
    return np.tile(prior, (rows, 1))


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = frame["directional_target"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    raw = pd.to_numeric(frame["raw_future_return"], errors="raise").to_numpy(dtype=float)
    band = pd.to_numeric(frame["decision_wait_band"], errors="raise").to_numpy(dtype=float)
    direction = np.where(predicted == 1, 1.0, -1.0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(np.mean((probabilities[:, 1] - y) ** 2)),
        "cost_aware_metric": float(np.mean(direction * raw - band)),
    }


def evaluate_set(research: pd.DataFrame, horizon: int, feature_set: str, family: str) -> dict[str, Any]:
    names = FEATURE_SETS[feature_set]
    _validate_features(names)
    missing = sorted(set(names) - set(research.columns))
    if missing:
        return {"status": "MISSING_FEATURES", "missing": missing, "feature_set": feature_set, "family": family}
    complete = research.loc[:, list(names)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    data = research.loc[complete].reset_index(drop=True)
    if len(data) < MIN_RESEARCH_ROWS:
        return {"status": "INSUFFICIENT_DATA", "rows": len(data), "feature_set": feature_set, "family": family}
    folds = purged_walk_forward(
        data["decision_timestamp_utc"], horizon,
        n_splits=N_SPLITS, min_validation_rows=MIN_VALIDATION_ROWS,
    )
    fold_metrics: list[dict[str, float]] = []
    prior_metrics: list[dict[str, float]] = []
    for fold in folds:
        train = data.iloc[fold.train_indices]
        validation = data.iloc[fold.validation_indices]
        if train["directional_target"].nunique() < 2 or validation["directional_target"].nunique() < 2:
            return {"status": "INSUFFICIENT_CLASS_VARIATION", "feature_set": feature_set, "family": family}
        model = _pipeline(family)
        model.fit(train.loc[:, list(names)], train["directional_target"])
        fold_metrics.append(_metrics(validation, _probabilities(model, validation.loc[:, list(names)])))
        prior_metrics.append(_metrics(validation, _class_prior(train["directional_target"], len(validation))))

    def mean(key: str, rows: list[dict[str, float]]) -> float:
        return float(np.mean([row[key] for row in rows]))

    return {
        "status": "EVALUATED",
        "feature_set": feature_set,
        "family": family,
        "feature_count": len(names),
        "rows": len(data),
        "fold_count": len(folds),
        "folds_beating_class_prior": int(sum(
            row["balanced_accuracy"] > prior["balanced_accuracy"]
            for row, prior in zip(fold_metrics, prior_metrics)
        )),
        "balanced_accuracy_mean": mean("balanced_accuracy", fold_metrics),
        "macro_f1_mean": mean("macro_f1", fold_metrics),
        "log_loss_mean": mean("log_loss", fold_metrics),
        "brier_mean": mean("brier", fold_metrics),
        "cost_aware_mean": mean("cost_aware_metric", fold_metrics),
        "class_prior_log_loss_mean": mean("log_loss", prior_metrics),
        "recent_fold_balanced_accuracy": float(fold_metrics[-1]["balanced_accuracy"]),
        "folds": fold_metrics,
    }


def evidence_gate(row: dict[str, Any], price_baseline: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("status") != "EVALUATED":
        return "NO_MACRO_EVIDENCE", [str(row.get("status"))]
    failed: list[str] = []
    if row["balanced_accuracy_mean"] < MIN_MEAN_BALANCED_ACCURACY:
        failed.append("mean_balanced_accuracy")
    if row["recent_fold_balanced_accuracy"] < MIN_RECENT_BALANCED_ACCURACY:
        failed.append("recent_fold_balanced_accuracy")
    if row["folds_beating_class_prior"] < MIN_FOLDS_BEATING_PRIOR:
        failed.append("fold_stability")
    if row["log_loss_mean"] > row["class_prior_log_loss_mean"] - MIN_LOG_LOSS_IMPROVEMENT_VS_PRIOR:
        failed.append("log_loss")
    delta = float(row["balanced_accuracy_mean"] - price_baseline["balanced_accuracy_mean"])
    if delta < MIN_DELTA_VS_PRICE_BASELINE:
        failed.append("delta_vs_price_only")
    return ("MACRO_CONTEXT_EVIDENCE_CANDIDATE" if not failed else "NO_MACRO_EVIDENCE", failed)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def run_research(root: Path = ROOT, persist: bool = True, refresh_context: bool = True) -> dict[str, Any]:
    source = load_xauusd_research_frame(root)
    context_manifest: dict[str, Any] | None = None
    if refresh_context:
        context, context_manifest = refresh_gold_macro_context(root)
    else:
        context = load_gold_macro_context(root)

    joined = join_context_asof(source, context)
    results: list[dict[str, Any]] = []
    quarantine: dict[str, Any] = {}

    for horizon, contract in TARGET_SPECS:
        label = f"{horizon}m:{contract.name}"
        directional = _directional_frame(joined, horizon, contract)
        research, quarantine_info = quarantine_split(directional, horizon)
        quarantine[label] = quarantine_info
        target_rows: list[dict[str, Any]] = []
        for family in MODEL_FAMILIES:
            for set_name in FEATURE_SETS:
                row = evaluate_set(research, horizon, set_name, family)
                row.update({
                    "asset_id": ASSET_ID,
                    "target_label": label,
                    "horizon_minutes": horizon,
                    "target_contract": asdict(contract),
                    "quarantine_evaluated": False,
                    "production_model_eligible": False,
                })
                target_rows.append(row)
            family_rows = [r for r in target_rows if r["family"] == family and r.get("status") == "EVALUATED"]
            baseline = next((r for r in family_rows if r["feature_set"] == "PRICE_CORE"), None)
            if baseline is None:
                continue
            for row in family_rows:
                row["delta_balanced_accuracy_vs_price_core"] = float(
                    row["balanced_accuracy_mean"] - baseline["balanced_accuracy_mean"]
                )
                if "MACRO" in row["feature_set"]:
                    status, failed = evidence_gate(row, baseline)
                    row["evidence_status"] = status
                    row["failed_gates"] = failed
        results.extend(target_rows)

    candidates = sorted(
        [row for row in results if row.get("evidence_status") == "MACRO_CONTEXT_EVIDENCE_CANDIDATE"],
        key=lambda row: (row["balanced_accuracy_mean"], -row["log_loss_mean"]),
        reverse=True,
    )
    decision = "MACRO_CONTEXT_EVIDENCE_FOUND" if candidates else "MACRO_CONTEXT_EVIDENCE_WEAK"
    report = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "verified_series": [asdict(spec) for spec in CONTEXT_SERIES],
        "provider_manifest": context_manifest,
        "current_vintage_data": True,
        "vintage_safe": False,
        "production_model_eligible": False,
        "target_specs": [{"horizon_minutes": h, "contract": asdict(c)} for h, c in TARGET_SPECS],
        "quarantine": quarantine,
        "results": results,
        "macro_context_candidate_count": len(candidates),
        "best_macro_context_candidate": candidates[0] if candidates else None,
        "decision": decision,
        "next_phase": (
            "ACQUIRE_VINTAGE_SAFE_OR_INTRADAY_INTERMARKET_DATA"
            if candidates else "EXPAND_VERIFIED_GOLD_CONTEXT_BEFORE_MODEL_COMPLEXITY"
        ),
        "production_integration": False,
        "model_promotion_performed": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }
    if persist:
        reports = asset_paths(ASSET_ID, root=root).reports
        _atomic_json(reports / "v10b3_macro_intermarket_evidence.json", report)
        flat: list[dict[str, Any]] = []
        for row in results:
            flat.append({
                "target_label": row.get("target_label"),
                "horizon_minutes": row.get("horizon_minutes"),
                "family": row.get("family"),
                "feature_set": row.get("feature_set"),
                "feature_count": row.get("feature_count"),
                "status": row.get("status"),
                "evidence_status": row.get("evidence_status"),
                "failed_gates": ";".join(row.get("failed_gates") or []),
                "rows": row.get("rows"),
                "folds_beating_class_prior": row.get("folds_beating_class_prior"),
                "balanced_accuracy_mean": row.get("balanced_accuracy_mean"),
                "macro_f1_mean": row.get("macro_f1_mean"),
                "log_loss_mean": row.get("log_loss_mean"),
                "brier_mean": row.get("brier_mean"),
                "cost_aware_mean": row.get("cost_aware_mean"),
                "recent_fold_balanced_accuracy": row.get("recent_fold_balanced_accuracy"),
                "delta_balanced_accuracy_vs_price_core": row.get("delta_balanced_accuracy_vs_price_core"),
            })
        _write_csv(reports / "v10b3_macro_intermarket_leaderboard.csv", flat)
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use existing verified context store; do not refresh FRED")
    args = parser.parse_args()
    report = run_research(ROOT, persist=not args.no_persist, refresh_context=not args.offline)
    print(json.dumps({
        "contract_version": report["contract_version"],
        "asset_id": report["asset_id"],
        "decision": report["decision"],
        "macro_context_candidate_count": report["macro_context_candidate_count"],
        "best_macro_context_candidate": report["best_macro_context_candidate"],
        "next_phase": report["next_phase"],
        "vintage_safe": False,
        "production_integration": False,
        "automatic_execution": "DISABLED",
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
