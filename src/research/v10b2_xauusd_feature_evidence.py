"""MarketFusion V1.0B.2 XAUUSD feature-evidence research.

Research-only phase. It keeps the V1.0B.1 final tail quarantined and asks whether
any existing causal XAUUSD feature family adds directional evidence before we
introduce macro/intermarket data or larger models.
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
from src.learning.v05c_walk_forward import purged_walk_forward
from src.research.v10b_xauusd_models import ASSET_ID, FEATURE_GROUPS, load_xauusd_research_frame, validate_feature_columns
from src.research.v10b1_xauusd_targets import TargetContract, apply_target_contract, exact_future_frame

CONTRACT_VERSION = "v1.0b2-xauusd-feature-evidence-v1"
RANDOM_STATE = 1901
QUARANTINE_FRACTION = 0.15
MIN_QUARANTINE_ROWS = 750
MIN_RESEARCH_ROWS = 3_000
N_SPLITS = 5
MIN_VALIDATION_ROWS = 300
MIN_MEAN_BALANCED_ACCURACY = 0.515
MIN_RECENT_BALANCED_ACCURACY = 0.51
MIN_FOLDS_BEATING_PRIOR = 3
MIN_LOG_LOSS_IMPROVEMENT = 0.002
MIN_GROUP_DELTA_BALANCED_ACCURACY = 0.002

TARGET_SPECS: tuple[tuple[int, TargetContract], ...] = (
    (15, TargetContract("V10B_REFERENCE", 1.5, 0.10, "Best V1.0B.1 walk-forward target.")),
    (30, TargetContract("SELECTIVE_MEDIUM", 2.0, 0.20, "Best V1.0B.1 final-holdout target.")),
)

GROUP_ORDER = (
    "TECHNICAL", "PRICE_ACTION", "STRUCTURE", "LIQUIDITY",
    "SUPPORT_RESISTANCE", "VOLATILITY", "SESSION", "REGIME", "PATTERNS",
)

FEATURE_SET_GROUPS: dict[str, tuple[str, ...]] = {
    "MARKET_CORE": ("MARKET_CORE",),
    **{f"MARKET_CORE_PLUS_{group}": ("MARKET_CORE", group) for group in GROUP_ORDER},
    "STRUCTURE_VOLATILITY": ("MARKET_CORE", "STRUCTURE", "VOLATILITY"),
    "STRUCTURE_LIQUIDITY": ("MARKET_CORE", "STRUCTURE", "LIQUIDITY"),
    "TECHNICAL_REGIME": ("MARKET_CORE", "TECHNICAL", "REGIME"),
    "ALL_EXISTING_PRICE_CONTEXT": ("MARKET_CORE",) + GROUP_ORDER,
}

MODEL_FAMILIES = ("LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING")


def feature_names(set_name: str) -> tuple[str, ...]:
    groups = FEATURE_SET_GROUPS[set_name]
    ordered: list[str] = []
    for group in groups:
        for name in FEATURE_GROUPS[group]:
            if name not in ordered:
                ordered.append(name)
    values = tuple(ordered)
    validate_feature_columns(values)
    return values


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
    raise ValueError(f"Unsupported feature-evidence model family: {family}")


def _probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(x), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("V1.0B.2 requires binary DOWN/UP probabilities")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid V1.0B.2 probability matrix")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("V1.0B.2 probabilities do not sum to one")
    return values


def _class_prior(train_y: pd.Series, rows: int) -> np.ndarray:
    counts = train_y.value_counts().reindex([0, 1], fill_value=0).astype(float)
    prior = ((counts + 1.0) / (counts.sum() + 2.0)).to_numpy(dtype=float)
    return np.tile(prior, (rows, 1))


def _previous_direction(frame: pd.DataFrame) -> np.ndarray:
    ret = pd.to_numeric(frame["m5_return_5m"], errors="coerce").fillna(0.0).to_numpy()
    predicted_up = ret >= 0
    values = np.empty((len(frame), 2), dtype=float)
    values[:, 0] = np.where(predicted_up, 0.10, 0.90)
    values[:, 1] = np.where(predicted_up, 0.90, 0.10)
    return values


def _metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    y = frame["directional_target"].astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    raw = pd.to_numeric(frame["raw_future_return"], errors="raise").to_numpy(dtype=float)
    band = pd.to_numeric(frame["decision_wait_band"], errors="raise").to_numpy(dtype=float)
    direction = np.where(predicted == 1, 1.0, -1.0)
    return {
        "accuracy": float((predicted == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier": float(np.mean((probabilities[:, 1] - y) ** 2)),
        "cost_aware_metric": float(np.mean(direction * raw - band)),
    }


def _directional_frame(source: pd.DataFrame, horizon: int, contract: TargetContract) -> pd.DataFrame:
    targeted = apply_target_contract(exact_future_frame(source, horizon), contract)
    work = targeted.loc[targeted["directional_target"].notna()].copy()
    work["decision_timestamp_utc"] = pd.to_datetime(work["decision_timestamp_utc"], utc=True, errors="raise")
    work["future_timestamp_utc"] = pd.to_datetime(work["future_timestamp_utc"], utc=True, errors="raise")
    work["directional_target"] = work["directional_target"].astype(int)
    return work.sort_values("decision_timestamp_utc").reset_index(drop=True)


def quarantine_split(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quarantine the V1.0B.1-style final tail and never evaluate it in this phase."""
    if len(frame) < MIN_RESEARCH_ROWS + MIN_QUARANTINE_ROWS:
        raise ValueError("INSUFFICIENT_DIRECTIONAL_DATA_FOR_QUARANTINE")
    quarantine_rows = max(MIN_QUARANTINE_ROWS, int(len(frame) * QUARANTINE_FRACTION))
    quarantine_start_index = len(frame) - quarantine_rows
    quarantine_start = pd.Timestamp(frame["decision_timestamp_utc"].iloc[quarantine_start_index])
    eligible = frame["future_timestamp_utc"] < quarantine_start
    research = frame.loc[eligible].copy().reset_index(drop=True)
    if len(research) < MIN_RESEARCH_ROWS:
        raise ValueError("INSUFFICIENT_RESEARCH_ROWS_AFTER_QUARANTINE_PURGE")
    info = {
        "quarantine_start": quarantine_start.isoformat(),
        "quarantine_rows": quarantine_rows,
        "research_rows": len(research),
        "quarantine_evaluated": False,
        "purge_ok": bool((research["future_timestamp_utc"] < quarantine_start).all()),
        "horizon_minutes": horizon,
    }
    return research, info


def evaluate_feature_set(research: pd.DataFrame, horizon: int, set_name: str, family: str) -> dict[str, Any]:
    names = feature_names(set_name)
    complete = research.loc[:, list(names)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    data = research.loc[complete].reset_index(drop=True)
    if len(data) < MIN_RESEARCH_ROWS:
        return {"status": "INSUFFICIENT_DATA", "feature_set": set_name, "family": family, "rows": len(data)}
    folds = purged_walk_forward(data["decision_timestamp_utc"], horizon, n_splits=N_SPLITS, min_validation_rows=MIN_VALIDATION_ROWS)
    fold_rows: list[dict[str, float]] = []
    prior_rows: list[dict[str, float]] = []
    previous_rows: list[dict[str, float]] = []
    for fold in folds:
        train = data.iloc[fold.train_indices]
        validation = data.iloc[fold.validation_indices]
        if train["directional_target"].nunique() < 2 or validation["directional_target"].nunique() < 2:
            return {"status": "INSUFFICIENT_CLASS_VARIATION", "feature_set": set_name, "family": family, "rows": len(data)}
        model = _pipeline(family)
        model.fit(train.loc[:, list(names)], train["directional_target"])
        fold_rows.append(_metrics(validation, _probabilities(model, validation.loc[:, list(names)])))
        prior_rows.append(_metrics(validation, _class_prior(train["directional_target"], len(validation))))
        previous_rows.append(_metrics(validation, _previous_direction(validation)))

    def mean(key: str, rows: list[dict[str, float]]) -> float:
        return float(np.mean([row[key] for row in rows]))

    folds_beating_prior = sum(
        current["balanced_accuracy"] > prior["balanced_accuracy"]
        for current, prior in zip(fold_rows, prior_rows)
    )
    return {
        "status": "EVALUATED",
        "feature_set": set_name,
        "family": family,
        "feature_groups": list(FEATURE_SET_GROUPS[set_name]),
        "feature_count": len(names),
        "rows": len(data),
        "fold_count": len(folds),
        "folds_beating_class_prior": int(folds_beating_prior),
        "balanced_accuracy_mean": mean("balanced_accuracy", fold_rows),
        "macro_f1_mean": mean("macro_f1", fold_rows),
        "log_loss_mean": mean("log_loss", fold_rows),
        "brier_mean": mean("brier", fold_rows),
        "cost_aware_mean": mean("cost_aware_metric", fold_rows),
        "class_prior_log_loss_mean": mean("log_loss", prior_rows),
        "previous_direction_cost_aware_mean": mean("cost_aware_metric", previous_rows),
        "recent_fold_balanced_accuracy": float(fold_rows[-1]["balanced_accuracy"]),
        "folds": fold_rows,
    }


def _evidence_status(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, list[str]]:
    failed: list[str] = []
    if row.get("status") != "EVALUATED":
        return "NO_FEATURE_EVIDENCE", [str(row.get("status"))]
    if row["balanced_accuracy_mean"] < MIN_MEAN_BALANCED_ACCURACY:
        failed.append("mean_balanced_accuracy")
    if row["recent_fold_balanced_accuracy"] < MIN_RECENT_BALANCED_ACCURACY:
        failed.append("recent_fold_balanced_accuracy")
    if row["folds_beating_class_prior"] < MIN_FOLDS_BEATING_PRIOR:
        failed.append("fold_stability")
    if row["log_loss_mean"] > row["class_prior_log_loss_mean"] - MIN_LOG_LOSS_IMPROVEMENT:
        failed.append("log_loss")
    if row["cost_aware_mean"] < row["previous_direction_cost_aware_mean"]:
        failed.append("cost_aware")
    delta = float(row["balanced_accuracy_mean"] - baseline["balanced_accuracy_mean"])
    if row["feature_set"] != "MARKET_CORE" and delta < MIN_GROUP_DELTA_BALANCED_ACCURACY:
        failed.append("feature_group_delta")
    return ("FEATURE_EVIDENCE_CANDIDATE" if not failed else "NO_FEATURE_EVIDENCE", failed)


def feature_diagnostics(research: pd.DataFrame, target_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(research["directional_target"], errors="coerce")
    all_names = feature_names("ALL_EXISTING_PRICE_CONTEXT")
    for name in all_names:
        values = pd.to_numeric(research[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        complete = values.notna() & y.notna()
        corr = values.loc[complete].corr(y.loc[complete], method="spearman") if complete.sum() >= 50 else np.nan
        rows.append({
            "target_label": target_label,
            "feature": name,
            "missing_fraction": float(values.isna().mean()),
            "unique_values": int(values.nunique(dropna=True)),
            "abs_spearman_to_direction": float(abs(corr)) if pd.notna(corr) else None,
        })
    return rows


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def run_research(root: Path = ROOT, persist: bool = True) -> dict[str, Any]:
    source = load_xauusd_research_frame(root)
    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    quarantine: dict[str, Any] = {}

    for horizon, contract in TARGET_SPECS:
        label = f"{horizon}m:{contract.name}"
        directional = _directional_frame(source, horizon, contract)
        research, quarantine_info = quarantine_split(directional, horizon)
        quarantine[label] = quarantine_info
        diagnostics.extend(feature_diagnostics(research, label))

        target_rows: list[dict[str, Any]] = []
        for family in MODEL_FAMILIES:
            for set_name in FEATURE_SET_GROUPS:
                row = evaluate_feature_set(research, horizon, set_name, family)
                row.update({
                    "asset_id": ASSET_ID,
                    "target_label": label,
                    "horizon_minutes": horizon,
                    "target_contract": asdict(contract),
                    "quarantine_evaluated": False,
                })
                target_rows.append(row)

            family_rows = [r for r in target_rows if r["family"] == family and r.get("status") == "EVALUATED"]
            baseline = next((r for r in family_rows if r["feature_set"] == "MARKET_CORE"), None)
            if baseline is None:
                continue
            for row in family_rows:
                status, failed = _evidence_status(row, baseline)
                row["evidence_status"] = status
                row["failed_gates"] = failed
                row["delta_balanced_accuracy_vs_market_core"] = float(
                    row["balanced_accuracy_mean"] - baseline["balanced_accuracy_mean"]
                )
        results.extend(target_rows)

    candidates = sorted(
        [r for r in results if r.get("evidence_status") == "FEATURE_EVIDENCE_CANDIDATE"],
        key=lambda r: (r["balanced_accuracy_mean"], -r["log_loss_mean"], r["cost_aware_mean"]),
        reverse=True,
    )
    decision = "FEATURE_EVIDENCE_FOUND" if candidates else "PRICE_ONLY_FEATURE_EVIDENCE_WEAK"
    report = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "asset_class": "PRECIOUS_METAL",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_rows": len(source),
        "target_specs": [{"horizon_minutes": h, "contract": asdict(c)} for h, c in TARGET_SPECS],
        "feature_sets": {name: list(groups) for name, groups in FEATURE_SET_GROUPS.items()},
        "model_families": list(MODEL_FAMILIES),
        "quarantine": quarantine,
        "results": results,
        "feature_evidence_candidate_count": len(candidates),
        "best_feature_evidence_candidate": candidates[0] if candidates else None,
        "decision": decision,
        "next_phase": "V1.0B.3_CONTROLLED_MODEL_RESEARCH" if candidates else "ADD_VERIFIED_GOLD_MACRO_INTERMARKET_CONTEXT",
        "production_integration": False,
        "model_promotion_performed": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }
    if persist:
        reports = asset_paths(ASSET_ID, root=root).reports
        _atomic_json(reports / "v10b2_feature_evidence.json", report)
        flat: list[dict[str, Any]] = []
        for row in results:
            flat.append({
                "asset_id": ASSET_ID,
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
                "delta_balanced_accuracy_vs_market_core": row.get("delta_balanced_accuracy_vs_market_core"),
            })
        _write_csv(reports / "v10b2_feature_leaderboard.csv", flat)
        _write_csv(reports / "v10b2_feature_diagnostics.csv", diagnostics)
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()
    report = run_research(ROOT, persist=not args.no_persist)
    print(json.dumps({
        "contract_version": report["contract_version"],
        "asset_id": report["asset_id"],
        "decision": report["decision"],
        "feature_evidence_candidate_count": report["feature_evidence_candidate_count"],
        "best_feature_evidence_candidate": report["best_feature_evidence_candidate"],
        "next_phase": report["next_phase"],
        "production_integration": False,
        "automatic_execution": "DISABLED",
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
