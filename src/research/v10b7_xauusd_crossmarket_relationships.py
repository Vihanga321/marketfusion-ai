"""MarketFusion V1.0B.7 causal cross-market relationship research for XAUUSD.

This is the last historical exploratory phase on the currently inspected sample.
It does not promote models. XAUUSD remains the only prediction target; MT5
context instruments remain read-only sensors. The previously exposed final tail
stays quarantined.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.assets.contracts import ROOT, asset_paths
from src.learning.v05c_walk_forward import purged_walk_forward
from src.research.v10b_xauusd_models import ASSET_ID, load_xauusd_research_frame
from src.research.v10b4_xauusd_intraday_context_audit import run_audit
from src.research.v10b5_xauusd_intraday_intermarket_evidence import (
    CONTEXT_COLLECTION_ROWS,
    CONTEXT_GROUPS,
    MIN_ALIGNED_RESEARCH_ROWS,
    MIN_BALANCED_ACCURACY_DELTA,
    MIN_FOLDS_BEATING_PAIRED_BASELINE,
    MIN_LOG_LOSS_IMPROVEMENT,
    MIN_MEAN_BALANCED_ACCURACY,
    MIN_RECENT_BALANCED_ACCURACY,
    MIN_VALIDATION_ROWS,
    N_SPLITS,
    PRICE_BASELINES,
    TARGET_SPECS,
    _class_prior,
    _directional_frame,
    _group_columns,
    _metrics,
    _pipeline,
    _previous_direction,
    _probabilities,
    _ready_symbols,
    collect_context_history,
    exact_context_join,
    quarantine_split,
)

CONTRACT_VERSION = "v1.0b7-xauusd-crossmarket-relationships-v1"
PREDICTION_TARGET = "XAUUSD"
EPSILON = 1e-12


@dataclass(frozen=True)
class RelationshipSpec:
    horizon_minutes: int
    target_contract_name: str
    baseline: str
    context_group: str
    model_family: str
    relationship_group: str
    selection_reason: str


# Frozen from the inspected V1.0B.5/V1.0B.6 evidence. Because these contracts
# were selected after historical inspection, B7 is explicitly promotion-ineligible.
SPECS: tuple[RelationshipSpec, ...] = (
    RelationshipSpec(
        15,
        "V10B_REFERENCE",
        "PRICE_CORE_NO_CLOCK",
        "SILVER",
        "LOGISTIC_REGRESSION",
        "XAU_XAG_RELATIONSHIPS",
        "Silver was the most persistent 15m directional helper before calibration; B6 showed calibration was not the missing ingredient.",
    ),
    RelationshipSpec(
        30,
        "SELECTIVE_MEDIUM",
        "PRICE_STRUCTURE_VOLATILITY_NO_CLOCK",
        "FX_USD",
        "HIST_GRADIENT_BOOSTING",
        "XAU_USD_RELATIONSHIPS",
        "FX_USD was the strongest 30m raw intermarket contract in B5; B6 showed output calibration erased its directional edge.",
    ),
)

_TARGET_LOOKUP = {(horizon, contract.name): contract for horizon, contract in TARGET_SPECS}

SILVER_RELATIONSHIP_COLUMNS = (
    "rel_xau_xag_divergence_5m",
    "rel_xau_xag_divergence_15m",
    "rel_xau_xag_divergence_60m",
    "rel_xau_xag_momentum_spread",
    "rel_xau_xag_direction_agreement_5m",
    "rel_xau_xag_direction_agreement_15m",
    "rel_xau_xag_vol_ratio_60m",
)

FX_RELATIONSHIP_COLUMNS = (
    "rel_usd_impulse_5m",
    "rel_usd_impulse_15m",
    "rel_usd_impulse_30m",
    "rel_usd_impulse_60m",
    "rel_fx_leg_agreement_5m",
    "rel_fx_leg_agreement_15m",
    "rel_fx_leg_agreement_60m",
    "rel_fx_leg_dispersion_15m",
    "rel_xau_inverse_usd_residual_5m",
    "rel_xau_inverse_usd_residual_15m",
    "rel_xau_inverse_usd_residual_60m",
    "rel_xau_usd_inverse_alignment_5m",
    "rel_xau_usd_inverse_alignment_15m",
    "rel_xau_usd_vol_ratio_60m",
    "rel_usd_impulse_acceleration",
)

RELATIONSHIP_COLUMNS = {
    "XAU_XAG_RELATIONSHIPS": SILVER_RELATIONSHIP_COLUMNS,
    "XAU_USD_RELATIONSHIPS": FX_RELATIONSHIP_COLUMNS,
}

for _spec in SPECS:
    if (_spec.horizon_minutes, _spec.target_contract_name) not in _TARGET_LOOKUP:
        raise RuntimeError("B7 spec references an unknown target contract")
    if _spec.baseline not in PRICE_BASELINES:
        raise RuntimeError("B7 spec references an unknown price baseline")
    if _spec.context_group not in CONTEXT_GROUPS:
        raise RuntimeError("B7 spec references an unknown context group")
    if _spec.relationship_group not in RELATIONSHIP_COLUMNS:
        raise RuntimeError("B7 spec references an unknown relationship group")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"Missing relationship source column: {name}")
    return pd.to_numeric(frame[name], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_relationship_features(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Add deterministic same-time/current-past cross-market relationship features."""
    work = frame.copy()
    xau5 = _numeric(work, "m5_return_5m")
    xau15 = _numeric(work, "m5_return_15m")
    xau60 = _numeric(work, "m5_return_60m")
    xau_vol60 = _numeric(work, "m5_volatility_60m")

    if group == "XAU_XAG_RELATIONSHIPS":
        xag5 = _numeric(work, "ctx_silver_ret_5m")
        xag15 = _numeric(work, "ctx_silver_ret_15m")
        xag60 = _numeric(work, "ctx_silver_ret_60m")
        xag_vol60 = _numeric(work, "ctx_silver_vol_60m")

        d5 = xau5 - xag5
        d15 = xau15 - xag15
        d60 = xau60 - xag60
        work["rel_xau_xag_divergence_5m"] = d5
        work["rel_xau_xag_divergence_15m"] = d15
        work["rel_xau_xag_divergence_60m"] = d60
        work["rel_xau_xag_momentum_spread"] = (d15 / 3.0) - (d60 / 12.0)
        work["rel_xau_xag_direction_agreement_5m"] = (
            np.sign(xau5) == np.sign(xag5)
        ).astype("int8")
        work["rel_xau_xag_direction_agreement_15m"] = (
            np.sign(xau15) == np.sign(xag15)
        ).astype("int8")
        work["rel_xau_xag_vol_ratio_60m"] = _safe_ratio(xau_vol60, xag_vol60)
        return work

    if group == "XAU_USD_RELATIONSHIPS":
        jpy5 = _numeric(work, "ctx_usdjpy_ret_5m")
        jpy15 = _numeric(work, "ctx_usdjpy_ret_15m")
        jpy30 = _numeric(work, "ctx_usdjpy_ret_30m")
        jpy60 = _numeric(work, "ctx_usdjpy_ret_60m")
        eur5 = _numeric(work, "ctx_eurusd_ret_5m")
        eur15 = _numeric(work, "ctx_eurusd_ret_15m")
        eur30 = _numeric(work, "ctx_eurusd_ret_30m")
        eur60 = _numeric(work, "ctx_eurusd_ret_60m")
        jpy_vol60 = _numeric(work, "ctx_usdjpy_vol_60m")
        eur_vol60 = _numeric(work, "ctx_eurusd_vol_60m")

        usd5 = 0.5 * (jpy5 - eur5)
        usd15 = 0.5 * (jpy15 - eur15)
        usd30 = 0.5 * (jpy30 - eur30)
        usd60 = 0.5 * (jpy60 - eur60)
        work["rel_usd_impulse_5m"] = usd5
        work["rel_usd_impulse_15m"] = usd15
        work["rel_usd_impulse_30m"] = usd30
        work["rel_usd_impulse_60m"] = usd60
        work["rel_fx_leg_agreement_5m"] = (
            np.sign(jpy5) == np.sign(-eur5)
        ).astype("int8")
        work["rel_fx_leg_agreement_15m"] = (
            np.sign(jpy15) == np.sign(-eur15)
        ).astype("int8")
        work["rel_fx_leg_agreement_60m"] = (
            np.sign(jpy60) == np.sign(-eur60)
        ).astype("int8")
        work["rel_fx_leg_dispersion_15m"] = (jpy15 + eur15).abs()
        work["rel_xau_inverse_usd_residual_5m"] = xau5 + usd5
        work["rel_xau_inverse_usd_residual_15m"] = xau15 + usd15
        work["rel_xau_inverse_usd_residual_60m"] = xau60 + usd60
        work["rel_xau_usd_inverse_alignment_5m"] = (
            np.sign(xau5) == np.sign(-usd5)
        ).astype("int8")
        work["rel_xau_usd_inverse_alignment_15m"] = (
            np.sign(xau15) == np.sign(-usd15)
        ).astype("int8")
        fx_vol60 = 0.5 * (jpy_vol60 + eur_vol60)
        work["rel_xau_usd_vol_ratio_60m"] = _safe_ratio(xau_vol60, fx_vol60)
        work["rel_usd_impulse_acceleration"] = usd5 - (usd15 / 3.0)
        return work

    raise ValueError(f"Unsupported relationship group: {group}")


def evaluate_relationship_contract(aligned: pd.DataFrame, spec: RelationshipSpec) -> dict[str, Any]:
    baseline_columns = PRICE_BASELINES[spec.baseline]
    raw_context_columns = _group_columns(spec.context_group)
    relationship_columns = RELATIONSHIP_COLUMNS[spec.relationship_group]
    raw_columns = baseline_columns + raw_context_columns
    enhanced_columns = raw_columns + relationship_columns

    enriched = add_relationship_features(aligned, spec.relationship_group)
    missing = sorted(set(enhanced_columns) - set(enriched.columns))
    if missing:
        return {"status": "MISSING_COLUMNS", "missing_columns": missing}

    complete = enriched.loc[:, list(enhanced_columns)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    data = enriched.loc[complete].reset_index(drop=True)
    if len(data) < MIN_ALIGNED_RESEARCH_ROWS:
        return {"status": "INSUFFICIENT_PREQUARANTINE_OVERLAP", "rows": int(len(data))}

    folds = purged_walk_forward(
        data["decision_broker_timestamp"],
        spec.horizon_minutes,
        n_splits=N_SPLITS,
        min_validation_rows=MIN_VALIDATION_ROWS,
    )

    raw_rows: list[dict[str, float]] = []
    enhanced_rows: list[dict[str, float]] = []
    prior_rows: list[dict[str, float]] = []
    previous_rows: list[dict[str, float]] = []

    for fold in folds:
        train = data.iloc[fold.train_indices]
        validation = data.iloc[fold.validation_indices]
        if train["directional_target"].nunique() < 2 or validation["directional_target"].nunique() < 2:
            return {"status": "INSUFFICIENT_CLASS_VARIATION", "rows": int(len(data))}

        raw_model = _pipeline(spec.model_family)
        raw_model.fit(train.loc[:, list(raw_columns)], train["directional_target"])
        raw_prob = _probabilities(raw_model, validation.loc[:, list(raw_columns)])
        raw_rows.append(_metrics(validation, raw_prob))

        enhanced_model = _pipeline(spec.model_family)
        enhanced_model.fit(train.loc[:, list(enhanced_columns)], train["directional_target"])
        enhanced_prob = _probabilities(enhanced_model, validation.loc[:, list(enhanced_columns)])
        enhanced_rows.append(_metrics(validation, enhanced_prob))

        prior_rows.append(_metrics(validation, _class_prior(train["directional_target"], len(validation))))
        previous_rows.append(_metrics(validation, _previous_direction(validation)))

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    beats = sum(
        enhanced["balanced_accuracy"] > raw["balanced_accuracy"]
        for enhanced, raw in zip(enhanced_rows, raw_rows)
    )

    return {
        "status": "EVALUATED",
        "rows": int(len(data)),
        "fold_count": int(len(folds)),
        "baseline": spec.baseline,
        "context_group": spec.context_group,
        "relationship_group": spec.relationship_group,
        "family": spec.model_family,
        "raw_context_feature_count": int(len(raw_columns)),
        "relationship_feature_count": int(len(relationship_columns)),
        "balanced_accuracy_mean": mean(enhanced_rows, "balanced_accuracy"),
        "raw_context_balanced_accuracy_mean": mean(raw_rows, "balanced_accuracy"),
        "balanced_accuracy_delta": mean(enhanced_rows, "balanced_accuracy") - mean(raw_rows, "balanced_accuracy"),
        "recent_fold_balanced_accuracy": float(enhanced_rows[-1]["balanced_accuracy"]),
        "folds_beating_raw_context": int(beats),
        "macro_f1_mean": mean(enhanced_rows, "macro_f1"),
        "raw_context_macro_f1_mean": mean(raw_rows, "macro_f1"),
        "log_loss_mean": mean(enhanced_rows, "log_loss"),
        "raw_context_log_loss_mean": mean(raw_rows, "log_loss"),
        "class_prior_log_loss_mean": mean(prior_rows, "log_loss"),
        "brier_mean": mean(enhanced_rows, "brier"),
        "raw_context_brier_mean": mean(raw_rows, "brier"),
        "cost_aware_mean": mean(enhanced_rows, "cost_aware_metric"),
        "raw_context_cost_aware_mean": mean(raw_rows, "cost_aware_metric"),
        "previous_direction_cost_aware_mean": mean(previous_rows, "cost_aware_metric"),
        "enhanced_folds": enhanced_rows,
        "raw_context_folds": raw_rows,
    }


def evidence_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("status") != "EVALUATED":
        return "NO_RELATIONSHIP_EVIDENCE", [str(row.get("status"))]
    failed: list[str] = []
    if row["balanced_accuracy_mean"] < MIN_MEAN_BALANCED_ACCURACY:
        failed.append("mean_balanced_accuracy")
    if row["recent_fold_balanced_accuracy"] < MIN_RECENT_BALANCED_ACCURACY:
        failed.append("recent_fold_balanced_accuracy")
    if row["folds_beating_raw_context"] < MIN_FOLDS_BEATING_PAIRED_BASELINE:
        failed.append("relationship_fold_stability")
    if row["balanced_accuracy_delta"] < MIN_BALANCED_ACCURACY_DELTA:
        failed.append("relationship_balanced_accuracy_delta")
    if row["log_loss_mean"] > row["raw_context_log_loss_mean"] - MIN_LOG_LOSS_IMPROVEMENT:
        failed.append("relationship_log_loss")
    if row["log_loss_mean"] >= row["class_prior_log_loss_mean"]:
        failed.append("class_prior_log_loss")
    if row["cost_aware_mean"] < row["raw_context_cost_aware_mean"]:
        failed.append("relationship_cost_aware")
    return (
        "CROSSMARKET_RELATIONSHIP_EVIDENCE_CANDIDATE" if not failed else "NO_RELATIONSHIP_EVIDENCE",
        failed,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: list[dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key not in {"enhanced_folds", "raw_context_folds"}}
        if isinstance(item.get("failed_gates"), list):
            item["failed_gates"] = ";".join(item["failed_gates"])
        flat.append(item)
    pd.DataFrame(flat).to_csv(path, index=False, lineterminator="\n")


def run_research(
    mt5: Any,
    *,
    root: Path = ROOT,
    persist: bool = True,
    context_rows: int = CONTEXT_COLLECTION_ROWS,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    captured = captured_at or datetime.now(timezone.utc)
    captured = captured.replace(tzinfo=timezone.utc) if captured.tzinfo is None else captured.astimezone(timezone.utc)

    audit = run_audit(mt5, root=root, persist=persist, captured_at=captured)
    ready_symbols = _ready_symbols(audit)
    context_frames, context_manifest = collect_context_history(
        mt5,
        ready_symbols,
        rows=context_rows,
        root=root,
        persist=persist,
    )
    usable = {
        family for family, item in context_manifest.items()
        if item.get("status") == "READY" and family in context_frames
    }

    source = load_xauusd_research_frame(root)
    results: list[dict[str, Any]] = []
    quarantine: dict[str, Any] = {}
    research_cache: dict[tuple[int, str], pd.DataFrame] = {}

    for spec in SPECS:
        contract = _TARGET_LOOKUP[(spec.horizon_minutes, spec.target_contract_name)]
        key = (spec.horizon_minutes, spec.target_contract_name)
        label = f"{spec.horizon_minutes}m:{spec.target_contract_name}"
        if key not in research_cache:
            directional = _directional_frame(source, spec.horizon_minutes, contract)
            research, quarantine_info = quarantine_split(directional, spec.horizon_minutes)
            research_cache[key] = research
            quarantine[label] = quarantine_info
        research = research_cache[key]

        missing_families = [family for family in CONTEXT_GROUPS[spec.context_group] if family not in usable]
        if missing_families:
            row: dict[str, Any] = {
                "status": "CONTEXT_GROUP_UNAVAILABLE",
                "missing_context_families": missing_families,
            }
        else:
            aligned = exact_context_join(research, context_frames, spec.context_group)
            row = evaluate_relationship_contract(aligned, spec)

        status, failed = evidence_status(row)
        row.update({
            "asset_id": ASSET_ID,
            "target_label": label,
            "horizon_minutes": spec.horizon_minutes,
            "target_contract": asdict(contract),
            "selection_reason": spec.selection_reason,
            "evidence_status": status,
            "failed_gates": failed,
            "quarantine_evaluated": False,
        })
        results.append(row)

    candidates = [row for row in results if row.get("evidence_status") == "CROSSMARKET_RELATIONSHIP_EVIDENCE_CANDIDATE"]
    candidates.sort(
        key=lambda row: (
            float(row.get("balanced_accuracy_delta", -999.0)),
            float(row.get("balanced_accuracy_mean", -999.0)),
            -float(row.get("log_loss_mean", 999.0)),
        ),
        reverse=True,
    )
    evaluated = [row for row in results if row.get("status") == "EVALUATED"]
    evaluated.sort(
        key=lambda row: (
            -len(row.get("failed_gates", [])),
            float(row.get("balanced_accuracy_mean", -999.0)),
            float(row.get("balanced_accuracy_delta", -999.0)),
        ),
        reverse=True,
    )

    decision = "CROSSMARKET_RELATIONSHIP_EVIDENCE_FOUND" if candidates else "CROSSMARKET_RELATIONSHIP_EVIDENCE_WEAK"
    if not evaluated:
        decision = "INSUFFICIENT_RELATIONSHIP_RESEARCH_HISTORY"
    next_phase = {
        "CROSSMARKET_RELATIONSHIP_EVIDENCE_FOUND": "FREEZE_RELATIONSHIP_CONTRACT_AND_START_FORWARD_POINT_IN_TIME_VALIDATION",
        "CROSSMARKET_RELATIONSHIP_EVIDENCE_WEAK": "STOP_HISTORICAL_TUNING_AND_COLLECT_NEW_FORWARD_POINT_IN_TIME_DATA",
        "INSUFFICIENT_RELATIONSHIP_RESEARCH_HISTORY": "COLLECT_MORE_POINT_IN_TIME_CONTEXT_BEFORE_RESEARCH",
    }[decision]

    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "prediction_target": PREDICTION_TARGET,
        "decision": decision,
        "relationship_candidate_count": int(len(candidates)),
        "best_relationship_candidate": candidates[0] if candidates else None,
        "best_screen_result": evaluated[0] if evaluated else None,
        "next_phase": next_phase,
        "historical_sample_reused": True,
        "last_historical_exploratory_phase": True,
        "promotion_eligible": False,
        "context_read_only": True,
        "verified_ready_symbols": ready_symbols,
        "context_manifest": context_manifest,
        "broker_clock_calibration": audit.get("broker_clock_calibration"),
        "historical_join_clock": "RAW_MT5_BROKER_EPOCH_KEY",
        "quarantine": quarantine,
        "quarantine_evaluated": False,
        "model_promotion_performed": False,
        "production_integration": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }

    if persist:
        reports = asset_paths(PREDICTION_TARGET, root).reports
        _atomic_json(reports / "v10b7_crossmarket_relationship_evidence.json", report)
        _write_csv(reports / "v10b7_crossmarket_relationship_leaderboard.csv", results)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--context-rows", type=int, default=CONTEXT_COLLECTION_ROWS)
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"MetaTrader5 unavailable: {exc}")
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    try:
        report = run_research(mt5, persist=not args.no_persist, context_rows=args.context_rows)
        compact = {
            "contract_version": report["contract_version"],
            "asset_id": report["asset_id"],
            "decision": report["decision"],
            "relationship_candidate_count": report["relationship_candidate_count"],
            "best_relationship_candidate": report["best_relationship_candidate"],
            "best_screen_result": report["best_screen_result"],
            "next_phase": report["next_phase"],
            "promotion_eligible": report["promotion_eligible"],
            "production_integration": report["production_integration"],
            "automatic_execution": report["automatic_execution"],
        }
        print(json.dumps(compact, indent=2, default=str))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
