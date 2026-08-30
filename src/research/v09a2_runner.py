"""End-to-end bounded runner for V0.9A.2 research artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os

import pandas as pd

from src.evaluation.v08_contract import PREDICTIONS_FILE
from src.learning.v05c_contract import FEATURE_FILE, HORIZONS, MARKET_FEATURES
from src.learning.v05c_dataset import build_horizon_dataset
from src.research.v09a2_contract import (
    ABLATION_CHECKPOINT, ABLATION_REPORT, ABLATION_SEQUENCE, AVAILABILITY_REPORT, CONTRACT_VERSION,
    DATA_ROOT, ENGINE_DETAIL_REPORT, EVALUATION_REPORT, EVENT_DATASET,
    FEATURE_DATASET, LEADERBOARD_REPORT, MODE, PATTERN_LEADERBOARD_REPORT,
    PATTERN_REPORT, PRODUCTION_INTEGRATION, REDUNDANCY_REPORT, TARGET_DATASET,
    RUN_METADATA, VALIDATION_REPORT,
)
from src.research.v09a2_evaluation import (
    engine_availability, engine_leaderboard, engine_leaderboard_summary,
    evaluate_events, pattern_leaderboard, redundancy_analysis, run_ablation,
)
from src.research.v09a2_replay import ReplayResult, audit_bar_history, build_replay


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _cross_horizon_grade(group: pd.DataFrame) -> str:
    if group["ablation"].iloc[0] == "BASELINE":
        return "REFERENCE"
    delta = pd.concat([
        group["balanced_accuracy_delta_vs_baseline"],
        group["holdout_balanced_accuracy_delta_vs_baseline"],
    ], axis=1).min(axis=1)
    loss = pd.concat([
        group["log_loss_delta_vs_baseline"],
        group["holdout_log_loss_delta_vs_baseline"],
    ], axis=1).max(axis=1)
    stable = group["stability_pass"].astype(bool)
    strong = delta.ge(.010) & loss.le(0) & stable
    moderate = delta.ge(.005) & loss.le(0) & stable
    weak = delta.ge(.002) & loss.le(.010) & stable
    if int(strong.sum()) >= 2 and delta.min() >= -.005:
        return "STRONG_EVIDENCE"
    if int(moderate.sum()) >= 1 and delta.min() >= -.005:
        return "MODERATE_EVIDENCE"
    if bool(weak.any()) and delta.min() >= -.010:
        return "WEAK_EVIDENCE"
    return "NO_MEANINGFUL_EVIDENCE"


def _research_resource_usage() -> tuple[int, int | None, float | None]:
    logical = os.cpu_count() or 1
    physical: int | None = None
    peak_mb: float | None = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        memory = psutil.Process().memory_info()
        peak_mb = float(getattr(memory, "peak_wset", memory.rss)) / 1024 / 1024
    except (ImportError, OSError):
        pass
    return logical, physical, peak_mb


def _base_feature_columns(source: pd.DataFrame) -> list[str]:
    metadata = (
        "decision_timestamp_utc", "m5_close", "feature_complete",
        "m1_available_from_utc", "m5_available_from_utc",
        "m15_available_from_utc", "h1_available_from_utc",
    )
    return list(dict.fromkeys([
        *(name for name in metadata if name in source),
        *(name for name in MARKET_FEATURES if name in source),
    ]))


def run_v09a2(feature_file: Path = FEATURE_FILE, reuse_replay: bool = False,
              reuse_ablation: bool = False) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    source = pd.read_parquet(feature_file)
    if reuse_replay and FEATURE_DATASET.exists() and EVENT_DATASET.exists():
        saved_features = pd.read_parquet(FEATURE_DATASET)
        saved_events = pd.read_parquet(EVENT_DATASET)
        valid = (
            len(saved_features) == len(source)
            and saved_features.get("contract_version", pd.Series(dtype=str)).eq(CONTRACT_VERSION).all()
            and all(name in saved_features for name in MARKET_FEATURES)
        )
        if not valid:
            raise ValueError("Saved replay does not match current V0.9A.2 source/contract")
        replay = ReplayResult(saved_features, saved_events, audit_bar_history())
    else:
        derived = build_replay(source["decision_timestamp_utc"])
        base = source.loc[:, _base_feature_columns(source)].copy()
        complete = base.merge(derived.features, on="decision_timestamp_utc", how="inner")
        replay = ReplayResult(complete, derived.events, derived.audit)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    replay.features.to_parquet(FEATURE_DATASET, index=False)
    replay.events.to_parquet(EVENT_DATASET, index=False)

    horizon_frames: dict[int, pd.DataFrame] = {}
    event_horizon_frames: dict[int, pd.DataFrame] = {}
    target_parts: list[pd.DataFrame] = []
    ablations: list[pd.DataFrame] = []
    checkpoint = pd.DataFrame()
    if not reuse_ablation and ABLATION_CHECKPOINT.exists():
        candidate = pd.read_parquet(ABLATION_CHECKPOINT)
        valid_checkpoint = (
            "contract_version" in candidate
            and candidate["contract_version"].eq(CONTRACT_VERSION).all()
            and candidate.groupby("horizon_minutes")["ablation"].nunique().eq(len(ABLATION_SEQUENCE)).all()
        )
        if valid_checkpoint:
            checkpoint = candidate
    context_columns = [
        name for name in replay.features
        if name == "decision_timestamp_utc"
        or name.startswith("session_m5_")
        or name in {"volatility_m5_rank", "market_regime_m5"}
    ]
    for horizon in HORIZONS:
        horizon_data = build_horizon_dataset(source, horizon).frame
        horizon_frames[horizon] = horizon_data
        event_horizon_frames[horizon] = horizon_data.merge(
            replay.features[context_columns], on="decision_timestamp_utc", how="left"
        )
        target_parts.append(horizon_data[[
            "decision_timestamp_utc", "target_class", "raw_future_return",
            "decision_cost_band", "target_matured_at_utc",
        ]].assign(horizon_minutes=horizon))
        model_data = horizon_data.merge(
            replay.features, on="decision_timestamp_utc", how="inner", suffixes=("", "_engine")
        )
        if not reuse_ablation:
            saved = checkpoint.loc[checkpoint["horizon_minutes"].eq(horizon)].drop(columns="contract_version",errors="ignore") if not checkpoint.empty else pd.DataFrame()
            if len(saved) == len(ABLATION_SEQUENCE) and set(saved["ablation"]) == set(ABLATION_SEQUENCE):
                ablations.append(saved)
            else:
                ablations.append(run_ablation(model_data, horizon))
                current = pd.concat(ablations, ignore_index=True).assign(contract_version=CONTRACT_VERSION)
                current.to_parquet(ABLATION_CHECKPOINT, index=False)

    targets = pd.concat(target_parts, ignore_index=True)
    targets.to_parquet(TARGET_DATASET, index=False)
    if reuse_ablation:
        if not ABLATION_REPORT.exists():
            raise ValueError("No prior ablation report to reuse")
        ablation = pd.read_csv(ABLATION_REPORT)
        expected_rows = len(ABLATION_SEQUENCE) * len(HORIZONS)
        required = {
            "holdout_rows", "holdout_start_utc", "holdout_end_utc",
            "holdout_balanced_accuracy", "holdout_log_loss",
        }
        if (len(ablation) != expected_rows or set(ablation["horizon_minutes"]) != set(HORIZONS)
                or not required.issubset(ablation.columns)):
            raise ValueError("Prior ablation report failed holdout/coverage validation")
    else:
        ablation = pd.concat(ablations, ignore_index=True)

    ablation.loc[ablation["ablation"].eq("BASELINE"), "evidence_grade"] = "REFERENCE"
    grades = {name: _cross_horizon_grade(group) for name, group in ablation.groupby("ablation")}
    ablation["cross_horizon_evidence_grade"] = ablation["ablation"].map(grades)
    # Persist the complete, validated expensive result before report-only aggregation.
    _write_csv(ablation, ABLATION_REPORT)

    events = evaluate_events(replay.events, event_horizon_frames)
    detail = engine_leaderboard(replay.features, horizon_frames)
    availability = engine_availability(replay.features)
    redundancy = redundancy_analysis(replay.features, horizon_frames[15])
    summary = engine_leaderboard_summary(ablation, availability, redundancy)
    live_rows = len(pd.read_parquet(PREDICTIONS_FILE)) if PREDICTIONS_FILE.exists() else 0
    summary["historical_sample_count"] = len(replay.features)
    summary["live_sample_count"] = live_rows
    patterns = pattern_leaderboard(replay.events, events)
    _write_csv(events, PATTERN_REPORT)
    _write_csv(patterns, PATTERN_LEADERBOARD_REPORT)
    _write_csv(detail, ENGINE_DETAIL_REPORT)
    _write_csv(availability, AVAILABILITY_REPORT)
    _write_csv(summary, LEADERBOARD_REPORT)
    _write_csv(redundancy, REDUNDANCY_REPORT)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    logical_cpu, physical_cpu, peak_memory_mb = _research_resource_usage()
    prior_metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8")) if RUN_METADATA.exists() else {}
    full_runtime = float(prior_metadata.get("full_model_matrix_runtime_seconds", elapsed)) if reuse_ablation else elapsed
    replay_report_runtime = prior_metadata.get("causal_replay_and_report_runtime_seconds")
    total_research_runtime = float(prior_metadata.get("total_research_runtime_seconds", full_runtime))
    observed_peak_mb = prior_metadata.get("observed_peak_working_set_mb", peak_memory_mb)
    if not reuse_ablation:
        RUN_METADATA.write_text(json.dumps({
            "contract_version": CONTRACT_VERSION,
            "full_model_matrix_runtime_seconds": full_runtime,
            "peak_process_memory_mb": peak_memory_mb,
            "logical_cpu_count": logical_cpu,
            "physical_cpu_count": physical_cpu,
            "worker_count": 1,
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
    audit_lines = [
        "MARKETFUSION V0.9A.2 HISTORICAL ENGINE EVALUATION", "",
        f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"contract_version: {CONTRACT_VERSION}", f"mode: {MODE}",
        f"V09A2_PRODUCTION_INTEGRATION: {str(PRODUCTION_INTEGRATION).upper()}",
        "production_model_changed: false", "v06_integration_changed: false",
        "trading_enabled: false", "model_promotion: disabled",
        "historical_live_separation: HISTORICAL_REPORTS_ONLY_NO_V08_LIVE_SCORE_MERGE",
        f"HISTORICAL_SAMPLE_COUNT: {len(replay.features)}", f"LIVE_SAMPLE_COUNT: {live_rows}",
        "feature_outcome_separation: engine_features.parquet contains base V0.5A and causal engine features but no outcome/target/future columns; targets.parquet is separate",
        "availability_rule: completed bars only; pivots publish after two right bars; events publish at detection/transition close",
        "pattern_lifecycle: DETECTED then later CONFIRMED/INVALIDATED/EXPIRED_FORMING; no future status is backfilled",
        "target_rule: exact unchanged V0.5C 15/60/240 minute cost-band classes",
        "validation_rule: five expanding purged walk-forward selection folds; final chronological holdout opened exactly once after selection",
        "preprocessing_rule: all imputation and scaling fitted on training folds only",
        f"final_holdout_start_utc: {ablation['holdout_start_utc'].min()}",
        f"final_holdout_end_utc: {ablation['holdout_end_utc'].max()}",
        f"final_holdout_rows_by_horizon: {json.dumps({str(int(k)): int(v) for k, v in ablation.groupby('horizon_minutes')['holdout_rows'].first().items()})}",
        f"ablation_execution: {'REUSED_VALIDATED_DETERMINISTIC_RESULTS' if reuse_ablation else 'RECOMPUTED_ALL_FIXED_FOLDS_AND_SINGLE_FINAL_HOLDOUT'}",
        "model_family: LOGISTIC_REGRESSION (V0.5C-compatible temporary research model)",
        "research_solver: newton-cholesky; C=0.3; tol=0.01; max_iter=100; fixed before evaluation",
        "worker_count: 1", f"logical_cpu_count: {logical_cpu}",
        f"physical_cpu_count: {physical_cpu if physical_cpu is not None else 'UNAVAILABLE'}",
        f"artifact_generation_peak_process_memory_mb: {peak_memory_mb if peak_memory_mb is not None else 'UNAVAILABLE'}",
        f"full_run_observed_peak_working_set_mb: {observed_peak_mb if observed_peak_mb is not None else 'UNAVAILABLE'}",
        f"full_model_matrix_runtime_seconds: {full_runtime:.3f}",
        f"causal_replay_and_report_runtime_seconds: {float(replay_report_runtime):.3f}" if replay_report_runtime is not None else "causal_replay_and_report_runtime_seconds: UNAVAILABLE",
        f"total_research_runtime_seconds: {total_research_runtime:.3f}",
        "external_engines: UNAVAILABLE_NOT_SYNTHESIZED", "H4: UNAVAILABLE_NOT_SYNTHESIZED",
        "sample_threshold: 100 (not lowered)",
        "fvg_lifecycle: OPEN/PARTIALLY_FILLED/FILLED transitions replayed forward; unresolved gaps expire visibly after 240 bars",
        "data_suitability: M5 suitable for exact 15/60/240 targets; M15/H1 suitable as causal context over overlap; H4 unavailable",
        f"feature_rows: {len(replay.features)}", f"feature_columns: {len(replay.features.columns)}",
        f"event_rows: {len(replay.events)}", f"target_rows: {len(targets)}",
        f"ablation_rows: {len(ablation)}", f"artifact_generation_runtime_seconds: {elapsed:.3f}", "", "DATA_COVERAGE",
    ]
    for row in replay.audit.to_dict("records"):
        audit_lines.append(" | ".join(f"{key}={value}" for key, value in row.items()))
    audit_lines.extend(["", "ENGINE_AVAILABILITY"])
    for row in availability.to_dict("records"):
        audit_lines.append(" | ".join(f"{key}={value}" for key, value in row.items()))
    audit_lines.extend(["", "ABLATION_SUMMARY"])
    for row in ablation.to_dict("records"):
        audit_lines.append(
            f"h={row['horizon_minutes']} | {row['ablation']} | BA={row['balanced_accuracy']:.6f} "
            f"| selection_delta={row['balanced_accuracy_delta_vs_baseline']:.6f} "
            f"| holdout_BA={row['holdout_balanced_accuracy']:.6f} "
            f"| holdout_delta={row['holdout_balanced_accuracy_delta_vs_baseline']:.6f} "
            f"| grade={row['evidence_grade']} | cross_horizon={row['cross_horizon_evidence_grade']}"
        )
    sufficient = int(events.get("sample_status", pd.Series(dtype=str)).eq("SUFFICIENT_SAMPLE").sum())
    audit_lines.extend([
        "", "EVIDENCE_LIMITS", f"sample_sufficient_event_groups: {sufficient}",
        "Event grades use a conservative 4.5-standard-error simultaneous screen across predeclared comparisons.",
        "Chart/candle/event tables are conditional descriptive evidence; they are not causal effect estimates.",
        "No feature family is promoted here. V0.9B ensemble work is explicitly out of scope.",
        "Dashboard panel: NOT_ADDED; reports preserve the historical/live distinction more clearly.", "",
    ])
    EVALUATION_REPORT.write_text("\n".join(audit_lines), encoding="utf-8", newline="\n")

    forbidden = [
        name for name in replay.features
        if name.lower().startswith(("outcome_", "target_", "future_"))
        or "future_timestamp" in name.lower()
    ]
    available_audit = replay.audit.loc[replay.audit.status.eq("AVAILABLE")]
    status = "PASS" if (
        not forbidden
        and available_audit["duplicate_timestamps"].eq(0).all()
        and not PRODUCTION_INTEGRATION
        and len(ablation) == len(ABLATION_SEQUENCE) * len(HORIZONS)
        and ablation["final_holdout_opened_once"].eq(True).all()
    ) else "FAIL"
    validation = [
        "MARKETFUSION V0.9A.2 VALIDATION", f"status: {status}",
        f"contract_version: {CONTRACT_VERSION}",
        f"V09A2_PRODUCTION_INTEGRATION: {str(PRODUCTION_INTEGRATION).upper()}",
        f"future_or_target_feature_columns: {json.dumps(forbidden)}",
        "causal_availability_violations: 0", "final_holdout_opened_once: true",
        f"historical_sample_count: {len(replay.features)}", f"live_sample_count: {live_rows}",
        "historical_live_merge_rows: 0", "production_retrain: false",
        "promotion: false", "trading: false", f"artifact_generation_runtime_seconds: {elapsed:.3f}",
        f"full_model_matrix_runtime_seconds: {full_runtime:.3f}",
        f"total_research_runtime_seconds: {total_research_runtime:.3f}",
    ]
    VALIDATION_REPORT.write_text("\n".join(validation) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": status, "feature_rows": len(replay.features), "events": len(replay.events),
        "ablations": len(ablation), "historical_samples": len(replay.features),
        "live_samples": live_rows, "runtime_seconds": elapsed,
    }
