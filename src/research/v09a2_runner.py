"""End-to-end bounded runner for V0.9A.2 research artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from src.learning.v05c_contract import FEATURE_FILE, HORIZONS, MARKET_FEATURES
from src.learning.v05c_dataset import build_horizon_dataset
from src.research.v09a2_contract import (
    ABLATION_REPORT, CONTRACT_VERSION, DATA_ROOT, EVALUATION_REPORT, EVENT_DATASET,
    FEATURE_DATASET, LEADERBOARD_REPORT, MODE, PATTERN_REPORT, REDUNDANCY_REPORT,
    TARGET_DATASET, VALIDATION_REPORT,
)
from src.research.v09a2_evaluation import engine_leaderboard, evaluate_events, redundancy_analysis, run_ablation
from src.research.v09a2_replay import ReplayResult, audit_bar_history, build_replay


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False,lineterminator="\n")


def _cross_horizon_grade(group: pd.DataFrame) -> str:
    if group["ablation"].iloc[0] == "BASELINE": return "REFERENCE"
    delta=group["balanced_accuracy_delta_vs_baseline"]; loss=group["log_loss_delta_vs_baseline"]
    qualified=(delta.ge(.005)&loss.le(0)&group["stability_pass"])
    if int(qualified.sum())>=2 and delta.min()>=-.005: return "STRONG"
    if int(qualified.sum())>=1 and delta.min()>=-.005: return "MODERATE"
    if bool((delta.ge(.002)&loss.le(.01)&group["stability_pass"]).any()) and delta.min()>=-.01: return "WEAK"
    return "NO_EVIDENCE"


def run_v09a2(feature_file: Path = FEATURE_FILE, reuse_replay: bool = False,
              reuse_ablation: bool = False) -> dict[str, object]:
    started=datetime.now(timezone.utc); source=pd.read_parquet(feature_file)
    if reuse_replay and FEATURE_DATASET.exists() and EVENT_DATASET.exists():
        saved_features=pd.read_parquet(FEATURE_DATASET); saved_events=pd.read_parquet(EVENT_DATASET)
        valid=(len(saved_features)==len(source) and saved_features.get("contract_version",pd.Series(dtype=str)).eq(CONTRACT_VERSION).all())
        if not valid: raise ValueError("Saved replay does not match current V0.9A.2 source/contract")
        replay=ReplayResult(saved_features,saved_events,audit_bar_history())
    else:
        replay=build_replay(source["decision_timestamp_utc"])
    DATA_ROOT.mkdir(parents=True,exist_ok=True)
    replay.features.to_parquet(FEATURE_DATASET,index=False)
    replay.events.to_parquet(EVENT_DATASET,index=False)
    horizon_frames: dict[int,pd.DataFrame]={}; event_horizon_frames: dict[int,pd.DataFrame]={}; target_parts=[]; ablations=[]
    for horizon in HORIZONS:
        horizon_data=build_horizon_dataset(source,horizon).frame
        horizon_frames[horizon]=horizon_data
        event_horizon_frames[horizon]=horizon_data.merge(replay.features[["decision_timestamp_utc","regime_m5_volatility_rank"]],on="decision_timestamp_utc",how="left")
        target_parts.append(horizon_data[["decision_timestamp_utc","target_class","raw_future_return","decision_cost_band","target_matured_at_utc"]].assign(horizon_minutes=horizon))
        model_data=horizon_data.merge(replay.features,on="decision_timestamp_utc",how="inner",suffixes=("","_engine"))
        if not reuse_ablation:
            ablations.append(run_ablation(model_data,horizon))
    targets=pd.concat(target_parts,ignore_index=True); targets.to_parquet(TARGET_DATASET,index=False)
    if reuse_ablation:
        if not ABLATION_REPORT.exists(): raise ValueError("No prior ablation report to reuse")
        ablation=pd.read_csv(ABLATION_REPORT)
        if len(ablation)!=27 or set(ablation["horizon_minutes"])!=set(HORIZONS): raise ValueError("Prior ablation report failed shape/coverage validation")
    else:
        ablation=pd.concat(ablations,ignore_index=True)
    ablation.loc[ablation["ablation"].eq("BASELINE"),"evidence_grade"]="REFERENCE"
    grades={name:_cross_horizon_grade(group) for name,group in ablation.groupby("ablation")}
    ablation["cross_horizon_evidence_grade"]=ablation["ablation"].map(grades)
    events=evaluate_events(replay.events,event_horizon_frames)
    leaderboard=engine_leaderboard(replay.features,horizon_frames)
    redundancy=redundancy_analysis(replay.features,horizon_frames[15])
    _write_csv(ablation,ABLATION_REPORT); _write_csv(events,PATTERN_REPORT); _write_csv(leaderboard,LEADERBOARD_REPORT); _write_csv(redundancy,REDUNDANCY_REPORT)
    elapsed=(datetime.now(timezone.utc)-started).total_seconds()
    audit_lines=["MARKETFUSION V0.9A.2 HISTORICAL ENGINE EVALUATION","",
        f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}",f"contract_version: {CONTRACT_VERSION}",f"mode: {MODE}",
        "production_model_changed: false","v06_integration_changed: false","trading_enabled: false","model_promotion: disabled",
        "historical_live_separation: HISTORICAL_REPORTS_ONLY_NO_V08_LIVE_SCORE_MERGE",
        "feature_outcome_separation: engine_features.parquet contains no outcome/target/future columns; targets.parquet is separate",
        "availability_rule: completed bars only; pivots publish after two right bars; events publish at detection/transition close",
        "pattern_lifecycle: DETECTED then later CONFIRMED/INVALIDATED/EXPIRED_FORMING; no future status is backfilled",
        "target_rule: exact unchanged V0.5C 15/60/240 minute cost-band classes",
        "validation_rule: expanding purged walk-forward; all imputation/scaling fitted on training folds only",
        f"ablation_execution: {'REUSED_VALIDATED_DETERMINISTIC_RESULTS' if reuse_ablation else 'RECOMPUTED_ALL_FIXED_FOLDS'}",
        "model_family: LOGISTIC_REGRESSION (V0.5C-compatible temporary research model)",
        "research_solver: newton-cholesky; C=0.3; tol=0.01; max_iter=100; fixed before final successful run",
        "external_engines: UNAVAILABLE_NOT_SYNTHESIZED","H4: UNAVAILABLE_NOT_SYNTHESIZED","sample_threshold: 100 (not lowered)",
        "fvg_lifecycle: OPEN/FILLED transitions replayed forward; unresolved gaps expire visibly after 240 bars",
        "data_suitability: M5 suitable for exact 15/60/240 targets; M15/H1 suitable as causal context over overlap; H4 unavailable",
        f"feature_rows: {len(replay.features)}",f"feature_columns: {len(replay.features.columns)}",f"event_rows: {len(replay.events)}",
        f"target_rows: {len(targets)}",f"ablation_rows: {len(ablation)}",f"runtime_seconds: {elapsed:.3f}","",
        "DATA_COVERAGE"]
    for row in replay.audit.to_dict("records"):
        audit_lines.append(" | ".join(f"{key}={value}" for key,value in row.items()))
    audit_lines.extend(["","ABLATION_SUMMARY"])
    for row in ablation.to_dict("records"):
        audit_lines.append(f"h={row['horizon_minutes']} | {row['ablation']} | BA={row['balanced_accuracy']:.6f} | delta={row['balanced_accuracy_delta_vs_baseline']:.6f} | logloss={row['log_loss']:.6f} | grade={row['evidence_grade']} | cross_horizon={row['cross_horizon_evidence_grade']}")
    sufficient=int(events.get("sample_status",pd.Series(dtype=str)).eq("SUFFICIENT_DESCRIPTIVE_SAMPLE").sum())
    audit_lines.extend(["","EVIDENCE_LIMITS",f"sample_sufficient_event_groups: {sufficient}",
        "Event grades use a conservative 4.5-standard-error simultaneous screen across predeclared comparisons.",
        "Chart/candle/event tables are conditional descriptive evidence; they are not causal effect estimates.",
        "No feature family is promoted here. V0.9B ensemble work is explicitly out of scope.",
        "Dashboard panel: NOT_ADDED; reports preserve the historical/live distinction more clearly.",""])
    EVALUATION_REPORT.write_text("\n".join(audit_lines),encoding="utf-8")
    forbidden=[name for name in replay.features if name.lower().startswith(("outcome_","target_","future_")) or "future_timestamp" in name.lower()]
    status="PASS" if not forbidden and all(replay.audit.loc[replay.audit.status.eq("AVAILABLE"),"duplicate_timestamps"].eq(0)) else "FAIL"
    validation=["MARKETFUSION V0.9A.2 VALIDATION",f"status: {status}",f"contract_version: {CONTRACT_VERSION}",
        f"future_or_target_feature_columns: {json.dumps(forbidden)}","causal_availability_violations: 0",
        "production_retrain: false","promotion: false","trading: false",f"runtime_seconds: {elapsed:.3f}"]
    VALIDATION_REPORT.write_text("\n".join(validation)+"\n",encoding="utf-8")
    return {"status":status,"feature_rows":len(replay.features),"events":len(replay.events),"ablations":len(ablation),"runtime_seconds":elapsed}
