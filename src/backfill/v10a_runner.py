"""CLI for MarketFusion V1.0A audited MT5 historical backfill."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from src.backfill.v10a_backfill import apply_backfill, build_backfill_audit
from src.backfill.v10a_contract import (
    APPLY_REPORT,
    BACKFILL_AUDIT_REPORT,
    CONTRACT_VERSION,
    FINAL_STATUSES,
    LATEST_STATUS_FILE,
    LOCK_FILE,
    MARKET_CORE_REPORT,
    MIN_COMMON_FEATURE_DAYS_FOR_APPLY,
    OVERLAP_REPORT,
    SOURCE_LABEL,
    TARGET_PROMOTION_HISTORY_DAYS,
    VALIDATION_REPORT,
)
from src.marketdata.v05a_contract import ROOT


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def _lock() -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"V1.0A lock exists: {LOCK_FILE}. Another audit/apply may be running.") from exc
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def _unlock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def _status_from_audit(audit, apply_requested: bool, applied: bool = False) -> str:
    if audit.failures:
        if any("overlap conflict" in item.lower() or "merge conflict" in item.lower() for item in audit.failures):
            return "FAIL_OVERLAP_CONFLICT"
        return "FAIL_DATASET_CAUSALITY"
    days = int(audit.market_core["eligible_trading_days"])
    if days < MIN_COMMON_FEATURE_DAYS_FOR_APPLY:
        return "INSUFFICIENT_COMMON_HISTORY"
    if not apply_requested:
        return "PASS_AUDIT_READY_TO_APPLY"
    if applied and days >= TARGET_PROMOTION_HISTORY_DAYS:
        return "PASS_APPLIED_HISTORY_GATE_MET"
    if applied:
        return "PASS_APPLIED_MARKET_CORE_EXPANDED"
    return "FAIL"


def _audit_report(audit, status: str) -> None:
    lines = [
        "MARKETFUSION V1.0A AUDITED MT5 HISTORICAL BACKFILL",
        f"contract_version: {CONTRACT_VERSION}",
        f"source_label: {SOURCE_LABEL}",
        f"captured_at_utc: {audit.captured_at_utc.isoformat()}",
        "mode: READ_ONLY_AUDIT",
        "trading_execution: DISABLED",
        "",
        "TIMEFRAMES",
    ]
    for row in audit.timeframe_rows:
        lines.append(
            "{timeframe}: requested={requested_rows} fetched={fetched_rows} before={stored_rows_before} "
            "candidate={candidate_rows_after} added={candidate_added_rows} first={candidate_first_open_utc} "
            "last={candidate_last_close_utc} conflicts={overlap_conflicts}".format(**row)
        )
    lines.extend([
        "", "OVERLAP",
        f"feature_dataset_overlap_rows: {audit.dataset_overlap['overlap_rows']}",
        f"feature_compared_non_null_values: {audit.dataset_overlap['compared_values']}",
        f"feature_mismatches: {audit.dataset_overlap['mismatches']}",
        f"previous_nulls_filled_by_warmup: {audit.dataset_overlap['filled_previous_nulls']}",
        "", "MARKET_CORE",
        f"eligible_rows: {audit.market_core['eligible_rows']}",
        f"eligible_trading_days: {audit.market_core['eligible_trading_days']}",
        f"calendar_span_days: {audit.market_core['calendar_span_days']}",
        f"first_decision_utc: {audit.market_core['first_decision_utc']}",
        f"last_decision_utc: {audit.market_core['last_decision_utc']}",
        f"formal_v05c_history_requirement_trading_days: {TARGET_PROMOTION_HISTORY_DAYS}",
        f"promotion_history_gate: {audit.market_core['promotion_history_gate']}",
        "", f"blocking_failures: {len(audit.failures)}",
    ])
    lines.extend(f"FAILURE: {item}" for item in audit.failures)
    lines.extend(f"WARNING: {item}" for item in audit.warnings)
    lines.append(f"V10A_AUDIT_STATUS: {status}")
    _write_text(BACKFILL_AUDIT_REPORT, lines)
    audit.bar_overlap.to_csv(OVERLAP_REPORT, index=False, lineterminator="\n")
    _write_text(MARKET_CORE_REPORT, [
        "MARKETFUSION V1.0A MARKET_CORE EXPANSION",
        "eligibility_rule: feature_complete == True AND 15m/60m/240m outcomes matured",
        f"eligible_rows: {audit.market_core['eligible_rows']}",
        f"eligible_trading_days: {audit.market_core['eligible_trading_days']}",
        f"calendar_span_days: {audit.market_core['calendar_span_days']}",
        f"first_decision_utc: {audit.market_core['first_decision_utc']}",
        f"last_decision_utc: {audit.market_core['last_decision_utc']}",
        f"v05c_promotion_history_requirement_trading_days: {TARGET_PROMOTION_HISTORY_DAYS}",
        f"promotion_history_gate: {audit.market_core['promotion_history_gate']}",
        "note: calendar span is reported separately and does not substitute for V0.5C eligible decision-day count.",
    ])


def run(apply: bool = False) -> dict[str, object]:
    _lock()
    try:
        audit = build_backfill_audit()
        audit_status = _status_from_audit(audit, apply_requested=False)
        _audit_report(audit, audit_status)
        applied = False
        backup = None
        post_apply = None
        if apply:
            if audit.failures:
                raise RuntimeError("Backfill apply refused because the audit has blocking failures")
            if int(audit.market_core["eligible_trading_days"]) < MIN_COMMON_FEATURE_DAYS_FOR_APPLY:
                raise RuntimeError(
                    f"Backfill apply refused: only {audit.market_core['eligible_trading_days']} fully-featured decision days; "
                    f"minimum audit floor is {MIN_COMMON_FEATURE_DAYS_FOR_APPLY}"
                )
            backup, post_apply = apply_backfill(audit)
            applied = True
        status = _status_from_audit(audit, apply_requested=apply, applied=applied)
        if status not in FINAL_STATUSES:
            raise RuntimeError(f"Unexpected V1.0A status: {status}")
        effective_market_core = post_apply if post_apply is not None else audit.market_core
        result: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "status": status,
            "git_commit": _git_commit(),
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "mode": "APPLY" if apply else "READ_ONLY_AUDIT",
            "source_label": SOURCE_LABEL,
            "trading_execution": False,
            "applied": applied,
            "backup_path": None if backup is None else str(backup),
            "timeframes": audit.timeframe_rows,
            "bar_overlap_conflicts": int(audit.bar_overlap["mismatch_rows"].sum()) if not audit.bar_overlap.empty else 0,
            "dataset_overlap_mismatches": int(audit.dataset_overlap["mismatches"]),
            "filled_previous_null_features": int(audit.dataset_overlap["filled_previous_nulls"]),
            "market_core": effective_market_core,
            "v05c_history_requirement_days": TARGET_PROMOTION_HISTORY_DAYS,
            "automatic_v05c_training": False,
            "automatic_model_promotion": False,
            "live_advisory_changed": False,
        }
        LATEST_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        LATEST_STATUS_FILE.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        _write_text(APPLY_REPORT, [
            "MARKETFUSION V1.0A APPLY REPORT",
            f"mode: {'APPLY' if apply else 'DRY_AUDIT_ONLY'}",
            f"applied: {str(applied).upper()}",
            f"backup_path: {result['backup_path']}",
            f"eligible_trading_days: {effective_market_core['eligible_trading_days']}",
            f"promotion_history_gate: {effective_market_core['promotion_history_gate']}",
            "automatic_v05c_training: NO", "automatic_model_promotion: NO", "live_advisory_changed: NO",
            f"V10A_STATUS: {status}",
        ])
        _write_text(VALIDATION_REPORT, [
            "MARKETFUSION V1.0A VALIDATION",
            f"commit: {result['git_commit']}", f"tested_at_utc: {result['created_at_utc']}", f"mode: {result['mode']}",
            f"timeframes: {','.join(row['timeframe'] for row in audit.timeframe_rows)}",
            f"bar_overlap_conflicts: {result['bar_overlap_conflicts']}",
            f"dataset_overlap_mismatches: {result['dataset_overlap_mismatches']}",
            f"blocking_failures: {len(audit.failures)}",
            f"eligible_rows: {effective_market_core['eligible_rows']}",
            f"eligible_trading_days: {effective_market_core['eligible_trading_days']}",
            f"calendar_span_days: {effective_market_core['calendar_span_days']}",
            f"promotion_history_gate: {effective_market_core['promotion_history_gate']}",
            "future_feature_violations: 0" if not audit.failures else "future_feature_violations: SEE_AUDIT",
            "outcome_timing_violations: 0" if not audit.failures else "outcome_timing_violations: SEE_AUDIT",
            "trading_execution: DISABLED", "automatic_promotion: FORBIDDEN", f"V10A_STATUS: {status}",
        ])
        return result
    finally:
        _unlock()


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketFusion V1.0A audited historical backfill")
    parser.add_argument("--apply", action="store_true", help="Apply the audited candidate stores with backup+rollback protection.")
    args = parser.parse_args()
    print(json.dumps(run(apply=args.apply), indent=2, default=str))


if __name__ == "__main__":
    main()
