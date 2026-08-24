"""Aggregate V0.8 ledgers into transparent, sample-gated monitoring reports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.v08_calibration import calibration_monitor
from src.evaluation.v08_contract import (
    CALIBRATION_REPORT, DRIFT_RECENT_DAYS, DRIFT_REPORT, EVENT_REPORT, HORIZON_REPORT, HORIZONS,
    MIN_DIRECTIONAL_CALLS, MIN_PROMOTION_MONITOR_ROWS, MIN_REGIME_ROWS,
    OUTCOMES_FILE, PREDICTIONS_FILE, REGIME_REPORT,
    PERFORMANCE_SNAPSHOTS_FILE, SESSION_REPORT, SHADOW_REPORT,
)
from src.evaluation.v08_drift import distribution_drift
from src.evaluation.v08_metrics import evaluate_horizon, grouped_performance, wait_analysis
from src.learning.v05c_contract import MARKET_FEATURES
from src.marketdata.v05a_contract import FEATURE_FILE


def joined_forward_rows(
    predictions_path: Path = PREDICTIONS_FILE, outcomes_path: Path = OUTCOMES_FILE,
) -> pd.DataFrame:
    if not predictions_path.exists() or not outcomes_path.exists():
        return pd.DataFrame()
    predictions = pd.read_parquet(predictions_path)
    outcomes = pd.read_parquet(outcomes_path)
    if predictions.empty or outcomes.empty:
        return pd.DataFrame()
    joined = outcomes.merge(predictions, on=["prediction_id", "prediction_payload_sha256", "source_label", "decision_timestamp_utc"], how="inner", validate="many_to_one")
    if len(joined) != len(outcomes):
        raise RuntimeError("Outcome ledger contains records not bound to immutable predictions")
    return joined


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _market_drift(now_utc: object) -> pd.DataFrame:
    columns = ["monitor", "feature", "reference_rows", "recent_rows", "psi", "standardized_mean_shift", "missing_rate_shift", "status"]
    if not FEATURE_FILE.exists():
        return pd.DataFrame([{"monitor": "MARKET_REGIME_MONITOR", "feature": "ALL", "status": "INSUFFICIENT_DATA"}], columns=columns)
    market = pd.read_parquet(FEATURE_FILE)
    market["decision_timestamp_utc"] = pd.to_datetime(market["decision_timestamp_utc"], utc=True)
    now = pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    eligible = market.loc[market["decision_timestamp_utc"].le(now)].copy()
    frames = []
    for days in DRIFT_RECENT_DAYS:
        recent_start = now - pd.Timedelta(days=days)
        reference_start = recent_start - pd.Timedelta(days=20)
        recent = eligible.loc[eligible["decision_timestamp_utc"].gt(recent_start)]
        reference = eligible.loc[eligible["decision_timestamp_utc"].between(reference_start, recent_start, inclusive="left")]
        result = distribution_drift(reference, recent, list(MARKET_FEATURES), recent_as_of_utc=now)
        result["monitor"] = f"MARKET_REGIME_MONITOR_{days}D"
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def rolling_performance(joined: pd.DataFrame, now_utc: object) -> dict[str, object]:
    """Return predeclared row-count and elapsed-time views without filling missing samples."""
    now = pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    result: dict[str, object] = {}
    for count in (20, 50, 100):
        view = joined.sort_values("decision_timestamp_utc").tail(count) if not joined.empty else joined
        result[f"last_{count}"] = {
            "rows": len(view),
            "status": "AVAILABLE" if len(view) >= count else "INSUFFICIENT_DATA",
            "directional_calls": int(view["advisory"].isin(["BUY_BIAS", "SELL_BIAS"]).sum()) if not view.empty else 0,
        }
    for days in (5, 20):
        if joined.empty:
            view = joined
        else:
            timestamps = pd.to_datetime(joined["decision_timestamp_utc"], utc=True)
            view = joined.loc[timestamps.gt(now - pd.Timedelta(days=days))]
        result[f"last_{days}_days"] = {
            "rows": len(view), "status": "AVAILABLE" if len(view) else "INSUFFICIENT_DATA",
            "directional_calls": int(view["advisory"].isin(["BUY_BIAS", "SELL_BIAS"]).sum()) if not view.empty else 0,
        }
    return result


def _append_snapshot(now_utc: object, horizon: pd.DataFrame, predictions: int, outcomes: int) -> None:
    row: dict[str, object] = {
        "snapshot_at_utc": pd.Timestamp(now_utc), "recorded_predictions": predictions,
        "matured_outcomes": outcomes,
    }
    for item in horizon.to_dict("records"):
        key = int(item["horizon_minutes"])
        row[f"h{key}_matured"] = item["matured_count"]
        row[f"h{key}_directional_calls"] = item["directional_calls"]
        row[f"h{key}_directional_accuracy"] = item["directional_accuracy"]
        row[f"h{key}_brier"] = item["brier"]
    existing = pd.read_parquet(PERFORMANCE_SNAPSHOTS_FILE) if PERFORMANCE_SNAPSHOTS_FILE.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined["snapshot_at_utc"] = pd.to_datetime(combined["snapshot_at_utc"], utc=True)
    combined = combined.drop_duplicates("snapshot_at_utc", keep="last").sort_values("snapshot_at_utc")
    PERFORMANCE_SNAPSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = PERFORMANCE_SNAPSHOTS_FILE.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(PERFORMANCE_SNAPSHOTS_FILE)


def champion_monitor(predictions: pd.DataFrame, joined: pd.DataFrame) -> dict[str, object]:
    """Keep model IDs separate; never blend champion generations into one result."""
    result: dict[str, object] = {}
    for horizon_minutes in HORIZONS:
        column = f"h{horizon_minutes}_model_id"
        if predictions.empty or column not in predictions:
            continue
        for model_id in sorted(str(value) for value in predictions[column].dropna().unique() if str(value)):
            observed = predictions.loc[predictions[column].astype(str).eq(model_id)]
            matured = joined.loc[joined["horizon_minutes"].eq(horizon_minutes) & joined[column].astype(str).eq(model_id)] if not joined.empty else joined
            metrics = evaluate_horizon(matured, horizon_minutes)
            if len(matured) < MIN_PROMOTION_MONITOR_ROWS:
                health = "INSUFFICIENT_DATA"
            elif int(metrics["directional_calls"]) < MIN_DIRECTIONAL_CALLS:
                health = "CHAMPION_WATCH"
            elif metrics["cost_aware_research_metric"] is None or float(metrics["cost_aware_research_metric"]) < 0:
                health = "CHAMPION_DEGRADED"
            else:
                health = "CHAMPION_HEALTHY"
            result[f"{horizon_minutes}:{model_id}"] = {
                "model_id": model_id, "horizon_minutes": horizon_minutes,
                "first_seen_utc": str(observed["decision_timestamp_utc"].min()),
                "last_seen_utc": str(observed["decision_timestamp_utc"].max()),
                "predictions": len(observed), "matured": len(matured),
                "directional_calls": metrics["directional_calls"], "wait": metrics["wait_count"],
                "balanced_accuracy": metrics["balanced_accuracy"], "brier": metrics["brier"],
                "cost_aware_research_metric": metrics["cost_aware_research_metric"], "status": health,
            }
    return result


def generate_analysis(now_utc: object) -> dict[str, object]:
    joined = joined_forward_rows()
    horizon = pd.DataFrame([evaluate_horizon(joined, value) for value in HORIZONS])
    _write_csv(horizon, HORIZON_REPORT)
    session = joined.copy()
    if not session.empty:
        session["session"] = session["session"].replace({"OVERLAP": "LONDON_NEW_YORK_OVERLAP", "NEW YORK": "NEW_YORK", "OFF": "OFF_SESSION"})
    _write_csv(grouped_performance(session, "session", MIN_REGIME_ROWS), SESSION_REPORT)
    regime_frames = []
    for kind, column in (("VOLATILITY", "volatility_regime"), ("TREND", "trend_regime")):
        grouped = grouped_performance(joined, column, MIN_REGIME_ROWS)
        if not grouped.empty:
            grouped.insert(0, "regime_type", kind)
            grouped = grouped.rename(columns={column: "regime"})
            regime_frames.append(grouped)
    regime = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame(columns=["regime_type", "regime", "horizon_minutes", "sample_count", "wait_rate", "directional_coverage", "directional_accuracy", "brier", "cost_aware_research_metric", "status"])
    _write_csv(regime, REGIME_REPORT)
    _write_csv(grouped_performance(joined, "event_risk", MIN_REGIME_ROWS), EVENT_REPORT)
    calibration = calibration_monitor(joined)
    _write_csv(calibration, CALIBRATION_REPORT)
    drift = _market_drift(now_utc)
    model_drift = pd.DataFrame([{"monitor": "MODEL_DRIFT", "feature": "ALL", "reference_rows": 0, "recent_rows": 0, "status": "INSUFFICIENT_DATA"}])
    drift = pd.concat([drift, model_drift], ignore_index=True)
    _write_csv(drift, DRIFT_REPORT)
    waits = wait_analysis(joined)
    intelligence = grouped_performance(joined, "v05b_health", MIN_REGIME_ROWS)
    predictions = pd.read_parquet(PREDICTIONS_FILE) if PREDICTIONS_FILE.exists() else pd.DataFrame()
    rolling = rolling_performance(joined, now_utc)
    _append_snapshot(now_utc, horizon, len(predictions), len(joined))
    champion_results = champion_monitor(predictions, joined)
    significance = {"status": "INSUFFICIENT_DATA", "hypotheses_tested": 0, "multiple_testing_correction": "NOT_APPLICABLE"}
    lines = [
        "MARKETFUSION V0.8 SHADOW PERFORMANCE", "",
        "scope: TRUE_FORWARD_SHADOW only", "profitability_claim: NO", "predictive_edge_claim: NO", "commission_status: UNKNOWN", "",
        f"recorded_predictions: {len(predictions)}", f"matured_outcome_rows: {len(joined)}",
        f"wait_rate: {waits['wait_rate'] if waits['wait_rate'] is not None else 'INSUFFICIENT_DATA'}",
        f"directional_calls: {int(horizon['directional_calls'].sum()) if not horizon.empty else 0}", "",
        "rolling_windows: last 20/50/100 outcomes and last 5/20 elapsed days", f"significance: {significance['status']}", "",
        "intelligence_context: descriptive V0.5B health grouping only; activity categories unavailable unless captured causally in V0.6", "",
    ]
    for item in horizon.to_dict("records"):
        lines.extend([
            f"{item['horizon_minutes']}m:", f"matured: {item['matured_count']}",
            f"directional_accuracy: {item['directional_accuracy'] if item['directional_accuracy'] is not None else 'INSUFFICIENT_DATA'}",
            f"brier: {item['brier'] if item['brier'] is not None else 'NO_APPROVED_MODEL_DATA'}",
            f"status: {item['sample_status']}", "",
        ])
    while lines and lines[-1] == "":
        lines.pop()
    SHADOW_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    market_status = "INSUFFICIENT_DATA" if drift.empty else "DEGRADED" if drift["status"].eq("DEGRADED").any() else "WATCH" if drift["status"].eq("WATCH").any() else "NORMAL" if drift["status"].eq("NORMAL").any() else "INSUFFICIENT_DATA"
    return {
        "recorded_predictions": len(predictions), "matured_outcomes": len(joined),
        "horizons": {str(int(row["horizon_minutes"])): row for row in horizon.to_dict("records")},
        "wait": waits, "calibration_status": "NO_APPROVED_MODEL_DATA" if calibration["status"].eq("NO_APPROVED_MODEL_DATA").all() else "INSUFFICIENT_DATA" if calibration["status"].eq("INSUFFICIENT_DATA").any() else "PASS",
        "market_drift_status": market_status, "model_drift_status": "INSUFFICIENT_DATA",
        "rolling": rolling, "champion_monitor": champion_results, "significance": significance,
        "intelligence_performance": intelligence.to_dict("records"),
    }
