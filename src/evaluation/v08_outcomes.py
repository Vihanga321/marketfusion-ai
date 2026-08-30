"""Attach exact, matured V0.5A outcomes to forward-shadow prediction IDs."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.evaluation.v08_contract import (
    APPROVED_MODEL_STATUSES, HORIZONS, OUTCOME_COLUMNS, OUTCOMES_FILE, SOURCE_LABEL,
)
from src.evaluation.v08_ledger import utc
from src.learning.v05c_dataset import decision_cost_band, target_class
from src.marketdata.v05a_contract import FEATURE_FILE


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _invalid_outcome(
    prediction: object, horizon: int, decision: pd.Timestamp, future: pd.Timestamp,
    matured: pd.Timestamp, now: pd.Timestamp, reason: str,
) -> dict[str, object]:
    return {
        "prediction_id": getattr(prediction, "prediction_id"),
        "prediction_payload_sha256": getattr(prediction, "prediction_payload_sha256"),
        "source_label": SOURCE_LABEL, "horizon_minutes": horizon,
        "decision_timestamp_utc": decision.isoformat(), "future_timestamp_utc": future.isoformat(),
        "matured_at_utc": matured.isoformat(), "entry_mid_or_reference_price": None,
        "future_price": None, "raw_return": None, "move_pips": None,
        "decision_spread_points": None, "cost_band": None, "target_class": None,
        "outcome_status": "INVALID", "evaluated_at_utc": now.isoformat(),
        "entry_reference_type": "COMPLETED_M5_CLOSE",
        "exit_reference_type": "EXACT_FUTURE_COMPLETED_M5_CLOSE",
        "evaluation_reason": reason,
    }


def matured_outcome_rows(
    predictions: pd.DataFrame, market: pd.DataFrame, now_utc: object,
    existing: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    now = utc(now_utc)
    existing = pd.DataFrame(columns=OUTCOME_COLUMNS) if existing is None else existing.copy()
    for column in OUTCOME_COLUMNS:
        if column not in existing:
            existing[column] = None
    existing = existing[list(OUTCOME_COLUMNS)]
    keys = {(str(row.prediction_id), int(row.horizon_minutes)) for row in existing.itertuples()}
    source = market.copy()
    source["decision_timestamp_utc"] = pd.to_datetime(source["decision_timestamp_utc"], utc=True, errors="raise")
    rows: list[dict[str, object]] = []
    stats = {"added": 0, "waiting": 0, "missing": 0, "not_applicable": 0, "invalid": 0, "contract_mismatch": 0, "early_violations": 0}
    for prediction in predictions.itertuples(index=False):
        if getattr(prediction, "source_label") != SOURCE_LABEL:
            continue
        decision = utc(getattr(prediction, "decision_timestamp_utc"))
        matched = source.loc[source["decision_timestamp_utc"].eq(decision)]
        for horizon in HORIZONS:
            key = (str(getattr(prediction, "prediction_id")), horizon)
            if key in keys:
                continue
            model_status = str(getattr(prediction, f"h{horizon}_status", "NO_APPROVED_MODEL"))
            if model_status not in APPROVED_MODEL_STATUSES:
                stats["not_applicable"] += 1
                continue
            future = decision + pd.Timedelta(minutes=horizon)
            if now < future:
                stats["waiting"] += 1
                continue
            if matched.empty:
                stats["missing"] += 1
                continue
            item = matched.iloc[-1]
            future_value = item.get(f"outcome_future_timestamp_{horizon}m")
            matured_value = item.get(f"outcome_matured_at_utc_{horizon}m")
            raw_value = item.get(f"outcome_future_return_{horizon}m")
            if pd.isna(future_value) or pd.isna(matured_value) or pd.isna(raw_value):
                stats["missing"] += 1
                continue
            source_future, source_matured = utc(future_value), utc(matured_value)
            if source_future != future or source_matured != future:
                stats["contract_mismatch"] += 1
                stats["invalid"] += 1
                rows.append(_invalid_outcome(prediction, horizon, decision, source_future, source_matured, now, "TARGET_TIMESTAMP_CONTRACT_MISMATCH"))
                keys.add(key)
                continue
            if source_matured > now:
                stats["early_violations"] += 1
                continue
            try:
                entry, raw_return = float(item["m5_close"]), float(raw_value)
                spread_points = float(item["m5_spread_points"])
                captured_entry = float(getattr(prediction, "market_mid"))
                valid_values = all(math.isfinite(value) for value in (entry, raw_return, spread_points, captured_entry))
                valid_values &= entry > 0 and spread_points >= 0 and abs(entry - captured_entry) <= 1e-9
            except (TypeError, ValueError):
                valid_values = False
            if not valid_values:
                stats["invalid"] += 1
                rows.append(_invalid_outcome(prediction, horizon, decision, source_future, source_matured, now, "INVALID_OR_MISMATCHED_REFERENCE_DATA"))
                keys.add(key)
                continue
            future_price = entry * (1.0 + raw_return)
            band = float(decision_cost_band(pd.DataFrame([item])).iloc[0])
            target = int(target_class(pd.Series([raw_return]), pd.Series([band])).iloc[0])
            rows.append({
                "prediction_id": getattr(prediction, "prediction_id"),
                "prediction_payload_sha256": getattr(prediction, "prediction_payload_sha256"),
                "source_label": SOURCE_LABEL, "horizon_minutes": horizon,
                "decision_timestamp_utc": decision.isoformat(), "future_timestamp_utc": future.isoformat(),
                "matured_at_utc": source_matured.isoformat(), "entry_mid_or_reference_price": entry,
                "future_price": future_price, "raw_return": raw_return,
                "move_pips": (future_price - entry) / 0.0001,
                "decision_spread_points": spread_points, "cost_band": band,
                "target_class": target, "outcome_status": "MATURED",
                "evaluated_at_utc": now.isoformat(), "entry_reference_type": "COMPLETED_M5_CLOSE",
                "exit_reference_type": "EXACT_FUTURE_COMPLETED_M5_CLOSE",
                "evaluation_reason": "EXACT_V05A_TARGET_EVENT_MATURED",
            })
            keys.add(key)
    stats["added"] = len(rows)
    appended = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    return appended[list(OUTCOME_COLUMNS)] if not appended.empty else pd.DataFrame(columns=OUTCOME_COLUMNS), stats


def attach_matured_outcomes(
    predictions_path: Path, market_path: Path = FEATURE_FILE, outcomes_path: Path = OUTCOMES_FILE,
    now_utc: object | None = None,
) -> dict[str, int | str]:
    if not predictions_path.exists() or not market_path.exists():
        return {"status": "INSUFFICIENT_DATA", "added": 0, "waiting": 0, "missing": 0, "not_applicable": 0, "invalid": 0, "contract_mismatch": 0, "early_violations": 0}
    predictions = pd.read_parquet(predictions_path)
    market = pd.read_parquet(market_path)
    existing = pd.read_parquet(outcomes_path) if outcomes_path.exists() else None
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)
    result, stats = matured_outcome_rows(predictions, market, now, existing)
    if len(result) != (0 if existing is None else len(existing)):
        _atomic(result, outcomes_path)
    return {"status": "PASS", **stats}
