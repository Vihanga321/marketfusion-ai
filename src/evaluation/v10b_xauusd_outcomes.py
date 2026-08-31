"""Mature XAUUSD forward-shadow outcomes using the V1.0A gold target contract.

This module deliberately does not call the EURUSD V0.5C cost-band helper.
Gold target class and cost band are taken from the already-audited XAUUSD
V1.0A target store at the original completed-M5 decision timestamp.
"""
from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from src.assets.contracts import ROOT, asset_paths, normalize_asset_id
from src.evaluation.v08_contract import (
    APPROVED_MODEL_STATUSES,
    HORIZONS,
    OUTCOME_COLUMNS,
    SOURCE_LABEL,
)
from src.evaluation.v08_ledger import utc

ASSET_ID = "XAUUSD"


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTCOME_COLUMNS)


def _invalid(
    prediction: object,
    horizon: int,
    decision: pd.Timestamp,
    future: pd.Timestamp,
    now: pd.Timestamp,
    reason: str,
) -> dict[str, object]:
    return {
        "prediction_id": getattr(prediction, "prediction_id"),
        "prediction_payload_sha256": getattr(prediction, "prediction_payload_sha256"),
        "source_label": SOURCE_LABEL,
        "horizon_minutes": horizon,
        "decision_timestamp_utc": decision.isoformat(),
        "future_timestamp_utc": future.isoformat(),
        "matured_at_utc": future.isoformat(),
        "entry_mid_or_reference_price": None,
        "future_price": None,
        "raw_return": None,
        "move_pips": None,
        "decision_spread_points": None,
        "cost_band": None,
        "target_class": None,
        "outcome_status": "INVALID",
        "evaluated_at_utc": now.isoformat(),
        "entry_reference_type": "COMPLETED_M5_CLOSE",
        "exit_reference_type": "EXACT_FUTURE_COMPLETED_M5_CLOSE",
        "evaluation_reason": reason,
    }


def matured_outcome_rows(
    predictions: pd.DataFrame,
    target_store: pd.DataFrame,
    now_utc: object,
    existing: pd.DataFrame | None = None,
    asset_id: str = ASSET_ID,
) -> tuple[pd.DataFrame, dict[str, int]]:
    asset = normalize_asset_id(asset_id)
    if asset != ASSET_ID:
        raise RuntimeError("V1.0B XAUUSD outcome evaluator rejects non-XAUUSD assets")
    now = utc(now_utc)
    existing = _empty() if existing is None else existing.copy()
    for column in OUTCOME_COLUMNS:
        if column not in existing:
            existing[column] = None
    existing = existing[list(OUTCOME_COLUMNS)]
    keys = {
        (str(row.prediction_id), int(row.horizon_minutes))
        for row in existing.itertuples(index=False)
    }

    source = target_store.copy()
    required = {"bar_close_utc", "close", "spread_points"}
    for horizon in HORIZONS:
        required.update({
            f"outcome_future_timestamp_{horizon}m",
            f"outcome_future_return_{horizon}m",
            f"target_neutral_band_{horizon}m",
            f"target_class_{horizon}m",
        })
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("Missing XAUUSD target-store columns: " + ", ".join(missing))
    source["decision_timestamp_utc"] = pd.to_datetime(source["bar_close_utc"], utc=True, errors="raise")

    rows: list[dict[str, object]] = []
    stats = {
        "added": 0, "waiting": 0, "missing": 0, "not_applicable": 0,
        "invalid": 0, "contract_mismatch": 0, "early_violations": 0,
    }
    for prediction in predictions.itertuples(index=False):
        if str(getattr(prediction, "source_label", "")) != SOURCE_LABEL:
            continue
        symbol = str(getattr(prediction, "symbol", "")).upper()
        if symbol != ASSET_ID:
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
            expected_future = decision + pd.Timedelta(minutes=horizon)
            if now < expected_future:
                stats["waiting"] += 1
                continue
            if matched.empty:
                stats["missing"] += 1
                continue
            item = matched.iloc[-1]
            future_value = item.get(f"outcome_future_timestamp_{horizon}m")
            raw_value = item.get(f"outcome_future_return_{horizon}m")
            band_value = item.get(f"target_neutral_band_{horizon}m")
            class_value = item.get(f"target_class_{horizon}m")
            if any(pd.isna(value) for value in (future_value, raw_value, band_value, class_value)):
                stats["missing"] += 1
                continue
            source_future = utc(future_value)
            if source_future != expected_future:
                stats["contract_mismatch"] += 1
                stats["invalid"] += 1
                rows.append(_invalid(
                    prediction, horizon, decision, source_future, now,
                    "XAUUSD_TARGET_TIMESTAMP_CONTRACT_MISMATCH",
                ))
                keys.add(key)
                continue
            if source_future > now:
                stats["early_violations"] += 1
                continue

            try:
                entry = float(item["close"])
                raw_return = float(raw_value)
                spread_points = float(item["spread_points"])
                cost_band = float(band_value)
                target_class = int(class_value)
                captured_entry = float(getattr(prediction, "market_mid"))
                finite = all(math.isfinite(value) for value in (
                    entry, raw_return, spread_points, cost_band, captured_entry,
                ))
                valid = (
                    finite and entry > 0 and spread_points >= 0 and cost_band >= 0
                    and target_class in (0, 1, 2)
                    and abs(entry - captured_entry) <= 1e-9
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                stats["invalid"] += 1
                rows.append(_invalid(
                    prediction, horizon, decision, expected_future, now,
                    "INVALID_OR_MISMATCHED_XAUUSD_REFERENCE_DATA",
                ))
                keys.add(key)
                continue

            future_price = entry * (1.0 + raw_return)
            rows.append({
                "prediction_id": getattr(prediction, "prediction_id"),
                "prediction_payload_sha256": getattr(prediction, "prediction_payload_sha256"),
                "source_label": SOURCE_LABEL,
                "horizon_minutes": horizon,
                "decision_timestamp_utc": decision.isoformat(),
                "future_timestamp_utc": expected_future.isoformat(),
                "matured_at_utc": source_future.isoformat(),
                "entry_mid_or_reference_price": entry,
                "future_price": future_price,
                "raw_return": raw_return,
                # OUTCOME_COLUMNS is shared with legacy EURUSD. Do not invent a
                # Forex pip convention for gold; use raw_return/price instead.
                "move_pips": None,
                "decision_spread_points": spread_points,
                "cost_band": cost_band,
                "target_class": target_class,
                "outcome_status": "MATURED",
                "evaluated_at_utc": now.isoformat(),
                "entry_reference_type": "COMPLETED_M5_CLOSE",
                "exit_reference_type": "EXACT_FUTURE_COMPLETED_M5_CLOSE",
                "evaluation_reason": "EXACT_XAUUSD_V10A_TARGET_EVENT_MATURED",
            })
            keys.add(key)

    stats["added"] = len(rows)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    return (
        combined[list(OUTCOME_COLUMNS)] if not combined.empty else _empty(),
        stats,
    )


def attach_matured_outcomes(
    predictions_path: Path | None = None,
    target_path: Path | None = None,
    outcomes_path: Path | None = None,
    now_utc: object | None = None,
    root: Path = ROOT,
) -> dict[str, int | str]:
    paths = asset_paths(ASSET_ID, root=root)
    predictions_path = predictions_path or paths.shadow_predictions
    target_path = target_path or (paths.features / "target_research.parquet")
    outcomes_path = outcomes_path or paths.shadow_outcomes
    if not predictions_path.exists() or not target_path.exists():
        return {
            "status": "INSUFFICIENT_DATA", "added": 0, "waiting": 0,
            "missing": 0, "not_applicable": 0, "invalid": 0,
            "contract_mismatch": 0, "early_violations": 0,
        }
    predictions = pd.read_parquet(predictions_path, engine="pyarrow")
    target_store = pd.read_parquet(target_path, engine="pyarrow")
    existing = pd.read_parquet(outcomes_path, engine="pyarrow") if outcomes_path.exists() else None
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)
    result, stats = matured_outcome_rows(predictions, target_store, now, existing)
    previous_rows = 0 if existing is None else len(existing)
    if len(result) != previous_rows:
        _atomic(result, outcomes_path)
    return {"status": "PASS", **stats}
