"""Attach exact, matured V0.5A outcomes to forward-shadow prediction IDs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.v08_contract import HORIZONS, OUTCOME_COLUMNS, OUTCOMES_FILE, SOURCE_LABEL
from src.evaluation.v08_ledger import utc
from src.learning.v05c_dataset import decision_cost_band, target_class
from src.marketdata.v05a_contract import FEATURE_FILE


def _atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def matured_outcome_rows(
    predictions: pd.DataFrame, market: pd.DataFrame, now_utc: object,
    existing: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    now = utc(now_utc)
    existing = pd.DataFrame(columns=OUTCOME_COLUMNS) if existing is None else existing.copy()
    keys = {(str(row.prediction_id), int(row.horizon_minutes)) for row in existing.itertuples()}
    source = market.copy()
    source["decision_timestamp_utc"] = pd.to_datetime(source["decision_timestamp_utc"], utc=True, errors="raise")
    rows: list[dict[str, object]] = []
    stats = {"added": 0, "waiting": 0, "missing": 0, "contract_mismatch": 0, "early_violations": 0}
    for prediction in predictions.itertuples(index=False):
        if getattr(prediction, "source_label") != SOURCE_LABEL:
            continue
        decision = utc(getattr(prediction, "decision_timestamp_utc"))
        matched = source.loc[source["decision_timestamp_utc"].eq(decision)]
        for horizon in HORIZONS:
            key = (str(getattr(prediction, "prediction_id")), horizon)
            if key in keys:
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
                continue
            if source_matured > now:
                stats["early_violations"] += 1
                continue
            entry = float(item["m5_close"])
            raw_return = float(raw_value)
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
                "decision_spread_points": float(item["m5_spread_points"]), "cost_band": band,
                "target_class": target, "outcome_status": "MATURED",
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
        return {"status": "INSUFFICIENT_DATA", "added": 0, "waiting": 0, "missing": 0, "contract_mismatch": 0, "early_violations": 0}
    predictions = pd.read_parquet(predictions_path)
    market = pd.read_parquet(market_path)
    existing = pd.read_parquet(outcomes_path) if outcomes_path.exists() else None
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else utc(now_utc)
    result, stats = matured_outcome_rows(predictions, market, now, existing)
    if len(result) != (0 if existing is None else len(existing)):
        _atomic(result, outcomes_path)
    return {"status": "PASS", **stats}
