"""MarketFusion V1.0C frozen XAUUSD forward point-in-time validation.

This is not a production trading model and cannot place orders.  Two hypotheses
selected before the forward start marker are frozen from prior exploratory
research and evaluated only on genuinely new completed-M5 observations.

The module deliberately separates:
- historical fitting, performed exactly once during initialization;
- immutable forward prediction recording;
- later exact-horizon outcome maturation; and
- predeclared forward evidence gates.

XAUUSD is the only prediction target.  XAGUSD, USDJPY and EURUSD are read-only
context sensors.  Automatic execution remains disabled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from src.assets.contracts import ROOT, asset_paths, discover_broker_symbol
from src.research.v10b_xauusd_models import build_research_features, load_xauusd_research_frame
from src.research.v10b1_xauusd_targets import TargetContract
from src.research.v10b4_xauusd_intraday_context_audit import run_audit
from src.research.v10b5_xauusd_intraday_intermarket_evidence import (
    CONTEXT_COLLECTION_ROWS,
    CONTEXT_GROUPS,
    PRICE_BASELINES,
    _directional_frame,
    _group_columns,
    _pipeline,
    _probabilities,
    _rates_to_context_features,
    _ready_symbols,
    collect_context_history,
    exact_context_join,
)

CONTRACT_VERSION = "v1.0c-xauusd-frozen-forward-validation-v1"
ASSET_ID = "XAUUSD"
FORWARD_DIR_NAME = "v10c_forward_validation"
MAX_FORWARD_RECORD_DELAY_MINUTES = 10
OUTCOME_MISSING_GRACE_MINUTES = 60
LIVE_XAU_ROWS = 700
LIVE_CONTEXT_ROWS = 700
OUTCOME_XAU_ROWS = 5_000

MIN_ACTIVE_TRADING_DAYS = 20
MIN_MATURED_OBSERVATIONS = 500
MIN_DIRECTIONAL_OUTCOMES = 400
STABILITY_BLOCKS = 4
MIN_BLOCK_DIRECTIONAL_ROWS = 80
MIN_MEAN_BALANCED_ACCURACY = 0.515
MIN_RECENT_BLOCK_BALANCED_ACCURACY = 0.510
MIN_BLOCKS_BEATING_PAIRED_BASELINE = 3
MIN_BALANCED_ACCURACY_DELTA = 0.003
MIN_LOG_LOSS_IMPROVEMENT = 0.002


@dataclass(frozen=True)
class FrozenContract:
    contract_id: str
    horizon_minutes: int
    target: TargetContract
    baseline: str
    context_group: str
    model_family: str
    selection_note: str


FROZEN_CONTRACTS: tuple[FrozenContract, ...] = (
    FrozenContract(
        contract_id="XAU15_SILVER_LR_V1",
        horizon_minutes=15,
        target=TargetContract("V10B_REFERENCE", 1.5, 0.10, "Frozen V1.0B reference target."),
        baseline="PRICE_CORE_NO_CLOCK",
        context_group="SILVER",
        model_family="LOGISTIC_REGRESSION",
        selection_note=(
            "Frozen after B5/B7: raw Silver context was the most persistent 15m directional helper; "
            "calibration and explicit XAU/XAG relationship expansion did not improve it."
        ),
    ),
    FrozenContract(
        contract_id="XAU30_FX_HGB_V1",
        horizon_minutes=30,
        target=TargetContract("SELECTIVE_MEDIUM", 2.0, 0.20, "Frozen V1.0B.1 selective-medium target."),
        baseline="PRICE_STRUCTURE_VOLATILITY_NO_CLOCK",
        context_group="FX_USD",
        model_family="HIST_GRADIENT_BOOSTING",
        selection_note=(
            "Frozen after B5: raw USDJPY+EURUSD HGB beat its paired baseline in 5/5 historical folds "
            "and failed only the class-prior probability gate before later calibration degraded direction."
        ),
    ),
)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _paths(root: Path = ROOT) -> dict[str, Path]:
    base = asset_paths(ASSET_ID, root).evaluation / FORWARD_DIR_NAME
    models = asset_paths(ASSET_ID, root).models / "forward_validation" / "v10c"
    return {
        "base": base,
        "manifest": base / "manifest.json",
        "predictions": base / "predictions.parquet",
        "outcomes": base / "outcomes.parquet",
        "status": base / "latest_status.json",
        "models": models,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_datetime64_any_dtype(selected[column]):
            selected[column] = pd.to_datetime(selected[column], utc=True).astype(str)
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(dtype=np.uint64)
    return sha256(hashed.tobytes()).hexdigest()


def _normalized_offset(audit: dict[str, Any]) -> int:
    calibration = audit.get("broker_clock_calibration") or {}
    if calibration.get("status") != "PASS":
        raise RuntimeError("Broker clock calibration is not PASS")
    offset = calibration.get("normalized_offset_seconds")
    if offset is None:
        raise RuntimeError("Broker clock calibration lacks normalized offset")
    return int(offset)


def _wait_band(row: pd.Series | dict[str, Any], contract: TargetContract) -> float:
    spread = float(row["spread_relative_to_price"])
    atr = float(row["atr_14"])
    close = float(row["bar_close"])
    if not all(math.isfinite(value) for value in (spread, atr, close)) or spread < 0 or atr <= 0 or close <= 0:
        raise ValueError("Invalid causal XAUUSD wait-band inputs")
    return max(contract.spread_multiplier * spread, contract.atr_multiplier * atr / close)


def _historical_training_frame(
    source: pd.DataFrame,
    context_frames: dict[str, pd.DataFrame],
    contract: FrozenContract,
    forward_start_broker: pd.Timestamp,
) -> pd.DataFrame:
    directional = _directional_frame(source, contract.horizon_minutes, contract.target)
    directional = directional.loc[
        pd.to_datetime(directional["future_timestamp_utc"], utc=True) < forward_start_broker
    ].copy()
    aligned = exact_context_join(directional, context_frames, contract.context_group)
    baseline_columns = PRICE_BASELINES[contract.baseline]
    context_columns = _group_columns(contract.context_group)
    required = list(baseline_columns + context_columns)
    complete = aligned.loc[:, required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    result = aligned.loc[complete].sort_values("decision_broker_timestamp").reset_index(drop=True)
    if len(result) < 3_000:
        raise RuntimeError(f"{contract.contract_id}: insufficient frozen historical training rows")
    if result["directional_target"].nunique() < 2:
        raise RuntimeError(f"{contract.contract_id}: historical training lacks both directional classes")
    return result


def initialize_forward_validation(
    mt5: Any,
    *,
    root: Path = ROOT,
    now_utc: object | None = None,
    context_rows: int = CONTEXT_COLLECTION_ROWS,
) -> dict[str, Any]:
    paths = _paths(root)
    if paths["manifest"].exists():
        raise RuntimeError(
            "V1.0C already initialized. The forward start marker is immutable; do not reset it to reuse observed data."
        )
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc)
    audit = run_audit(mt5, root=root, persist=False, captured_at=now.to_pydatetime())
    offset = _normalized_offset(audit)
    ready = _ready_symbols(audit)
    required_families = {family for contract in FROZEN_CONTRACTS for family in CONTEXT_GROUPS[contract.context_group]}
    missing = sorted(required_families - set(ready))
    if missing:
        raise RuntimeError("Missing frozen forward context families: " + ", ".join(missing))
    frozen_symbols = {family: ready[family] for family in sorted(required_families)}

    context_frames, context_manifest = collect_context_history(
        mt5, frozen_symbols, rows=context_rows, root=root, persist=False
    )
    unavailable = [family for family in required_families if family not in context_frames]
    if unavailable:
        raise RuntimeError("Insufficient initialization context history: " + ", ".join(sorted(unavailable)))

    source = load_xauusd_research_frame(root)
    forward_start_broker = now + pd.Timedelta(seconds=offset)
    paths["models"].mkdir(parents=True, exist_ok=True)
    model_manifest: dict[str, Any] = {}

    for contract in FROZEN_CONTRACTS:
        training = _historical_training_frame(source, context_frames, contract, forward_start_broker)
        baseline_columns = PRICE_BASELINES[contract.baseline]
        context_columns = _group_columns(contract.context_group)
        all_columns = baseline_columns + context_columns

        baseline_model = _pipeline(contract.model_family)
        context_model = _pipeline(contract.model_family)
        baseline_model.fit(training.loc[:, list(baseline_columns)], training["directional_target"])
        context_model.fit(training.loc[:, list(all_columns)], training["directional_target"])

        baseline_path = paths["models"] / f"{contract.contract_id}_baseline.joblib"
        context_path = paths["models"] / f"{contract.contract_id}_context.joblib"
        joblib.dump(baseline_model, baseline_path)
        joblib.dump(context_model, context_path)

        counts = training["directional_target"].value_counts().reindex([0, 1], fill_value=0).astype(float)
        prior = ((counts + 1.0) / (counts.sum() + 2.0)).to_numpy(dtype=float)
        hash_columns = ["decision_broker_timestamp", "directional_target", *all_columns]
        model_manifest[contract.contract_id] = {
            "contract": asdict(contract),
            "baseline_features": list(baseline_columns),
            "context_features": list(context_columns),
            "training_rows": int(len(training)),
            "training_first_broker_epoch": pd.Timestamp(training["decision_broker_timestamp"].iloc[0]).isoformat(),
            "training_last_broker_epoch": pd.Timestamp(training["decision_broker_timestamp"].iloc[-1]).isoformat(),
            "training_data_sha256": _frame_sha(training, hash_columns),
            "baseline_model_path": str(baseline_path.relative_to(root)),
            "baseline_model_sha256": _file_sha(baseline_path),
            "context_model_path": str(context_path.relative_to(root)),
            "context_model_sha256": _file_sha(context_path),
            "frozen_training_class_prior": {"down": float(prior[0]), "up": float(prior[1])},
            "forward_validation_only": True,
            "promotion_eligible": False,
        }

    manifest: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "initialized_at_utc": now.isoformat(),
        "forward_start_utc": now.isoformat(),
        "forward_start_broker_epoch": forward_start_broker.isoformat(),
        "broker_clock_offset_seconds": offset,
        "broker_clock_calibration": audit.get("broker_clock_calibration"),
        "frozen_context_symbols": frozen_symbols,
        "context_initialization_manifest": context_manifest,
        "models": model_manifest,
        "evaluation_contract": {
            "minimum_active_trading_days": MIN_ACTIVE_TRADING_DAYS,
            "minimum_matured_observations_per_contract": MIN_MATURED_OBSERVATIONS,
            "minimum_directional_outcomes_per_contract": MIN_DIRECTIONAL_OUTCOMES,
            "stability_blocks": STABILITY_BLOCKS,
            "minimum_directional_rows_per_block": MIN_BLOCK_DIRECTIONAL_ROWS,
            "minimum_balanced_accuracy": MIN_MEAN_BALANCED_ACCURACY,
            "minimum_recent_block_balanced_accuracy": MIN_RECENT_BLOCK_BALANCED_ACCURACY,
            "minimum_blocks_beating_paired_baseline": MIN_BLOCKS_BEATING_PAIRED_BASELINE,
            "minimum_balanced_accuracy_delta": MIN_BALANCED_ACCURACY_DELTA,
            "minimum_paired_log_loss_improvement": MIN_LOG_LOSS_IMPROVEMENT,
            "context_log_loss_must_beat_frozen_training_class_prior": True,
            "context_cost_aware_must_not_be_worse_than_paired_baseline": True,
            "gates_frozen_at_initialization": True,
        },
        "historical_sample_tuning_stopped": True,
        "retraining_during_forward_window": False,
        "backfill_before_forward_start": False,
        "maximum_forward_record_delay_minutes": MAX_FORWARD_RECORD_DELAY_MINUTES,
        "forming_m5_bar_excluded": True,
        "exact_broker_epoch_context_join": True,
        "model_promotion_performed": False,
        "promotion_eligible": False,
        "production_integration": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
        "git_commit": _git_commit(),
    }
    _atomic_json(paths["manifest"], manifest)
    return manifest


def _xau_completed_features(mt5: Any, symbol: str, point: float, rows: int = LIVE_XAU_ROWS) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"MT5 symbol_select failed for {symbol}")
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, "TIMEFRAME_M5"), 1, rows)
    if rates is None:
        raise RuntimeError(f"MT5 copy_rates_from_pos failed for {symbol}: {mt5.last_error()}")
    frame = pd.DataFrame(rates).sort_values("time").reset_index(drop=True)
    if len(frame) < 100:
        raise RuntimeError("Insufficient completed XAUUSD M5 history for causal forward features")
    open_ = pd.to_numeric(frame["open"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise")
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    ret5 = close.pct_change()
    spread_points = pd.to_numeric(frame.get("spread", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    info = mt5.symbol_info(symbol)
    current_spread = float(getattr(info, "spread", 0.0)) if info is not None else 0.0
    spread_points = spread_points.fillna(current_spread).clip(lower=0)
    decision = pd.to_datetime(frame["time"], unit="s", utc=True, errors="raise") + pd.Timedelta(minutes=5)

    base = pd.DataFrame({
        "decision_timestamp_utc": decision,
        "bar_open": open_, "bar_high": high, "bar_low": low, "bar_close": close,
        "m5_return_5m": ret5,
        "m5_return_15m": close.pct_change(3),
        "m5_return_60m": close.pct_change(12),
        "m5_return_240m": close.pct_change(48),
        "m5_volatility_60m": ret5.rolling(12, min_periods=12).std(),
        "m5_volatility_240m": ret5.rolling(48, min_periods=48).std(),
        "m5_spread_points": spread_points,
        "spread_relative_to_price": spread_points * float(point) / close,
        "spread_relative_to_atr": spread_points * float(point) / atr.replace(0, np.nan),
        "atr_14": atr,
        "utc_hour": decision.dt.hour,
        "day_of_week": decision.dt.dayofweek,
    })
    features = build_research_features(base)
    features["decision_broker_timestamp"] = pd.to_datetime(features["decision_timestamp_utc"], utc=True)
    return features.sort_values("decision_broker_timestamp").reset_index(drop=True)


def _live_context_frames(mt5: Any, symbols: dict[str, str], rows: int = LIVE_CONTEXT_ROWS) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for family, symbol in symbols.items():
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Frozen context symbol unavailable: {family}={symbol}")
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, "TIMEFRAME_M5"), 1, rows)
        if rates is None:
            raise RuntimeError(f"MT5 context history unavailable: {family}={symbol}")
        result[family] = _rates_to_context_features(rates, family)
    return result


def _manifest_model(root: Path, item: dict[str, Any], key: str) -> Any:
    path = root / str(item[f"{key}_model_path"])
    if not path.exists() or _file_sha(path) != item[f"{key}_model_sha256"]:
        raise RuntimeError(f"Frozen {key} model artifact missing or SHA mismatch")
    return joblib.load(path)


def _prediction_payload_sha(record: dict[str, Any]) -> str:
    payload = {key: record[key] for key in sorted(record) if key not in {"prediction_payload_sha256", "recorded_at_utc"}}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _append_prediction(path: Path, record: dict[str, Any]) -> str:
    incoming = _prediction_payload_sha(record)
    if incoming != record.get("prediction_payload_sha256"):
        raise ValueError("Forward prediction payload SHA mismatch")
    existing = pd.read_parquet(path, engine="pyarrow") if path.exists() else pd.DataFrame()
    if not existing.empty:
        same = existing.loc[
            existing["contract_id"].astype(str).eq(str(record["contract_id"]))
            & existing["decision_broker_timestamp"].astype(str).eq(str(record["decision_broker_timestamp"]))
        ]
        if not same.empty:
            if str(same.iloc[0]["prediction_payload_sha256"]) == incoming:
                return "DUPLICATE_IDEMPOTENT"
            raise RuntimeError("FORWARD_PREDICTION_CONFLICT")
    combined = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    _atomic_parquet(path, combined)
    return "APPENDED"


def record_forward_predictions(mt5: Any, *, root: Path = ROOT, now_utc: object | None = None) -> dict[str, Any]:
    paths = _paths(root)
    if not paths["manifest"].exists():
        raise RuntimeError("V1.0C is not initialized; run --initialize first")
    manifest = _json(paths["manifest"])
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc)
    audit = run_audit(mt5, root=root, persist=False, captured_at=now.to_pydatetime())
    offset = _normalized_offset(audit)
    if offset != int(manifest["broker_clock_offset_seconds"]):
        raise RuntimeError("Broker clock normalized offset changed during frozen forward validation")
    ready = _ready_symbols(audit)
    frozen_symbols = {str(k): str(v) for k, v in (manifest["frozen_context_symbols"] or {}).items()}
    for family, symbol in frozen_symbols.items():
        if ready.get(family) != symbol:
            raise RuntimeError(f"Frozen context identity changed: {family} expected {symbol}, got {ready.get(family)}")

    xau_spec = discover_broker_symbol(mt5, ASSET_ID)
    xau = _xau_completed_features(mt5, xau_spec.broker_symbol, xau_spec.point)
    context = _live_context_frames(mt5, frozen_symbols)
    forward_start_broker = _utc(manifest["forward_start_broker_epoch"])
    existing = pd.read_parquet(paths["predictions"], engine="pyarrow") if paths["predictions"].exists() else pd.DataFrame()
    added = 0
    skipped = 0
    cycles: list[dict[str, Any]] = []

    for contract in FROZEN_CONTRACTS:
        aligned = exact_context_join(xau, context, contract.context_group)
        if aligned.empty:
            cycles.append({"contract_id": contract.contract_id, "status": "NO_EXACT_CONTEXT_ALIGNMENT"})
            continue
        aligned["normalized_decision_utc"] = pd.to_datetime(aligned["decision_broker_timestamp"], utc=True) - pd.Timedelta(seconds=offset)
        eligible = aligned.loc[
            (aligned["decision_broker_timestamp"] > forward_start_broker)
            & (aligned["normalized_decision_utc"] <= now)
            & (aligned["normalized_decision_utc"] >= now - pd.Timedelta(minutes=MAX_FORWARD_RECORD_DELAY_MINUTES))
        ].copy()
        if eligible.empty:
            cycles.append({"contract_id": contract.contract_id, "status": "NO_NEW_FORWARD_DECISION"})
            continue
        row = eligible.iloc[-1]
        broker_key = pd.Timestamp(row["decision_broker_timestamp"]).isoformat()
        if not existing.empty and bool((
            existing["contract_id"].astype(str).eq(contract.contract_id)
            & existing["decision_broker_timestamp"].astype(str).eq(broker_key)
        ).any()):
            skipped += 1
            cycles.append({"contract_id": contract.contract_id, "status": "ALREADY_RECORDED", "decision": broker_key})
            continue

        model_info = manifest["models"][contract.contract_id]
        baseline_columns = tuple(model_info["baseline_features"])
        context_columns = tuple(model_info["context_features"])
        all_columns = baseline_columns + context_columns
        values = row.loc[list(all_columns)].replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            cycles.append({"contract_id": contract.contract_id, "status": "FEATURE_ROW_INCOMPLETE"})
            continue
        baseline_model = _manifest_model(root, model_info, "baseline")
        context_model = _manifest_model(root, model_info, "context")
        baseline_prob = _probabilities(baseline_model, pd.DataFrame([row.loc[list(baseline_columns)].to_dict()]))[0]
        context_prob = _probabilities(context_model, pd.DataFrame([row.loc[list(all_columns)].to_dict()]))[0]
        band = _wait_band(row, contract.target)
        record: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "contract_id": contract.contract_id,
            "prediction_id": sha256(f"{contract.contract_id}|{broker_key}|{manifest['forward_start_utc']}".encode()).hexdigest(),
            "prediction_payload_sha256": None,
            "decision_broker_timestamp": broker_key,
            "normalized_decision_utc": pd.Timestamp(row["normalized_decision_utc"]).isoformat(),
            "recorded_at_utc": now.isoformat(),
            "horizon_minutes": contract.horizon_minutes,
            "target_name": contract.target.name,
            "xau_broker_symbol": xau_spec.broker_symbol,
            "context_group": contract.context_group,
            "context_symbols_json": json.dumps({family: frozen_symbols[family] for family in CONTEXT_GROUPS[contract.context_group]}, sort_keys=True),
            "broker_clock_offset_seconds": offset,
            "entry_close": float(row["bar_close"]),
            "atr_14": float(row["atr_14"]),
            "spread_relative_to_price": float(row["spread_relative_to_price"]),
            "decision_wait_band": float(band),
            "baseline_prob_down": float(baseline_prob[0]),
            "baseline_prob_up": float(baseline_prob[1]),
            "context_prob_down": float(context_prob[0]),
            "context_prob_up": float(context_prob[1]),
            "baseline_direction": int(np.argmax(baseline_prob)),
            "context_direction": int(np.argmax(context_prob)),
            "baseline_model_sha256": model_info["baseline_model_sha256"],
            "context_model_sha256": model_info["context_model_sha256"],
            "forward_validation_only": True,
            "promotion_eligible": False,
            "automatic_execution": "DISABLED",
        }
        record["prediction_payload_sha256"] = _prediction_payload_sha(record)
        status = _append_prediction(paths["predictions"], record)
        if status == "APPENDED":
            added += 1
        cycles.append({"contract_id": contract.contract_id, "status": status, "decision": broker_key})
        existing = pd.read_parquet(paths["predictions"], engine="pyarrow")
    return {"status": "PASS", "added": added, "skipped": skipped, "contracts": cycles}


def _completed_xau_close_map(mt5: Any, symbol: str, rows: int = OUTCOME_XAU_ROWS) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, "TIMEFRAME_M5"), 1, rows)
    if rates is None:
        raise RuntimeError(f"MT5 XAUUSD outcome bars unavailable: {mt5.last_error()}")
    frame = pd.DataFrame(rates).sort_values("time").reset_index(drop=True)
    result = pd.DataFrame({
        "decision_broker_timestamp": pd.to_datetime(frame["time"], unit="s", utc=True) + pd.Timedelta(minutes=5),
        "close": pd.to_numeric(frame["close"], errors="raise"),
    })
    if result["decision_broker_timestamp"].duplicated().any():
        raise RuntimeError("Duplicate XAUUSD completed-M5 outcome keys")
    return result


def mature_forward_outcomes(mt5: Any, *, root: Path = ROOT, now_utc: object | None = None) -> dict[str, Any]:
    paths = _paths(root)
    if not paths["predictions"].exists():
        return {"status": "NO_PREDICTIONS", "added": 0, "waiting": 0, "invalid": 0}
    manifest = _json(paths["manifest"])
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc)
    offset = int(manifest["broker_clock_offset_seconds"])
    symbol = discover_broker_symbol(mt5, ASSET_ID).broker_symbol
    closes = _completed_xau_close_map(mt5, symbol)
    predictions = pd.read_parquet(paths["predictions"], engine="pyarrow")
    existing = pd.read_parquet(paths["outcomes"], engine="pyarrow") if paths["outcomes"].exists() else pd.DataFrame()
    existing_ids = set(existing["prediction_id"].astype(str)) if not existing.empty else set()
    rows: list[dict[str, Any]] = []
    waiting = 0
    invalid = 0

    for prediction in predictions.itertuples(index=False):
        prediction_id = str(prediction.prediction_id)
        if prediction_id in existing_ids:
            continue
        decision_broker = _utc(prediction.decision_broker_timestamp)
        horizon = int(prediction.horizon_minutes)
        future_broker = decision_broker + pd.Timedelta(minutes=horizon)
        normalized_future = future_broker - pd.Timedelta(seconds=offset)
        if now < normalized_future:
            waiting += 1
            continue
        matched = closes.loc[closes["decision_broker_timestamp"].eq(future_broker)]
        if matched.empty:
            if now < normalized_future + pd.Timedelta(minutes=OUTCOME_MISSING_GRACE_MINUTES):
                waiting += 1
                continue
            status = "INVALID_MISSING_EXACT_FUTURE_M5"
            raw_return = None
            target = None
            directional = False
            future_close = None
            invalid += 1
        else:
            future_close = float(matched.iloc[-1]["close"])
            entry = float(prediction.entry_close)
            band = float(prediction.decision_wait_band)
            raw_return = future_close / entry - 1.0
            directional = abs(raw_return) > band
            target = (1 if raw_return > 0 else 0) if directional else None
            status = "MATURED"
        rows.append({
            "contract_version": CONTRACT_VERSION,
            "prediction_id": prediction_id,
            "prediction_payload_sha256": str(prediction.prediction_payload_sha256),
            "contract_id": str(prediction.contract_id),
            "decision_broker_timestamp": decision_broker.isoformat(),
            "normalized_decision_utc": str(prediction.normalized_decision_utc),
            "horizon_minutes": horizon,
            "future_broker_timestamp": future_broker.isoformat(),
            "normalized_future_utc": normalized_future.isoformat(),
            "entry_close": float(prediction.entry_close),
            "future_close": future_close,
            "raw_future_return": raw_return,
            "decision_wait_band": float(prediction.decision_wait_band),
            "directional_eligible": bool(directional),
            "directional_target": target,
            "outcome_status": status,
            "evaluated_at_utc": now.isoformat(),
            "entry_reference_type": "EXACT_COMPLETED_M5_CLOSE",
            "exit_reference_type": "EXACT_FUTURE_COMPLETED_M5_CLOSE",
        })
    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        if combined["prediction_id"].duplicated().any():
            raise RuntimeError("Duplicate immutable V1.0C outcome prediction_id")
        _atomic_parquet(paths["outcomes"], combined)
    return {"status": "PASS", "added": len(rows), "waiting": waiting, "invalid": invalid}


def _metric_set(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    y = frame["directional_target"].astype(int).to_numpy()
    prob = frame[[f"{prefix}_prob_down", f"{prefix}_prob_up"]].to_numpy(dtype=float)
    predicted = prob.argmax(axis=1)
    raw = frame["raw_future_return"].to_numpy(dtype=float)
    band = frame["decision_wait_band"].to_numpy(dtype=float)
    direction = np.where(predicted == 1, 1.0, -1.0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(np.mean((prob[:, 1] - y) ** 2)),
        "cost_aware": float(np.mean(direction * raw - band)),
    }


def _prior_log_loss(frame: pd.DataFrame, prior: dict[str, Any]) -> float:
    p = np.array([float(prior["down"]), float(prior["up"])], dtype=float)
    probabilities = np.tile(p, (len(frame), 1))
    return float(log_loss(frame["directional_target"].astype(int), probabilities, labels=[0, 1]))


def evaluate_forward_contract(
    merged: pd.DataFrame,
    *,
    frozen_prior: dict[str, Any],
) -> dict[str, Any]:
    matured = merged.loc[merged["outcome_status"].eq("MATURED")].copy()
    active_days = int(pd.to_datetime(matured["normalized_decision_utc"], utc=True).dt.date.nunique()) if not matured.empty else 0
    directional = matured.loc[matured["directional_eligible"].astype(bool)].sort_values("normalized_decision_utc").reset_index(drop=True)
    sample_ready = (
        active_days >= MIN_ACTIVE_TRADING_DAYS
        and len(matured) >= MIN_MATURED_OBSERVATIONS
        and len(directional) >= MIN_DIRECTIONAL_OUTCOMES
    )
    result: dict[str, Any] = {
        "active_trading_days": active_days,
        "matured_observations": int(len(matured)),
        "directional_outcomes": int(len(directional)),
        "wait_or_neutral_outcomes": int(len(matured) - len(directional)),
        "minimum_sample_ready": bool(sample_ready),
        "decision": "COLLECTING",
        "failed_gates": [],
    }
    if len(directional) < 2 or directional["directional_target"].nunique() < 2:
        return result

    context = _metric_set(directional, "context")
    baseline = _metric_set(directional, "baseline")
    prior_loss = _prior_log_loss(directional, frozen_prior)
    result.update({
        "context": context,
        "paired_baseline": baseline,
        "frozen_training_class_prior_log_loss": prior_loss,
        "balanced_accuracy_delta": context["balanced_accuracy"] - baseline["balanced_accuracy"],
        "paired_log_loss_improvement": baseline["log_loss"] - context["log_loss"],
    })
    if not sample_ready:
        return result

    blocks = np.array_split(directional, STABILITY_BLOCKS)
    block_rows: list[dict[str, Any]] = []
    blocks_beating = 0
    block_contract_ok = True
    for index, block in enumerate(blocks, start=1):
        if len(block) < MIN_BLOCK_DIRECTIONAL_ROWS or block["directional_target"].nunique() < 2:
            block_contract_ok = False
            block_rows.append({"block": index, "rows": int(len(block)), "status": "INSUFFICIENT_BLOCK_DATA"})
            continue
        c = _metric_set(block, "context")
        b = _metric_set(block, "baseline")
        if c["balanced_accuracy"] > b["balanced_accuracy"]:
            blocks_beating += 1
        block_rows.append({"block": index, "rows": int(len(block)), "context": c, "paired_baseline": b})
    recent_ba = (
        float(block_rows[-1]["context"]["balanced_accuracy"])
        if block_contract_ok and "context" in block_rows[-1] else None
    )
    failed: list[str] = []
    if context["balanced_accuracy"] < MIN_MEAN_BALANCED_ACCURACY:
        failed.append("mean_balanced_accuracy")
    if recent_ba is None or recent_ba < MIN_RECENT_BLOCK_BALANCED_ACCURACY:
        failed.append("recent_block_balanced_accuracy")
    if not block_contract_ok or blocks_beating < MIN_BLOCKS_BEATING_PAIRED_BASELINE:
        failed.append("paired_block_stability")
    if result["balanced_accuracy_delta"] < MIN_BALANCED_ACCURACY_DELTA:
        failed.append("balanced_accuracy_delta")
    if result["paired_log_loss_improvement"] < MIN_LOG_LOSS_IMPROVEMENT:
        failed.append("paired_log_loss")
    if context["log_loss"] >= prior_loss:
        failed.append("frozen_class_prior_log_loss")
    if context["cost_aware"] < baseline["cost_aware"]:
        failed.append("paired_cost_aware")
    result.update({
        "stability_blocks": block_rows,
        "blocks_beating_paired_baseline": int(blocks_beating),
        "recent_block_balanced_accuracy": recent_ba,
        "failed_gates": failed,
        "decision": "FORWARD_EVIDENCE_PASS" if not failed else "FORWARD_EVIDENCE_FAIL",
    })
    return result


def forward_status(*, root: Path = ROOT, now_utc: object | None = None) -> dict[str, Any]:
    paths = _paths(root)
    if not paths["manifest"].exists():
        return {"contract_version": CONTRACT_VERSION, "status": "NOT_INITIALIZED", "automatic_execution": "DISABLED"}
    manifest = _json(paths["manifest"])
    predictions = pd.read_parquet(paths["predictions"], engine="pyarrow") if paths["predictions"].exists() else pd.DataFrame()
    outcomes = pd.read_parquet(paths["outcomes"], engine="pyarrow") if paths["outcomes"].exists() else pd.DataFrame()
    contracts: dict[str, Any] = {}
    all_ready = True
    any_pass = False
    for contract in FROZEN_CONTRACTS:
        if predictions.empty or outcomes.empty:
            merged = pd.DataFrame(columns=["outcome_status", "directional_eligible", "normalized_decision_utc"])
        else:
            pred = predictions.loc[predictions["contract_id"].eq(contract.contract_id)].copy()
            out = outcomes.loc[outcomes["contract_id"].eq(contract.contract_id)].copy()
            merged = pred.merge(out, on=["prediction_id", "contract_id"], how="inner", suffixes=("", "_out"), validate="one_to_one")
            if "normalized_decision_utc_out" in merged:
                merged["normalized_decision_utc"] = merged["normalized_decision_utc_out"]
            if "decision_wait_band_out" in merged:
                merged["decision_wait_band"] = merged["decision_wait_band_out"]
        result = evaluate_forward_contract(
            merged,
            frozen_prior=manifest["models"][contract.contract_id]["frozen_training_class_prior"],
        )
        contracts[contract.contract_id] = result
        all_ready = all_ready and bool(result.get("minimum_sample_ready"))
        any_pass = any_pass or result.get("decision") == "FORWARD_EVIDENCE_PASS"
    if not all_ready:
        decision = "COLLECTING_FORWARD_DATA"
    else:
        decision = "FORWARD_EVIDENCE_FOUND" if any_pass else "FORWARD_EVIDENCE_WEAK"
    payload = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "updated_at_utc": (pd.Timestamp.now(tz="UTC") if now_utc is None else _utc(now_utc)).isoformat(),
        "decision": decision,
        "forward_start_utc": manifest["forward_start_utc"],
        "contracts": contracts,
        "historical_tuning_stopped": True,
        "models_frozen": True,
        "retraining_during_forward_window": False,
        "promotion_eligible": False,
        "model_promotion_performed": False,
        "production_integration": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }
    _atomic_json(paths["status"], payload)
    return payload


def run_cycle(mt5: Any, *, root: Path = ROOT, now_utc: object | None = None) -> dict[str, Any]:
    prediction_cycle = record_forward_predictions(mt5, root=root, now_utc=now_utc)
    outcome_cycle = mature_forward_outcomes(mt5, root=root, now_utc=now_utc)
    status = forward_status(root=root, now_utc=now_utc)
    return {"prediction_cycle": prediction_cycle, "outcome_cycle": outcome_cycle, "forward_status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--context-rows", type=int, default=CONTEXT_COLLECTION_ROWS)
    args = parser.parse_args()

    if args.status:
        print(json.dumps(forward_status(), indent=2, default=str))
        return 0

    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"MetaTrader5 unavailable: {exc}")
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    try:
        if args.initialize:
            print(json.dumps(initialize_forward_validation(mt5, context_rows=args.context_rows), indent=2, default=str))
            return 0
        while True:
            print(json.dumps(run_cycle(mt5), indent=2, default=str))
            if not args.continuous:
                return 0
            time.sleep(max(30, int(args.interval_seconds)))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
