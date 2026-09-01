"""MarketFusion V1.0B.5 XAUUSD intraday intermarket evidence research.

XAUUSD remains the only prediction target. Other MT5 instruments are read-only
context sensors. This phase uses exact completed-M5 broker-clock joins and keeps
the previously exposed XAUUSD tail quarantined. It performs no model promotion
or production integration.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
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
from src.marketdata.mt5_continuous_store import atomic_write_parquet
from src.research.v10b_xauusd_models import (
    ASSET_ID,
    FEATURE_GROUPS,
    load_xauusd_research_frame,
    validate_feature_columns,
)
from src.research.v10b1_xauusd_targets import TargetContract, apply_target_contract, exact_future_frame
from src.research.v10b4_xauusd_intraday_context_audit import run_audit

CONTRACT_VERSION = "v1.0b5-xauusd-intraday-intermarket-evidence-v1"
PREDICTION_TARGET = "XAUUSD"
RANDOM_STATE = 2301

CONTEXT_COLLECTION_ROWS = 50_000
MIN_CONTEXT_ROWS = 2_000
MIN_ALIGNED_RESEARCH_ROWS = 3_000
N_SPLITS = 5
MIN_VALIDATION_ROWS = 250

QUARANTINE_FRACTION = 0.15
MIN_QUARANTINE_ROWS = 750
MIN_SOURCE_RESEARCH_ROWS = 3_000

MIN_MEAN_BALANCED_ACCURACY = 0.515
MIN_RECENT_BALANCED_ACCURACY = 0.510
MIN_FOLDS_BEATING_PAIRED_BASELINE = 3
MIN_BALANCED_ACCURACY_DELTA = 0.003
MIN_LOG_LOSS_IMPROVEMENT = 0.002

TARGET_SPECS: tuple[tuple[int, TargetContract], ...] = (
    (15, TargetContract("V10B_REFERENCE", 1.5, 0.10, "V1.0B.1 reference target.")),
    (30, TargetContract("SELECTIVE_MEDIUM", 2.0, 0.20, "V1.0B.1 selective-medium target.")),
)

ALLOWED_CONTEXT_FAMILIES = ("SILVER", "USDJPY", "EURUSD", "US500", "US100")
CONTEXT_GROUPS: dict[str, tuple[str, ...]] = {
    "SILVER": ("SILVER",),
    "FX_USD": ("USDJPY", "EURUSD"),
    "EQUITY_RISK": ("US500", "US100"),
    "SILVER_PLUS_FX": ("SILVER", "USDJPY", "EURUSD"),
    "ALL_VERIFIED_INTRADAY": ("SILVER", "USDJPY", "EURUSD", "US500", "US100"),
}

CLOCK_FEATURES = {"utc_hour", "day_of_week"}
PRICE_BASELINES: dict[str, tuple[str, ...]] = {
    "PRICE_CORE_NO_CLOCK": tuple(name for name in FEATURE_GROUPS["MARKET_CORE"] if name not in CLOCK_FEATURES),
    "PRICE_STRUCTURE_VOLATILITY_NO_CLOCK": tuple(dict.fromkeys(
        tuple(name for name in FEATURE_GROUPS["MARKET_CORE"] if name not in CLOCK_FEATURES)
        + FEATURE_GROUPS["STRUCTURE"]
        + FEATURE_GROUPS["VOLATILITY"]
    )),
}

MODEL_FAMILIES = ("LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING")

for _columns in PRICE_BASELINES.values():
    validate_feature_columns(_columns)
    if CLOCK_FEATURES.intersection(_columns):
        raise RuntimeError("V1.0B.5 price baselines must not use legacy wall-clock fields")


def _pipeline(family: str) -> Pipeline:
    if family == "LOGISTIC_REGRESSION":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
    if family == "HIST_GRADIENT_BOOSTING":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unsupported V1.0B.5 model family: {family}")


def _probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(x), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("V1.0B.5 requires binary DOWN/UP probabilities")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid V1.0B.5 probability matrix")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("V1.0B.5 probabilities do not sum to one")
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
    work["decision_timestamp_utc"] = pd.to_datetime(
        work["decision_timestamp_utc"], utc=True, errors="raise"
    )
    work["future_timestamp_utc"] = pd.to_datetime(
        work["future_timestamp_utc"], utc=True, errors="raise"
    )
    work["directional_target"] = work["directional_target"].astype(int)
    return work.sort_values("decision_timestamp_utc").reset_index(drop=True)


def quarantine_split(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep the same final-tail quarantine contract used by V1.0B.2/V1.0B.3."""
    if len(frame) < MIN_SOURCE_RESEARCH_ROWS + MIN_QUARANTINE_ROWS:
        raise ValueError("INSUFFICIENT_DIRECTIONAL_DATA_FOR_QUARANTINE")
    quarantine_rows = max(MIN_QUARANTINE_ROWS, int(len(frame) * QUARANTINE_FRACTION))
    quarantine_start_index = len(frame) - quarantine_rows
    quarantine_start = pd.Timestamp(frame["decision_timestamp_utc"].iloc[quarantine_start_index])
    eligible = frame["future_timestamp_utc"] < quarantine_start
    research = frame.loc[eligible].copy().reset_index(drop=True)
    if len(research) < MIN_SOURCE_RESEARCH_ROWS:
        raise ValueError("INSUFFICIENT_RESEARCH_ROWS_AFTER_QUARANTINE_PURGE")
    return research, {
        "quarantine_start_broker_epoch": quarantine_start.isoformat(),
        "quarantine_rows": int(quarantine_rows),
        "research_rows_before_context_alignment": int(len(research)),
        "quarantine_evaluated": False,
        "purge_ok": bool((research["future_timestamp_utc"] < quarantine_start).all()),
        "horizon_minutes": int(horizon),
        "timestamp_semantics": "MT5_BROKER_EPOCH_KEY_NOT_WALL_CLOCK_UTC",
    }


def _context_feature_columns(family: str) -> tuple[str, ...]:
    prefix = family.lower()
    return (
        f"ctx_{prefix}_ret_5m",
        f"ctx_{prefix}_ret_15m",
        f"ctx_{prefix}_ret_30m",
        f"ctx_{prefix}_ret_60m",
        f"ctx_{prefix}_vol_60m",
        f"ctx_{prefix}_range_fraction",
    )


def _rates_to_context_features(rates: Any, family: str) -> pd.DataFrame:
    """Convert completed M5 MT5 rates to causal features keyed by broker epoch.

    The timestamp is intentionally retained as a broker-clock key rather than
    re-labelled as wall-clock UTC. This lets old XAUUSD research rows and new
    context rows join exactly without inventing historical DST corrections.
    """
    frame = pd.DataFrame(rates)
    required = {"time", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{family} MT5 response missing fields: {', '.join(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=("decision_broker_timestamp",) + _context_feature_columns(family))

    timestamp = pd.to_datetime(
        pd.to_numeric(frame["time"], errors="raise"),
        unit="s",
        utc=True,
        errors="raise",
    )
    if timestamp.duplicated().any():
        raise ValueError(f"{family} context contains duplicate broker timestamps")

    work = pd.DataFrame({
        "decision_broker_timestamp": timestamp + pd.Timedelta(minutes=5),
        "open": pd.to_numeric(frame["open"], errors="raise"),
        "high": pd.to_numeric(frame["high"], errors="raise"),
        "low": pd.to_numeric(frame["low"], errors="raise"),
        "close": pd.to_numeric(frame["close"], errors="raise"),
    }).sort_values("decision_broker_timestamp").reset_index(drop=True)

    invalid = (
        (work["low"] > work["high"])
        | (work["open"] < work["low"])
        | (work["open"] > work["high"])
        | (work["close"] < work["low"])
        | (work["close"] > work["high"])
        | (work["close"] <= 0)
    )
    if invalid.any():
        raise ValueError(f"{family} context contains invalid OHLC geometry")

    prefix = family.lower()
    ret5 = work["close"].pct_change(1)
    work[f"ctx_{prefix}_ret_5m"] = ret5
    work[f"ctx_{prefix}_ret_15m"] = work["close"].pct_change(3)
    work[f"ctx_{prefix}_ret_30m"] = work["close"].pct_change(6)
    work[f"ctx_{prefix}_ret_60m"] = work["close"].pct_change(12)
    work[f"ctx_{prefix}_vol_60m"] = ret5.rolling(12, min_periods=12).std()
    work[f"ctx_{prefix}_range_fraction"] = (work["high"] - work["low"]) / work["close"]

    keep = ["decision_broker_timestamp", *_context_feature_columns(family)]
    return work.loc[:, keep]


def collect_context_history(
    mt5: Any,
    ready_families: dict[str, str],
    *,
    rows: int = CONTEXT_COLLECTION_ROWS,
    root: Path = ROOT,
    persist: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Collect deep completed-M5 context history using broker-discovered symbols."""
    if rows < MIN_CONTEXT_ROWS:
        raise ValueError("context collection rows below minimum evidence contract")

    frames: dict[str, pd.DataFrame] = {}
    manifest: dict[str, dict[str, Any]] = {}
    output_dir = asset_paths(PREDICTION_TARGET, root).market / "intermarket" / "v10b5"

    for family, symbol in ready_families.items():
        if family not in ALLOWED_CONTEXT_FAMILIES:
            continue
        if not mt5.symbol_select(symbol, True):
            manifest[family] = {
                "status": "UNAVAILABLE",
                "broker_symbol": symbol,
                "reason": "SYMBOL_SELECT_FAILED",
            }
            continue

        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, "TIMEFRAME_M5"), 1, rows)
        if rates is None:
            manifest[family] = {
                "status": "UNAVAILABLE",
                "broker_symbol": symbol,
                "reason": f"MT5_COPY_RATES_FAILED:{mt5.last_error()}",
            }
            continue

        features = _rates_to_context_features(rates, family)
        if len(features) < MIN_CONTEXT_ROWS:
            manifest[family] = {
                "status": "LIMITED",
                "broker_symbol": symbol,
                "rows": int(len(features)),
                "reason": "INSUFFICIENT_COMPLETED_M5_HISTORY",
            }
            continue

        frames[family] = features
        manifest[family] = {
            "status": "READY",
            "broker_symbol": symbol,
            "rows": int(len(features)),
            "first_decision_broker_timestamp": features["decision_broker_timestamp"].iloc[0].isoformat(),
            "last_decision_broker_timestamp": features["decision_broker_timestamp"].iloc[-1].isoformat(),
            "timestamp_semantics": "MT5_BROKER_EPOCH_KEY_NOT_WALL_CLOCK_UTC",
            "forming_bar_excluded": True,
        }
        if persist:
            atomic_write_parquet(features, output_dir / f"{family}_{symbol}_M5.parquet")

    return frames, manifest


def _ready_symbols(audit: dict[str, Any]) -> dict[str, str]:
    if audit.get("decision") not in {
        "INTRADAY_CONTEXT_READY_FOR_RESEARCH",
        "LIMITED_INTRADAY_CONTEXT_AVAILABLE",
    }:
        raise RuntimeError("V1.0B.4 audit did not authorize intraday context research")
    calibration = audit.get("broker_clock_calibration", {})
    if calibration.get("status") != "PASS":
        raise RuntimeError("V1.0B.4 broker clock calibration is not PASS")

    ready: dict[str, str] = {}
    for item in audit.get("families", []):
        family = str(item.get("family", ""))
        symbol = item.get("broker_symbol")
        if family in ALLOWED_CONTEXT_FAMILIES and item.get("status") == "READY" and symbol:
            ready[family] = str(symbol)
    if not ready:
        raise RuntimeError("No verified V1.0B.4 intraday context families are READY")
    return ready


def _group_columns(group_name: str) -> tuple[str, ...]:
    families = CONTEXT_GROUPS[group_name]
    names: list[str] = []
    for family in families:
        names.extend(_context_feature_columns(family))
    return tuple(names)


def exact_context_join(
    research: pd.DataFrame,
    context_frames: dict[str, pd.DataFrame],
    group_name: str,
) -> pd.DataFrame:
    """Inner-join context on the exact completed-M5 broker timestamp only."""
    required_families = CONTEXT_GROUPS[group_name]
    if any(family not in context_frames for family in required_families):
        return pd.DataFrame()

    result = research.copy()
    result["decision_broker_timestamp"] = pd.to_datetime(
        result["decision_timestamp_utc"], utc=True, errors="raise"
    )
    for family in required_families:
        context = context_frames[family].copy()
        context["decision_broker_timestamp"] = pd.to_datetime(
            context["decision_broker_timestamp"], utc=True, errors="raise"
        )
        if context["decision_broker_timestamp"].duplicated().any():
            raise ValueError(f"Duplicate exact-join timestamps in {family}")
        result = result.merge(
            context,
            on="decision_broker_timestamp",
            how="inner",
            validate="one_to_one",
        )

    result = result.sort_values("decision_broker_timestamp").reset_index(drop=True)
    if result["decision_broker_timestamp"].duplicated().any():
        raise ValueError("Exact context join produced duplicate decisions")
    return result


def _paired_evaluation(
    aligned: pd.DataFrame,
    *,
    horizon: int,
    baseline_name: str,
    context_group: str,
    family: str,
) -> dict[str, Any]:
    baseline_columns = PRICE_BASELINES[baseline_name]
    context_columns = _group_columns(context_group)
    all_columns = baseline_columns + context_columns

    missing = sorted(set(all_columns) - set(aligned.columns))
    if missing:
        return {
            "status": "MISSING_COLUMNS",
            "missing_columns": missing,
            "baseline": baseline_name,
            "context_group": context_group,
            "family": family,
        }

    complete = aligned.loc[:, list(all_columns)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    data = aligned.loc[complete].reset_index(drop=True)
    if len(data) < MIN_ALIGNED_RESEARCH_ROWS:
        return {
            "status": "INSUFFICIENT_PREQUARANTINE_OVERLAP",
            "rows": int(len(data)),
            "baseline": baseline_name,
            "context_group": context_group,
            "family": family,
        }

    folds = purged_walk_forward(
        data["decision_broker_timestamp"],
        horizon,
        n_splits=N_SPLITS,
        min_validation_rows=MIN_VALIDATION_ROWS,
    )

    baseline_fold_rows: list[dict[str, float]] = []
    context_fold_rows: list[dict[str, float]] = []
    prior_fold_rows: list[dict[str, float]] = []
    previous_fold_rows: list[dict[str, float]] = []

    for fold in folds:
        train = data.iloc[fold.train_indices]
        validation = data.iloc[fold.validation_indices]
        if train["directional_target"].nunique() < 2 or validation["directional_target"].nunique() < 2:
            return {
                "status": "INSUFFICIENT_CLASS_VARIATION",
                "rows": int(len(data)),
                "baseline": baseline_name,
                "context_group": context_group,
                "family": family,
            }

        baseline_model = _pipeline(family)
        baseline_model.fit(train.loc[:, list(baseline_columns)], train["directional_target"])
        baseline_prob = _probabilities(baseline_model, validation.loc[:, list(baseline_columns)])
        baseline_fold_rows.append(_metrics(validation, baseline_prob))

        context_model = _pipeline(family)
        context_model.fit(train.loc[:, list(all_columns)], train["directional_target"])
        context_prob = _probabilities(context_model, validation.loc[:, list(all_columns)])
        context_fold_rows.append(_metrics(validation, context_prob))

        prior_fold_rows.append(_metrics(
            validation,
            _class_prior(train["directional_target"], len(validation)),
        ))
        previous_fold_rows.append(_metrics(validation, _previous_direction(validation)))

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    context_beats_baseline = sum(
        context["balanced_accuracy"] > baseline["balanced_accuracy"]
        for context, baseline in zip(context_fold_rows, baseline_fold_rows)
    )

    return {
        "status": "EVALUATED",
        "baseline": baseline_name,
        "context_group": context_group,
        "context_families": list(CONTEXT_GROUPS[context_group]),
        "family": family,
        "rows": int(len(data)),
        "fold_count": int(len(folds)),
        "baseline_feature_count": int(len(baseline_columns)),
        "context_feature_count": int(len(context_columns)),
        "balanced_accuracy_mean": mean(context_fold_rows, "balanced_accuracy"),
        "baseline_balanced_accuracy_mean": mean(baseline_fold_rows, "balanced_accuracy"),
        "balanced_accuracy_delta": (
            mean(context_fold_rows, "balanced_accuracy")
            - mean(baseline_fold_rows, "balanced_accuracy")
        ),
        "macro_f1_mean": mean(context_fold_rows, "macro_f1"),
        "baseline_macro_f1_mean": mean(baseline_fold_rows, "macro_f1"),
        "log_loss_mean": mean(context_fold_rows, "log_loss"),
        "baseline_log_loss_mean": mean(baseline_fold_rows, "log_loss"),
        "class_prior_log_loss_mean": mean(prior_fold_rows, "log_loss"),
        "brier_mean": mean(context_fold_rows, "brier"),
        "baseline_brier_mean": mean(baseline_fold_rows, "brier"),
        "cost_aware_mean": mean(context_fold_rows, "cost_aware_metric"),
        "baseline_cost_aware_mean": mean(baseline_fold_rows, "cost_aware_metric"),
        "previous_direction_cost_aware_mean": mean(previous_fold_rows, "cost_aware_metric"),
        "recent_fold_balanced_accuracy": float(context_fold_rows[-1]["balanced_accuracy"]),
        "folds_beating_paired_baseline": int(context_beats_baseline),
        "context_folds": context_fold_rows,
        "baseline_folds": baseline_fold_rows,
    }


def _evidence_status(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("status") != "EVALUATED":
        return "NO_INTRADAY_CONTEXT_EVIDENCE", [str(row.get("status"))]

    failed: list[str] = []
    if row["balanced_accuracy_mean"] < MIN_MEAN_BALANCED_ACCURACY:
        failed.append("mean_balanced_accuracy")
    if row["recent_fold_balanced_accuracy"] < MIN_RECENT_BALANCED_ACCURACY:
        failed.append("recent_fold_balanced_accuracy")
    if row["folds_beating_paired_baseline"] < MIN_FOLDS_BEATING_PAIRED_BASELINE:
        failed.append("paired_fold_stability")
    if row["balanced_accuracy_delta"] < MIN_BALANCED_ACCURACY_DELTA:
        failed.append("balanced_accuracy_delta")
    if row["log_loss_mean"] > row["baseline_log_loss_mean"] - MIN_LOG_LOSS_IMPROVEMENT:
        failed.append("paired_log_loss")
    if row["log_loss_mean"] >= row["class_prior_log_loss_mean"]:
        failed.append("class_prior_log_loss")
    if row["cost_aware_mean"] < row["baseline_cost_aware_mean"]:
        failed.append("paired_cost_aware")
    return (
        "INTRADAY_CONTEXT_EVIDENCE_CANDIDATE" if not failed else "NO_INTRADAY_CONTEXT_EVIDENCE",
        failed,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: list[dict[str, Any]] = []
    for row in rows:
        copy = {key: value for key, value in row.items() if key not in {"context_folds", "baseline_folds"}}
        if isinstance(copy.get("failed_gates"), list):
            copy["failed_gates"] = ";".join(copy["failed_gates"])
        if isinstance(copy.get("context_families"), list):
            copy["context_families"] = ";".join(copy["context_families"])
        flat.append(copy)
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
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    else:
        captured = captured.astimezone(timezone.utc)

    audit = run_audit(mt5, root=root, persist=persist, captured_at=captured)
    ready_symbols = _ready_symbols(audit)
    context_frames, context_manifest = collect_context_history(
        mt5,
        ready_symbols,
        rows=context_rows,
        root=root,
        persist=persist,
    )
    usable_context = {
        family for family, item in context_manifest.items()
        if item.get("status") == "READY" and family in context_frames
    }

    source = load_xauusd_research_frame(root)
    results: list[dict[str, Any]] = []
    quarantine: dict[str, Any] = {}
    overlap_summary: dict[str, Any] = {}

    for horizon, contract in TARGET_SPECS:
        label = f"{horizon}m:{contract.name}"
        directional = _directional_frame(source, horizon, contract)
        research, quarantine_info = quarantine_split(directional, horizon)
        quarantine[label] = quarantine_info

        for context_group, required in CONTEXT_GROUPS.items():
            missing_families = [family for family in required if family not in usable_context]
            if missing_families:
                for baseline_name in PRICE_BASELINES:
                    for family in MODEL_FAMILIES:
                        results.append({
                            "asset_id": ASSET_ID,
                            "target_label": label,
                            "horizon_minutes": horizon,
                            "target_contract": asdict(contract),
                            "status": "CONTEXT_GROUP_UNAVAILABLE",
                            "evidence_status": "NO_INTRADAY_CONTEXT_EVIDENCE",
                            "failed_gates": ["CONTEXT_GROUP_UNAVAILABLE"],
                            "missing_context_families": missing_families,
                            "baseline": baseline_name,
                            "context_group": context_group,
                            "family": family,
                            "quarantine_evaluated": False,
                        })
                continue

            aligned = exact_context_join(research, context_frames, context_group)
            overlap_summary[f"{label}:{context_group}"] = {
                "rows": int(len(aligned)),
                "first_decision_broker_timestamp": (
                    aligned["decision_broker_timestamp"].iloc[0].isoformat()
                    if len(aligned) else None
                ),
                "last_decision_broker_timestamp": (
                    aligned["decision_broker_timestamp"].iloc[-1].isoformat()
                    if len(aligned) else None
                ),
                "exact_join_only": True,
                "forward_fill": False,
            }

            for baseline_name in PRICE_BASELINES:
                for family in MODEL_FAMILIES:
                    row = _paired_evaluation(
                        aligned,
                        horizon=horizon,
                        baseline_name=baseline_name,
                        context_group=context_group,
                        family=family,
                    )
                    evidence_status, failed = _evidence_status(row)
                    row.update({
                        "asset_id": ASSET_ID,
                        "target_label": label,
                        "horizon_minutes": horizon,
                        "target_contract": asdict(contract),
                        "evidence_status": evidence_status,
                        "failed_gates": failed,
                        "quarantine_evaluated": False,
                    })
                    results.append(row)

    candidates = [
        row for row in results
        if row.get("evidence_status") == "INTRADAY_CONTEXT_EVIDENCE_CANDIDATE"
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("balanced_accuracy_delta", -999.0)),
            float(row.get("balanced_accuracy_mean", -999.0)),
            -float(row.get("log_loss_mean", 999.0)),
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None

    evaluated = [row for row in results if row.get("status") == "EVALUATED"]
    evaluated.sort(
        key=lambda row: (
            float(row.get("balanced_accuracy_delta", -999.0)),
            float(row.get("balanced_accuracy_mean", -999.0)),
        ),
        reverse=True,
    )
    best_screen = evaluated[0] if evaluated else None

    decision = (
        "INTRADAY_CONTEXT_EVIDENCE_FOUND"
        if candidates else
        "INTRADAY_CONTEXT_EVIDENCE_WEAK"
    )
    if not evaluated:
        decision = "INSUFFICIENT_PREQUARANTINE_INTERMARKET_HISTORY"

    next_phase = {
        "INTRADAY_CONTEXT_EVIDENCE_FOUND": "V1.0B.6_CONTROLLED_XAUUSD_MODEL_RESEARCH",
        "INTRADAY_CONTEXT_EVIDENCE_WEAK": "EXPAND_OR_WAIT_FOR_VERIFIED_INTRADAY_CONTEXT",
        "INSUFFICIENT_PREQUARANTINE_INTERMARKET_HISTORY": "COLLECT_MORE_POINT_IN_TIME_CONTEXT_BEFORE_RESEARCH",
    }[decision]

    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "prediction_target": PREDICTION_TARGET,
        "decision": decision,
        "intraday_context_candidate_count": int(len(candidates)),
        "best_intraday_context_candidate": best,
        "best_screen_result": best_screen,
        "next_phase": next_phase,
        "context_read_only": True,
        "verified_ready_symbols": ready_symbols,
        "context_manifest": context_manifest,
        "broker_clock_calibration": audit.get("broker_clock_calibration"),
        "historical_join_clock": "RAW_MT5_BROKER_EPOCH_KEY",
        "wall_clock_features_used": False,
        "historical_dst_correction_invented": False,
        "exact_completed_m5_join_only": True,
        "forming_bar_excluded": True,
        "quarantine": quarantine,
        "overlap_summary": overlap_summary,
        "evaluated_rows": int(len(evaluated)),
        "model_promotion_performed": False,
        "production_integration": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }

    if persist:
        reports = asset_paths(PREDICTION_TARGET, root).reports
        _atomic_json(reports / "v10b5_intraday_intermarket_evidence.json", report)
        _write_csv(reports / "v10b5_intraday_intermarket_leaderboard.csv", results)

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
        report = run_research(
            mt5,
            persist=not args.no_persist,
            context_rows=args.context_rows,
        )
        compact = {
            "contract_version": report["contract_version"],
            "asset_id": report["asset_id"],
            "decision": report["decision"],
            "intraday_context_candidate_count": report["intraday_context_candidate_count"],
            "best_intraday_context_candidate": report["best_intraday_context_candidate"],
            "best_screen_result": report["best_screen_result"],
            "next_phase": report["next_phase"],
            "production_integration": report["production_integration"],
            "automatic_execution": report["automatic_execution"],
        }
        print(json.dumps(compact, indent=2, default=str))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
