"""Causal V0.5C dataset, target, feature-group, and fingerprint gates."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.intelligence.v05b_contract import CONTEXT_HISTORY_FILE
from src.learning.v05c_contract import (
    COMMISSION_STATUS,
    COST_BAND_MULTIPLIER,
    EURUSD_POINT_SIZE,
    HORIZONS,
    MARKET_FEATURES,
    MIN_COST_POINTS,
    MIN_INTELLIGENCE_CALENDAR_DAYS,
    MIN_INTELLIGENCE_JOINED_ROWS,
    MIN_LIVE_SURPRISE_ROWS,
    MIN_MARKET_CALENDAR_DAYS,
    MIN_MARKET_ROWS,
    ROOT,
)


@dataclass(frozen=True)
class HorizonDataset:
    horizon_minutes: int
    frame: pd.DataFrame
    features: tuple[str, ...]
    fingerprint: str
    feature_contract_hash: str


def validate_feature_registry(features: tuple[str, ...] | list[str]) -> None:
    if not features:
        raise ValueError("Feature registry is empty")
    duplicates = pd.Index(features).duplicated()
    if duplicates.any():
        raise ValueError("Duplicate feature names are forbidden")
    forbidden = [column for column in features if column.lower().startswith(("outcome_", "future_", "target_")) or "future_timestamp" in column.lower()]
    if forbidden:
        raise ValueError("Outcome/future/target-derived fields are forbidden: " + ", ".join(forbidden))


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def validate_availability(frame: pd.DataFrame) -> int:
    decision = _utc(frame["decision_timestamp_utc"])
    violations = 0
    for column in ("m1_available_from_utc", "m5_available_from_utc", "m15_available_from_utc", "h1_available_from_utc"):
        if column not in frame:
            raise ValueError(f"Missing availability column: {column}")
        available = pd.to_datetime(frame[column], utc=True, errors="coerce")
        violations += int((available.notna() & available.gt(decision)).sum())
    if violations:
        raise ValueError(f"Future feature availability violations: {violations}")
    return violations


def decision_cost_band(frame: pd.DataFrame) -> pd.Series:
    """Observable spread/noise threshold using decision-time values only."""
    spread = pd.to_numeric(frame["m5_spread_points"], errors="raise")
    close = pd.to_numeric(frame["m5_close"], errors="raise")
    if (spread < 0).any() or (close <= 0).any():
        raise ValueError("Invalid decision-time spread or close")
    conservative_points = spread.clip(lower=MIN_COST_POINTS)
    return conservative_points * EURUSD_POINT_SIZE * COST_BAND_MULTIPLIER / close


def target_class(raw_return: pd.Series, cost_band: pd.Series) -> pd.Series:
    result = pd.Series(1, index=raw_return.index, dtype="int8")
    result.loc[raw_return > cost_band] = 2
    result.loc[raw_return < -cost_band] = 0
    return result


def _fingerprint_frame(frame: pd.DataFrame, contract: dict[str, object]) -> str:
    canonical = frame.copy()
    for column in canonical:
        if isinstance(canonical[column].dtype, pd.DatetimeTZDtype):
            canonical[column] = canonical[column].astype("int64")
    hashed = pd.util.hash_pandas_object(canonical, index=False, categorize=False).to_numpy(dtype="uint64")
    digest = sha256()
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def feature_contract_hash(features: tuple[str, ...], horizon_minutes: int) -> str:
    payload = {
        "features": list(features),
        "horizon_minutes": horizon_minutes,
        "availability_rule": "feature_complete == True; every MARKET_CORE feature is available at or before decision time",
        "target": "DOWN if return < -decision_cost_band; NEUTRAL inside band; UP if return > band",
        "cost": {"point_size": EURUSD_POINT_SIZE, "minimum_points": MIN_COST_POINTS, "multiplier": COST_BAND_MULTIPLIER, "commission_status": COMMISSION_STATUS},
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_horizon_dataset(source: pd.DataFrame, horizon_minutes: int, as_of: object | None = None) -> HorizonDataset:
    if horizon_minutes not in HORIZONS:
        raise ValueError(f"Unsupported independent horizon: {horizon_minutes}")
    features = tuple(MARKET_FEATURES)
    validate_feature_registry(features)
    return_column = f"outcome_future_return_{horizon_minutes}m"
    future_column = f"outcome_future_timestamp_{horizon_minutes}m"
    matured_column = f"outcome_matured_at_utc_{horizon_minutes}m"
    required = {"decision_timestamp_utc", "m5_close", "m5_spread_points", return_column, future_column, matured_column, *features}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("Missing V0.5A training columns: " + ", ".join(missing))
    validate_availability(source)

    work = source.copy()
    work["decision_timestamp_utc"] = _utc(work["decision_timestamp_utc"])
    work[future_column] = pd.to_datetime(work[future_column], utc=True, errors="coerce")
    work[matured_column] = pd.to_datetime(work[matured_column], utc=True, errors="coerce")
    cutoff = work["decision_timestamp_utc"].max() if as_of is None else pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")

    complete = work["feature_complete"].fillna(False).astype(bool) if "feature_complete" in work else work.loc[:, list(features)].notna().all(axis=1)
    matured = work[return_column].notna() & work[future_column].notna() & work[matured_column].notna()
    matured &= work[matured_column].le(cutoff)
    eligible = work.loc[complete & matured].copy()

    expected = eligible["decision_timestamp_utc"] + pd.Timedelta(minutes=horizon_minutes)
    wrong = eligible[future_column].ne(expected) | eligible[matured_column].ne(expected)
    if wrong.any():
        raise ValueError(f"Matured {horizon_minutes}m labels violate exact horizon timing: {int(wrong.sum())}")
    if eligible.loc[:, list(features)].isna().any(axis=None):
        raise ValueError("feature_complete rows contain missing MARKET_CORE features")

    eligible["raw_future_return"] = pd.to_numeric(eligible[return_column], errors="raise")
    eligible["decision_cost_band"] = decision_cost_band(eligible)
    eligible["target_class"] = target_class(eligible["raw_future_return"], eligible["decision_cost_band"])
    eligible["target_matured_at_utc"] = eligible[matured_column]
    ordered = ["decision_timestamp_utc", *features, "target_class", "raw_future_return", "decision_cost_band", "target_matured_at_utc"]
    eligible = eligible[ordered].sort_values("decision_timestamp_utc").reset_index(drop=True)
    contract_hash = feature_contract_hash(features, horizon_minutes)
    fingerprint = _fingerprint_frame(eligible, {"horizon_minutes": horizon_minutes, "feature_contract_hash": contract_hash, "availability_rule": "feature_complete", "rows": len(eligible), "columns": ordered})
    return HorizonDataset(horizon_minutes, eligible, features, fingerprint, contract_hash)


def load_horizon_dataset(path: Path, horizon_minutes: int) -> HorizonDataset:
    return build_horizon_dataset(pd.read_parquet(path, engine="pyarrow"), horizon_minutes)


def feature_group_eligibility(market: pd.DataFrame, intelligence: pd.DataFrame | None = None, live_surprises: pd.DataFrame | None = None, intelligence_validation_passed: bool = False) -> pd.DataFrame:
    decisions = _utc(market["decision_timestamp_utc"])
    complete = market["feature_complete"].fillna(False).astype(bool) if "feature_complete" in market else market.loc[:, list(MARKET_FEATURES)].notna().all(axis=1)
    all_matured = pd.Series(True, index=market.index)
    for horizon in HORIZONS:
        all_matured &= market[f"outcome_future_return_{horizon}m"].notna()
        matured_column = f"outcome_matured_at_utc_{horizon}m"
        if matured_column in market:
            all_matured &= pd.to_datetime(market[matured_column], utc=True, errors="coerce").notna()
    market_mask = complete & all_matured
    market_rows = int(market_mask.sum())
    market_days = int(decisions.loc[market_mask].dt.floor("D").nunique())

    intelligence = intelligence if intelligence is not None else pd.DataFrame()
    if intelligence.empty:
        intelligence_rows = intelligence_days = 0
    else:
        intel_time = _utc(intelligence["decision_timestamp_utc"])
        eligible_time = intel_time <= decisions.max()
        intelligence_rows = int(eligible_time.sum())
        intelligence_days = int(intel_time.loc[eligible_time].dt.floor("D").nunique())
    surprise_rows = 0 if live_surprises is None else len(live_surprises)
    rows = [
        {"group": "MARKET_CORE", "eligible": market_rows >= MIN_MARKET_ROWS and market_days >= MIN_MARKET_CALENDAR_DAYS, "rows": market_rows, "days": market_days, "reason": f"requires feature_complete plus all horizons matured, >= {MIN_MARKET_ROWS} rows and >= {MIN_MARKET_CALENDAR_DAYS} eligible decision days"},
        {"group": "INTELLIGENCE_V05B", "eligible": intelligence_rows >= MIN_INTELLIGENCE_JOINED_ROWS and intelligence_days >= MIN_INTELLIGENCE_CALENDAR_DAYS and intelligence_validation_passed, "rows": intelligence_rows, "days": intelligence_days, "reason": f"INSUFFICIENT_HISTORY until >= {MIN_INTELLIGENCE_JOINED_ROWS} causal joined rows, >= {MIN_INTELLIGENCE_CALENDAR_DAYS} days, and chronological validation"},
        {"group": "EVENT_MEMORY", "eligible": False, "rows": 0, "days": 0, "reason": "RESEARCH_ONLY_DISABLED: no independently audited V0.5C join contract"},
        {"group": "LIVE_SURPRISE", "eligible": surprise_rows >= MIN_LIVE_SURPRISE_ROWS, "rows": surprise_rows, "days": 0, "reason": f"INSUFFICIENT_HISTORY until >= {MIN_LIVE_SURPRISE_ROWS} verified live surprise samples"},
    ]
    return pd.DataFrame(rows)


def load_feature_group_eligibility(market: pd.DataFrame) -> pd.DataFrame:
    intelligence = pd.read_parquet(CONTEXT_HISTORY_FILE) if CONTEXT_HISTORY_FILE.exists() else pd.DataFrame()
    surprise_path = ROOT / "data" / "mt5" / "calendar" / "v04d_live_verified_surprises.csv"
    surprises = pd.read_csv(surprise_path) if surprise_path.exists() else pd.DataFrame()
    return feature_group_eligibility(market, intelligence, surprises)
