"""Controlled XAUUSD-only model research and promotion for MarketFusion V1.0B.

This module is research/shadow only. It never places orders and never reuses
EURUSD model artifacts. Historical labels come from the V1.0A causal target
store; a final chronological holdout is kept outside feature/model selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.assets.contracts import ROOT, asset_paths, normalize_asset_id
from src.assets.registry import approved_champions, model_id
from src.learning.v05c_models import multiclass_brier
from src.learning.v05c_walk_forward import purged_walk_forward

CONTRACT_VERSION = "v1.0b-xauusd-controlled-model-research-v1"
ASSET_ID = "XAUUSD"
HORIZONS = (15, 60, 240)
CLASS_LABELS = (0, 1, 2)
CLASS_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
RANDOM_STATE = 1705

FINAL_HOLDOUT_FRACTION = 0.15
MIN_FINAL_HOLDOUT_ROWS = 500
MIN_RESEARCH_ROWS = 2_000
MIN_PROMOTION_HISTORY_DAYS = 90
MIN_PROMOTION_FOLDS = 5
MIN_CALIBRATION_ROWS = 250
MIN_BASE_TRAIN_ROWS = 1_000

PROMOTION_BALANCED_ACCURACY_MARGIN = 0.02
PROMOTION_MACRO_F1_MARGIN = 0.02
PROMOTION_LOG_LOSS_MARGIN = 0.01
MAX_RECENT_FOLD_DEGRADATION = 0.08
MAX_WORST_FOLD_DEFICIT = 0.05

BASELINES = ("MAJORITY_CLASS", "CLASS_PRIOR", "PREVIOUS_DIRECTION")

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "MARKET_CORE": (
        "m5_return_5m", "m5_return_15m", "m5_return_60m", "m5_return_240m",
        "m5_volatility_60m", "m5_volatility_240m", "m5_spread_points",
        "spread_relative_to_price", "spread_relative_to_atr", "atr_14",
        "utc_hour", "day_of_week",
    ),
    "TECHNICAL": (
        "ema_12_gap_atr", "ema_26_gap_atr", "ema_12_26_gap_atr",
        "sma_20_gap_atr", "rsi_14", "close_zscore_20",
    ),
    "PRICE_ACTION": (
        "bar_range_atr", "body_atr", "body_fraction", "upper_wick_fraction",
        "lower_wick_fraction", "close_location",
    ),
    "STRUCTURE": (
        "distance_prior_high_20_atr", "distance_prior_low_20_atr",
        "distance_prior_high_50_atr", "distance_prior_low_50_atr",
        "breakout_high_20", "breakdown_low_20", "trend_return_12",
    ),
    "LIQUIDITY": (
        "sweep_high_20", "sweep_low_20", "range_expansion_atr",
        "spread_atr_ratio",
    ),
    "SUPPORT_RESISTANCE": (
        "distance_support_20_atr", "distance_resistance_20_atr",
        "distance_support_50_atr", "distance_resistance_50_atr",
        "distance_ema_20_atr", "distance_ema_50_atr",
    ),
    "VOLATILITY": (
        "atr_fraction", "volatility_ratio_60_240", "range_volatility_20",
        "spread_price_fraction", "spread_atr_ratio",
    ),
    "SESSION": (
        "hour_sin", "hour_cos", "is_sydney_session", "is_tokyo_session",
        "is_london_session", "is_new_york_session", "is_london_ny_overlap",
    ),
    "REGIME": (
        "trend_strength_atr", "volatility_regime_low", "volatility_regime_high",
        "trend_regime_up", "trend_regime_down",
    ),
    "PATTERNS": (
        "pattern_doji", "pattern_bullish_engulfing", "pattern_bearish_engulfing",
        "pattern_bullish_pin", "pattern_bearish_pin", "pattern_inside_bar",
        "pattern_outside_bar",
    ),
}


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def _rsi(close: pd.Series, periods: int = 14) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gain = delta.clip(lower=0).rolling(periods, min_periods=periods).mean()
    loss = (-delta.clip(upper=0)).rolling(periods, min_periods=periods).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.where(loss.ne(0), 100.0).where(gain.ne(0), 0.0)


def _session_features(timestamp: pd.Series) -> pd.DataFrame:
    """DST-aware local-clock flags matching the canonical session schedules."""
    stamp = _utc(timestamp)
    weekday = stamp.dt.dayofweek.lt(5)
    sydney = stamp.dt.tz_convert("Australia/Sydney")
    tokyo = stamp.dt.tz_convert("Asia/Tokyo")
    london = stamp.dt.tz_convert("Europe/London")
    new_york = stamp.dt.tz_convert("America/New_York")
    flags = pd.DataFrame(index=timestamp.index)
    flags["is_sydney_session"] = (weekday & sydney.dt.hour.between(8, 16)).astype("int8")
    flags["is_tokyo_session"] = (weekday & tokyo.dt.hour.between(9, 17)).astype("int8")
    flags["is_london_session"] = (weekday & london.dt.hour.between(8, 16)).astype("int8")
    flags["is_new_york_session"] = (weekday & new_york.dt.hour.between(8, 16)).astype("int8")
    flags["is_london_ny_overlap"] = (
        flags["is_london_session"].astype(bool) & flags["is_new_york_session"].astype(bool)
    ).astype("int8")
    hour = stamp.dt.hour + stamp.dt.minute / 60.0
    flags["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    flags["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return flags


def build_research_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic completed-M5 features using only current/past bars."""
    required = {
        "decision_timestamp_utc", "bar_open", "bar_high", "bar_low", "bar_close",
        "m5_return_5m", "m5_return_15m", "m5_return_60m", "m5_return_240m",
        "m5_volatility_60m", "m5_volatility_240m", "m5_spread_points",
        "spread_relative_to_price", "spread_relative_to_atr", "atr_14",
        "utc_hour", "day_of_week",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing XAUUSD research columns: " + ", ".join(missing))

    result = frame.sort_values("decision_timestamp_utc").reset_index(drop=True).copy()
    result["decision_timestamp_utc"] = _utc(result["decision_timestamp_utc"])
    if result["decision_timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate XAUUSD decision timestamps are forbidden")

    open_ = pd.to_numeric(result["bar_open"], errors="coerce")
    high = pd.to_numeric(result["bar_high"], errors="coerce")
    low = pd.to_numeric(result["bar_low"], errors="coerce")
    close = pd.to_numeric(result["bar_close"], errors="coerce")
    invalid_ohlc = (low > high) | (open_ < low) | (open_ > high) | (close < low) | (close > high)
    if invalid_ohlc.fillna(True).any():
        raise ValueError("Invalid XAUUSD OHLC geometry")
    atr = pd.to_numeric(result["atr_14"], errors="coerce").replace(0, np.nan)
    bar_range = (high - low).clip(lower=0)
    body = close - open_
    abs_body = body.abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()

    result["ema_12_gap_atr"] = (close - ema12) / atr
    result["ema_26_gap_atr"] = (close - ema26) / atr
    result["ema_12_26_gap_atr"] = (ema12 - ema26) / atr
    result["sma_20_gap_atr"] = (close - sma20) / atr
    result["rsi_14"] = _rsi(close) / 100.0
    result["close_zscore_20"] = (close - sma20) / std20.replace(0, np.nan)

    result["bar_range_atr"] = bar_range / atr
    result["body_atr"] = body / atr
    result["body_fraction"] = _safe_div(body, bar_range)
    result["upper_wick_fraction"] = _safe_div(upper_wick, bar_range)
    result["lower_wick_fraction"] = _safe_div(lower_wick, bar_range)
    result["close_location"] = _safe_div(close - low, bar_range)

    prior_high20 = high.shift(1).rolling(20, min_periods=20).max()
    prior_low20 = low.shift(1).rolling(20, min_periods=20).min()
    prior_high50 = high.shift(1).rolling(50, min_periods=50).max()
    prior_low50 = low.shift(1).rolling(50, min_periods=50).min()
    result["distance_prior_high_20_atr"] = (prior_high20 - close) / atr
    result["distance_prior_low_20_atr"] = (close - prior_low20) / atr
    result["distance_prior_high_50_atr"] = (prior_high50 - close) / atr
    result["distance_prior_low_50_atr"] = (close - prior_low50) / atr
    result["breakout_high_20"] = (close > prior_high20).astype("int8")
    result["breakdown_low_20"] = (close < prior_low20).astype("int8")
    result["trend_return_12"] = close.pct_change(12)

    result["sweep_high_20"] = ((high > prior_high20) & (close <= prior_high20)).astype("int8")
    result["sweep_low_20"] = ((low < prior_low20) & (close >= prior_low20)).astype("int8")
    result["range_expansion_atr"] = bar_range / atr
    result["spread_atr_ratio"] = pd.to_numeric(result["spread_relative_to_atr"], errors="coerce")

    result["distance_support_20_atr"] = (close - prior_low20) / atr
    result["distance_resistance_20_atr"] = (prior_high20 - close) / atr
    result["distance_support_50_atr"] = (close - prior_low50) / atr
    result["distance_resistance_50_atr"] = (prior_high50 - close) / atr
    result["distance_ema_20_atr"] = (close - ema20) / atr
    result["distance_ema_50_atr"] = (close - ema50) / atr

    result["atr_fraction"] = atr / close.replace(0, np.nan)
    result["volatility_ratio_60_240"] = _safe_div(
        result["m5_volatility_60m"], result["m5_volatility_240m"]
    )
    result["range_volatility_20"] = _safe_div(
        bar_range.rolling(20, min_periods=20).mean(), atr
    )
    result["spread_price_fraction"] = pd.to_numeric(result["spread_relative_to_price"], errors="coerce")

    sessions = _session_features(result["decision_timestamp_utc"])
    for column in sessions:
        result[column] = sessions[column]

    trend_strength = (ema20 - ema50) / atr
    volatility = pd.to_numeric(result["m5_volatility_60m"], errors="coerce")
    rolling_low = volatility.rolling(500, min_periods=100).quantile(1 / 3)
    rolling_high = volatility.rolling(500, min_periods=100).quantile(2 / 3)
    result["trend_strength_atr"] = trend_strength
    result["volatility_regime_low"] = (volatility < rolling_low).astype("int8")
    result["volatility_regime_high"] = (volatility > rolling_high).astype("int8")
    result["trend_regime_up"] = (trend_strength > 0.25).astype("int8")
    result["trend_regime_down"] = (trend_strength < -0.25).astype("int8")

    previous_open = open_.shift(1)
    previous_close = close.shift(1)
    previous_high = high.shift(1)
    previous_low = low.shift(1)
    previous_body = previous_close - previous_open
    doji_threshold = (0.10 * bar_range).fillna(np.inf)
    result["pattern_doji"] = (abs_body <= doji_threshold).astype("int8")
    result["pattern_bullish_engulfing"] = (
        (previous_body < 0) & (body > 0) & (open_ <= previous_close) & (close >= previous_open)
    ).astype("int8")
    result["pattern_bearish_engulfing"] = (
        (previous_body > 0) & (body < 0) & (open_ >= previous_close) & (close <= previous_open)
    ).astype("int8")
    result["pattern_bullish_pin"] = (
        (lower_wick >= 2.0 * abs_body) & (upper_wick <= abs_body) & (body >= 0)
    ).astype("int8")
    result["pattern_bearish_pin"] = (
        (upper_wick >= 2.0 * abs_body) & (lower_wick <= abs_body) & (body <= 0)
    ).astype("int8")
    result["pattern_inside_bar"] = ((high < previous_high) & (low > previous_low)).astype("int8")
    result["pattern_outside_bar"] = ((high > previous_high) & (low < previous_low)).astype("int8")
    return result


def load_xauusd_research_frame(root: Path = ROOT) -> pd.DataFrame:
    """Join V1.0A completed-bar features and causal targets by decision timestamp."""
    paths = asset_paths(ASSET_ID, root=root)
    feature_path = paths.features / "features.parquet"
    target_path = paths.features / "target_research.parquet"
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            "XAUUSD V1.0A stores are missing. Run the V1.0A XAUUSD collector/audit locally first."
        )
    features = pd.read_parquet(feature_path, engine="pyarrow").copy()
    targets = pd.read_parquet(target_path, engine="pyarrow").copy()
    if "decision_timestamp_utc" not in features or "bar_close_utc" not in targets:
        raise ValueError("Invalid V1.0A XAUUSD feature/target contract")
    features["decision_timestamp_utc"] = _utc(features["decision_timestamp_utc"])
    targets["decision_timestamp_utc"] = _utc(targets["bar_close_utc"])

    bar_columns = ["decision_timestamp_utc", "open", "high", "low", "close", "spread_points"]
    target_columns: list[str] = []
    for horizon in HORIZONS:
        target_columns.extend([
            f"outcome_future_timestamp_{horizon}m",
            f"outcome_future_return_{horizon}m",
            f"target_neutral_band_{horizon}m",
            f"target_class_{horizon}m",
        ])
    missing = sorted(set(bar_columns + target_columns) - set(targets.columns))
    if missing:
        raise ValueError("Missing V1.0A target columns: " + ", ".join(missing))
    selected = targets.loc[:, bar_columns + target_columns].rename(columns={
        "open": "bar_open", "high": "bar_high", "low": "bar_low",
        "close": "bar_close", "spread_points": "bar_spread_points",
    })
    merged = features.merge(selected, on="decision_timestamp_utc", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("XAUUSD feature/target join produced zero rows")
    if "close" in merged.columns:
        close_gap = (
            pd.to_numeric(merged["close"], errors="coerce")
            - pd.to_numeric(merged["bar_close"], errors="coerce")
        ).abs()
        if close_gap.fillna(np.inf).gt(1e-9).any():
            raise ValueError("XAUUSD feature/bar close mismatch")
    return build_research_features(merged)


def feature_set(group: str) -> tuple[str, ...]:
    if group == "BASELINE":
        groups = ("MARKET_CORE",)
    elif group == "ALL_V10B":
        groups = tuple(FEATURE_GROUPS)
    elif group.startswith("BASELINE_PLUS_"):
        name = group.removeprefix("BASELINE_PLUS_")
        if name not in FEATURE_GROUPS or name == "MARKET_CORE":
            raise ValueError(f"Unknown ablation feature group: {group}")
        groups = ("MARKET_CORE", name)
    else:
        raise ValueError(f"Unknown feature set: {group}")
    ordered: list[str] = []
    for name in groups:
        for column in FEATURE_GROUPS[name]:
            if column not in ordered:
                ordered.append(column)
    return tuple(ordered)


def ablation_names() -> tuple[str, ...]:
    return (
        "BASELINE",
        *tuple(f"BASELINE_PLUS_{name}" for name in FEATURE_GROUPS if name != "MARKET_CORE"),
        "ALL_V10B",
    )


def validate_feature_columns(columns: Iterable[str]) -> None:
    values = tuple(columns)
    if not values:
        raise ValueError("Feature set cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate model features are forbidden")
    forbidden = [
        column for column in values
        if column.lower().startswith(("outcome_", "future_", "target_"))
        or "future_timestamp" in column.lower()
    ]
    if forbidden:
        raise ValueError("Target/future fields are forbidden from model features: " + ", ".join(forbidden))


@dataclass(frozen=True)
class HorizonData:
    horizon_minutes: int
    frame: pd.DataFrame
    feature_names: tuple[str, ...]


def build_horizon_data(source: pd.DataFrame, horizon_minutes: int, features: tuple[str, ...]) -> HorizonData:
    if horizon_minutes not in HORIZONS:
        raise ValueError("Unsupported XAUUSD horizon")
    validate_feature_columns(features)
    target = f"target_class_{horizon_minutes}m"
    future_time = f"outcome_future_timestamp_{horizon_minutes}m"
    raw_return = f"outcome_future_return_{horizon_minutes}m"
    neutral = f"target_neutral_band_{horizon_minutes}m"
    required = {"decision_timestamp_utc", target, future_time, raw_return, neutral, *features}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError("Missing V1.0B columns: " + ", ".join(missing))
    work = source.copy()
    work["decision_timestamp_utc"] = _utc(work["decision_timestamp_utc"])
    work[future_time] = pd.to_datetime(work[future_time], utc=True, errors="coerce")
    complete = work.loc[:, list(features)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    matured = work[target].notna() & work[future_time].notna() & work[raw_return].notna() & work[neutral].notna()
    eligible = work.loc[complete & matured].copy()
    expected = eligible["decision_timestamp_utc"] + pd.Timedelta(minutes=horizon_minutes)
    if eligible[future_time].ne(expected).any():
        raise ValueError(f"XAUUSD {horizon_minutes}m target violates exact completed-bar horizon")
    if (eligible[future_time] <= eligible["decision_timestamp_utc"]).any():
        raise ValueError("Future target timestamp must be strictly after decision time")
    eligible["target_class"] = pd.to_numeric(eligible[target], errors="raise").astype(int)
    if not set(eligible["target_class"].unique()).issubset(set(CLASS_LABELS)):
        raise ValueError("Unexpected target class")
    eligible["raw_future_return"] = pd.to_numeric(eligible[raw_return], errors="raise")
    eligible["decision_cost_band"] = pd.to_numeric(eligible[neutral], errors="raise")
    keep = [
        "decision_timestamp_utc", *features, "target_class",
        "raw_future_return", "decision_cost_band", future_time,
    ]
    eligible = eligible.loc[:, keep].sort_values("decision_timestamp_utc").reset_index(drop=True)
    if eligible["decision_timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate eligible decision timestamps")
    return HorizonData(horizon_minutes, eligible, features)


@dataclass(frozen=True)
class ChronologicalSplit:
    research_indices: np.ndarray
    holdout_indices: np.ndarray
    holdout_start: pd.Timestamp


def final_holdout_split(data: HorizonData) -> ChronologicalSplit:
    times = _utc(data.frame["decision_timestamp_utc"]).reset_index(drop=True)
    if len(times) < MIN_RESEARCH_ROWS + MIN_FINAL_HOLDOUT_ROWS:
        raise ValueError("INSUFFICIENT_DATA for independent final holdout")
    holdout_rows = max(MIN_FINAL_HOLDOUT_ROWS, int(len(times) * FINAL_HOLDOUT_FRACTION))
    holdout_start_index = len(times) - holdout_rows
    holdout_start = times.iloc[holdout_start_index]
    research_mask = (times + pd.Timedelta(minutes=data.horizon_minutes)) < holdout_start
    research = np.flatnonzero(research_mask.to_numpy())
    holdout = np.arange(holdout_start_index, len(times))
    if len(research) < MIN_RESEARCH_ROWS or len(holdout) < MIN_FINAL_HOLDOUT_ROWS:
        raise ValueError("INSUFFICIENT_DATA after final-holdout purge")
    if (times.iloc[research] + pd.Timedelta(minutes=data.horizon_minutes) >= holdout_start).any():
        raise ValueError("Final holdout purge failed")
    return ChronologicalSplit(research, holdout, holdout_start)


def available_model_families() -> tuple[list[str], list[str]]:
    families = ["LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING", "RANDOM_FOREST"]
    unavailable: list[str] = []
    try:
        import xgboost  # noqa: F401
        families.append("XGBOOST")
    except (ImportError, OSError) as exc:
        unavailable.append(f"XGBOOST:{type(exc).__name__}")
    return families, unavailable


def _build_estimator(family: str) -> Pipeline:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    if family == "LOGISTIC_REGRESSION":
        return Pipeline([
            ("imputer", imputer),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(
                max_iter=3_000, class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ])
    if family == "HIST_GRADIENT_BOOSTING":
        return Pipeline([
            ("imputer", imputer),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=0.1, random_state=RANDOM_STATE,
            )),
        ])
    if family == "RANDOM_FOREST":
        return Pipeline([
            ("imputer", imputer),
            ("classifier", RandomForestClassifier(
                n_estimators=160, max_depth=10, min_samples_leaf=8,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ])
    if family == "XGBOOST":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:
            raise RuntimeError("XGBoost is unavailable in this environment") from exc
        return Pipeline([
            ("imputer", imputer),
            ("classifier", XGBClassifier(
                objective="multi:softprob", num_class=3, n_estimators=120,
                max_depth=3, learning_rate=0.04, subsample=0.9,
                colsample_bytree=0.9, eval_metric="mlogloss", n_jobs=1,
                random_state=RANDOM_STATE,
            )),
        ])
    raise ValueError(f"Unsupported V1.0B model family: {family}")


def validate_probability_matrix(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected Nx3 DOWN/NEUTRAL/UP probabilities")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain NaN/Infinity")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities outside [0,1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities do not sum to one")


def _aligned_probabilities(model: Any, x: Any) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    aligned = np.full((len(raw), 3), 1e-12, dtype=float)
    for source_index, label in enumerate(model.classes_):
        aligned[:, CLASS_LABELS.index(int(label))] = raw[:, source_index]
    row_sum = aligned.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sum)) or np.any(row_sum <= 0):
        raise ValueError("Invalid probability normalization")
    aligned /= row_sum
    validate_probability_matrix(aligned)
    return aligned


@dataclass
class TemporalModelBundle:
    asset_id: str
    horizon_minutes: int
    family: str
    feature_names: tuple[str, ...]
    base_model: Any
    calibrator: Any | None
    calibration_status: str
    feature_contract_hash: str

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        ordered = x.loc[:, list(self.feature_names)]
        base_probability = _aligned_probabilities(self.base_model, ordered)
        if self.calibrator is None:
            return base_probability
        logs = np.log(np.clip(base_probability, 1e-12, 1.0))
        return _aligned_probabilities(self.calibrator, logs)


def _feature_hash(features: tuple[str, ...], horizon: int) -> str:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "asset_id": ASSET_ID,
        "horizon_minutes": horizon,
        "features": list(features),
        "target": "V1.0A_COST_1_5X_ATR_10_EXACT_FUTURE_COMPLETED_M5",
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_temporal_model(
    family: str,
    x: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    horizon_minutes: int,
) -> TemporalModelBundle:
    validate_feature_columns(tuple(x.columns))
    times = _utc(timestamps).reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = pd.to_numeric(y, errors="raise").astype(int).reset_index(drop=True)
    if y.nunique() != 3:
        raise ValueError("Training partition must contain all three XAUUSD classes")
    calibration_rows = max(MIN_CALIBRATION_ROWS, int(len(x) * 0.20))
    start = len(x) - calibration_rows
    if start > 0:
        calibration_start = times.iloc[start]
        base_indices = np.flatnonzero(
            ((times + pd.Timedelta(minutes=horizon_minutes)) < calibration_start).to_numpy()
        )
        calibration_indices = np.arange(start, len(times))
        enough = (
            len(base_indices) >= MIN_BASE_TRAIN_ROWS
            and len(calibration_indices) >= MIN_CALIBRATION_ROWS
            and y.iloc[base_indices].nunique() == 3
            and y.iloc[calibration_indices].nunique() == 3
        )
        if enough:
            base = _build_estimator(family)
            base.fit(x.iloc[base_indices], y.iloc[base_indices])
            base_probability = _aligned_probabilities(base, x.iloc[calibration_indices])
            calibrator = LogisticRegression(max_iter=3_000, random_state=RANDOM_STATE)
            calibrator.fit(
                np.log(np.clip(base_probability, 1e-12, 1.0)),
                y.iloc[calibration_indices],
            )
            return TemporalModelBundle(
                ASSET_ID, horizon_minutes, family, tuple(x.columns), base, calibrator,
                "SIGMOID_TEMPORAL_HOLDOUT", _feature_hash(tuple(x.columns), horizon_minutes),
            )
    base = _build_estimator(family)
    base.fit(x, y)
    return TemporalModelBundle(
        ASSET_ID, horizon_minutes, family, tuple(x.columns), base, None,
        "INSUFFICIENT_SAMPLE", _feature_hash(tuple(x.columns), horizon_minutes),
    )


def baseline_probabilities(name: str, train_target: pd.Series, validation: pd.DataFrame) -> np.ndarray:
    counts = train_target.value_counts().reindex(CLASS_LABELS, fill_value=0).astype(float)
    priors = ((counts + 1.0) / (counts.sum() + len(CLASS_LABELS))).to_numpy()
    if name == "CLASS_PRIOR":
        result = np.tile(priors, (len(validation), 1))
    elif name == "MAJORITY_CLASS":
        majority = int(counts.idxmax())
        result = np.full((len(validation), 3), 0.01)
        result[:, majority] = 0.98
    elif name == "PREVIOUS_DIRECTION":
        signal = pd.to_numeric(validation["m5_return_5m"], errors="coerce").fillna(0.0).to_numpy()
        predicted = np.where(signal > 0, 2, np.where(signal < 0, 0, 1))
        result = np.full((len(validation), 3), 0.05)
        result[np.arange(len(validation)), predicted] = 0.90
    else:
        raise ValueError(f"Unknown baseline: {name}")
    result /= result.sum(axis=1, keepdims=True)
    validate_probability_matrix(result)
    return result


def evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    raw_returns: pd.Series,
    cost_band: pd.Series,
) -> dict[str, float]:
    validate_probability_matrix(probabilities)
    y = pd.to_numeric(y_true, errors="raise").astype(int).to_numpy()
    predicted = probabilities.argmax(axis=1)
    raw = pd.to_numeric(raw_returns, errors="raise").to_numpy(dtype=float)
    cost = pd.to_numeric(cost_band, errors="raise").to_numpy(dtype=float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=list(CLASS_LABELS), zero_division=0
    )
    acted = predicted != 1
    direction = predicted - 1
    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=list(CLASS_LABELS))),
        "brier": float(multiclass_brier(y, probabilities)),
        "coverage": float(acted.mean()),
        "cost_aware_metric": (
            float(np.mean(direction[acted] * raw[acted] - cost[acted]))
            if acted.any() else float("nan")
        ),
    }
    for index, name in CLASS_NAMES.items():
        result[f"{name.lower()}_precision"] = float(precision[index])
        result[f"{name.lower()}_recall"] = float(recall[index])
        result[f"{name.lower()}_f1"] = float(f1[index])
    return result


def _stability(candidate: list[dict[str, float]], baseline: list[dict[str, float]]) -> tuple[str, str]:
    if len(candidate) < MIN_PROMOTION_FOLDS or len(candidate) != len(baseline):
        return "INSUFFICIENT_DATA", "not enough comparable chronological folds"
    c = np.asarray([row["balanced_accuracy"] for row in candidate], dtype=float)
    b = np.asarray([row["balanced_accuracy"] for row in baseline], dtype=float)
    worst_ok = float(c.min()) >= float(b.mean()) - MAX_WORST_FOLD_DEFICIT
    recent_ok = float(c[-1]) >= float(c[:-1].mean()) - MAX_RECENT_FOLD_DEGRADATION
    beating = int((c > b).sum())
    passed = worst_ok and recent_ok and beating >= int(np.ceil(len(c) / 2))
    return (
        "PASS" if passed else "FAIL",
        f"worst_ok={worst_ok};recent_ok={recent_ok};folds_beating_majority={beating}/{len(c)}",
    )


def _evaluate_walk_forward(
    data: HorizonData,
    research_indices: np.ndarray,
    family: str,
) -> tuple[dict[str, float], list[dict[str, float]], str, str]:
    research = data.frame.iloc[research_indices].reset_index(drop=True)
    folds = purged_walk_forward(
        research["decision_timestamp_utc"], data.horizon_minutes,
        n_splits=5, min_validation_rows=250,
    )
    all_y: list[int] = []
    all_prob: list[np.ndarray] = []
    all_raw: list[float] = []
    all_cost: list[float] = []
    fold_metrics: list[dict[str, float]] = []
    majority_metrics: list[dict[str, float]] = []
    statuses: list[str] = []
    for fold in folds:
        train = research.iloc[fold.train_indices]
        validation = research.iloc[fold.validation_indices]
        if family in BASELINES:
            probabilities = baseline_probabilities(family, train["target_class"], validation)
            status = "BASELINE_NOT_CALIBRATED"
        else:
            model = fit_temporal_model(
                family, train.loc[:, list(data.feature_names)], train["target_class"],
                train["decision_timestamp_utc"], data.horizon_minutes,
            )
            probabilities = model.predict_proba(validation.loc[:, list(data.feature_names)])
            status = model.calibration_status
        statuses.append(status)
        metrics = evaluate(
            validation["target_class"], probabilities,
            validation["raw_future_return"], validation["decision_cost_band"],
        )
        fold_metrics.append(metrics)
        majority_probability = baseline_probabilities("MAJORITY_CLASS", train["target_class"], validation)
        majority_metrics.append(evaluate(
            validation["target_class"], majority_probability,
            validation["raw_future_return"], validation["decision_cost_band"],
        ))
        all_y.extend(validation["target_class"].astype(int).tolist())
        all_prob.extend(probabilities)
        all_raw.extend(validation["raw_future_return"].astype(float).tolist())
        all_cost.extend(validation["decision_cost_band"].astype(float).tolist())
    aggregate = evaluate(pd.Series(all_y), np.asarray(all_prob), pd.Series(all_raw), pd.Series(all_cost))
    stability_status, stability_reason = (
        ("BASELINE", "comparison baseline")
        if family in BASELINES else _stability(fold_metrics, majority_metrics)
    )
    aggregate["fold_count"] = float(len(folds))
    aggregate["calibration_ok"] = float(
        family in BASELINES or all(item == "SIGMOID_TEMPORAL_HOLDOUT" for item in statuses)
    )
    return aggregate, fold_metrics, stability_status, stability_reason


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (float(row["balanced_accuracy"]), float(row["macro_f1"]), -float(row["log_loss"]))


def run_ablation(source: pd.DataFrame, horizon_minutes: int) -> list[dict[str, Any]]:
    """Screen feature groups using logistic regression; final holdout remains untouched."""
    rows: list[dict[str, Any]] = []
    for name in ablation_names():
        features = feature_set(name)
        data = build_horizon_data(source, horizon_minutes, features)
        split = final_holdout_split(data)
        metrics, _, stability, reason = _evaluate_walk_forward(
            data, split.research_indices, "LOGISTIC_REGRESSION"
        )
        rows.append({
            "asset_id": ASSET_ID, "horizon_minutes": horizon_minutes,
            "ablation": name, "feature_count": len(features),
            "evaluation_scope": "RESEARCH_WALK_FORWARD_ONLY",
            "stability_status": stability, "stability_reason": reason, **metrics,
        })
    return rows


def _candidate_feature_sets(ablation_rows: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(ablation_rows, key=_rank_key, reverse=True)
    selected = ["BASELINE", "ALL_V10B"]
    for row in ranked:
        name = str(row["ablation"])
        if name not in selected:
            selected.append(name)
        if len(selected) >= 3:
            break
    return selected


def _history_days(frame: pd.DataFrame) -> int:
    return int(_utc(frame["decision_timestamp_utc"]).dt.floor("D").nunique())


def _promotion_decision(
    candidate_walk: dict[str, Any], candidate_holdout: dict[str, Any],
    majority_holdout: dict[str, Any], prior_holdout: dict[str, Any],
    previous_holdout: dict[str, Any], stability_status: str,
    calibration_status: str, history_days: int, fold_count: int,
) -> tuple[bool, str]:
    gates = {
        "history_days": history_days >= MIN_PROMOTION_HISTORY_DAYS,
        "walk_forward_folds": fold_count >= MIN_PROMOTION_FOLDS,
        "walk_forward_stability": stability_status == "PASS",
        "calibration": calibration_status == "SIGMOID_TEMPORAL_HOLDOUT",
        "holdout_balanced_accuracy_margin": (
            float(candidate_holdout["balanced_accuracy"])
            >= float(majority_holdout["balanced_accuracy"]) + PROMOTION_BALANCED_ACCURACY_MARGIN
        ),
        "holdout_macro_f1_margin": (
            float(candidate_holdout["macro_f1"])
            >= float(majority_holdout["macro_f1"]) + PROMOTION_MACRO_F1_MARGIN
        ),
        "holdout_log_loss_margin": (
            float(candidate_holdout["log_loss"])
            <= float(prior_holdout["log_loss"]) - PROMOTION_LOG_LOSS_MARGIN
        ),
        "holdout_cost_aware_not_worse": (
            float(candidate_holdout["cost_aware_metric"])
            >= float(previous_holdout["cost_aware_metric"]) - 1e-6
        ),
        "walk_forward_metrics_finite": all(
            np.isfinite(float(candidate_walk[key]))
            for key in ("balanced_accuracy", "macro_f1", "log_loss", "brier")
        ),
        "holdout_metrics_finite": all(
            np.isfinite(float(candidate_holdout[key]))
            for key in ("balanced_accuracy", "macro_f1", "log_loss", "brier")
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return (
        not failed,
        "all conservative V1.0B promotion gates passed" if not failed
        else "failed gates: " + ", ".join(failed),
    )


def _artifact_digest(bundle: TemporalModelBundle) -> str:
    payload = json.dumps({
        "asset_id": bundle.asset_id, "horizon": bundle.horizon_minutes,
        "family": bundle.family, "features": list(bundle.feature_names),
        "feature_contract_hash": bundle.feature_contract_hash,
    }, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8", newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _save_bundle(bundle: TemporalModelBundle, path: Path) -> str:
    if path.exists():
        raise RuntimeError(f"Immutable XAUUSD artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, temporary)
    digest = sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    return digest


def _append_asset_registry(path: Path, record: dict[str, Any]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("XAUUSD registry must be a JSON list")
        existing = payload
    for item in existing:
        if normalize_asset_id(str(item.get("asset_id") or item.get("symbol"))) != ASSET_ID:
            raise RuntimeError("Cross-symbol registry contamination rejected")
        if item.get("model_id") == record["model_id"]:
            if item != record:
                raise RuntimeError("Immutable XAUUSD registry mutation rejected")
            return
    if record["promotion_status"] == "APPROVED_CHAMPION":
        for item in existing:
            if (
                int(item.get("horizon_minutes", -1)) == int(record["horizon_minutes"])
                and item.get("promotion_status") == "APPROVED_CHAMPION"
            ):
                raise RuntimeError(
                    "Existing XAUUSD champion requires an explicit supersession policy; silent replacement rejected"
                )
    existing.append(record)
    _write_json(path, existing)


def _registry_record(
    model_identity: str, bundle: TemporalModelBundle, artifact_path: Path,
    artifact_sha: str, research: pd.DataFrame, metrics: dict[str, Any],
    promotion_status: str, rejection_reason: str, root: Path,
) -> dict[str, Any]:
    relative = artifact_path.resolve().relative_to(root.resolve()).as_posix()
    return {
        "asset_id": ASSET_ID, "symbol": ASSET_ID, "asset_class": "PRECIOUS_METAL",
        "model_id": model_identity, "horizon_minutes": bundle.horizon_minutes,
        "family": bundle.family, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "training_start": pd.Timestamp(research["decision_timestamp_utc"].iloc[0]).isoformat(),
        "training_end": pd.Timestamp(research["decision_timestamp_utc"].iloc[-1]).isoformat(),
        "training_rows": int(len(research)), "feature_count": len(bundle.feature_names),
        "feature_names": list(bundle.feature_names), "feature_contract_hash": bundle.feature_contract_hash,
        "target_contract_version": CONTRACT_VERSION, "calibration_status": bundle.calibration_status,
        "balanced_accuracy": float(metrics["balanced_accuracy"]), "macro_f1": float(metrics["macro_f1"]),
        "log_loss": float(metrics["log_loss"]), "brier": float(metrics["brier"]),
        "promotion_status": promotion_status, "rejection_reason": rejection_reason,
        "artifact_path": relative, "artifact_sha256": artifact_sha, "git_commit": _git_commit(root),
        "trading_enabled": False, "manual_confirmation_required": True,
    }


def train_horizon(
    source: pd.DataFrame, horizon_minutes: int, root: Path = ROOT, persist: bool = True,
) -> dict[str, Any]:
    ablation = run_ablation(source, horizon_minutes)
    selected_sets = _candidate_feature_sets(ablation)
    families, unavailable = available_model_families()
    candidates: list[dict[str, Any]] = []
    data_by_set: dict[str, tuple[HorizonData, ChronologicalSplit]] = {}
    for set_name in selected_sets:
        data = build_horizon_data(source, horizon_minutes, feature_set(set_name))
        split = final_holdout_split(data)
        data_by_set[set_name] = (data, split)
        for family in families:
            metrics, folds, stability, reason = _evaluate_walk_forward(
                data, split.research_indices, family
            )
            candidates.append({
                "asset_id": ASSET_ID, "horizon_minutes": horizon_minutes,
                "family": family, "feature_set": set_name,
                "feature_count": len(data.feature_names),
                "evaluation_scope": "RESEARCH_WALK_FORWARD_ONLY",
                "stability_status": stability, "stability_reason": reason,
                "fold_count": len(folds), **metrics,
            })
    if not candidates:
        return {
            "horizon_minutes": horizon_minutes, "status": "NO_APPROVED_MODEL",
            "reason": "NO_AVAILABLE_MODEL_FAMILIES", "unavailable_families": unavailable,
            "ablation": ablation, "candidates": [],
        }

    best = sorted(candidates, key=_rank_key, reverse=True)[0]
    data, split = data_by_set[str(best["feature_set"])]
    research = data.frame.iloc[split.research_indices].reset_index(drop=True)
    holdout = data.frame.iloc[split.holdout_indices].reset_index(drop=True)
    final_model = fit_temporal_model(
        str(best["family"]), research.loc[:, list(data.feature_names)],
        research["target_class"], research["decision_timestamp_utc"], horizon_minutes,
    )
    holdout_probability = final_model.predict_proba(holdout.loc[:, list(data.feature_names)])
    holdout_metrics = evaluate(
        holdout["target_class"], holdout_probability,
        holdout["raw_future_return"], holdout["decision_cost_band"],
    )
    baseline_holdout: dict[str, dict[str, float]] = {}
    for baseline in BASELINES:
        probabilities = baseline_probabilities(baseline, research["target_class"], holdout)
        baseline_holdout[baseline] = evaluate(
            holdout["target_class"], probabilities,
            holdout["raw_future_return"], holdout["decision_cost_band"],
        )
    promote, reason = _promotion_decision(
        best, holdout_metrics, baseline_holdout["MAJORITY_CLASS"],
        baseline_holdout["CLASS_PRIOR"], baseline_holdout["PREVIOUS_DIRECTION"],
        str(best["stability_status"]), final_model.calibration_status,
        _history_days(research), int(best["fold_count"]),
    )
    promotion_status = "APPROVED_CHAMPION" if promote else "NO_APPROVED_MODEL"
    record = None
    if persist:
        paths = asset_paths(ASSET_ID, root=root)
        digest = _artifact_digest(final_model)
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
        identity = model_id(ASSET_ID, final_model.family, horizon_minutes, stamp, digest)
        artifact_path = paths.models / "artifacts" / f"{identity}.joblib"
        artifact_sha = _save_bundle(final_model, artifact_path)
        record = _registry_record(
            identity, final_model, artifact_path, artifact_sha, research,
            holdout_metrics, promotion_status, "" if promote else reason, root,
        )
        _append_asset_registry(paths.model_registry, record)

    return {
        "horizon_minutes": horizon_minutes, "status": promotion_status, "reason": reason,
        "selected_family": str(best["family"]), "selected_feature_set": str(best["feature_set"]),
        "selected_feature_count": int(best["feature_count"]),
        "calibration_status": final_model.calibration_status,
        "research_walk_forward": best, "final_holdout": holdout_metrics,
        "holdout_rows": len(holdout), "holdout_start": split.holdout_start.isoformat(),
        "baselines_final_holdout": baseline_holdout, "ablation": ablation,
        "candidates": candidates, "unavailable_families": unavailable,
        "registry_record": record,
    }


def _target_distribution(source: pd.DataFrame, horizon: int) -> dict[str, Any]:
    values = pd.to_numeric(source[f"target_class_{horizon}m"], errors="coerce").dropna().astype(int)
    counts = values.value_counts().reindex(CLASS_LABELS, fill_value=0)
    total = int(counts.sum())
    return {
        CLASS_NAMES[label]: {
            "rows": int(counts[label]), "fraction": float(counts[label] / total) if total else None,
        }
        for label in CLASS_LABELS
    }


def run_research(root: Path = ROOT, persist: bool = True) -> dict[str, Any]:
    source = load_xauusd_research_frame(root)
    paths = asset_paths(ASSET_ID, root=root)
    results: dict[str, Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        result = train_horizon(source, horizon, root=root, persist=persist)
        results[str(horizon)] = result
        candidate_rows.extend(result.get("candidates", []))
        ablation_rows.extend(result.get("ablation", []))
    report = {
        "contract_version": CONTRACT_VERSION, "asset_id": ASSET_ID,
        "asset_class": "PRECIOUS_METAL", "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows_joined": len(source), "range_start": source["decision_timestamp_utc"].min().isoformat(),
        "range_end": source["decision_timestamp_utc"].max().isoformat(),
        "targets": {str(h): _target_distribution(source, h) for h in HORIZONS},
        "horizons": results,
        "production_integration": bool(
            any(result.get("status") == "APPROVED_CHAMPION" for result in results.values())
        ),
        "automatic_execution": "DISABLED", "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }
    if persist:
        _write_json(paths.reports / "v10b_model_research.json", report)
        _write_csv(paths.reports / "v10b_candidates.csv", candidate_rows)
        _write_csv(paths.reports / "v10b_ablation.csv", ablation_rows)
    return report


def load_champion_bundle(record: dict[str, Any], root: Path = ROOT) -> TemporalModelBundle:
    asset = normalize_asset_id(str(record.get("asset_id") or record.get("symbol")))
    if asset != ASSET_ID:
        raise RuntimeError("Cross-symbol model artifact rejected")
    if record.get("promotion_status") != "APPROVED_CHAMPION":
        raise RuntimeError("Only an approved XAUUSD champion may be loaded")
    artifact = Path(str(record["artifact_path"]))
    if not artifact.is_absolute():
        artifact = root / artifact
    if not artifact.exists():
        raise FileNotFoundError(f"Approved XAUUSD artifact missing: {artifact}")
    actual = sha256(artifact.read_bytes()).hexdigest()
    if actual != str(record.get("artifact_sha256")):
        raise RuntimeError("XAUUSD champion artifact SHA mismatch")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, TemporalModelBundle):
        raise TypeError("Unexpected XAUUSD model artifact type")
    if bundle.asset_id != ASSET_ID or bundle.horizon_minutes != int(record["horizon_minutes"]):
        raise RuntimeError("XAUUSD artifact identity mismatch")
    if bundle.feature_contract_hash != str(record.get("feature_contract_hash")):
        raise RuntimeError("XAUUSD artifact feature contract mismatch")
    return bundle


def infer_latest(source: pd.DataFrame, root: Path = ROOT) -> dict[str, Any]:
    """Read-only shadow inference from approved XAUUSD champions."""
    champions = approved_champions(ASSET_ID, asset_paths(ASSET_ID, root=root).model_registry)
    horizons: dict[str, Any] = {}
    latest_time = None
    for horizon in HORIZONS:
        record = champions.get(horizon)
        if record is None:
            horizons[str(horizon)] = {
                "model_status": "NO_APPROVED_MODEL", "model_id": None,
                "prob_down": None, "prob_neutral": None, "prob_up": None,
                "shadow_direction": "WAIT", "decision_gate": "WAIT_NO_APPROVED_MODEL",
            }
            continue
        bundle = load_champion_bundle(record, root=root)
        missing = sorted(set(bundle.feature_names) - set(source.columns))
        if missing:
            raise RuntimeError("Live XAUUSD feature contract missing: " + ", ".join(missing))
        complete = source.loc[:, list(bundle.feature_names)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        eligible = source.loc[complete].sort_values("decision_timestamp_utc")
        if eligible.empty:
            horizons[str(horizon)] = {
                "model_status": "APPROVED_CHAMPION", "model_id": record["model_id"],
                "prob_down": None, "prob_neutral": None, "prob_up": None,
                "shadow_direction": "WAIT", "decision_gate": "WAIT_INSUFFICIENT_FEATURES",
            }
            continue
        row = eligible.iloc[[-1]]
        latest_time = pd.Timestamp(row["decision_timestamp_utc"].iloc[0])
        probabilities = bundle.predict_proba(row.loc[:, list(bundle.feature_names)])[0]
        direction = CLASS_NAMES[int(np.argmax(probabilities))]
        horizons[str(horizon)] = {
            "model_status": "APPROVED_CHAMPION", "model_id": record["model_id"],
            "feature_contract_hash": bundle.feature_contract_hash,
            "prob_down": float(probabilities[0]), "prob_neutral": float(probabilities[1]),
            "prob_up": float(probabilities[2]), "shadow_direction": direction,
            "decision_gate": "PASS_SHADOW_MODEL_AVAILABLE",
        }
    return {
        "asset_id": ASSET_ID,
        "decision_timestamp_utc": None if latest_time is None else latest_time.isoformat(),
        "horizons": horizons, "trading_enabled": False,
        "manual_confirmation_required": True,
    }


def infer_latest_from_disk(root: Path = ROOT) -> dict[str, Any]:
    return infer_latest(load_xauusd_research_frame(root), root=root)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true", help="Evaluate without writing artifacts/registry/reports")
    args = parser.parse_args()
    report = run_research(ROOT, persist=not args.no_persist)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
