"""Frozen research contract for V0.9A.2 historical engine evaluation."""
from __future__ import annotations

from pathlib import Path

from src.learning.v05c_contract import HORIZONS, MARKET_FEATURES, ROOT

CONTRACT_VERSION = "v0.9a2-historical-engine-evaluation-v1"
MODE = "RESEARCH_ONLY"
TIMEFRAMES = ("M5", "M15", "H1", "H4")
AVAILABLE_TIMEFRAMES = ("M5", "M15", "H1")
MODEL_FAMILY = "LOGISTIC_REGRESSION"
MIN_EVENT_SAMPLES = 100
MIN_SUBGROUP_SAMPLES = 100
RANDOM_STATE = 1705

DATA_ROOT = ROOT / "data" / "research" / "v09a2"
FEATURE_DATASET = DATA_ROOT / "engine_features.parquet"
TARGET_DATASET = DATA_ROOT / "targets.parquet"
EVENT_DATASET = DATA_ROOT / "engine_events.parquet"

REPORT_ROOT = ROOT / "reports"
EVALUATION_REPORT = REPORT_ROOT / "v09a2_engine_evaluation.txt"
PATTERN_REPORT = REPORT_ROOT / "v09a2_pattern_evaluation.csv"
ABLATION_REPORT = REPORT_ROOT / "v09a2_ablation.csv"
REDUNDANCY_REPORT = REPORT_ROOT / "v09a2_feature_redundancy.csv"
LEADERBOARD_REPORT = REPORT_ROOT / "v09a2_engine_leaderboard.csv"
VALIDATION_REPORT = REPORT_ROOT / "v09a2_validation.txt"

CHART_PATTERNS = (
    "DOUBLE_TOP", "DOUBLE_BOTTOM", "TRIPLE_TOP", "TRIPLE_BOTTOM",
    "HEAD_AND_SHOULDERS", "INVERSE_HEAD_AND_SHOULDERS",
    "RISING_WEDGE", "FALLING_WEDGE", "BULL_FLAG", "BEAR_FLAG",
    "BULL_PENNANT", "BEAR_PENNANT", "ASCENDING_TRIANGLE",
    "DESCENDING_TRIANGLE", "SYMMETRICAL_TRIANGLE", "EXPANDING_TRIANGLE",
    "RECTANGLE_RANGE",
)

CANDLE_PATTERNS = (
    "DOJI", "HAMMER", "HANGING_MAN", "BULLISH_PIN_BAR", "SHOOTING_STAR",
    "INVERTED_HAMMER", "BEARISH_PIN_BAR", "BULLISH_ENGULFING",
    "BEARISH_ENGULFING", "INSIDE_BAR", "OUTSIDE_BAR", "MORNING_STAR",
    "EVENING_STAR", "THREE_WHITE_SOLDIERS", "THREE_BLACK_CROWS",
)

FEATURE_GROUPS = {
    "BASELINE": tuple(MARKET_FEATURES),
    "TECHNICAL": ("technical_",),
    "PRICE_ACTION": ("price_action_",),
    "PATTERNS": ("pattern_",),
    "STRUCTURE": ("structure_",),
    "LIQUIDITY": ("liquidity_",),
    "SUPPORT_RESISTANCE": ("support_resistance_",),
    "REGIME": ("regime_",),
}

ABLATION_SEQUENCE = (
    "BASELINE", "+TECHNICAL", "+PRICE_ACTION", "+PATTERNS", "+STRUCTURE",
    "+LIQUIDITY", "+SUPPORT_RESISTANCE", "+REGIME", "+ALL_V09",
)


def grade_evidence(sample_count: int, ba_delta: float | None, stable: bool,
                   log_loss_delta: float | None = None) -> str:
    """Deterministic, conservative grading; never relaxes the sample gate."""
    if sample_count < MIN_EVENT_SAMPLES or ba_delta is None:
        return "INSUFFICIENT_DATA"
    loss_delta = 0.0 if log_loss_delta is None else log_loss_delta
    if stable and ba_delta >= 0.010 and loss_delta <= 0.0:
        return "STRONG"
    if stable and ba_delta >= 0.005 and loss_delta <= 0.0:
        return "MODERATE"
    if stable and ba_delta >= 0.002 and loss_delta <= 0.010:
        return "WEAK"
    return "NO_EVIDENCE"
