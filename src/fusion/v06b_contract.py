"""Static, reviewable contract for V0.6B risk fusion."""
from __future__ import annotations

from pathlib import Path

from src.marketdata.v05a_contract import ROOT

CONTRACT_VERSION = "v0.6b-risk-fusion-advisory-v1"
SYMBOL = "EURUSD"
HORIZONS = (15, 60, 240)
HORIZON_WEIGHTS = {15: 0.40, 60: 0.40, 240: 0.20}
MIN_DIRECTION_PROBABILITY = 0.55
MIN_DIRECTION_MARGIN = 0.08
MIN_APPROVED_HORIZONS = 2
MAX_MARKET_AGE_MINUTES = 15
MAX_INFERENCE_AGE_MINUTES = 5
MAX_MODEL_ARTIFACT_AGE_DAYS = 45
MAX_INTELLIGENCE_AGE_MINUTES = 30
MAX_EVENT_CAPTURE_AGE_HOURS = 24
MIN_REGIME_HISTORY_ROWS = 20
MIN_TRADE_WINDOW_MINUTES = 10
DEFAULT_POLL_SECONDS = 15

RUNTIME_ROOT = ROOT / "data" / "runtime" / "v06"
LATEST_ADVISORY_FILE = RUNTIME_ROOT / "latest_v06b_advisory.json"

# Only identities emitted by the audited V0.4D consensus gate are accepted.
TARGET_EVENT_CODES = frozenset({
    "nonfarm-payrolls", "unemployment-rate", "consumer-price-index",
    "core-consumer-price-index", "cpi", "core-cpi",
})
TARGET_EVENT_NAMES = frozenset({
    "nonfarm payrolls", "unemployment rate", "consumer price index",
    "core consumer price index", "cpi", "core cpi",
})

EVENT_POLICY = (
    ("BLOCK", -15.0, 15.0),
    ("POST_RELEASE_VOLATILITY", -60.0, -15.0),
    ("ELEVATED", 15.0, 60.0),
    ("WATCH", 60.0, 240.0),
)

REASON_PRIORITIES = {
    "RUNTIME_ERROR": 100,
    "NO_APPROVED_MODEL": 95,
    "MODEL_INTEGRITY_FAILURE": 94,
    "MALFORMED_PROBABILITIES": 93,
    "STALE_MARKET_DATA": 90,
    "STALE_MODEL_DATA": 89,
    "EVENT_DATA_INCOMPLETE": 87,
    "EVENT_BLOCK_WINDOW": 86,
    "EXTREME_SPREAD": 84,
    "EXTREME_VOLATILITY": 83,
    "INSUFFICIENT_FEATURES": 80,
    "INSUFFICIENT_APPROVED_HORIZONS": 79,
    "HORIZON_DISAGREEMENT": 77,
    "TRADE_WINDOW_TOO_SHORT": 76,
    "LOW_DIRECTIONAL_CONFIDENCE": 75,
    "INTELLIGENCE_STALE": 50,
    "INTELLIGENCE_DEGRADED": 45,
    "MEMORY_UNAVAILABLE": 30,
    "DIRECTIONAL_CONSENSUS": 10,
}

ALLOWED_ACTIONS = ("WAIT", "BUY_BIAS", "SELL_BIAS")
TRADING_ENABLED = False
MANUAL_CONFIRMATION_REQUIRED = True
