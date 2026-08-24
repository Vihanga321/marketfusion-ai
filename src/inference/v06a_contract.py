"""Static contract for MarketFusion V0.6A real-time shadow inference."""
from __future__ import annotations

from pathlib import Path

from src.learning.v05c_contract import HORIZONS, MARKET_FEATURES, MODEL_ROOT, RUNTIME_REGISTRY, ROOT
from src.marketdata.v05a_contract import FEATURE_FILE

CONTRACT_VERSION = "v0.6a-realtime-shadow-inference-v1"
SYMBOL = "EURUSD"
RUNTIME_ROOT = ROOT / "data" / "inference" / "v06a"
LATEST_STATUS_FILE = RUNTIME_ROOT / "latest_shadow_prediction.json"
HISTORY_FILE = RUNTIME_ROOT / "shadow_prediction_history.parquet"
FEATURE_SOURCE = FEATURE_FILE
V05C_REGISTRY = RUNTIME_REGISTRY
V05C_MODEL_ROOT = MODEL_ROOT
VERIFIED_CONSENSUS_FILE = ROOT / "data" / "mt5" / "calendar" / "v04d_live_verified_consensus.csv"

# The production candidates are trained on completed M5 decisions. V0.6A therefore
# never invents partial-bar features from the live tick stream.
MAX_FEATURE_AGE_MINUTES = 15
DEFAULT_POLL_SECONDS = 30

# Explicit research-only probability gates. These thresholds do not imply an edge.
MIN_DIRECTION_PROBABILITY = 0.55
MIN_DIRECTION_MARGIN = 0.10
MEDIUM_CONFIDENCE_PROBABILITY = 0.60
MEDIUM_CONFIDENCE_MARGIN = 0.12
HIGH_CONFIDENCE_PROBABILITY = 0.70
HIGH_CONFIDENCE_MARGIN = 0.20

# This first guard covers only the exact audited V0.4D target releases. Broader event
# risk belongs in V0.6B and must not be silently inferred here.
TARGET_EVENT_BLOCK_BEFORE_MINUTES = 30
TARGET_EVENT_BLOCK_AFTER_MINUTES = 15

CLASS_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
REQUIRED_FEATURES = tuple(MARKET_FEATURES)
SHADOW_ONLY = True
TRADING_EXECUTION = False

OUTPUT_STATUSES = (
    "PASS_FAIL_CLOSED_NO_CHAMPION",
    "PASS_SHADOW_RUNNING",
    "FAIL_CLOSED",
)
