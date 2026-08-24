"""Static security and data contract for the V0.7 local dashboard API."""
from __future__ import annotations

from pathlib import Path

from src.marketdata.v05a_contract import ROOT

CONTRACT_VERSION = "v0.7-local-dashboard-api-v1"
HOST = "127.0.0.1"
PORT = 8765
VITE_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")
V06_STATE_FILE = ROOT / "data" / "runtime" / "v06" / "latest_marketfusion_state.json"
V06_HISTORY_FILE = ROOT / "data" / "runtime" / "v06" / "state_history.parquet"
MARKET_ROOT = ROOT / "data" / "mt5" / "continuous"
V05A_STATUS_FILE = MARKET_ROOT / "runtime_status.json"
V05B_STATUS_FILE = ROOT / "data" / "intelligence" / "runtime_status.json"
V05B_CONTEXT_FILE = ROOT / "data" / "intelligence" / "context" / "v05b_context_history.parquet"
V04D_EVENTS_FILE = ROOT / "data" / "mt5" / "calendar" / "v04d_live_verified_consensus.csv"
V05C_CANDIDATES_FILE = ROOT / "reports" / "v05c_candidate_results.csv"
V05C_TRAINING_FILE = ROOT / "reports" / "v05c_training_quality.txt"
V05C_ELIGIBILITY_FILE = ROOT / "reports" / "v05c_feature_group_eligibility.txt"
V08_STATUS_FILE = ROOT / "data" / "evaluation" / "v08" / "latest_status.json"
MAX_CANDLE_LIMIT = 1000
DEFAULT_CANDLE_LIMIT = 300
MAX_HISTORY_LIMIT = 1000
STATE_STALE_SECONDS = 45
TIMEFRAMES = {
    "M1": MARKET_ROOT / "EURUSD_M1.parquet",
    "M5": MARKET_ROOT / "EURUSD_M5.parquet",
    "M15": MARKET_ROOT / "EURUSD_M15.parquet",
    "H1": MARKET_ROOT / "EURUSD_H1.parquet",
}
REQUIRED_STATE_KEYS = frozenset({
    "contract_version", "system", "market", "predictions", "decision",
    "trade_window", "health", "reasons",
})
FORBIDDEN_RESPONSE_KEYS = frozenset({
    "account", "account_number", "login", "password", "token", "api_key",
    "apikey", "balance", "equity", "positions", "orders", "order",
    "credentials", "secret",
})
TRADING_ROUTE_FRAGMENTS = ("trade", "order", "buy", "sell", "position")
