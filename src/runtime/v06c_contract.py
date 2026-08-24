"""Static contract for the V0.6C dashboard state."""
from __future__ import annotations

from src.fusion.v06b_contract import RUNTIME_ROOT

CONTRACT_VERSION = "v0.6c-unified-runtime-v1"
LATEST_STATE_FILE = RUNTIME_ROOT / "latest_marketfusion_state.json"
STATE_HISTORY_FILE = RUNTIME_ROOT / "state_history.parquet"
SHADOW_PERFORMANCE_FILE = RUNTIME_ROOT / "shadow_performance.parquet"
RUNTIME_LOCK_FILE = RUNTIME_ROOT / "v06_runtime.lock"
DEFAULT_POLL_SECONDS = 15
LOCAL_TIMEZONE = "Asia/Colombo"
OVERALL_STATUSES = (
    "PASS", "PASS_DEGRADED", "PASS_FAIL_CLOSED_NO_CHAMPION",
    "PASS_WITH_LIMITED_COVERAGE", "FAIL_CLOSED", "FAIL",
)
