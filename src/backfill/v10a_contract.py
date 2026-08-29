"""Static contract for V1.0A audited MT5 historical backfill.

V1.0A extends the existing V0.5A MARKET_CORE history without changing feature
definitions or enabling trade execution. Historical bars are explicitly marked as
research backfill; they are not forward-observed shadow records.
"""
from __future__ import annotations

from pathlib import Path

from src.learning.v05c_contract import MIN_PROMOTION_HISTORY_DAYS
from src.marketdata.v05a_contract import ROOT, SYMBOL, TIMEFRAMES

CONTRACT_VERSION = "v1.0a-audited-mt5-backfill-v1"
SOURCE_LABEL = "MT5_HISTORICAL_BACKFILL"

DATA_ROOT = ROOT / "data" / "backfill" / "v10a"
STAGING_DIR = DATA_ROOT / "staging"
BACKUP_DIR = DATA_ROOT / "backups"
CONFLICT_DIR = DATA_ROOT / "conflicts"
LATEST_STATUS_FILE = DATA_ROOT / "latest_status.json"
LOCK_FILE = DATA_ROOT / "v10a_backfill.lock"
CANDIDATE_FEATURE_FILE = STAGING_DIR / "v10a_candidate_features.parquet"

REPORT_ROOT = ROOT / "reports"
CONTRACT_REPORT = REPORT_ROOT / "v10a_contract.txt"
BACKFILL_AUDIT_REPORT = REPORT_ROOT / "v10a_backfill_audit.txt"
OVERLAP_REPORT = REPORT_ROOT / "v10a_overlap_audit.csv"
MARKET_CORE_REPORT = REPORT_ROOT / "v10a_market_core_expansion.txt"
APPLY_REPORT = REPORT_ROOT / "v10a_apply_report.txt"
VALIDATION_REPORT = REPORT_ROOT / "v10a_validation.txt"

# V0.9 proved at least 100k M1 bars are available. V1.0A deliberately requests
# up to 200k M1 bars so the audit can discover deeper terminal history when the
# broker/MT5 Max-bars setting permits it. Counts are ceilings, not date promises.
FETCH_LIMITS = {
    "M1": 200_000,
    "M5": 100_000,
    "M15": 60_000,
    "H1": 30_000,
}
if set(FETCH_LIMITS) != set(TIMEFRAMES):
    raise RuntimeError("V1.0A fetch limits must cover exactly the V0.5A timeframes")

TARGET_PROMOTION_HISTORY_DAYS = MIN_PROMOTION_HISTORY_DAYS
MIN_COMMON_FEATURE_DAYS_FOR_APPLY = 20

OVERLAP_IMMUTABLE_FIELDS = (
    "bar_close_utc", "open", "high", "low", "close",
    "tick_volume", "spread_points", "real_volume",
)
PRICE_TOLERANCE = 1e-12

FINAL_STATUSES = (
    "PASS_AUDIT_READY_TO_APPLY",
    "PASS_APPLIED_MARKET_CORE_EXPANDED",
    "PASS_APPLIED_HISTORY_GATE_MET",
    "INSUFFICIENT_COMMON_HISTORY",
    "FAIL_OVERLAP_CONFLICT",
    "FAIL_DATASET_CAUSALITY",
    "FAIL",
)
