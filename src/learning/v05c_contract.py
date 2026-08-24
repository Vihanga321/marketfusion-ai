"""Static V0.5C scientific and artifact contract."""
from __future__ import annotations

from pathlib import Path

from src.marketdata.v05a_contract import FEATURE_COLUMNS, FEATURE_FILE

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "v0.5c-controlled-daily-learning-v1"
HORIZONS = (15, 60, 240)
CLASS_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
CLASS_LABELS = (0, 1, 2)

SOURCE_FILE = FEATURE_FILE
MODEL_ROOT = ROOT / "models" / "v05c"
RUNTIME_REGISTRY = MODEL_ROOT / "registry.json"
REPORT_ROOT = ROOT / "reports"
TRAINING_QUALITY_REPORT = REPORT_ROOT / "v05c_training_quality.txt"
TARGET_CONTRACT_REPORT = REPORT_ROOT / "v05c_target_contract.txt"
FEATURE_REGISTRY_REPORT = REPORT_ROOT / "v05c_feature_registry.csv"
FEATURE_GROUP_REPORT = REPORT_ROOT / "v05c_feature_group_eligibility.txt"
WALK_FORWARD_REPORT = REPORT_ROOT / "v05c_walk_forward_audit.csv"
CANDIDATE_REPORT = REPORT_ROOT / "v05c_candidate_results.csv"
MODEL_REGISTRY_REPORT = REPORT_ROOT / "v05c_model_registry.csv"
CALIBRATION_REPORT = REPORT_ROOT / "v05c_calibration_audit.csv"
STABILITY_REPORT = REPORT_ROOT / "v05c_stability_audit.csv"

MARKET_FEATURES = tuple(FEATURE_COLUMNS)
FEATURE_GROUPS = ("MARKET_CORE", "INTELLIGENCE_V05B", "EVENT_MEMORY", "LIVE_SURPRISE")

# All thresholds are explicit here so activation and promotion policy is reviewable.
MIN_MARKET_ROWS = 2_000
MIN_MARKET_CALENDAR_DAYS = 20
MIN_INTELLIGENCE_JOINED_ROWS = 2_000
MIN_INTELLIGENCE_CALENDAR_DAYS = 90
MIN_LIVE_SURPRISE_ROWS = 100
MIN_TRAINING_ROWS = 2_000
MIN_WALK_FORWARD_FOLDS = 5
MIN_VALIDATION_ROWS = 250
MIN_CALIBRATION_ROWS = 250
MIN_BASE_TRAINING_ROWS = 1_000
MIN_PROMOTION_HISTORY_DAYS = 90
N_WALK_FORWARD_FOLDS = 5

# Conservative observable spread/noise band. No unknown commission is fabricated.
EURUSD_POINT_SIZE = 0.00001
MIN_COST_POINTS = 1.0
COST_BAND_MULTIPLIER = 1.25
COMMISSION_STATUS = "UNKNOWN"

PROMOTION_BALANCED_ACCURACY_MARGIN = 0.02
PROMOTION_MACRO_F1_MARGIN = 0.02
PROMOTION_LOG_LOSS_MARGIN = 0.01
MAX_RECENT_FOLD_DEGRADATION = 0.08
MAX_WORST_FOLD_DEFICIT = 0.05

RANDOM_STATE = 1705

MODEL_REGISTRY_COLUMNS = (
    "model_id", "horizon_minutes", "family", "created_at_utc",
    "training_data_fingerprint", "feature_contract_hash", "feature_count",
    "train_start", "train_end", "training_rows", "walk_forward_folds",
    "balanced_accuracy", "macro_f1", "log_loss", "brier", "coverage",
    "cost_aware_metric", "stability_status", "calibration_status",
    "promotion_status", "rejection_reason", "git_commit", "artifact_path",
    "artifact_sha256",
)
