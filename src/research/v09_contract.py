"""Static scientific contract for MarketFusion V0.9 challenger hardening."""
from __future__ import annotations

from pathlib import Path

from src.evaluation.v08_contract import PREDEFINED_COST_MULTIPLIERS
from src.learning.v05c_contract import MIN_PROMOTION_HISTORY_DAYS, ROOT

CONTRACT_VERSION = "v0.9-challenger-hardening-v1"
PRIMARY_SOURCE_EXPERIMENT = "H60_ABLATE_VOLUME"
PRIMARY_HORIZON = 60
PRIMARY_V08_SPEC_ID = "ABLATE_VOLUME"

REPORT_ROOT = ROOT / "reports"
V08_SCORECARD = REPORT_ROOT / "v08_model_improvement_scorecard.csv"
V08_RESEARCH_AUDIT = REPORT_ROOT / "v08_model_research_audit.json"
REPRO_REPORT = REPORT_ROOT / "v09_reproducibility_audit.txt"
SCORECARD_REPORT = REPORT_ROOT / "v09_candidate_scorecard.csv"
NESTED_AUDIT_REPORT = REPORT_ROOT / "v09_nested_temporal_audit.csv"
FEATURE_IMPORTANCE_REPORT = REPORT_ROOT / "v09_feature_importance.csv"
FEATURE_STABILITY_REPORT = REPORT_ROOT / "v09_feature_stability.csv"
ABSTENTION_REPORT = REPORT_ROOT / "v09_abstention_research.csv"
TARGET_REPORT = REPORT_ROOT / "v09_target_robustness.csv"
REGIME_REPORT = REPORT_ROOT / "v09_regime_robustness.csv"
CALIBRATION_REPORT = REPORT_ROOT / "v09_calibration.csv"
MT5_DEPTH_REPORT = REPORT_ROOT / "v09_mt5_history_depth_audit.txt"
HANDOFF_REPORT = REPORT_ROOT / "v09_v05c_challenger_candidates.json"
VALIDATION_REPORT = REPORT_ROOT / "v09_validation.txt"
STATUS_FILE = ROOT / "data" / "research" / "v09" / "latest_status.json"

# Fixed in advance. V0.9 may compare these values but must not expand the grids after
# observing the final temporal holdout.
TARGET_MULTIPLIERS = tuple(float(value) for value in PREDEFINED_COST_MULTIPLIERS)
TOP_PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60)
DIRECTIONAL_MARGIN_THRESHOLDS = (0.05, 0.08, 0.10)

FINAL_HOLDOUT_FRACTION = 0.15
N_OUTER_FOLDS = 4
N_INNER_FOLDS = 3
MIN_FINAL_HOLDOUT_ROWS = 500
MIN_OUTER_TRAIN_ROWS = 1500
MIN_INNER_TRAIN_ROWS = 800
MIN_VALIDATION_ROWS = 250
MIN_REGIME_ROWS = 100
REPRO_TOLERANCE = 1e-6
RANDOM_STATE = 1705

# Research readiness is deliberately weaker than formal promotion. Formal promotion
# remains V0.5C's job and its 90-day gate is imported, never redefined or weakened.
RESEARCH_BASELINE_BA_MARGIN = 0.01
MAX_RECENT_OUTER_DEGRADATION = 0.08
MIN_RESEARCH_COVERAGE = 0.05
MAX_RESEARCH_COVERAGE = 0.95

ALLOWED_FINAL_STATUSES = (
    "PASS_RESEARCH_READY_HISTORY_INSUFFICIENT",
    "PASS_RESEARCH_READY",
    "PASS_RESEARCH_REJECTED",
    "INSUFFICIENT_HISTORY",
    "RESEARCH_RESULT_NOT_REPRODUCIBLE",
    "FAIL",
)
