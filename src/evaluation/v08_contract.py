"""Static contracts for V0.8 forward-shadow monitoring and research separation."""
from __future__ import annotations

from pathlib import Path

from src.marketdata.v05a_contract import ROOT

CONTRACT_VERSION = "v0.8-forward-shadow-monitor-v1"
OBSERVATION_CONTRACT_VERSION = "v0.8-live-shadow-observation-v2"
SOURCE_LABEL = "TRUE_FORWARD_SHADOW"
RESEARCH_LABEL = "BACKTEST_RESEARCH"
HORIZONS = (15, 60, 240)
DATA_ROOT = ROOT / "data" / "evaluation" / "v08"
PREDICTIONS_FILE = DATA_ROOT / "shadow_predictions.parquet"
OUTCOMES_FILE = DATA_ROOT / "shadow_outcomes.parquet"
WINDOW_OUTCOMES_FILE = DATA_ROOT / "shadow_window_outcomes.parquet"
PERFORMANCE_SNAPSHOTS_FILE = DATA_ROOT / "performance_snapshots.parquet"
PROVIDER_UPTIME_FILE = DATA_ROOT / "provider_uptime.parquet"
LATEST_STATUS_FILE = DATA_ROOT / "latest_status.json"
CONFLICT_DIR = DATA_ROOT / "conflicts"
MONITOR_LOCK_FILE = DATA_ROOT / "v08_monitor.lock"

REPORT_ROOT = ROOT / "reports"
CONTRACT_REPORT = REPORT_ROOT / "v08_contract.txt"
SHADOW_REPORT = REPORT_ROOT / "v08_shadow_performance.txt"
HORIZON_REPORT = REPORT_ROOT / "v08_horizon_performance.csv"
SESSION_REPORT = REPORT_ROOT / "v08_session_performance.csv"
REGIME_REPORT = REPORT_ROOT / "v08_regime_performance.csv"
EVENT_REPORT = REPORT_ROOT / "v08_event_performance.csv"
CALIBRATION_REPORT = REPORT_ROOT / "v08_calibration_monitor.csv"
DRIFT_REPORT = REPORT_ROOT / "v08_drift_monitor.csv"
PROVIDER_REPORT = REPORT_ROOT / "v08_provider_uptime.csv"
SCORECARD_REPORT = REPORT_ROOT / "v08_model_improvement_scorecard.csv"
VALIDATION_REPORT = REPORT_ROOT / "v08_validation.txt"
LIVE_SHADOW_REPORT = REPORT_ROOT / "v08_live_shadow_validation.txt"

MIN_FORWARD_SHADOW_ROWS = 100
MIN_DIRECTIONAL_CALLS = 50
MIN_CALIBRATION_ROWS = 100
MIN_REGIME_ROWS = 30
MIN_PROMOTION_MONITOR_ROWS = 250
MAX_FORWARD_RECORD_DELAY_MINUTES = 10
DEFAULT_MONITOR_SECONDS = 300
DRIFT_RECENT_DAYS = (1, 5, 20)
CALIBRATION_BINS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0000001)
PREDEFINED_COST_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0)
DESCRIPTIVE_BOOTSTRAP_SEED = 20260824

LEGACY_PREDICTION_COLUMNS = (
    "prediction_id", "prediction_payload_sha256", "source_label",
    "decision_timestamp_utc", "recorded_at_utc", "source_generated_at_utc",
    "symbol", "market_mid", "spread_points", "session", "trend_regime",
    "volatility_regime", "spread_regime",
    *tuple(f"h{h}_{field}" for h in HORIZONS for field in (
        "model_id", "status", "p_down", "p_neutral", "p_up", "shadow_direction",
    )),
    "fusion_agreement", "weighted_down", "weighted_neutral", "weighted_up",
    "advisory", "confidence", "risk", "blockers", "event_risk",
    "minutes_to_event", "trade_window_status", "suggested_start_utc",
    "suggested_end_utc", "next_reassessment_utc", "v05b_health",
    "event_health", "model_health", "git_commit", "v06_contract",
    "feature_contract_hash", "manual_execution_only",
)

# The immutable decision ledger remains the durable source of truth.  These
# fields add the live-validation capture context without creating a parallel
# inference or evaluation store.  Legacy rows are intentionally readable.
V1_PREDICTION_COLUMNS = LEGACY_PREDICTION_COLUMNS + (
    "observation_contract_version", "market_status", "active_sessions", "market_bid", "market_ask",
    "data_freshness", "market_age_seconds", "latest_tick_utc", "latest_m5_utc",
    "latest_m15_utc", "feature_timestamp_utc", "feature_complete", "source_fresh",
    "target_event_guard", "v06b_gate", "v06c_status", "v06c_gate",
    "trading_enabled", "manual_confirmation_required",
    "entry_reference_type", "exit_reference_type",
    *tuple(f"h{h}_gate" for h in HORIZONS),
)

PREDICTION_COLUMNS = V1_PREDICTION_COLUMNS[:-len(HORIZONS)] + (
    "engine_snapshot_contract", "engine_snapshot_json",
    "chart_pattern_type", "chart_pattern_score", "chart_pattern_timeframe",
    "nearest_support_distance_price", "nearest_resistance_distance_price",
    "structure_event_type", "structure_event_time_utc",
    "liquidity_event_type", "liquidity_event_time_utc", "engine_observational_only",
    *tuple(f"h{h}_gate" for h in HORIZONS),
)

OUTCOME_COLUMNS = (
    "prediction_id", "prediction_payload_sha256", "source_label", "horizon_minutes",
    "decision_timestamp_utc", "future_timestamp_utc", "matured_at_utc",
    "entry_mid_or_reference_price", "future_price", "raw_return", "move_pips",
    "decision_spread_points", "cost_band", "target_class", "outcome_status",
    "evaluated_at_utc", "entry_reference_type", "exit_reference_type",
    "evaluation_reason",
)

APPROVED_MODEL_STATUSES = frozenset({"APPROVED_CHAMPION"})
LIVE_DATA_FRESHNESS = frozenset({"LIVE", "FRESH"})
EVALUATION_STATES = (
    "PENDING", "PENDING_DATA", "EVALUATED", "INVALID", "NO_APPROVED_MODEL",
)
MAX_OBSERVATION_API_LIMIT = 200

FORBIDDEN_PREDICTION_FRAGMENTS = ("outcome", "future_price", "raw_return", "move_pips", "target_class")
FINAL_STATUSES = (
    "PASS_MONITORING_NO_CHAMPION", "PASS_SHADOW_EVALUATION",
    "PASS_SHADOW_EVALUATION_DEGRADED_PROVIDERS", "INSUFFICIENT_DATA", "FAIL",
)
