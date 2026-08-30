export type AdvisoryAction = "WAIT" | "BUY_BIAS" | "SELL_BIAS";
export type Confidence = "VERY_LOW" | "LOW" | "MEDIUM" | "HIGH";
export type ConnectionState = "CONNECTING" | "LIVE" | "STALE" | "DISCONNECTED";
export type Timeframe = "M1" | "M5" | "M15" | "H1";

export interface DualTime { utc: string | null; asia_colombo: string | null }
export interface FreshnessState { status: string; observed_at_utc?: string | null; age_minutes?: number | null }
export interface RegimeState {
  volatility_regime?: string;
  trend_regime?: string;
  trend_score?: number | null;
  current_volatility?: number | null;
}
export interface SpreadState { status?: string; current_points?: number | null; recent_median_points?: number | null; ratio?: number | null }
export interface MarketState {
  close?: number | null;
  freshness?: FreshnessState;
  session?: string;
  regime?: RegimeState;
  spread?: SpreadState;
}
export interface HorizonPrediction {
  model_id?: string | null;
  model_status: string;
  model_created_at_utc?: string;
  calibration_status?: string;
  prob_down?: number | null;
  prob_neutral?: number | null;
  prob_up?: number | null;
  top_class?: string;
  top_probability?: number;
  shadow_direction?: string;
  decision_gate?: string;
  probability_confidence_band?: Confidence;
}
export interface FusionProbability { down: number; neutral: number; up: number }
export interface FusionState {
  status?: string;
  direction?: string;
  confidence?: Confidence;
  probabilities?: FusionProbability | null;
  valid_horizons?: number[];
  margin?: number;
  horizon_directions?: Record<string, string>;
}
export interface PredictionState {
  v06a_status?: string;
  horizons: Record<string, HorizonPrediction>;
  fusion?: FusionState | null;
}
export interface DecisionState {
  action: AdvisoryAction;
  action_meaning?: string;
  direction?: string;
  confidence: Confidence;
  confidence_meaning?: string;
  gate: string;
  manual_confirmation_required: boolean;
  trading_enabled: false;
  decision_time: DualTime;
  next_reassessment: DualTime;
}
export interface TradeWindowState { status: string; start_utc: string | null; end_utc: string | null; horizon_minutes?: number | null }
export interface AuditedEvent { event_id?: string | null; event_code?: string; event_name?: string; event_timestamp_utc?: string }
export interface EventRiskState {
  status?: string;
  data_status?: string;
  blocks_direction?: boolean;
  minutes_to_event?: number | null;
  nearest_event?: AuditedEvent | null;
  freshness?: FreshnessState;
}
export interface IntelligenceState {
  status?: string;
  news_15m?: number | null;
  news_60m?: number | null;
  news_240m?: number | null;
  high_impact_240m?: number | null;
  provider_errors?: number;
}
export interface HealthState { sources: Record<string, string>; failed_sources: string[]; degraded_sources: string[] }
export interface Reason { code: string; priority: number; blocking: boolean; source: string; message: string; evidence?: unknown }
export interface SystemState {
  status: string;
  mode: string;
  symbol: string;
  generated_time: DualTime;
  registry?: { status?: string; champion_count?: number; reloaded_this_cycle?: boolean };
}
export interface MarketFusionState {
  contract_version: string;
  system: SystemState;
  market: MarketState;
  predictions: PredictionState;
  decision: DecisionState;
  trade_window: TradeWindowState;
  event?: EventRiskState;
  intelligence?: IntelligenceState;
  health: HealthState;
  reasons: Reason[];
}
export interface Candle { time: string; open: number; high: number; low: number; close: number; volume: number | null }
export interface CandleResponse { status: string; timeframe: Timeframe; count: number; candles: Candle[] }
export interface MarketSummary { status: string; captured_at_utc: string | null; bid: number | null; ask: number | null; mid: number | null; spread_points: number | null }
export interface IntelligenceDetail {
  status: string;
  captured_at_utc: string | null;
  news_counts: Record<string, number | null>;
  topic_counts_24h: Record<string, number | null>;
  macro: Record<string, number | null>;
  provider_health: Record<string, string>;
}
export interface AuditedEventDetail { event_id: string | number | null; event_name: string | null; event_code: string | null; event_timestamp_utc: string | null; forecast_value: number | null; consensus_status: string | null }
export interface EventsResponse { status: string; events: AuditedEventDetail[] }
export interface ResearchCandidate { family: string; balanced_accuracy: number; promotion_status: string }
export interface ResearchState { status: string; approved_champions: number; candidates: Record<string, ResearchCandidate>; last_training_utc: string | null; market_core_rows: number | null; history_days: number | null; v05b_eligibility: string; live_surprise_samples: number; next_recommended_training: string }
export interface V08HorizonPerformance { matured_count: number; directional_calls: number; wait_rate: number | null; directional_accuracy: number | null; brier: number | null; sample_status: string }
export interface V08Status {
  contract_version: string;
  status: string;
  performance: {
    recorded_predictions: number;
    matured_outcomes: number;
    horizons: Record<string, V08HorizonPerformance>;
    wait: { wait_rate: number | null };
    calibration_status: string;
    market_drift_status: string;
    model_drift_status: string;
  };
  champions: Record<string, string>;
  provider_health: { status: string; providers: Record<string, string>; uptime_percentage: Record<string, number> };
  research: { status: string; experiments: number; best_experiment: string | null; decision: string };
  manual_execution_only: true;
  trading_enabled: false;
}
export interface ShadowObservation {
  observation_id: string;
  decision_timestamp_utc: string;
  evaluation_due_utc: string | null;
  horizon_minutes: number;
  model_id: string | null;
  model_status: string;
  predicted_class: string | null;
  prob_down: number | null;
  prob_neutral: number | null;
  prob_up: number | null;
  model_confidence: number | null;
  v06c_decision: string;
  v06c_gate: string | null;
  actual_class: string | null;
  actual_return: number | null;
  direction_correct: boolean | null;
  evaluation_status: string;
  blocking_reasons: string[];
}
export interface ShadowHorizonSummary {
  horizon_minutes: number;
  total_recorded: number;
  sample_count: number;
  sample_status: string;
  PENDING: number;
  PENDING_DATA: number;
  EVALUATED: number;
  INVALID: number;
  NO_APPROVED_MODEL: number;
  accuracy: number | null;
  balanced_accuracy: number | null;
  macro_f1: number | null;
  mean_log_loss: number | null;
  mean_brier_score: number | null;
  average_confidence: number | null;
}
export interface ShadowSummary {
  contract_version: string;
  status: string;
  recorder: { status: string; trigger: string; last_cycle_status?: string; last_cycle_reason: string; updated_at_utc?: string | null };
  total_observations: number;
  performance_observations: number;
  counts: Record<string, number>;
  horizons: Record<string, ShadowHorizonSummary>;
  latest_observation: ShadowObservation | null;
}
export interface ShadowRecent { contract_version: string; status: string; count: number; limit: number; items: ShadowObservation[]; trading_enabled: false }
export interface OperatorStatus {
  contract_version: string;
  current_time: DualTime;
  market: {
    status: "OPEN" | "CLOSED_WEEKEND" | "OPENING_SOON" | "CLOSING_SOON" | "UNKNOWN";
    market_open: boolean;
    reason: string;
    next_market_open_utc: string | null;
    next_market_close_utc: string | null;
  };
  session: {
    current_session: string;
    active_sessions: string[];
    overlap: boolean;
    next_session: string | null;
    next_session_change_utc: string | null;
    next_transition: string;
  };
  data_freshness: {
    status: "LIVE" | "FRESH" | "DELAYED" | "STALE" | "UNAVAILABLE";
    reason: string;
    basis: string;
    latest_market_tick_utc: string | null;
    latest_completed_m5_utc: string | null;
    latest_completed_m15_utc: string | null;
    feature_row_utc: string | null;
    market_age_seconds: number | null;
    tick_age_seconds: number | null;
    m5_age_seconds: number | null;
    m15_age_seconds: number | null;
    feature_age_seconds: number | null;
    market_closed_context: boolean;
  };
  reassessment: { status: string; at_utc: string | null; reason: string };
  trading_window: {
    status: string;
    reason: string;
    trading_state: "READY" | "WAIT" | "BLOCKED" | "MARKET_CLOSED";
    start_utc: string | null;
    end_utc: string | null;
    horizon_minutes?: number | null;
    execution: "DISABLED";
    trading_enabled: false;
    manual_confirmation_required: true;
  };
  system_health: { status: "PASS" | "DEGRADED" | "FAIL"; reason: string };
  trading_state: "READY" | "WAIT" | "BLOCKED" | "MARKET_CLOSED";
  wait_explanation: string;
  runtime_gate?: string | null;
  runtime_status?: string | null;
  mode: string;
}
export interface EngineOutput { engine_name: string; status: string; direction_score: number | null; confidence: number | null; regime: string | null; feature_count: number; input_freshness: string; reason_codes: string[]; components: Record<string, unknown> }
export interface EngineEvent { event_id: string; event_type: string; timeframe: string; detected_at_utc: string; status: string; direction?: string }
export interface PatternRecord { pattern_id: string; pattern_type: string; timeframe: string; detected_at_utc: string; confirmed_at_utc: string | null; status: string; direction: string; score: number; confidence: number; upper_boundary: number | null; lower_boundary: number | null }
export interface EngineLevel { level_id: string; kind: string; source: string; price: number; distance_price?: number; distance_atr?: number }
export interface TimeframeIntelligence {
  timeframe: string; latest_completed_bar_utc: string; structure_state: string; patterns: PatternRecord[];
  structure_events: EngineEvent[]; price_action_events: EngineEvent[];
  support_resistance: { nearest_support: EngineLevel | null; nearest_resistance: EngineLevel | null; dynamic: { name: string; price: number }[] };
  liquidity: { sweeps: EngineEvent[]; fair_value_gaps: (EngineEvent & { lower_boundary: number; upper_boundary: number })[]; nearest_liquidity: Record<string, unknown> | null };
}
export interface EngineStatus { contract_version: string; status: string; decision_timestamp_utc: string | null; market_status?: string; data_freshness?: string; engines: Record<string, EngineOutput>; external_engines: Record<string, EngineOutput>; timeframes?: Record<string, TimeframeIntelligence>; recent_patterns?: PatternRecord[]; event_history?: EngineEvent[]; informational_agreement?: { status: string; conflict_resolution: string }; timeframe_availability?: Record<string, { status: string; reason?: string }>; not_implemented_features?: Record<string, { status: string; reason: string }>; observational_only: boolean; v06_integration: boolean; trading_enabled?: false }
