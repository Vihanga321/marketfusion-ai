import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { MarketFusionState, OperatorStatus, ShadowRecent, ShadowSummary } from "./types/marketfusion";
import { localTime } from "./utils/format";

const state: MarketFusionState = {
  contract_version: "v0.6c-unified-runtime-v1",
  system: { status: "PASS_FAIL_CLOSED_NO_CHAMPION", mode: "SHADOW_ADVISORY_ONLY", symbol: "EURUSD", generated_time: { utc: "2026-08-24T12:00:00Z", asia_colombo: "2026-08-24T17:30:00+05:30" }, registry: { champion_count: 0 } },
  market: { close: 1.166, freshness: { status: "FRESH", age_minutes: 2, observed_at_utc: "2026-08-24T11:58:00Z" }, session: "OVERLAP", regime: { trend_regime: "DOWN", volatility_regime: "NORMAL" }, spread: { status: "NORMAL", current_points: 1 } },
  predictions: { v06a_status: "PASS_FAIL_CLOSED_NO_CHAMPION", horizons: {
    "15": { model_status: "NO_APPROVED_MODEL", model_id: null, prob_down: null, prob_neutral: null, prob_up: null, decision_gate: "WAIT_NO_MODEL" },
    "60": { model_status: "NO_APPROVED_MODEL", model_id: null, prob_down: null, prob_neutral: null, prob_up: null, decision_gate: "WAIT_NO_MODEL" },
    "240": { model_status: "NO_APPROVED_MODEL", model_id: null, prob_down: null, prob_neutral: null, prob_up: null, decision_gate: "WAIT_NO_MODEL" }
  }, fusion: { status: "NO_APPROVED_MODEL", direction: "WAIT", confidence: "VERY_LOW", probabilities: null, valid_horizons: [] } },
  decision: { action: "WAIT", direction: "WAIT", action_meaning: "No directional advice is available.", confidence: "VERY_LOW", gate: "WAIT_NO_MODEL", manual_confirmation_required: true, trading_enabled: false, decision_time: { utc: "2026-08-24T11:55:00Z", asia_colombo: null }, next_reassessment: { utc: "2026-08-24T12:05:00Z", asia_colombo: null } },
  trade_window: { status: "NOT_APPLICABLE_WAIT", start_utc: null, end_utc: null, horizon_minutes: null },
  event: { status: "NONE", nearest_event: null }, intelligence: { status: "DEGRADED", provider_errors: 5 },
  health: { sources: { v05a_market: "PASS", v05b_intelligence: "PASS_DEGRADED" }, failed_sources: [], degraded_sources: ["v05b_intelligence"] },
  reasons: [{ code: "NO_APPROVED_MODEL", priority: 95, blocking: true, source: "V0.5C/V0.6A", message: "No approved V0.5C champion", evidence: null }]
};

const operator: OperatorStatus = {
  contract_version: "v0.7-operator-status-v1",
  current_time: { utc: "2026-08-24T12:00:00Z", asia_colombo: "2026-08-24T17:30:00+05:30" },
  market: { status: "OPEN", market_open: true, reason: "FX_WEEK_OPEN", next_market_open_utc: null, next_market_close_utc: "2026-08-28T21:00:00Z" },
  session: { current_session: "LONDON + NEW YORK", active_sessions: ["LONDON", "NEW_YORK"], overlap: true, next_session: "SYDNEY", next_session_change_utc: "2026-08-24T16:00:00Z", next_transition: "NEW_YORK CLOSE" },
  data_freshness: { status: "FRESH", reason: "CURRENT_COMPLETED_MARKET_DATA", basis: "LATEST_AVAILABLE_SOURCE", latest_market_tick_utc: "2026-08-24T11:59:58Z", latest_completed_m5_utc: "2026-08-24T11:55:00Z", latest_completed_m15_utc: "2026-08-24T11:45:00Z", feature_row_utc: "2026-08-24T11:58:00Z", market_age_seconds: 120, tick_age_seconds: 2, m5_age_seconds: 300, m15_age_seconds: 900, feature_age_seconds: 120, market_closed_context: false },
  reassessment: { status: "SCHEDULED", at_utc: "2026-08-24T12:05:00Z", reason: "V06_NEXT_REASSESSMENT" },
  trading_window: { status: "WAITING_FOR_MODEL", reason: "NO_APPROVED_MODEL", trading_state: "WAIT", start_utc: null, end_utc: null, horizon_minutes: null, execution: "DISABLED", trading_enabled: false, manual_confirmation_required: true },
  system_health: { status: "PASS", reason: "RUNTIME_AND_MARKET_PIPELINE_OPERATIONAL" },
  trading_state: "WAIT",
  wait_explanation: "Waiting for an approved model before a directional advisory can be published.",
  runtime_gate: "WAIT_NO_MODEL",
  runtime_status: "PASS_FAIL_CLOSED_NO_CHAMPION",
  mode: "SHADOW_ADVISORY_ONLY"
};

function response(data: object, freshness = "FRESH") {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(data), headers: new Headers({ "X-MarketFusion-State-Freshness": freshness }) } as Response);
}

const emptyShadow: ShadowSummary = {
  contract_version: "v0.8-live-shadow-observation-v1", status: "NO_LIVE_SHADOW_OBSERVATIONS_YET",
  recorder: { status: "RUNNING", trigger: "NEW_COMPLETED_CAUSAL_DECISION_ROW", last_cycle_reason: "MARKET_CLOSED_WEEKEND" },
  total_observations: 0, performance_observations: 0,
  counts: { PENDING: 0, PENDING_DATA: 0, EVALUATED: 0, INVALID: 0, NO_APPROVED_MODEL: 0 },
  horizons: { "15": { horizon_minutes: 15, total_recorded: 0, sample_count: 0, sample_status: "INSUFFICIENT_DATA", PENDING: 0, PENDING_DATA: 0, EVALUATED: 0, INVALID: 0, NO_APPROVED_MODEL: 0, accuracy: null, balanced_accuracy: null, macro_f1: null, mean_log_loss: null, mean_brier_score: null, average_confidence: null } }, latest_observation: null
};
const emptyRecent: ShadowRecent = { contract_version: "v0.8-live-shadow-observation-v1", status: "PASS", count: 0, limit: 20, items: [], trading_enabled: false };

function mockHealthyFetch(operatorValue: OperatorStatus = operator, shadow: ShadowSummary = emptyShadow, recent: ShadowRecent = emptyRecent) {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/state") return response(state);
    if (url === "/api/operator/status") return response(operatorValue);
    if (url.startsWith("/api/market/candles")) return response({ status: "PASS", timeframe: "M5", count: 0, candles: [] });
    if (url === "/api/market/summary") return response({ status: "PASS_RUNNING", captured_at_utc: null, bid: 1.16, ask: 1.17, mid: 1.165, spread_points: 1 });
    if (url === "/api/intelligence") return response({ status: "PASS_RUNNING_DEGRADED", captured_at_utc: null, news_counts: { "15": 0, "60": 0, "240": 0, "1440": 1 }, macro: {}, provider_health: { FED_RSS: "OK", ECB: "OK", BLS: "OK", GDELT: "DEGRADED", FRED: "DEGRADED" } });
    if (url === "/api/events") return response({ status: "PASS", events: [] });
    if (url === "/api/v08/status") return response({ contract_version: "v0.8-forward-shadow-monitor-v1", status: "PASS_MONITORING_NO_CHAMPION", performance: { recorded_predictions: 1, matured_outcomes: 0, horizons: { "15": { matured_count: 0, directional_calls: 0, wait_rate: null, directional_accuracy: null, brier: null, sample_status: "INSUFFICIENT_DATA" }, "60": { matured_count: 0, directional_calls: 0, wait_rate: null, directional_accuracy: null, brier: null, sample_status: "INSUFFICIENT_DATA" }, "240": { matured_count: 0, directional_calls: 0, wait_rate: null, directional_accuracy: null, brier: null, sample_status: "INSUFFICIENT_DATA" } }, wait: { wait_rate: null }, calibration_status: "NO_APPROVED_MODEL_DATA", market_drift_status: "INSUFFICIENT_DATA", model_drift_status: "INSUFFICIENT_DATA" }, champions: { "15": "NONE", "60": "NONE", "240": "NONE" }, provider_health: { status: "PASS", providers: {}, uptime_percentage: {} }, research: { status: "AVAILABLE_NOT_RUN", experiments: 0, best_experiment: null, decision: "INSUFFICIENT_DATA" }, manual_execution_only: true, trading_enabled: false });
    if (url === "/api/evaluation/shadow/summary") return response(shadow);
    if (url.startsWith("/api/evaluation/shadow/recent")) return response(recent);
    return response({ status: "RESEARCH_ONLY", approved_champions: 0, candidates: {}, last_training_utc: null, market_core_rows: 8558, history_days: 39, v05b_eligibility: "INSUFFICIENT_HISTORY", live_surprise_samples: 0, next_recommended_training: "DAILY_MANUAL_SCHEDULE" });
  }));
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("MarketFusion dashboard safety UI", () => {
  it("displays intentional WAIT state with no champion", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("CURRENT ADVISORY")).toBeInTheDocument(); expect(screen.getAllByText("WAIT").length).toBeGreaterThan(0); });
  it("displays all three no-approved-model cards", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findAllByText("NO APPROVED MODEL")).toHaveLength(3); });
  it("uses N/A and does not invent probability fills", async () => { mockHealthyFetch(); const { container } = render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.getAllByText("N/A").length).toBeGreaterThan(0); expect(container.querySelectorAll(".prob-fill")).toHaveLength(0); });
  it("contains no buy sell or order controls", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.queryByRole("button", { name: /buy|sell|order|close/i })).not.toBeInTheDocument(); });
  it("shows manual confirmation and disabled execution", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.getAllByText("REQUIRED").length).toBeGreaterThan(0); expect(screen.getAllByText("DISABLED").length).toBeGreaterThan(0); });
  it("connection loss forces safety WAIT and blocks the window", async () => { vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline")))); render(<App />); expect(await screen.findByText("CONNECTION LOST")).toBeInTheDocument(); expect(screen.getByText("SAFETY ADVISORY")).toBeInTheDocument(); expect(screen.getAllByText("WAIT").length).toBeGreaterThan(0); expect(screen.getByText("WAITING FOR DATA")).toBeInTheDocument(); expect(screen.getAllByText("BLOCKED").length).toBeGreaterThan(0); });
  it("makes degraded providers visible", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.getAllByText("DEGRADED").length).toBeGreaterThanOrEqual(2); expect(screen.getByText("GDELT")).toBeInTheDocument(); expect(screen.getByText("FRED")).toBeInTheDocument(); });
  it("shows event countdown only when an event timestamp exists", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("AUDITED EVENT DATA UNAVAILABLE")).toBeInTheDocument(); expect(screen.queryByText(/BLOCK.*15m/)).not.toBeInTheDocument(); });
  it("maps WAIT_NO_MODEL to a clear operator state", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("WAITING FOR MODEL")).toBeInTheDocument(); expect(screen.getByText("Waiting for an approved model before a directional advisory can be published.")).toBeInTheDocument(); });
  it("formats Sri Lanka time via the timezone utility", () => { expect(localTime("2026-08-24T12:00:00Z")).toBe("17:30"); });
  it("separates system health from trading state", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("SYSTEM HEALTH"); expect(screen.getByText("TRADING STATE")).toBeInTheDocument(); expect(screen.getAllByText("PASS").length).toBeGreaterThan(0); expect(screen.getAllByText("WAIT").length).toBeGreaterThan(0); });
  it("shows a confirmed weekend closure without marking health failed", async () => {
    const weekend: OperatorStatus = { ...operator, market: { status: "CLOSED_WEEKEND", market_open: false, reason: "FX_WEEKEND_CLOSE", next_market_open_utc: "2026-08-30T21:00:00Z", next_market_close_utc: "2026-09-04T21:00:00Z" }, session: { current_session: "WEEKEND", active_sessions: [], overlap: false, next_session: "SYDNEY", next_session_change_utc: "2026-08-30T21:00:00Z", next_transition: "MARKET OPEN" }, data_freshness: { ...operator.data_freshness, status: "STALE", reason: "STALE_MARKET_CLOSED", market_closed_context: true }, trading_window: { ...operator.trading_window, status: "MARKET_CLOSED", reason: "MARKET_CLOSED_WEEKEND", trading_state: "MARKET_CLOSED" }, trading_state: "MARKET_CLOSED" };
    mockHealthyFetch(weekend); render(<App />); expect(await screen.findByText("CLOSED — WEEKEND")).toBeInTheDocument(); expect(screen.getByText("STALE — MARKET CLOSED")).toBeInTheDocument(); expect(screen.getAllByText("PASS").length).toBeGreaterThan(0);
  });
  it("shows degraded intelligence without changing market core", async () => { mockHealthyFetch(); render(<App />); await waitFor(() => expect(screen.getByText("V05A MARKET")).toBeInTheDocument()); expect(screen.getByText("V05B INTELLIGENCE")).toBeInTheDocument(); });
  it("shows V0.8 insufficient forward history without fake metrics", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText(/Shadow Performance/i)).toBeInTheDocument(); expect(screen.getAllByText("INSUFFICIENT DATA").length).toBeGreaterThan(0); });
  it("shows the live shadow recorder running with an honest no-data state", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText(/Live Shadow Validation/i)).toBeInTheDocument(); expect(screen.getByText("NO LIVE SHADOW OBSERVATIONS YET")).toBeInTheDocument(); expect(screen.getByText("MARKET CLOSED WEEKEND")).toBeInTheDocument(); });
  it("renders a pending real observation without fake actual data", async () => {
    const observation = { observation_id: "a".repeat(64), decision_timestamp_utc: "2026-08-24T12:00:00Z", evaluation_due_utc: "2026-08-24T12:15:00Z", horizon_minutes: 15, model_id: "champion", model_status: "APPROVED_CHAMPION", predicted_class: "DOWN", prob_down: .7, prob_neutral: .2, prob_up: .1, model_confidence: .7, v06c_decision: "WAIT", v06c_gate: "WAIT_EVENT_DATA_INCOMPLETE", actual_class: null, actual_return: null, direction_correct: null, evaluation_status: "PENDING", blocking_reasons: ["EVENT_DATA_INCOMPLETE"] };
    const summary = { ...emptyShadow, status: "AVAILABLE", performance_observations: 1, total_observations: 3, counts: { ...emptyShadow.counts, PENDING: 1, NO_APPROVED_MODEL: 2 }, latest_observation: observation };
    const recent = { ...emptyRecent, count: 1, items: [observation] };
    mockHealthyFetch(operator, summary, recent); render(<App />); expect(await screen.findByText("LATEST PREDICTION")).toBeInTheDocument(); expect(screen.getAllByText("PENDING").length).toBeGreaterThan(0); expect(screen.getAllByText("N/A").length).toBeGreaterThan(0);
  });
  it("renders an evaluated observation with actual result", async () => {
    const observation = { observation_id: "b".repeat(64), decision_timestamp_utc: "2026-08-24T12:00:00Z", evaluation_due_utc: "2026-08-24T12:15:00Z", horizon_minutes: 15, model_id: "champion", model_status: "APPROVED_CHAMPION", predicted_class: "DOWN", prob_down: .7, prob_neutral: .2, prob_up: .1, model_confidence: .7, v06c_decision: "WAIT", v06c_gate: "WAIT_EVENT_DATA_INCOMPLETE", actual_class: "DOWN", actual_return: -.0014, direction_correct: true, evaluation_status: "EVALUATED", blocking_reasons: ["EVENT_DATA_INCOMPLETE"] };
    const summary = { ...emptyShadow, status: "AVAILABLE", performance_observations: 1, total_observations: 3, counts: { ...emptyShadow.counts, EVALUATED: 1, NO_APPROVED_MODEL: 2 }, latest_observation: observation };
    mockHealthyFetch(operator, summary, { ...emptyRecent, count: 1, items: [observation] }); render(<App />); expect(await screen.findAllByText("CORRECT")).not.toHaveLength(0); expect(screen.getAllByText("DOWN").length).toBeGreaterThan(0); expect(screen.getAllByText("-0.140%").length).toBeGreaterThan(0);
  });
});
