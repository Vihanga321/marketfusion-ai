import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { MarketFusionState } from "./types/marketfusion";
import { localTime } from "./utils/format";

const state: MarketFusionState = {
  contract_version: "v0.6c-unified-runtime-v1",
  system: { status: "PASS_FAIL_CLOSED_NO_CHAMPION", mode: "SHADOW_ADVISORY_ONLY", symbol: "EURUSD", generated_time: { utc: "2026-08-24T12:00:00Z", asia_colombo: "2026-08-24T17:30:00+05:30" }, registry: { champion_count: 0 } },
  market: { close: 1.166, freshness: { status: "FRESH", age_minutes: 2, observed_at_utc: "2026-08-24T11:58:00Z" }, session: "OVERLAP", regime: { trend_regime: "DOWN", volatility_regime: "NORMAL" }, spread: { status: "NORMAL", current_points: 1 } },
  predictions: { v06a_status: "PASS_FAIL_CLOSED_NO_CHAMPION", horizons: {
    "15": { model_status: "NO_APPROVED_MODEL", model_id: null, prob_down: null, prob_neutral: null, prob_up: null },
    "60": { model_status: "NO_APPROVED_MODEL", model_id: null, prob_down: null, prob_neutral: null, prob_up: null },
    "240": { model_status: "NO_APPROVED_MODEL", model_id: null, prob_down: null, prob_neutral: null, prob_up: null }
  }, fusion: { status: "NO_APPROVED_MODEL", direction: "WAIT", confidence: "VERY_LOW", probabilities: null, valid_horizons: [] } },
  decision: { action: "WAIT", confidence: "VERY_LOW", gate: "WAIT_NO_MODEL", manual_confirmation_required: true, trading_enabled: false, decision_time: { utc: "2026-08-24T11:55:00Z", asia_colombo: null }, next_reassessment: { utc: "2026-08-24T12:05:00Z", asia_colombo: null } },
  trade_window: { status: "NOT_APPLICABLE_WAIT", start_utc: null, end_utc: null, horizon_minutes: null },
  event: { status: "NONE", nearest_event: null }, intelligence: { status: "DEGRADED", provider_errors: 5 },
  health: { sources: { v05a_market: "PASS", v05b_intelligence: "PASS_DEGRADED" }, failed_sources: [], degraded_sources: ["v05b_intelligence"] },
  reasons: [{ code: "NO_APPROVED_MODEL", priority: 95, blocking: true, source: "V0.5C/V0.6A", message: "No approved V0.5C champion", evidence: null }]
};

function response(data: object, freshness = "FRESH") {
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(data), headers: new Headers({ "X-MarketFusion-State-Freshness": freshness }) } as Response);
}

function mockHealthyFetch() {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/state") return response(state);
    if (url.startsWith("/api/market/candles")) return response({ status: "PASS", timeframe: "M5", count: 0, candles: [] });
    if (url === "/api/market/summary") return response({ status: "PASS_RUNNING", captured_at_utc: null, bid: 1.16, ask: 1.17, mid: 1.165, spread_points: 1 });
    if (url === "/api/intelligence") return response({ status: "PASS_RUNNING_DEGRADED", captured_at_utc: null, news_counts: { "15": 0, "60": 0, "240": 0, "1440": 1 }, macro: {}, provider_health: { FED_RSS: "OK", ECB: "OK", BLS: "OK", GDELT: "DEGRADED", FRED: "DEGRADED" } });
    if (url === "/api/events") return response({ status: "PASS", events: [] });
    return response({ status: "RESEARCH_ONLY", approved_champions: 0, candidates: {}, last_training_utc: null, market_core_rows: 8558, history_days: 39, v05b_eligibility: "INSUFFICIENT_HISTORY", live_surprise_samples: 0, next_recommended_training: "DAILY_MANUAL_SCHEDULE" });
  }));
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("MarketFusion dashboard safety UI", () => {
  it("displays intentional WAIT state with no champion", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("CURRENT ADVISORY")).toBeInTheDocument(); expect(screen.getAllByText("WAIT").length).toBeGreaterThan(0); });
  it("displays all three no-approved-model cards", async () => { mockHealthyFetch(); render(<App />); expect((await screen.findAllByText("NO APPROVED MODEL"))).toHaveLength(3); });
  it("renders null probabilities as dashes", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(9); });
  it("contains no buy sell or order controls", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.queryByRole("button", { name: /buy|sell|order|close/i })).not.toBeInTheDocument(); });
  it("shows manual execution and disabled trading labels", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("MANUAL EXECUTION")).toBeInTheDocument(); expect(screen.getAllByText("DISABLED").length).toBeGreaterThan(0); });
  it("connection loss forces safety WAIT", async () => { vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new Error("offline")))); render(<App />); expect(await screen.findByText("CONNECTION LOST")).toBeInTheDocument(); expect(screen.getByText("SAFETY ADVISORY")).toBeInTheDocument(); expect(screen.getAllByText("WAIT").length).toBeGreaterThan(0); });
  it("makes degraded providers visible", async () => { mockHealthyFetch(); render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(screen.getAllByText("DEGRADED").length).toBeGreaterThanOrEqual(2); expect(screen.getByText("GDELT")).toBeInTheDocument(); expect(screen.getByText("FRED")).toBeInTheDocument(); });
  it("shows event countdown only when event timestamp exists", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("AUDITED EVENT DATA UNAVAILABLE")).toBeInTheDocument(); expect(screen.queryByText("BLOCK −15m")).not.toBeInTheDocument(); });
  it("keeps trade window empty for WAIT_NO_MODEL", async () => { mockHealthyFetch(); render(<App />); expect(await screen.findByText("WAIT_NO_MODEL")).toBeInTheDocument(); expect(screen.getByText("No manual trade window is available. No approved model.")).toBeInTheDocument(); });
  it("formats Sri Lanka time via the timezone utility", () => { expect(localTime("2026-08-24T12:00:00Z")).toBe("17:30"); });
  it("does not display fake percentage bars", async () => { mockHealthyFetch(); const { container } = render(<App />); await screen.findByText("CURRENT ADVISORY"); expect(container.querySelectorAll(".prob-fill")).toHaveLength(0); });
  it("shows degraded intelligence without changing market core", async () => { mockHealthyFetch(); render(<App />); await waitFor(() => expect(screen.getByText("V05A MARKET")).toBeInTheDocument()); expect(screen.getByText("V05B INTELLIGENCE")).toBeInTheDocument(); });
});
