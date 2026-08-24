import type { CandleResponse, EventsResponse, IntelligenceDetail, MarketFusionState, MarketSummary, ResearchState, Timeframe } from "../types/marketfusion";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<{ data: T; response: Response }> {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return { data: await response.json() as T, response };
}

export const fetchState = (signal?: AbortSignal) => getJson<MarketFusionState>("/api/state", signal);
export const fetchCandles = (timeframe: Timeframe, signal?: AbortSignal) => getJson<CandleResponse>(`/api/market/candles?timeframe=${timeframe}&limit=300`, signal);
export const fetchMarketSummary = (signal?: AbortSignal) => getJson<MarketSummary>("/api/market/summary", signal);
export const fetchIntelligence = (signal?: AbortSignal) => getJson<IntelligenceDetail>("/api/intelligence", signal);
export const fetchResearch = (signal?: AbortSignal) => getJson<ResearchState>("/api/research", signal);
export const fetchEvents = (signal?: AbortSignal) => getJson<EventsResponse>("/api/events", signal);
