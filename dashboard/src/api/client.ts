import type { CandleResponse, EngineStatus, EventsResponse, IntelligenceDetail, MarketFusionState, MarketSummary, OperatorStatus, PatternRecord, ResearchState, ShadowRecent, ShadowSummary, Timeframe, V08Status } from "../types/marketfusion";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<{ data: T; response: Response }> {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return { data: await response.json() as T, response };
}

export type AssetId = "EURUSD" | "XAUUSD";
const assetPath = (path: string, symbol: AssetId) => symbol === "EURUSD" ? path : `${path}${path.includes("?") ? "&" : "?"}symbol=${symbol}`;

export const fetchState = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<MarketFusionState>(assetPath("/api/state", symbol), signal);
export const fetchCandles = (timeframe: Timeframe, signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<CandleResponse>(assetPath(`/api/market/candles?timeframe=${timeframe}&limit=300`, symbol), signal);
export const fetchMarketSummary = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<MarketSummary>(assetPath("/api/market/summary", symbol), signal);
export const fetchOperatorStatus = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<OperatorStatus>(assetPath("/api/operator/status", symbol), signal);
export const fetchIntelligence = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<IntelligenceDetail>(assetPath("/api/intelligence", symbol), signal);
export const fetchResearch = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<ResearchState>(assetPath("/api/research", symbol), signal);
export const fetchEvents = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<EventsResponse>(assetPath("/api/events", symbol), signal);
export const fetchV08Status = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<V08Status>(assetPath("/api/v08/status", symbol), signal);
export const fetchShadowSummary = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<ShadowSummary>(assetPath("/api/evaluation/shadow/summary", symbol), signal);
export const fetchShadowRecent = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<ShadowRecent>(assetPath("/api/evaluation/shadow/recent?limit=20", symbol), signal);
export const fetchEngineStatus = (signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<EngineStatus>(assetPath("/api/engines/status", symbol), signal);
export const fetchRecentPatterns = (limit = 20, signal?: AbortSignal, symbol: AssetId = "EURUSD") => getJson<{ status: string; count: number; items: PatternRecord[] }>(assetPath(`/api/engines/patterns/recent?limit=${limit}`, symbol), signal);
