import { useEffect, useState } from "react";
import { fetchEvents, fetchIntelligence, fetchMarketSummary, fetchOperatorStatus, fetchResearch, fetchV08Status } from "../api/client";
import type { EventsResponse, IntelligenceDetail, MarketSummary, OperatorStatus, ResearchState, V08Status } from "../types/marketfusion";

export function useSupplementary(refreshMs = 15000) {
  const [market, setMarket] = useState<MarketSummary | null>(null);
  const [intelligence, setIntelligence] = useState<IntelligenceDetail | null>(null);
  const [research, setResearch] = useState<ResearchState | null>(null);
  const [events, setEvents] = useState<EventsResponse | null>(null);
  const [monitoring, setMonitoring] = useState<V08Status | null>(null);
  const [operator, setOperator] = useState<OperatorStatus | null>(null);
  useEffect(() => {
    let active = true; let controller: AbortController | null = null;
    const load = async () => {
      controller?.abort(); controller = new AbortController();
      try {
        const [marketResult, intelResult, researchResult, eventResult, monitoringResult, operatorResult] = await Promise.all([
          fetchMarketSummary(controller.signal), fetchIntelligence(controller.signal), fetchResearch(controller.signal), fetchEvents(controller.signal), fetchV08Status(controller.signal), fetchOperatorStatus(controller.signal)
        ]);
        if (active) { setMarket(marketResult.data); setIntelligence(intelResult.data); setResearch(researchResult.data); setEvents(eventResult.data); setMonitoring(monitoringResult.data); setOperator(operatorResult.data); }
      } catch { /* Primary state hook owns connection status. */ }
    };
    void load(); const timer = window.setInterval(() => void load(), refreshMs);
    return () => { active = false; controller?.abort(); window.clearInterval(timer); };
  }, [refreshMs]);
  return { market, intelligence, research, events, monitoring, operator };
}
