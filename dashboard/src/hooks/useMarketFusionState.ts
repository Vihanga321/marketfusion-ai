import { useEffect, useRef, useState } from "react";
import { fetchState } from "../api/client";
import type { ConnectionState, MarketFusionState } from "../types/marketfusion";

export function useMarketFusionState(intervalMs = 3000) {
  const [state, setState] = useState<MarketFusionState | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("CONNECTING");
  const [lastSuccess, setLastSuccess] = useState<Date | null>(null);
  const inFlight = useRef(false);

  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;
    const poll = async () => {
      if (!active || inFlight.current) return;
      inFlight.current = true;
      controller = new AbortController();
      try {
        const { data, response } = await fetchState(controller.signal);
        if (!active) return;
        setState(data);
        setConnection(response.headers.get("X-MarketFusion-State-Freshness") === "FRESH" ? "LIVE" : "STALE");
        setLastSuccess(new Date());
      } catch {
        if (active) setConnection("DISCONNECTED");
      } finally {
        inFlight.current = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), intervalMs);
    return () => { active = false; controller?.abort(); window.clearInterval(timer); inFlight.current = false; };
  }, [intervalMs]);
  return { state, connection, lastSuccess };
}
