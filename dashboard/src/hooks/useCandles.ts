import { useEffect, useState } from "react";
import { fetchCandles } from "../api/client";
import type { AssetId } from "../api/client";
import type { Candle, Timeframe } from "../types/marketfusion";

export function useCandles(timeframe: Timeframe, symbol: AssetId = "EURUSD", refreshMs = 2000) {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [status, setStatus] = useState("LOADING");
  useEffect(() => {
    let active = true;
    let controller: AbortController | null = null;
    const load = async () => {
      controller?.abort(); controller = new AbortController();
      try {
        const { data } = await fetchCandles(timeframe, controller.signal, symbol);
        if (active) { setCandles(data.candles); setStatus(data.status); }
      } catch { if (active) setStatus("DISCONNECTED"); }
    };
    void load();
    const timer = window.setInterval(() => void load(), refreshMs);
    return () => { active = false; controller?.abort(); window.clearInterval(timer); };
  }, [timeframe, symbol, refreshMs]);
  return { candles, status };
}
