import { useEffect, useState } from "react";
import { FullDashboardShell } from "./components/FullDashboardShell";
import { useCandles } from "./hooks/useCandles";
import { useForwardValidation } from "./hooks/useForwardValidation";
import { useMarketFusionState } from "./hooks/useMarketFusionState";
import { useRealtimeQuote } from "./hooks/useRealtimeQuote";
import { useSupplementary } from "./hooks/useSupplementary";
import type { Timeframe } from "./types/marketfusion";
import type { AssetId } from "./api/client";
import "./full-dashboard.css";

export default function App() {
  const configured = (import.meta.env.VITE_MARKETFUSION_SYMBOL === "XAUUSD" ? "XAUUSD" : "EURUSD") as AssetId;
  const [selectedAsset, setSelectedAsset] = useState<AssetId>(configured);
  const { state, connection, lastSuccess } = useMarketFusionState(selectedAsset);
  const realtime = useRealtimeQuote(selectedAsset);
  const [timeframe, setTimeframe] = useState<Timeframe>("M5");
  const { candles, status: candleStatus } = useCandles(timeframe, selectedAsset);
  const supplementary = useSupplementary(selectedAsset);
  const forward = useForwardValidation(selectedAsset);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const snapshot = realtime.snapshot;
    const isLive = realtime.connection === "LIVE";
    window.dispatchEvent(new CustomEvent("marketfusion-live-quote", {
      detail: {
        symbol: selectedAsset,
        timeframe,
        connection: realtime.connection,
        mid: isLive ? snapshot?.mid ?? null : null,
        sequence: snapshot?.sequence ?? null,
        normalizedTickUtc: isLive ? snapshot?.normalized_tick_utc ?? null : null,
        partialM1: isLive ? snapshot?.partial_m1 ?? null : null,
      },
    }));
  }, [selectedAsset, timeframe, realtime.connection, realtime.snapshot]);

  const refresh = () => {
    supplementary.refresh();
    forward.refresh();
  };

  return (
    <FullDashboardShell
      selectedAsset={selectedAsset}
      onAssetChange={setSelectedAsset}
      state={state}
      connection={connection}
      lastSuccess={lastSuccess}
      operator={supplementary.operator}
      quote={realtime.quote ?? supplementary.market}
      intelligence={supplementary.intelligence}
      research={supplementary.research}
      events={supplementary.events}
      monitoring={supplementary.monitoring}
      shadowSummary={supplementary.shadowSummary}
      shadowRecent={supplementary.shadowRecent}
      engines={supplementary.engines}
      candles={candles}
      candleStatus={candleStatus}
      timeframe={timeframe}
      setTimeframe={setTimeframe}
      now={now}
      refresh={refresh}
      forwardValidation={forward.status}
      forwardConnection={forward.connection}
    />
  );
}
