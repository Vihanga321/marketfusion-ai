import { useEffect, useState } from "react";
import { DashboardShell } from "./components/DashboardShell";
import { useCandles } from "./hooks/useCandles";
import { useMarketFusionState } from "./hooks/useMarketFusionState";
import { useSupplementary } from "./hooks/useSupplementary";
import type { Timeframe } from "./types/marketfusion";
import type { AssetId } from "./api/client";

export default function App() {
  const configured = (import.meta.env.VITE_MARKETFUSION_SYMBOL === "XAUUSD" ? "XAUUSD" : "EURUSD") as AssetId;
  const [selectedAsset, setSelectedAsset] = useState<AssetId>(configured);
  const { state, connection, lastSuccess } = useMarketFusionState(selectedAsset);
  const [timeframe, setTimeframe] = useState<Timeframe>("M5");
  const { candles, status: candleStatus } = useCandles(timeframe, selectedAsset);
  const supplementary = useSupplementary(selectedAsset);
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, []);
  return <><label className="asset-selector"><span>ACTIVE ASSET</span><select aria-label="Asset" value={selectedAsset} onChange={event => setSelectedAsset(event.target.value as AssetId)}><option value="EURUSD">EUR/USD</option><option value="XAUUSD">XAU/USD</option></select><small>{selectedAsset === "XAUUSD" ? "PRECIOUS METAL" : "FOREX"}</small></label><DashboardShell selectedAsset={selectedAsset} onAssetChange={setSelectedAsset} state={state} connection={connection} lastSuccess={lastSuccess} operator={supplementary.operator} quote={supplementary.market} intelligence={supplementary.intelligence} research={supplementary.research} events={supplementary.events} monitoring={supplementary.monitoring} shadowSummary={supplementary.shadowSummary} shadowRecent={supplementary.shadowRecent} engines={supplementary.engines} candles={candles} candleStatus={candleStatus} timeframe={timeframe} setTimeframe={setTimeframe} now={now} refresh={supplementary.refresh} /></>;
}
