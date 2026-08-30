import { useEffect, useState } from "react";
import type { OperatorStatus } from "../types/marketfusion";
import { countdownState, formatCountdown, formatColomboOpening } from "../utils/format";

export function MarketOpenCountdown({ operator, now, serverOffsetMs = 0 }: { operator: OperatorStatus | null; now: number; serverOffsetMs?: number }) {
  const market = operator?.market;
  const target = market?.next_market_open_utc;
  const state = countdownState(Boolean(market?.market_open), target, now + serverOffsetMs);
  if (market?.market_open) return <div className="market-countdown market-countdown-open"><span>MARKET OPEN</span></div>;
  if (state === "UNKNOWN") return <div className="market-countdown market-countdown-unknown"><span>NEXT OPEN — UNKNOWN</span></div>;
  const remaining = Math.max(0, new Date(target as string).getTime() - (now + serverOffsetMs));
  const openingSoon = remaining <= 30 * 60 * 1000;
  const expired = remaining === 0;
  return <div className={`market-countdown ${openingSoon ? "market-countdown-soon" : ""} ${remaining <= 5 * 60 * 1000 ? "market-countdown-near" : ""}`}>
    <span>{expired ? "OPENING..." : openingSoon ? "OPENING SOON" : "OPENS IN"}</span>
    {!expired && <b>{formatCountdown(remaining)}</b>}
    <small>NEXT OPEN · {formatColomboOpening(target)}</small>
  </div>;
}

export function useServerClock(operator: OperatorStatus | null): number {
  const [offsetMs, setOffsetMs] = useState(0);
  const serverUtc = operator?.current_time.utc;
  useEffect(() => {
    if (!serverUtc) return;
    const serverMs = new Date(serverUtc).getTime();
    if (Number.isFinite(serverMs)) setOffsetMs(serverMs - Date.now());
  }, [serverUtc]);
  return offsetMs;
}
