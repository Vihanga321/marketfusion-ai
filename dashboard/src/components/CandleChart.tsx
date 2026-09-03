import { useEffect, useRef, useState } from "react";
import type { Candle } from "../types/marketfusion";

type LiveQuoteEventDetail = {
  connection?: string;
  timeframe?: string;
  mid?: number | null;
  normalizedTickUtc?: string | null;
  partialM1?: {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number | null;
    is_complete?: false;
    usage?: string;
  } | null;
};

type DisplayFormingCandle = {
  timeframe: string;
  candle: Candle;
};

const TIMEFRAME_MINUTES: Record<string, number> = { M1: 1, M5: 5, M15: 15, H1: 60 };

function bucketStart(timestamp: string | null | undefined, timeframe: string) {
  const minutes = TIMEFRAME_MINUTES[timeframe];
  if (!minutes) return null;
  const parsed = timestamp ? new Date(timestamp) : new Date();
  if (Number.isNaN(parsed.getTime())) return null;
  const bucketMs = minutes * 60_000;
  return new Date(Math.floor(parsed.getTime() / bucketMs) * bucketMs).toISOString();
}

export function CandleChart({ candles, livePrice = null, live = false }: { candles: Candle[]; livePrice?: number | null; live?: boolean }) {
  const [eventLivePrice, setEventLivePrice] = useState<number | null>(null);
  const [eventLive, setEventLive] = useState(false);
  const [displayForming, setDisplayForming] = useState<DisplayFormingCandle | null>(null);
  const latestClosedRef = useRef<Candle | null>(candles.at(-1) ?? null);

  useEffect(() => {
    latestClosedRef.current = candles.at(-1) ?? null;
  }, [candles]);

  useEffect(() => {
    const handle = (event: Event) => {
      const detail = (event as CustomEvent<LiveQuoteEventDetail>).detail ?? {};
      const isLive = String(detail.connection ?? "").toUpperCase() === "LIVE";
      const mid = detail.mid;
      const timeframe = String(detail.timeframe ?? "").toUpperCase();
      setEventLive(isLive);
      setEventLivePrice(isLive && typeof mid === "number" && Number.isFinite(mid) ? mid : null);

      if (!isLive || typeof mid !== "number" || !Number.isFinite(mid) || !TIMEFRAME_MINUTES[timeframe]) {
        setDisplayForming(null);
        return;
      }

      const partial = detail.partialM1;
      if (timeframe === "M1" && partial && [partial.open, partial.high, partial.low, partial.close].every(Number.isFinite)) {
        setDisplayForming({
          timeframe,
          candle: {
            time: partial.time,
            open: partial.open,
            high: partial.high,
            low: partial.low,
            close: partial.close,
            volume: partial.volume ?? null,
          },
        });
        return;
      }

      // M5/M15/H1 forming bars are browser-only visual approximations seeded
      // from the latest completed close and advanced by the live mid price.
      // They never enter the causal completed-candle/model path.
      const time = bucketStart(detail.normalizedTickUtc, timeframe);
      if (!time) {
        setDisplayForming(null);
        return;
      }

      setDisplayForming((current) => {
        if (current?.timeframe === timeframe && current.candle.time === time) {
          return {
            timeframe,
            candle: {
              ...current.candle,
              high: Math.max(current.candle.high, mid),
              low: Math.min(current.candle.low, mid),
              close: mid,
            },
          };
        }
        const previousClose = current?.timeframe === timeframe ? current.candle.close : latestClosedRef.current?.close;
        const open = typeof previousClose === "number" && Number.isFinite(previousClose) ? previousClose : mid;
        return {
          timeframe,
          candle: {
            time,
            open,
            high: Math.max(open, mid),
            low: Math.min(open, mid),
            close: mid,
            volume: null,
          },
        };
      });
    };
    window.addEventListener("marketfusion-live-quote", handle);
    return () => window.removeEventListener("marketfusion-live-quote", handle);
  }, []);

  const effectiveLivePrice = livePrice ?? eventLivePrice;
  const effectiveLive = live || eventLive;
  const eventPartial = displayForming?.candle ?? null;
  const completed = candles.slice(eventPartial ? -89 : -90);
  const visible = eventPartial && !completed.some((candle) => candle.time === eventPartial.time)
    ? [...completed, eventPartial]
    : completed;
  if (!visible.length) return <div className="chart-empty">MARKET DATA UNAVAILABLE</div>;
  const width = 960, height = 330, pad = 18;
  const highs = visible.map(c => c.high); const lows = visible.map(c => c.low);
  if (effectiveLivePrice != null && Number.isFinite(effectiveLivePrice)) { highs.push(effectiveLivePrice); lows.push(effectiveLivePrice); }
  const high = Math.max(...highs); const low = Math.min(...lows);
  const range = Math.max(high - low, 0.00001); const step = (width - pad * 2) / visible.length;
  const y = (value: number) => pad + ((high - value) / range) * (height - pad * 2);
  const digits = high > 100 ? 2 : 5;
  return <svg className="candle-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Completed causal candles with a display-only realtime forming candle and live price overlay">
    <defs><pattern id="grid" width="80" height="55" patternUnits="userSpaceOnUse"><path d="M 80 0 L 0 0 0 55" fill="none" stroke="#17304a" strokeWidth="1" /></pattern></defs>
    <rect width={width} height={height} fill="url(#grid)" />
    {visible.map((candle, index) => {
      const isPartial = eventPartial?.time === candle.time;
      const x = pad + index * step + step / 2; const up = candle.close >= candle.open; const color = up ? "#20c997" : "#f0657a";
      const bodyTop = y(Math.max(candle.open, candle.close)); const bodyHeight = Math.max(1.5, Math.abs(y(candle.open) - y(candle.close)));
      return <g key={`${candle.time}-${isPartial ? "live" : "closed"}`} aria-label={isPartial ? `Display-only forming ${displayForming?.timeframe ?? ""} candle` : "Completed candle"}><line x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} stroke={color} strokeWidth={isPartial ? "1.6" : "1"} /><rect x={x - Math.max(1, step * .28)} y={bodyTop} width={Math.max(2, step * .56)} height={bodyHeight} fill={color} rx=".5" opacity={isPartial ? "0.92" : "1"} />{isPartial && <circle cx={x} cy={y(candle.close)} r="2.5" fill="#8ee8ff" />}</g>;
    })}
    {effectiveLivePrice != null && Number.isFinite(effectiveLivePrice) && <g aria-label={effectiveLive ? "Live display price" : "Stale display price"}>
      <line x1={pad} x2={width - pad} y1={y(effectiveLivePrice)} y2={y(effectiveLivePrice)} stroke={effectiveLive ? "#4dd7ff" : "#f0b44d"} strokeWidth="1.2" strokeDasharray="5 4" opacity="0.9" />
      <circle cx={width - pad - 2} cy={y(effectiveLivePrice)} r="4" fill={effectiveLive ? "#4dd7ff" : "#f0b44d"} />
      <rect x={width - 112} y={Math.max(4, y(effectiveLivePrice) - 14)} width="104" height="22" rx="4" fill="#071521" opacity="0.94" />
      <text x={width - 14} y={Math.max(18, y(effectiveLivePrice) + 2)} textAnchor="end" fill={effectiveLive ? "#8ee8ff" : "#ffd37a"} fontSize="11">{effectiveLive ? "LIVE" : "STALE"} {effectiveLivePrice.toFixed(digits)}</text>
    </g>}
    <text x={width - 10} y={18} textAnchor="end" fill="#7f94aa" fontSize="12">{high.toFixed(digits)}</text>
    <text x={width - 10} y={height - 8} textAnchor="end" fill="#7f94aa" fontSize="12">{low.toFixed(digits)}</text>
  </svg>;
}
