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

function axisTime(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  }).format(parsed).replace(",", "");
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
  const completed = candles.slice(eventPartial ? -109 : -110);
  const visible = eventPartial && !completed.some((candle) => candle.time === eventPartial.time)
    ? [...completed, eventPartial]
    : completed;

  if (!visible.length) return <div className="chart-empty">MARKET DATA UNAVAILABLE</div>;

  // MT5-like geometry while preserving MarketFusion's existing green/red colors.
  const width = 960;
  const height = 360;
  const leftPad = 12;
  const rightPad = 72;
  const topPad = 12;
  const bottomPad = 28;
  const volumeHeight = 48;
  const volumeGap = 8;
  const priceBottom = height - bottomPad - volumeHeight - volumeGap;
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = priceBottom - topPad;

  const highs = visible.map((c) => c.high);
  const lows = visible.map((c) => c.low);
  if (effectiveLivePrice != null && Number.isFinite(effectiveLivePrice)) {
    highs.push(effectiveLivePrice);
    lows.push(effectiveLivePrice);
  }
  const rawHigh = Math.max(...highs);
  const rawLow = Math.min(...lows);
  const rawRange = Math.max(rawHigh - rawLow, 0.00001);
  const high = rawHigh + rawRange * 0.035;
  const low = rawLow - rawRange * 0.035;
  const range = Math.max(high - low, 0.00001);
  const step = plotWidth / visible.length;
  const y = (value: number) => topPad + ((high - value) / range) * plotHeight;
  const digits = high > 100 ? 2 : 5;

  const volumes = visible.map((c) => typeof c.volume === "number" && Number.isFinite(c.volume) ? Math.max(0, c.volume) : 0);
  const maxVolume = Math.max(...volumes, 0);
  const volumeTop = priceBottom + volumeGap;
  const volumeBottom = height - bottomPad;
  const volumeY = (value: number) => maxVolume > 0
    ? volumeBottom - (value / maxVolume) * volumeHeight
    : volumeBottom;

  const horizontalTicks = 6;
  const timeTickIndexes = Array.from(new Set([0, Math.floor((visible.length - 1) * 0.25), Math.floor((visible.length - 1) * 0.5), Math.floor((visible.length - 1) * 0.75), visible.length - 1]));

  return <svg className="candle-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="MT5-style MarketFusion chart with completed causal candles and a display-only realtime forming candle">
    <rect x="0" y="0" width={width} height={height} fill="#071521" />

    {/* MT5-style dashed grid */}
    {Array.from({ length: horizontalTicks + 1 }, (_, index) => {
      const gy = topPad + (plotHeight / horizontalTicks) * index;
      return <line key={`h-${index}`} x1={leftPad} x2={width - rightPad} y1={gy} y2={gy} stroke="#17304a" strokeWidth="1" strokeDasharray="3 4" />;
    })}
    {Array.from({ length: 12 }, (_, index) => {
      const gx = leftPad + (plotWidth / 11) * index;
      return <line key={`v-${index}`} x1={gx} x2={gx} y1={topPad} y2={volumeBottom} stroke="#17304a" strokeWidth="1" strokeDasharray="3 4" />;
    })}

    {/* Candles: same MarketFusion colors, slimmer MT5-like bodies/wicks */}
    {visible.map((candle, index) => {
      const isPartial = eventPartial?.time === candle.time;
      const x = leftPad + index * step + step / 2;
      const up = candle.close >= candle.open;
      const color = up ? "#20c997" : "#f0657a";
      const bodyTop = y(Math.max(candle.open, candle.close));
      const bodyHeight = Math.max(1.1, Math.abs(y(candle.open) - y(candle.close)));
      const bodyWidth = Math.max(1.4, Math.min(5.2, step * 0.48));
      return <g key={`${candle.time}-${isPartial ? "live" : "closed"}`} aria-label={isPartial ? `Display-only forming ${displayForming?.timeframe ?? ""} candle` : "Completed candle"}>
        <line x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} stroke={color} strokeWidth={isPartial ? "1.35" : "1"} />
        <rect x={x - bodyWidth / 2} y={bodyTop} width={bodyWidth} height={bodyHeight} fill={color} stroke={color} strokeWidth="0.7" />
      </g>;
    })}

    {/* Actual volume only; no fabricated bars when volume is unavailable. */}
    {maxVolume > 0 && visible.map((candle, index) => {
      const value = volumes[index];
      if (value <= 0) return null;
      const x = leftPad + index * step + step / 2;
      const up = candle.close >= candle.open;
      const color = up ? "#20c997" : "#f0657a";
      const barWidth = Math.max(1, Math.min(3.2, step * 0.32));
      const vy = volumeY(value);
      return <rect key={`vol-${candle.time}`} x={x - barWidth / 2} y={vy} width={barWidth} height={Math.max(1, volumeBottom - vy)} fill={color} opacity="0.75" />;
    })}

    {/* Right-side price scale, closer to MT5. */}
    {Array.from({ length: horizontalTicks + 1 }, (_, index) => {
      const price = high - (range / horizontalTicks) * index;
      const gy = topPad + (plotHeight / horizontalTicks) * index;
      return <text key={`p-${index}`} x={width - rightPad + 7} y={gy + 4} fill="#7f94aa" fontSize="11">{price.toFixed(digits)}</text>;
    })}

    {/* Bottom date/time scale. */}
    {timeTickIndexes.map((index) => {
      const candle = visible[index];
      if (!candle) return null;
      const x = leftPad + index * step + step / 2;
      const anchor = index === 0 ? "start" : index === visible.length - 1 ? "end" : "middle";
      const labelX = index === 0 ? leftPad : index === visible.length - 1 ? width - rightPad : x;
      return <text key={`t-${candle.time}`} x={labelX} y={height - 8} textAnchor={anchor} fill="#7f94aa" fontSize="10">{axisTime(candle.time)}</text>;
    })}

    {/* Display-only live price line. */}
    {effectiveLivePrice != null && Number.isFinite(effectiveLivePrice) && <g aria-label={effectiveLive ? "Live display price" : "Stale display price"}>
      <line x1={leftPad} x2={width - rightPad} y1={y(effectiveLivePrice)} y2={y(effectiveLivePrice)} stroke={effectiveLive ? "#4dd7ff" : "#f0b44d"} strokeWidth="1" strokeDasharray="5 4" opacity="0.9" />
      <rect x={width - rightPad + 2} y={Math.max(2, y(effectiveLivePrice) - 10)} width={rightPad - 5} height="20" rx="2" fill="#10263a" />
      <text x={width - 5} y={Math.max(15, y(effectiveLivePrice) + 4)} textAnchor="end" fill={effectiveLive ? "#8ee8ff" : "#ffd37a"} fontSize="11">{effectiveLivePrice.toFixed(digits)}</text>
    </g>}

    <line x1={width - rightPad} x2={width - rightPad} y1={topPad} y2={volumeBottom} stroke="#29455f" strokeWidth="1" />
    <line x1={leftPad} x2={width - rightPad} y1={volumeBottom} y2={volumeBottom} stroke="#29455f" strokeWidth="1" />
  </svg>;
}
