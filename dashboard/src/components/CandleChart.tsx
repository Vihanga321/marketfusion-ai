import type { Candle } from "../types/marketfusion";

export function CandleChart({ candles, livePrice = null, live = false }: { candles: Candle[]; livePrice?: number | null; live?: boolean }) {
  const visible = candles.slice(-90);
  if (!visible.length) return <div className="chart-empty">MARKET DATA UNAVAILABLE</div>;
  const width = 960, height = 330, pad = 18;
  const highs = visible.map(c => c.high); const lows = visible.map(c => c.low);
  if (livePrice != null && Number.isFinite(livePrice)) { highs.push(livePrice); lows.push(livePrice); }
  const high = Math.max(...highs); const low = Math.min(...lows);
  const range = Math.max(high - low, 0.00001); const step = (width - pad * 2) / visible.length;
  const y = (value: number) => pad + ((high - value) / range) * (height - pad * 2);
  const digits = high > 100 ? 2 : 5;
  return <svg className="candle-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Completed candle chart with display-only live price overlay">
    <defs><pattern id="grid" width="80" height="55" patternUnits="userSpaceOnUse"><path d="M 80 0 L 0 0 0 55" fill="none" stroke="#17304a" strokeWidth="1" /></pattern></defs>
    <rect width={width} height={height} fill="url(#grid)" />
    {visible.map((candle, index) => {
      const x = pad + index * step + step / 2; const up = candle.close >= candle.open; const color = up ? "#20c997" : "#f0657a";
      const bodyTop = y(Math.max(candle.open, candle.close)); const bodyHeight = Math.max(1.5, Math.abs(y(candle.open) - y(candle.close)));
      return <g key={candle.time}><line x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} stroke={color} strokeWidth="1" /><rect x={x - Math.max(1, step * .28)} y={bodyTop} width={Math.max(2, step * .56)} height={bodyHeight} fill={color} rx=".5" /></g>;
    })}
    {livePrice != null && Number.isFinite(livePrice) && <g aria-label={live ? "Live display price" : "Stale display price"}>
      <line x1={pad} x2={width - pad} y1={y(livePrice)} y2={y(livePrice)} stroke={live ? "#4dd7ff" : "#f0b44d"} strokeWidth="1.2" strokeDasharray="5 4" opacity="0.9" />
      <circle cx={width - pad - 2} cy={y(livePrice)} r="4" fill={live ? "#4dd7ff" : "#f0b44d"} />
      <rect x={width - 112} y={Math.max(4, y(livePrice) - 14)} width="104" height="22" rx="4" fill="#071521" opacity="0.94" />
      <text x={width - 14} y={Math.max(18, y(livePrice) + 2)} textAnchor="end" fill={live ? "#8ee8ff" : "#ffd37a"} fontSize="11">{live ? "LIVE" : "STALE"} {livePrice.toFixed(digits)}</text>
    </g>}
    <text x={width - 10} y={18} textAnchor="end" fill="#7f94aa" fontSize="12">{high.toFixed(digits)}</text>
    <text x={width - 10} y={height - 8} textAnchor="end" fill="#7f94aa" fontSize="12">{low.toFixed(digits)}</text>
  </svg>;
}
