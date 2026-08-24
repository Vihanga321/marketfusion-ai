import type { Candle } from "../types/marketfusion";

export function CandleChart({ candles }: { candles: Candle[] }) {
  const visible = candles.slice(-90);
  if (!visible.length) return <div className="chart-empty">MARKET DATA UNAVAILABLE</div>;
  const width = 960, height = 330, pad = 18;
  const high = Math.max(...visible.map(c => c.high)); const low = Math.min(...visible.map(c => c.low));
  const range = Math.max(high - low, 0.00001); const step = (width - pad * 2) / visible.length;
  const y = (value: number) => pad + ((high - value) / range) * (height - pad * 2);
  return <svg className="candle-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="EUR/USD completed candle chart">
    <defs><pattern id="grid" width="80" height="55" patternUnits="userSpaceOnUse"><path d="M 80 0 L 0 0 0 55" fill="none" stroke="#17304a" strokeWidth="1" /></pattern></defs>
    <rect width={width} height={height} fill="url(#grid)" />
    {visible.map((candle, index) => {
      const x = pad + index * step + step / 2; const up = candle.close >= candle.open; const color = up ? "#20c997" : "#f0657a";
      const bodyTop = y(Math.max(candle.open, candle.close)); const bodyHeight = Math.max(1.5, Math.abs(y(candle.open) - y(candle.close)));
      return <g key={candle.time}><line x1={x} x2={x} y1={y(candle.high)} y2={y(candle.low)} stroke={color} strokeWidth="1" /><rect x={x - Math.max(1, step * .28)} y={bodyTop} width={Math.max(2, step * .56)} height={bodyHeight} fill={color} rx=".5" /></g>;
    })}
    <text x={width - 10} y={18} textAnchor="end" fill="#7f94aa" fontSize="12">{high.toFixed(5)}</text>
    <text x={width - 10} y={height - 8} textAnchor="end" fill="#7f94aa" fontSize="12">{low.toFixed(5)}</text>
  </svg>;
}
