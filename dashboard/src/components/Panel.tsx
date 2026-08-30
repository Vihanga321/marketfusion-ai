import type { ReactNode } from "react";
import { MarketOpenBadge } from "./MarketOpenBadge";

export function Panel({ title, eyebrow, className = "", children, action }: { title: string; eyebrow?: string; className?: string; children: ReactNode; action?: ReactNode }) {
  const resolvedAction = action ?? (className.includes("decision-panel") ? <MarketOpenBadge /> : null);
  return <section className={`panel ${className}`}>
    <header className="panel-header"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{resolvedAction}</header>
    {children}
  </section>;
}

export function Metric({ label, value, tone = "muted", sub }: { label: string; value: ReactNode; tone?: string; sub?: ReactNode }) {
  return <div className="metric"><span>{label}</span><strong className={`tone-${tone}`}>{value}</strong>{sub && <small>{sub}</small>}</div>;
}

export function StatusDot({ label, value, tone }: { label: string; value: string; tone?: string }) {
  const resolved = tone ?? (/(PASS|LIVE|OK|CONNECTED|NORMAL|FRESH)/.test(value) ? "good" : /(DEGRADED|WAIT|NO CHAMPION|CAUTION)/.test(value) ? "warn" : "bad");
  return <div className="status-chip"><i className={`dot tone-${resolved}`} /><span>{label}</span><b>{value.replaceAll("_", " ")}</b></div>;
}
