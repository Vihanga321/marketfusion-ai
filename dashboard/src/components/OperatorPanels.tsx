import { CalendarClock, Clock3, Database, Radio, ShieldCheck } from "lucide-react";
import type { OperatorStatus } from "../types/marketfusion";
import { ageFrom, dash, label, localTime, until, utcTime } from "../utils/format";
import { Metric, Panel } from "./Panel";

function marketLabel(status?: string): string {
  if (status === "OPEN") return "LIVE";
  if (status === "CLOSED_WEEKEND") return "CLOSED — WEEKEND";
  return label(status);
}

export function OperatorPanels({ operator, now, connected }: { operator: OperatorStatus | null; now: number; connected: boolean }) {
  const market = operator?.market;
  const data = operator?.data_freshness;
  const session = operator?.session;
  const reassessment = operator?.reassessment;
  const window = operator?.trading_window;
  const dataLabel = data?.status === "STALE" && data.market_closed_context ? "STALE — MARKET CLOSED" : label(data?.status);
  const currentIso = new Date(now).toISOString();
  const transitionTarget = session?.next_session_change_utc;
  return <div className="operator-grid">
    <Panel title="Market Status" eyebrow="FOREX WEEK CALENDAR" className="operator-panel">
      <div className={`operator-hero tone-${market?.market_open ? "good" : "warn"}`}><Radio size={18} /><strong>{marketLabel(market?.status)}</strong></div>
      <div className="operator-lines"><span>Current LKT</span><b>{localTime(currentIso, true)}</b><span>Current UTC</span><b>{utcTime(currentIso)}</b><span>Last M5</span><b>{localTime(data?.latest_completed_m5_utc)} · {ageFrom(data?.latest_completed_m5_utc, now)}</b><span>Next market open</span><b>{market?.market_open ? "OPEN" : until(market?.next_market_open_utc, now)}</b></div>
    </Panel>

    <Panel title="Session" eyebrow="DST-AWARE GLOBAL WINDOWS" className="operator-panel">
      <div className="operator-hero tone-info"><CalendarClock size={18} /><strong>{label(session?.current_session)}</strong></div>
      <div className="operator-lines"><span>Active</span><b>{session?.active_sessions.length ? session.active_sessions.map(label).join(" + ") : dash}</b><span>Overlap</span><b>{session?.overlap ? "ACTIVE" : "NONE"}</b><span>{label(session?.next_transition)}</span><b>{until(transitionTarget, now)}</b></div>
    </Panel>

    <Panel title="Data Freshness" eyebrow="V0.5A COMPLETED DATA" className="operator-panel">
      <div className={`operator-hero tone-${data?.status === "LIVE" || data?.status === "FRESH" ? "good" : "warn"}`}><Database size={18} /><strong>{dataLabel}</strong></div>
      <div className="operator-lines"><span>Market tick</span><b>{localTime(data?.latest_market_tick_utc)} · {ageFrom(data?.latest_market_tick_utc, now)}</b><span>Completed M5</span><b>{localTime(data?.latest_completed_m5_utc)} · {ageFrom(data?.latest_completed_m5_utc, now)}</b><span>Completed M15</span><b>{localTime(data?.latest_completed_m15_utc)} · {ageFrom(data?.latest_completed_m15_utc, now)}</b><span>Feature row</span><b>{localTime(data?.feature_row_utc)} · {ageFrom(data?.feature_row_utc, now)}</b></div>
    </Panel>

    <Panel title="Next AI Reassessment" eyebrow="V0.6 PUBLISHED SCHEDULE" className="operator-panel">
      <div className="operator-hero tone-info"><Clock3 size={18} /><strong>{reassessment?.status === "UNAVAILABLE" ? "UNAVAILABLE" : reassessment?.status === "OVERDUE" ? "OVERDUE" : until(reassessment?.at_utc, now)}</strong></div>
      <div className="operator-lines"><span>Scheduled LKT</span><b>{localTime(reassessment?.at_utc)}</b><span>Scheduled UTC</span><b>{utcTime(reassessment?.at_utc)}</b></div><p className="operator-reason">{reassessment?.reason ?? "Runtime reassessment data unavailable."}</p>
    </Panel>

    <Panel title="Trading Window" eyebrow="ADVISORY ONLY" className="operator-panel">
      <div className="operator-hero tone-warn"><ShieldCheck size={18} /><strong>{connected ? label(window?.status) : "WAITING FOR DATA"}</strong></div>
      <div className="operator-lines"><span>Trading state</span><b>{connected ? label(operator?.trading_state) : "BLOCKED"}</b><span>Reason</span><b>{connected ? label(window?.reason) : "RUNTIME CONNECTION UNAVAILABLE"}</b><span>Execution</span><b>DISABLED</b><span>Manual confirmation</span><b>REQUIRED</b></div>
    </Panel>
  </div>;
}
