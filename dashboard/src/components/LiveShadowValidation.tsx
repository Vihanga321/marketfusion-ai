import { ShieldCheck } from "lucide-react";
import type { ShadowRecent, ShadowSummary } from "../types/marketfusion";
import { dash, label, localTime, number, probability } from "../utils/format";
import { Metric, Panel, StatusDot } from "./Panel";

interface Props { summary: ShadowSummary | null; recent: ShadowRecent | null }

export function LiveShadowValidation({ summary, recent }: Props) {
  const horizon = summary?.horizons?.["15"];
  const latest = summary?.performance_observations ? summary.latest_observation : null;
  const empty = !summary || summary.performance_observations === 0;
  return <section className="shadow-validation">
    <Panel title="Live Shadow Validation" eyebrow="V0.8 REAL FORWARD OBSERVATIONS" action={<StatusDot label="RECORDER" value={summary?.recorder?.status ?? "UNAVAILABLE"} tone={summary?.recorder?.status === "RUNNING" ? "good" : "warn"} />}>
      <div className="shadow-summary-grid">
        <Metric label="OBSERVATIONS" value={summary?.performance_observations ?? dash} sub={`${summary?.total_observations ?? 0} horizon states retained`} />
        <Metric label="PENDING" value={(summary?.counts?.PENDING ?? 0) + (summary?.counts?.PENDING_DATA ?? 0)} />
        <Metric label="EVALUATED" value={summary?.counts?.EVALUATED ?? 0} />
        <Metric label="INVALID" value={summary?.counts?.INVALID ?? 0} tone={(summary?.counts?.INVALID ?? 0) ? "bad" : "muted"} />
        <Metric label="15M ACCURACY" value={horizon?.accuracy == null ? label(horizon?.sample_status) : probability(horizon.accuracy)} />
        <Metric label="MACRO F1" value={horizon?.macro_f1 == null ? dash : number(horizon.macro_f1, 4)} />
        <Metric label="BRIER" value={horizon?.mean_brier_score == null ? dash : number(horizon.mean_brier_score, 4)} />
        <Metric label="LOG LOSS" value={horizon?.mean_log_loss == null ? dash : number(horizon.mean_log_loss, 4)} />
      </div>
      {empty ? <div className="shadow-empty"><ShieldCheck size={22} /><b>NO LIVE SHADOW OBSERVATIONS YET</b><span>{label(summary?.recorder?.last_cycle_reason ?? "WAITING_FOR_FRESH_MARKET_DECISION")}</span></div> : latest && <div className="latest-shadow">
        <div><span>LATEST PREDICTION</span><b>{latest.predicted_class ?? dash}</b><small>{localTime(latest.decision_timestamp_utc, true)} LKT · {latest.horizon_minutes}m</small></div>
        <Metric label="CONFIDENCE" value={probability(latest.model_confidence)} />
        <Metric label="V0.6 DECISION" value={label(latest.v06c_decision)} />
        <Metric label="EVALUATION DUE" value={localTime(latest.evaluation_due_utc, true)} />
        <Metric label="ACTUAL" value={latest.actual_class ?? dash} />
        <Metric label="RETURN" value={latest.actual_return == null ? dash : `${(latest.actual_return * 100).toFixed(3)}%`} />
        <Metric label="RESULT" value={latest.direction_correct == null ? label(latest.evaluation_status) : latest.direction_correct ? "CORRECT" : "INCORRECT"} tone={latest.direction_correct == null ? "warn" : latest.direction_correct ? "good" : "bad"} />
      </div>}
      <h3 className="section-label">RECENT OBSERVATIONS</h3>
      {recent?.items?.length ? <div className="shadow-table-wrap"><table className="shadow-table"><thead><tr><th>Time</th><th>Horizon</th><th>Prediction</th><th>Confidence</th><th>V0.6 Decision</th><th>Gate</th><th>Actual</th><th>Return</th><th>Result</th><th>Status</th></tr></thead><tbody>{recent.items.map(item => <tr key={item.observation_id}><td>{localTime(item.decision_timestamp_utc)}</td><td>{item.horizon_minutes}m</td><td>{item.predicted_class ?? "N/A"}</td><td>{probability(item.model_confidence)}</td><td>{label(item.v06c_decision)}</td><td title={item.blocking_reasons?.join(", ")}>{label(item.v06c_gate)}</td><td>{item.actual_class ?? "N/A"}</td><td>{item.actual_return == null ? "N/A" : `${(item.actual_return * 100).toFixed(3)}%`}</td><td>{item.direction_correct == null ? "N/A" : item.direction_correct ? "CORRECT" : "INCORRECT"}</td><td>{label(item.evaluation_status)}</td></tr>)}</tbody></table></div> : <div className="empty-state">NO RECENT SHADOW OBSERVATIONS</div>}
    </Panel>
  </section>;
}
