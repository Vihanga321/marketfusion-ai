import { useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clipboard,
  Clock3,
  Database,
  FlaskConical,
  Gauge,
  HeartPulse,
  LayoutDashboard,
  LineChart,
  ListChecks,
  LockKeyhole,
  Network,
  RefreshCw,
  ScrollText,
  Server,
  Settings2,
  ShieldCheck,
  Sparkles,
  Wifi,
} from "lucide-react";
import { CandleChart } from "./CandleChart";
import { LiveShadowValidation } from "./LiveShadowValidation";
import { PredictionCard } from "./PredictionCard";
import type { AssetId } from "../api/client";
import type {
  EngineStatus,
  HorizonPrediction,
  MarketFusionState,
  OperatorStatus,
  ShadowRecent,
  ShadowSummary,
  Timeframe,
  V08Status,
} from "../types/marketfusion";
import type { ForwardContractStatus, ForwardValidationStatus } from "../types/forwardValidation";

const tabs = [
  ["overview", "Overview", LayoutDashboard],
  ["market", "Market", LineChart],
  ["models", "Models", Gauge],
  ["engines", "Engines", Sparkles],
  ["validation", "Validation", ListChecks],
  ["intermarket", "Intermarket", Network],
  ["research", "Research", FlaskConical],
  ["health", "Health", HeartPulse],
  ["logs", "Logs", ScrollText],
  ["system", "System", Settings2],
] as const;
type Tab = (typeof tabs)[number][0];

export type FullDashboardProps = {
  selectedAsset: AssetId;
  onAssetChange: (value: AssetId) => void;
  state: MarketFusionState | null;
  connection: string;
  lastSuccess: Date | null;
  operator: OperatorStatus | null;
  quote: { status?: string; bid?: number | null; ask?: number | null; mid?: number | null; spread_points?: number | null } | null;
  intelligence: any;
  research: any;
  events: any;
  monitoring: V08Status | null;
  shadowSummary: ShadowSummary | null;
  shadowRecent: ShadowRecent | null;
  engines: EngineStatus | null;
  candles: any[];
  candleStatus: string;
  timeframe: Timeframe;
  setTimeframe: (value: Timeframe) => void;
  now: number;
  refresh: () => void;
  forwardValidation: ForwardValidationStatus | null;
  forwardConnection: string;
};

type PageProps = FullDashboardProps & { setTab?: (tab: Tab) => void };

const human = (value?: string | null) => String(value ?? "UNKNOWN").replaceAll("_", " ");
const pct = (value?: number | null, digits = 1) => value == null || Number.isNaN(value) ? "--" : `${(value * 100).toFixed(digits)}%`;
const num = (value?: number | null, digits = 4) => value == null || Number.isNaN(value) ? "--" : value.toFixed(digits);
const time = (value?: string | null) => {
  if (!value) return "--";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "--" : new Intl.DateTimeFormat("en-LK", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(parsed);
};
const dateTime = (value?: string | null) => {
  if (!value) return "--";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "--" : new Intl.DateTimeFormat("en-LK", { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(parsed);
};

function statusTone(value?: string | null) {
  const text = String(value ?? "").toUpperCase();
  if (/(FAIL|ERROR|INVALID|STOPPED|DISCONNECTED)/.test(text)) return "bad";
  if (/(WAIT|STALE|DEGRADED|UNAVAILABLE|BLOCKED|INSUFFICIENT|CLOSED)/.test(text)) return "warn";
  if (/(PASS|LIVE|FRESH|RUNNING|OPEN|CONNECTED|READY|OK|APPENDED)/.test(text)) return "good";
  return "info";
}

function StatusPill({ label, value }: { label: string; value?: string | null }) {
  return <span className={`mf-status mf-status-${statusTone(value)}`}><i className="mf-status-dot" /><span>{label}</span><b>{human(value)}</b></span>;
}

function Card({ title, eyebrow, children, className = "", action }: { title: string; eyebrow?: string; children: ReactNode; className?: string; action?: ReactNode }) {
  return <section className={`mf-card ${className}`}><div className="mf-card-head"><div>{eyebrow && <span className="mf-eyebrow">{eyebrow}</span>}<h3>{title}</h3></div>{action}</div>{children}</section>;
}

function Metric({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return <div className="mf-metric"><span>{label}</span><strong>{value}</strong>{sub != null && <small>{sub}</small>}</div>;
}

function Progress({ label, value, target }: { label: string; value: number; target: number }) {
  const amount = target > 0 ? Math.max(0, Math.min(100, value / target * 100)) : 0;
  return <div className="mf-progress-row"><div><span>{label}</span><b>{value} / {target}</b></div><div className="mf-progress-track"><i style={{ width: `${amount}%` }} /></div><small>{amount.toFixed(1)}%</small></div>;
}

function ForwardContractCard({ id, contract }: { id: string; contract: ForwardContractStatus }) {
  return <Card title={id} eyebrow={contract.minimum_sample_ready ? "MINIMUM SAMPLE READY" : "FROZEN FORWARD CONTRACT"} className="mf-contract-card">
    <div className="mf-contract-top"><StatusPill label="STATE" value={contract.decision} /><span className="mf-lock"><LockKeyhole size={14} />Models frozen</span></div>
    <div className="mf-contract-progress">
      <Progress label="Trading days" value={contract.active_trading_days ?? 0} target={20} />
      <Progress label="Matured" value={contract.matured_observations ?? 0} target={500} />
      <Progress label="Directional" value={contract.directional_outcomes ?? 0} target={400} />
    </div>
    <div className="mf-metrics-grid mf-metrics-grid-4">
      <Metric label="Context BA" value={pct(contract.context?.balanced_accuracy)} />
      <Metric label="Baseline BA" value={pct(contract.paired_baseline?.balanced_accuracy)} />
      <Metric label="BA delta" value={pct(contract.balanced_accuracy_delta)} />
      <Metric label="Log-loss Δ" value={num(contract.paired_log_loss_improvement, 5)} />
    </div>
    <div className="mf-contract-note"><span>Gate evaluation</span><b>{contract.minimum_sample_ready ? contract.failed_gates?.length ? contract.failed_gates.join(" · ") : "No failed gates reported" : "Waiting for minimum sample"}</b></div>
  </Card>;
}

function ForwardOverview({ status, connection }: { status: ForwardValidationStatus | null; connection: string }) {
  if (!status) return <Card title="V1.0C Forward Validation" eyebrow="READ ONLY"><div className="mf-empty-state"><Database size={27} /><div><b>Forward status {connection === "LOADING" ? "loading" : "unavailable"}</b><span>The console does not fabricate progress when the backend status file is unavailable.</span></div></div></Card>;
  const contracts = Object.values(status.contracts ?? {});
  const minDays = contracts.length ? Math.min(...contracts.map((c) => c.active_trading_days ?? 0)) : 0;
  const minMatured = contracts.length ? Math.min(...contracts.map((c) => c.matured_observations ?? 0)) : 0;
  const minDirectional = contracts.length ? Math.min(...contracts.map((c) => c.directional_outcomes ?? 0)) : 0;
  const overall = Math.min(minDays / 20, minMatured / 500, minDirectional / 400, 1) * 100;
  return <Card title="V1.0C Forward Validation" eyebrow="FROZEN POINT-IN-TIME EXPERIMENT" className="mf-forward-overview">
    <div className="mf-forward-head"><div><StatusPill label="DECISION" value={status.decision} /><span>Started {dateTime(status.forward_start_utc)}</span></div><strong>{overall.toFixed(1)}%</strong></div>
    <div className="mf-progress-track mf-progress-large"><i style={{ width: `${overall}%` }} /></div>
    <div className="mf-metrics-grid mf-metrics-grid-4"><Metric label="Active days" value={`${minDays}/20`} /><Metric label="Matured floor" value={`${minMatured}/500`} /><Metric label="Directional floor" value={`${minDirectional}/400`} /><Metric label="Contracts" value={contracts.length} /></div>
    <div className="mf-safety-line"><ShieldCheck size={15} /><span>Historical tuning stopped · retraining disabled · promotion disabled during validation</span></div>
  </Card>;
}

function OverviewPage(props: PageProps) {
  const [why, setWhy] = useState(false);
  const priceDigits = props.selectedAsset === "XAUUSD" ? 2 : 5;
  const mainEngines = Object.values(props.engines?.engines ?? {});
  const externalEngines = Object.values(props.engines?.external_engines ?? {});
  const decision = props.connection === "LIVE" ? props.state?.decision.action ?? "WAIT" : "WAIT";
  return <div className="mf-page-stack">
    <div className="mf-hero-grid">
      <Card title="Live Market" eyebrow={`${props.selectedAsset === "XAUUSD" ? "GOLD / U.S. DOLLAR" : "EURO / U.S. DOLLAR"} · DISPLAY PATH`} className="mf-price-card">
        <div className="mf-price-row"><div><span>{props.selectedAsset === "XAUUSD" ? "XAU/USD" : "EUR/USD"}</span><strong>{props.quote?.mid == null ? "LIVE FEED UNAVAILABLE" : props.quote.mid.toFixed(priceDigits)}</strong></div><StatusPill label="QUOTE" value={props.quote?.bid != null ? "LIVE" : "UNAVAILABLE"} /></div>
        <div className="mf-metrics-grid mf-metrics-grid-4"><Metric label="BID" value={props.quote?.bid == null ? "--" : props.quote.bid.toFixed(priceDigits)} /><Metric label="ASK" value={props.quote?.ask == null ? "--" : props.quote.ask.toFixed(priceDigits)} /><Metric label="SPREAD" value={props.quote?.spread_points == null ? "--" : `${props.quote.spread_points.toFixed(1)} pts`} /><Metric label="SESSION" value={props.operator?.session.current_session ?? "UNKNOWN"} /></div>
        <div className="mf-mini-chart"><CandleChart candles={props.candles.slice(-90)} /></div>
      </Card>
      <Card title="Current Advisory" eyebrow="V0.6C · READ ONLY" className="mf-advisory-card">
        <div className="mf-advisory-decision"><span>DECISION</span><strong>{human(decision)}</strong><p>{props.state?.decision.action_meaning ?? "No directional advice is available; the system remains fail-closed."}</p></div>
        <div className="mf-metrics-grid mf-metrics-grid-2"><Metric label="CONFIDENCE" value={human(props.state?.decision.confidence)} /><Metric label="GATE" value={human(props.state?.decision.gate)} /><Metric label="MARKET" value={props.operator?.market.market_open ? "OPEN" : "CLOSED"} /><Metric label="MODE" value="SHADOW ADVISORY" /></div>
        <button className="mf-text-button" onClick={() => setWhy((value) => !value)}>{why ? "HIDE REASONS" : "WHY WAIT?"}</button>
        {why && <div className="mf-reasons">{props.state?.reasons?.length ? props.state.reasons.map((reason, index) => <div key={`${reason.code}-${index}`}><b>{index + 1}</b><span>{reason.message}<small>{human(reason.code)}</small></span></div>) : <span>No backend reason is available.</span>}</div>}
        <div className="mf-safety-line"><CheckCircle2 size={15} />Automatic execution disabled · manual confirmation required</div>
      </Card>
    </div>
    <div className="mf-kpi-strip">
      <Metric label="MT5" value={props.quote?.status?.startsWith("PASS") ? "CONNECTED" : "UNAVAILABLE"} sub="broker feed" />
      <Metric label="FEATURES" value={props.operator?.data_freshness.status ?? "UNKNOWN"} sub={time(props.operator?.data_freshness.feature_row_utc)} />
      <Metric label="ENGINES" value={mainEngines.length + externalEngines.length} sub="observational" />
      <Metric label="V0.8 OBS" value={props.shadowSummary?.performance_observations ?? 0} sub="evaluated" />
      <Metric label="SYSTEM" value={props.operator?.system_health.status ?? "UNKNOWN"} sub="read only" />
    </div>
    <ForwardOverview status={props.forwardValidation} connection={props.forwardConnection} />
    <div className="mf-three-grid">
      <Card title="Model Registry" eyebrow="APPROVAL STATE"><div className="mf-list">{[15, 60, 240].map((h) => <div key={h}><span>{h} MIN</span><b>{human(props.state?.predictions.horizons[String(h)]?.model_status)}</b></div>)}</div><button className="mf-text-button" onClick={() => props.setTab?.("models")}>VIEW MODEL DETAILS</button></Card>
      <Card title="Intermarket Context" eyebrow="RESEARCH SENSORS"><div className="mf-list">{externalEngines.length ? externalEngines.slice(0, 5).map((engine) => <div key={engine.engine_name}><span>{human(engine.engine_name)}</span><b>{human(engine.status)}</b></div>) : <div><span>External engine feed</span><b>NO DATA</b></div>}</div><button className="mf-text-button" onClick={() => props.setTab?.("intermarket")}>VIEW INTERMARKET</button></Card>
      <Card title="Runtime Health" eyebrow="24/5 SERVER"><div className="mf-list"><div><span>API</span><b>{props.connection}</b></div><div><span>Market</span><b>{human(props.operator?.market.status)}</b></div><div><span>Freshness</span><b>{human(props.operator?.data_freshness.status)}</b></div><div><span>Forward status</span><b>{props.forwardConnection}</b></div></div><button className="mf-text-button" onClick={() => props.setTab?.("health")}>VIEW HEALTH</button></Card>
    </div>
  </div>;
}

function MarketPage(props: PageProps) {
  const digits = props.selectedAsset === "XAUUSD" ? 2 : 5;
  return <div className="mf-page-stack">
    <div className="mf-section-title"><div><span>COMPLETED-CANDLE MARKET VIEW</span><h2>Live Market</h2></div><p>Live quotes are display-only. Model inputs remain on completed causal candles.</p></div>
    <Card title={`${props.selectedAsset === "XAUUSD" ? "XAU/USD" : "EUR/USD"} Chart`} eyebrow="CAUSAL CANDLES" action={<div className="mf-timeframes">{(["M1", "M5", "M15", "H1"] as Timeframe[]).map((item) => <button key={item} className={props.timeframe === item ? "active" : ""} onClick={() => props.setTimeframe(item)}>{item}</button>)}</div>}>
      <div className="mf-market-headline"><strong>{props.quote?.mid == null ? "--" : props.quote.mid.toFixed(digits)}</strong><StatusPill label="CANDLES" value={props.candleStatus} /></div><CandleChart candles={props.candles} />
      <div className="mf-metrics-grid mf-metrics-grid-4"><Metric label="Last completed M5" value={time(props.operator?.data_freshness.latest_completed_m5_utc)} /><Metric label="Last completed M15" value={time(props.operator?.data_freshness.latest_completed_m15_utc)} /><Metric label="Feature row" value={time(props.operator?.data_freshness.feature_row_utc)} /><Metric label="Freshness" value={human(props.operator?.data_freshness.status)} /></div>
    </Card>
    <div className="mf-two-grid"><Card title="Quote Detail" eyebrow="LIVE UI PATH"><div className="mf-list"><div><span>Bid</span><b>{props.quote?.bid == null ? "--" : props.quote.bid.toFixed(digits)}</b></div><div><span>Ask</span><b>{props.quote?.ask == null ? "--" : props.quote.ask.toFixed(digits)}</b></div><div><span>Spread points</span><b>{props.quote?.spread_points == null ? "--" : props.quote.spread_points.toFixed(2)}</b></div><div><span>Latest tick</span><b>{time(props.operator?.data_freshness.latest_market_tick_utc)}</b></div></div></Card><Card title="Session & Market" eyebrow="OPERATOR CONTEXT"><div className="mf-list"><div><span>Market</span><b>{human(props.operator?.market.status)}</b></div><div><span>Session</span><b>{props.operator?.session.current_session ?? "UNKNOWN"}</b></div><div><span>Overlap</span><b>{props.operator?.session.overlap ? "YES" : "NO"}</b></div><div><span>Next transition</span><b>{props.operator?.session.next_transition ?? "--"}</b></div></div></Card></div>
  </div>;
}

function ModelsPage(props: PageProps) {
  const [details, setDetails] = useState<number | null>(null);
  return <div className="mf-page-stack">
    <div className="mf-section-title"><div><span>MODEL REGISTRY</span><h2>Model Status</h2></div><p>Model cards are read-only. Missing or unapproved horizons stay unavailable.</p></div>
    <div className="mf-model-grid">{[15, 60, 240].map((horizon) => { const item = props.state?.predictions.horizons[String(horizon)] as HorizonPrediction | undefined; return <div key={horizon} className="mf-model-wrap"><PredictionCard label={`${horizon} MIN`} prediction={item} predictionTime={props.state?.decision.decision_time.utc} featureTime={props.operator?.data_freshness.feature_row_utc} now={props.now} /><button className="mf-text-button" onClick={() => setDetails(details === horizon ? null : horizon)}>{details === horizon ? "HIDE DETAILS" : "DETAILS"}</button>{details === horizon && <div className="mf-detail"><span>Model ID <b>{item?.model_id ?? "N/A"}</b></span><span>Status <b>{human(item?.model_status)}</b></span><span>Gate <b>{human(item?.decision_gate)}</b></span></div>}</div>; })}</div>
    {!!Object.keys(props.forwardValidation?.contracts ?? {}).length && <div className="mf-two-grid">{Object.entries(props.forwardValidation?.contracts ?? {}).map(([id, contract]) => <ForwardContractCard key={id} id={id} contract={contract} />)}</div>}
    <Card title="Promotion Boundary" eyebrow="SAFETY"><div className="mf-callout"><ShieldCheck size={22} /><div><b>No automatic promotion</b><span>A forward pass only justifies controlled later research. Production integration remains disabled.</span></div></div></Card>
  </div>;
}

function EnginesPage(props: PageProps) {
  const all = [...Object.values(props.engines?.engines ?? {}), ...Object.values(props.engines?.external_engines ?? {})];
  return <div className="mf-page-stack"><div className="mf-section-title"><div><span>OBSERVATIONAL ENGINE LAYER</span><h2>Engine Intelligence</h2></div><p>Scores are research context, not calibrated trading probabilities.</p></div><div className="mf-engine-grid">{all.length ? all.map((engine) => <Card key={engine.engine_name} title={human(engine.engine_name)} eyebrow="RESEARCH ENGINE"><div className="mf-engine-score"><strong>{engine.direction_score == null ? "--" : `${engine.direction_score >= 0 ? "+" : ""}${engine.direction_score.toFixed(2)}`}</strong><StatusPill label="STATE" value={engine.status} /></div><div className="mf-metrics-grid mf-metrics-grid-2"><Metric label="Confidence" value={pct(engine.confidence)} /><Metric label="Features" value={engine.feature_count ?? 0} /><Metric label="Regime" value={human(engine.regime)} /><Metric label="Freshness" value={human(engine.input_freshness)} /></div><small className="mf-muted">{engine.reason_codes?.join(" · ") || "No engine reason code"}</small></Card>) : <Card title="Engine feed"><div className="mf-empty-state"><Sparkles size={26} /><span>No engine output is currently available.</span></div></Card>}</div></div>;
}

function ValidationPage(props: PageProps) {
  return <div className="mf-page-stack">
    <div className="mf-section-title"><div><span>POINT-IN-TIME EVIDENCE</span><h2>Forward Validation</h2></div><p>No historical predictions are reconstructed and no profitability claim is made.</p></div>
    <ForwardOverview status={props.forwardValidation} connection={props.forwardConnection} />
    {!!Object.keys(props.forwardValidation?.contracts ?? {}).length && <div className="mf-two-grid">{Object.entries(props.forwardValidation?.contracts ?? {}).map(([id, contract]) => <ForwardContractCard key={id} id={id} contract={contract} />)}</div>}
    <Card title="V0.8 Shadow Ledger" eyebrow="OBSERVATIONAL MONITOR"><LiveShadowValidation summary={props.shadowSummary} recent={props.shadowRecent} /></Card>
    <Card title="Frozen Gates" eyebrow="V1.0C MINIMUMS"><div className="mf-gate-grid">{["20 active trading days", "500 matured observations", "400 directional outcomes", "4 chronological stability blocks", "Balanced accuracy ≥ 0.515", "Recent block BA ≥ 0.510", "≥ 3/4 blocks beat baseline", "BA delta ≥ 0.003", "Paired log-loss improvement ≥ 0.002", "Context log loss beats class prior", "Cost-aware non-worse than baseline"].map((item) => <div key={item}><CheckCircle2 size={14} /><span>{item}</span></div>)}</div></Card>
  </div>;
}

function IntermarketPage(props: PageProps) {
  const engines = Object.values(props.engines?.external_engines ?? {});
  const sensors = props.selectedAsset === "XAUUSD" ? ["XAGUSD / Silver", "USDJPY", "EURUSD", "US500", "USTEC"] : ["EURUSD primary market"];
  return <div className="mf-page-stack"><div className="mf-section-title"><div><span>CAUSAL CONTEXT SENSORS</span><h2>Intermarket</h2></div><p>Context is synchronized to completed decision rows. No forming candle or future information is allowed.</p></div><div className="mf-sensor-grid">{sensors.map((sensor) => <Card key={sensor} title={sensor} eyebrow="FROZEN SENSOR"><StatusPill label="CONFIG" value="READY" /><small className="mf-muted">Identity frozen for research; live availability remains broker-dependent.</small></Card>)}</div><Card title="External Engine Outputs" eyebrow="LIVE CONTEXT"><div className="mf-table"><div className="mf-table-head"><span>Engine</span><span>Status</span><span>Score</span><span>Confidence</span><span>Freshness</span></div>{engines.length ? engines.map((engine) => <div key={engine.engine_name}><span>{human(engine.engine_name)}</span><span>{human(engine.status)}</span><span>{engine.direction_score == null ? "--" : engine.direction_score.toFixed(3)}</span><span>{pct(engine.confidence)}</span><span>{human(engine.input_freshness)}</span></div>) : <div><span>No external engine rows</span><span>--</span><span>--</span><span>--</span><span>--</span></div>}</div></Card></div>;
}

function ResearchPage(props: PageProps) {
  const milestones = [
    ["V1.0A", "XAUUSD infrastructure", "PASS"], ["V1.0B", "First model research", "0/3 APPROVED"],
    ["V1.0B.1", "Target / horizon research", "NO TARGET CANDIDATE"], ["V1.0B.2", "Price feature evidence", "WEAK"],
    ["V1.0B.3", "Fed macro context", "WEAK"], ["V1.0B.4", "Intraday intermarket audit", "READY FOR RESEARCH"],
    ["V1.0B.5", "Intraday context evidence", "WEAK"], ["V1.0B.6", "Temporal calibration", "WEAK"],
    ["V1.0B.7", "Cross-market relationships", "WEAK · STOP TUNING"], ["V1.0C", "Frozen forward validation", props.forwardValidation?.decision ?? "NOT ACTIVE"],
  ];
  return <div className="mf-page-stack"><div className="mf-section-title"><div><span>RESEARCH HISTORY</span><h2>Research Lab</h2></div><p>Historical results remain visible for auditability. This page cannot start training or modify parameters.</p></div><Card title="Current Research State" eyebrow="BACKEND"><div className="mf-metrics-grid mf-metrics-grid-4"><Metric label="Status" value={human(props.research?.status)} /><Metric label="Approved champions" value={props.research?.approved_champions ?? 0} /><Metric label="History rows" value={props.research?.market_core_rows ?? "--"} /><Metric label="Next recommendation" value={human(props.research?.next_recommended_training)} /></div></Card><Card title="XAUUSD Research Timeline" eyebrow="V1.0 SERIES"><div className="mf-timeline">{milestones.map(([phase, name, result]) => <div key={phase}><b>{phase}</b><span>{name}</span><strong>{human(result)}</strong></div>)}</div></Card><Card title="Research Boundary" eyebrow="FROZEN WINDOW"><div className="mf-callout"><LockKeyhole size={22} /><div><b>Historical tuning stopped</b><span>No retraining, recalibration, feature search, threshold changes or parameter fishing during forward validation.</span></div></div></Card></div>;
}

function HealthPage(props: PageProps) {
  const sources = Object.entries(props.state?.health.sources ?? {});
  const providers = Object.entries(props.monitoring?.provider_health?.providers ?? {});
  return <div className="mf-page-stack"><div className="mf-section-title"><div><span>24/5 OPERATIONAL READINESS</span><h2>System Health</h2></div><p>Missing telemetry is shown as unavailable instead of being guessed.</p></div><div className="mf-health-grid"><Card title="MT5 / Market Feed" eyebrow="BROKER"><StatusPill label="MT5" value={props.quote?.status?.startsWith("PASS") ? "CONNECTED" : "UNAVAILABLE"} /><div className="mf-list"><div><span>Quote freshness</span><b>{human(props.operator?.data_freshness.status)}</b></div><div><span>Last tick</span><b>{time(props.operator?.data_freshness.latest_market_tick_utc)}</b></div><div><span>Last M5</span><b>{time(props.operator?.data_freshness.latest_completed_m5_utc)}</b></div></div></Card><Card title="API / Runtime" eyebrow="LOCALHOST"><StatusPill label="API" value={props.connection} /><div className="mf-list"><div><span>System health</span><b>{human(props.operator?.system_health.status)}</b></div><div><span>Mode</span><b>SHADOW ADVISORY ONLY</b></div><div><span>Last state</span><b>{props.lastSuccess ? props.lastSuccess.toLocaleTimeString() : "--"}</b></div></div></Card><Card title="Forward Collector" eyebrow="V1.0C"><StatusPill label="FORWARD" value={props.forwardValidation?.decision ?? props.forwardConnection} /><div className="mf-list"><div><span>Updated</span><b>{time(props.forwardValidation?.updated_at_utc)}</b></div><div><span>Models frozen</span><b>{props.forwardValidation ? props.forwardValidation.models_frozen === false ? "NO" : "YES" : "--"}</b></div><div><span>Retraining</span><b>{props.forwardValidation ? props.forwardValidation.retraining_during_forward_window ? "UNEXPECTED" : "DISABLED" : "--"}</b></div></div></Card></div><div className="mf-two-grid"><Card title="Runtime Sources" eyebrow="STATE HEALTH"><div className="mf-list">{sources.length ? sources.map(([key, value]) => <div key={key}><span>{human(key)}</span><b>{human(String(value))}</b></div>) : <div><span>Source telemetry</span><b>UNAVAILABLE</b></div>}</div></Card><Card title="Provider Health" eyebrow="V0.8"><div className="mf-list">{providers.length ? providers.map(([key, value]) => <div key={key}><span>{human(key)}</span><b>{human(String(value))}</b></div>) : <div><span>Provider telemetry</span><b>UNAVAILABLE</b></div>}</div></Card></div><Card title="Machine Telemetry" eyebrow="DEDICATED PC"><div className="mf-empty-state"><Server size={27} /><div><b>CPU / RAM / disk endpoint not exposed yet</b><span>The UI intentionally leaves machine values unavailable until a read-only backend source exists.</span></div></div></Card></div>;
}

function LogsPage(props: PageProps) {
  const reasons = props.state?.reasons ?? [];
  const observations = props.shadowRecent?.items ?? [];
  return <div className="mf-page-stack"><div className="mf-section-title"><div><span>READ-ONLY AUDIT VIEW</span><h2>Operational Logs</h2></div><p>Only real backend reason codes and shadow observations are shown here.</p></div><div className="mf-two-grid"><Card title="Current Reason Codes" eyebrow="V0.6 GATES"><div className="mf-log-list">{reasons.length ? reasons.map((reason, index) => <div key={`${reason.code}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><b>{human(reason.code)}</b><p>{reason.message}</p></div>) : <div><span>--</span><b>NO ACTIVE REASONS</b><p>No reason rows are currently available.</p></div>}</div></Card><Card title="Recent Shadow Observations" eyebrow="V0.8"><div className="mf-log-list">{observations.length ? observations.slice(0, 12).map((item) => <div key={item.observation_id}><span>{time(item.decision_timestamp_utc)}</span><b>{human(item.evaluation_status)}</b><p>{item.horizon_minutes}m · {human(item.predicted_class)} → {human(item.actual_class)}</p></div>) : <div><span>--</span><b>NO OBSERVATIONS</b><p>No recent shadow observations are available.</p></div>}</div></Card></div></div>;
}

function SystemPage(props: PageProps) {
  return <div className="mf-page-stack"><div className="mf-section-title"><div><span>LOCAL RESEARCH SERVER</span><h2>System Diagnostics</h2></div><p>The dashboard has no trade execution controls and remains fail-closed.</p></div><div className="mf-system-grid"><Card title="Market Data" eyebrow="V0.5A"><StatusPill label="STATUS" value={props.state?.health.sources.v05a_market ?? props.operator?.data_freshness.status} /><small className="mf-muted">Live display + completed-candle causal features.</small></Card><Card title="Intelligence" eyebrow="V0.5B"><StatusPill label="STATUS" value={props.state?.health.sources.v05b_intelligence ?? props.intelligence?.status} /><small className="mf-muted">Research context and provider telemetry.</small></Card><Card title="Decision Runtime" eyebrow="V0.6"><StatusPill label="STATUS" value={props.state?.system.status} /><small className="mf-muted">Gate: {human(props.state?.decision.gate)}</small></Card><Card title="Dashboard API" eyebrow="V0.7"><StatusPill label="STATUS" value={props.connection} /><small className="mf-muted">Localhost-only read surface.</small></Card><Card title="Monitoring" eyebrow="V0.8"><StatusPill label="STATUS" value={props.monitoring?.status} /><small className="mf-muted">Shadow observation monitor.</small></Card><Card title="Forward Experiment" eyebrow="V1.0C"><StatusPill label="STATUS" value={props.forwardValidation?.decision ?? props.forwardConnection} /><small className="mf-muted">Frozen point-in-time validation.</small></Card></div><Card title="Safety Center" eyebrow="NON-NEGOTIABLE"><div className="mf-safety-grid"><div><ShieldCheck size={20} /><span>Automatic execution</span><b>DISABLED</b></div><div><LockKeyhole size={20} /><span>Runtime</span><b>SHADOW ADVISORY ONLY</b></div><div><CheckCircle2 size={20} /><span>Manual confirmation</span><b>REQUIRED</b></div><div><Database size={20} /><span>Forward backfill</span><b>PROHIBITED</b></div><div><BrainCircuit size={20} /><span>Retraining in window</span><b>DISABLED</b></div><div><Server size={20} /><span>Production integration</span><b>NOT ENABLED</b></div></div></Card></div>;
}

export function FullDashboardShell(props: FullDashboardProps) {
  const [tab, setTab] = useState<Tab>("overview");
  const [copied, setCopied] = useState(false);
  const clock = useMemo(() => new Date(props.now).toLocaleTimeString("en-LK", { hour12: false }), [props.now]);
  const assetClass = props.selectedAsset === "XAUUSD" ? "PRECIOUS METAL" : "FOREX";
  const display = props.selectedAsset === "XAUUSD" ? "XAU/USD" : "EUR/USD";

  const copyDiagnostics = async () => {
    const lines = ["MarketFusion diagnostics", `Asset ${props.selectedAsset}`, `API ${props.connection}`, `MT5 ${props.quote?.status?.startsWith("PASS") ? "CONNECTED" : "UNAVAILABLE"}`, `Market ${props.operator?.market.status ?? "UNKNOWN"}`, `Freshness ${props.operator?.data_freshness.status ?? "UNKNOWN"}`, `Decision ${props.state?.decision.action ?? "WAIT"}`, `Gate ${props.state?.decision.gate ?? "UNKNOWN"}`, `Forward ${props.forwardValidation?.decision ?? props.forwardConnection}`, "Automatic execution DISABLED", "Manual confirmation REQUIRED"];
    await navigator.clipboard?.writeText(lines.join("\n"));
    setCopied(true); window.setTimeout(() => setCopied(false), 1400);
  };

  const shared = { ...props, setTab };
  const content = tab === "market" ? <MarketPage {...shared} /> : tab === "models" ? <ModelsPage {...shared} /> : tab === "engines" ? <EnginesPage {...shared} /> : tab === "validation" ? <ValidationPage {...shared} /> : tab === "intermarket" ? <IntermarketPage {...shared} /> : tab === "research" ? <ResearchPage {...shared} /> : tab === "health" ? <HealthPage {...shared} /> : tab === "logs" ? <LogsPage {...shared} /> : tab === "system" ? <SystemPage {...shared} /> : <OverviewPage {...shared} />;

  return <main className="mf-shell">
    <header className="mf-topbar"><div className="mf-brand"><div className="mf-brand-mark"><BrainCircuit size={23} /></div><div><h1>MARKETFUSION <span>AI</span></h1><p>XAUUSD research · forward validation · system operations</p></div></div><div className="mf-top-status"><StatusPill label="MT5" value={props.quote?.status?.startsWith("PASS") ? "CONNECTED" : "UNAVAILABLE"} /><StatusPill label="API" value={props.connection} /><StatusPill label="MARKET" value={props.operator?.market.market_open ? "OPEN" : "CLOSED"} /><StatusPill label="FORWARD" value={props.forwardValidation?.decision ?? props.forwardConnection} /></div><div className="mf-clock"><Clock3 size={15} /><b>{clock}</b><span>LKT</span></div></header>
    <aside className="mf-sidebar"><div className="mf-asset-box"><span>ACTIVE ASSET</span><select aria-label="Asset" value={props.selectedAsset} onChange={(event) => props.onAssetChange(event.target.value as AssetId)}><option value="XAUUSD">XAU/USD · Gold</option><option value="EURUSD">EUR/USD</option></select><small>{assetClass}</small></div><nav aria-label="Dashboard sections">{tabs.map(([id, title, Icon]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}><Icon size={16} /><span>{title}</span></button>)}</nav><div className="mf-sidebar-safety"><ShieldCheck size={18} /><div><b>SHADOW / MANUAL</b><span>Execution disabled</span></div></div></aside>
    <section className="mf-main"><div className="mf-command-row"><div><span>{display}</span><b>{props.selectedAsset} RESEARCH</b></div><div className="mf-command-actions"><button onClick={props.refresh}><RefreshCw size={15} /><span>REFRESH</span></button><button onClick={copyDiagnostics}><Clipboard size={15} /><span>{copied ? "COPIED" : "DIAGNOSTICS"}</span></button></div></div>{props.connection !== "LIVE" && <div className="mf-banner"><Wifi size={16} /><b>STATE {props.connection}</b><span>Directional advice remains suppressed and fail-closed.</span></div>}{content}</section>
    <footer className="mf-footer"><span><Activity size={13} />Read-only research terminal</span><span>Automatic execution: <b>DISABLED</b></span><span>Runtime: <b>SHADOW_ADVISORY_ONLY</b></span><span>Manual confirmation: <b>REQUIRED</b></span><BarChart3 size={14} /></footer>
  </main>;
}
