import { useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clipboard,
  Clock3,
  Gauge,
  LayoutDashboard,
  LineChart,
  ListChecks,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sparkles,
  Wifi,
} from "lucide-react";
import { CandleChart } from "./CandleChart";
import { LiveShadowValidation } from "./LiveShadowValidation";
import { MarketOpenCountdown, useServerClock } from "./MarketOpenCountdown";
import { Metric, Panel, StatusDot } from "./Panel";
import { PredictionCard } from "./PredictionCard";
import { OperatorPanels } from "./OperatorPanels";
import type { AssetId } from "../api/client";
import type {
  AdvisoryAction,
  EngineOutput,
  EngineStatus,
  HorizonPrediction,
  MarketFusionState,
  OperatorStatus,
  ShadowRecent,
  ShadowSummary,
  Timeframe,
  V08Status,
} from "../types/marketfusion";
import {
  ageFrom,
  countdown,
  dash,
  label,
  localTime,
  number,
  probability,
  until,
  utcTime,
} from "../utils/format";

const tabs = [
  ["overview", "Overview", LayoutDashboard],
  ["market", "Market", LineChart],
  ["models", "Models", Gauge],
  ["engines", "Engines", Sparkles],
  ["validation", "Validation", ListChecks],
  ["system", "System", Settings2],
] as const;
type Tab = (typeof tabs)[number][0];

type Props = {
  selectedAsset: AssetId;
  onAssetChange: (value: AssetId) => void;
  state: MarketFusionState | null;
  connection: string;
  lastSuccess: Date | null;
  operator: OperatorStatus | null;
  quote: {
    status?: string;
    bid?: number | null;
    ask?: number | null;
    mid?: number | null;
    spread_points?: number | null;
  } | null;
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
};

function statusTone(value?: string): string | undefined {
  return value && /(FAIL|INVALID|STOPPED|DISCONNECTED)/.test(value)
    ? "bad"
    : value && /(WAIT|STALE|DEGRADED|UNAVAILABLE|BLOCKED)/.test(value)
      ? "warn"
      : value && /(PASS|LIVE|FRESH|RUNNING|OPEN|CONNECTED|OK)/.test(value)
        ? "good"
        : "info";
}
function reasonLabel(code?: string | null): string {
  return code === "WAIT_STALE_MARKET_DATA"
    ? "WAIT — FEATURE DATA STALE"
    : label(code);
}
function EngineCard({
  engine,
  detailed,
  onToggle,
}: {
  engine: EngineOutput;
  detailed: boolean;
  onToggle: () => void;
}) {
  return (
    <article className="engine-card">
      <div className="engine-card-head">
        <span>{label(engine.engine_name)}</span>
        <StatusDot
          label="STATUS"
          value={engine.status}
          tone={statusTone(engine.status)}
        />
      </div>
      <strong
        className={
          engine.direction_score == null
            ? "tone-muted"
            : engine.direction_score > 0.1
              ? "tone-good"
              : engine.direction_score < -0.1
                ? "tone-down"
                : "tone-muted"
        }
      >
        {engine.direction_score == null
          ? "UNAVAILABLE"
          : `${engine.direction_score >= 0 ? "+" : ""}${engine.direction_score.toFixed(2)}`}
      </strong>
      <div className="engine-meta">
        <span>{engine.regime ?? "NO REGIME"}</span>
        <span>
          {engine.confidence == null
            ? "--"
            : `${Math.round(engine.confidence * 100)}%`}
        </span>
      </div>
      <small>{engine.reason_codes.join(" · ") || "No reason available"}</small>
      <button className="text-button" onClick={onToggle}>
        {detailed ? "HIDE DETAILS" : "DETAILS"}
      </button>
      {detailed && (
        <div className="engine-details">
          <span>
            Freshness <b>{engine.input_freshness}</b>
          </span>
          <span>
            Features <b>{engine.feature_count}</b>
          </span>
          {Object.entries(engine.components).map(([key, value]) => (
            <span key={key}>
              {label(key)} <b>{String(value)}</b>
            </span>
          ))}
        </div>
      )}
    </article>
  );
}
function WhyPanel({
  state,
  open,
  onToggle,
}: {
  state: MarketFusionState | null;
  open: boolean;
  onToggle: () => void;
}) {
  const reasons = state?.reasons ?? [];
  return (
    <>
      <button className="why-button" onClick={onToggle} aria-expanded={open}>
        <ShieldCheck size={16} /> {open ? "HIDE REASONS" : "WHY WAIT?"}
      </button>
      {open && (
        <div className="why-panel">
          <h3>Why MarketFusion is waiting</h3>
          {reasons.length ? (
            reasons.map((item, index) => (
              <div key={`${item.code}-${index}`}>
                <b>{index + 1}.</b>
                <span>
                  {item.message}
                  <small>{item.code}</small>
                </span>
              </div>
            ))
          ) : (
            <p>No backend reason is available.</p>
          )}
        </div>
      )}
    </>
  );
}
function MarketCard({
  operator,
  quote,
  candles,
  candleStatus,
  timeframe,
  setTimeframe,
  selectedAsset,
}: Pick<
  Props,
  | "operator"
  | "quote"
  | "candles"
  | "candleStatus"
  | "timeframe"
  | "setTimeframe"
  | "selectedAsset"
>) {
  const data = operator?.data_freshness;
  const live = quote?.bid != null && quote?.ask != null;
  const display = selectedAsset === "XAUUSD" ? "XAU/USD" : "EUR/USD";
  const name =
    selectedAsset === "XAUUSD"
      ? "GOLD / U.S. DOLLAR · PRECIOUS METAL"
      : "EURO / U.S. DOLLAR · FOREX";
  const price = (value?: number | null) =>
    value == null ? dash : value.toFixed(selectedAsset === "XAUUSD" ? 2 : 5);
  return (
    <Panel
      title="Live Market"
      eyebrow={`${name} · COMPLETED CANDLES`}
      className="market-card"
      action={
        <div className="timeframe-tabs">
          {(["M1", "M5", "M15", "H1"] as Timeframe[]).map((item) => (
            <button
              className={item === timeframe ? "active" : ""}
              onClick={() => setTimeframe(item)}
              key={item}
            >
              {item}
            </button>
          ))}
        </div>
      }
    >
      <div className="market-price">
        <span>{display}</span>
        <strong>{live ? price(quote?.mid) : "LIVE FEED UNAVAILABLE"}</strong>
        <StatusDot
          label="QUOTE"
          value={live ? "LIVE" : "UNAVAILABLE"}
          tone={live ? "good" : "warn"}
        />
      </div>
      <div className="quote-metrics">
        <Metric label="BID" value={live ? price(quote?.bid) : dash} />
        <Metric label="ASK" value={live ? price(quote?.ask) : dash} />
        <Metric
          label="SPREAD"
          value={
            live
              ? selectedAsset === "XAUUSD"
                ? `${(quote?.spread_points ?? 0).toFixed(1)} points`
                : `${((quote?.spread_points ?? 0) / 10).toFixed(1)} pips`
              : dash
          }
        />
        <Metric
          label="FEATURE FRESHNESS"
          value={data?.status ?? "UNAVAILABLE"}
        />
      </div>
      <CandleChart candles={candles} />
      <div className="chart-footer">
        <span>
          <Activity size={13} /> {candleStatus}
        </span>
        <span>
          {timeframe} {timeframe === "M5" ? "COMPLETED" : "HISTORICAL"}
        </span>
        <span>Latest {localTime(candles.at(-1)?.time, true)} LKT</span>
      </div>
    </Panel>
  );
}
function AdvisoryCard({
  state,
  operator,
  connection,
  why,
  setWhy,
}: {
  state: MarketFusionState | null;
  operator: OperatorStatus | null;
  connection: string;
  why: boolean;
  setWhy: (value: boolean) => void;
}) {
  const action: AdvisoryAction =
    connection === "LIVE" && state ? state.decision.action : "WAIT";
  const risk = state?.reasons.some((item) => item.blocking)
    ? "BLOCKING"
    : "NORMAL";
  return (
    <Panel
      title="Current Advisory"
      eyebrow="V0.6C · READ ONLY"
      className="advisory-card"
    >
      <div className="advisory-main">
        <span>DECISION</span>
        <strong>{action.replace("_", " ")}</strong>
        <p>
          {state?.decision.action_meaning ??
            "Directional advice is unavailable; the system remains fail-closed."}
        </p>
      </div>
      <div className="advisory-grid">
        <Metric
          label="CONFIDENCE"
          value={state?.decision.confidence ?? "VERY LOW"}
        />
        <Metric
          label="RISK"
          value={risk}
          tone={risk === "NORMAL" ? "good" : "warn"}
        />
        <Metric
          label="MARKET"
          value={operator?.market.status === "OPEN" ? "OPEN" : "CLOSED"}
        />
        <Metric label="GATE" value={reasonLabel(state?.decision.gate)} />
      </div>
      <WhyPanel state={state} open={why} onToggle={() => setWhy(!why)} />
      <div className="execution-note">
        <CheckCircle2 size={16} /> Execution disabled · manual confirmation
        required
      </div>
    </Panel>
  );
}
function SummaryCards({
  operator,
  state,
  monitoring,
  now,
}: Pick<Props, "operator" | "state" | "monitoring" | "now">) {
  return (
    <div className="summary-grid">
      <Panel title="Market" eyebrow="STATE">
        <StatusDot
          label="MARKET"
          value={operator?.market.status ?? "UNKNOWN"}
          tone={statusTone(operator?.market.status)}
        />
        <small>{operator?.session.current_session ?? "UNKNOWN"}</small>
      </Panel>
      <Panel title="Quotes" eyebrow="FRESHNESS">
        <StatusDot
          label="QUOTE"
          value={operator?.data_freshness.status ?? "UNKNOWN"}
          tone={statusTone(operator?.data_freshness.status)}
        />
        <small>
          {ageFrom(operator?.data_freshness.latest_market_tick_utc, now)} old
        </small>
      </Panel>
      <Panel title="Features" eyebrow="CAUSAL DATA">
        <StatusDot
          label="FEATURES"
          value={operator?.data_freshness.status ?? "UNKNOWN"}
          tone={statusTone(operator?.data_freshness.status)}
        />
        <small>
          {localTime(operator?.data_freshness.feature_row_utc, true)}
        </small>
      </Panel>
      <Panel title="Models" eyebrow="APPROVAL">
        <strong className="summary-number">
          {state?.predictions.fusion?.valid_horizons?.length ?? 0}/3
        </strong>
        <small>approved horizons</small>
      </Panel>
      <Panel title="Next AI" eyebrow="REASSESSMENT">
        <strong className="summary-number">
          {until(operator?.reassessment.at_utc, now)}
        </strong>
        <small>{operator?.reassessment.status ?? "UNKNOWN"}</small>
      </Panel>
      <Panel title="Trading" eyebrow="SAFETY">
        <StatusDot
          label="STATE"
          value={operator?.trading_state ?? "UNKNOWN"}
          tone="warn"
        />
        <small>automatic execution disabled</small>
      </Panel>
    </div>
  );
}
function Models({
  state,
  operator,
  now,
}: Pick<Props, "state" | "operator" | "now">) {
  const [details, setDetails] = useState<number | null>(null);
  return (
    <section className="tab-content">
      <div className="section-intro">
        <div>
          <span className="eyebrow">V0.6A SHADOW INFERENCE</span>
          <h2>Model Status</h2>
        </div>
        <p>
          Approved models are shown with real probabilities. Missing horizons
          remain unavailable.
        </p>
      </div>
      <div className="model-grid">
        {[
          [15, "15 MIN"],
          [60, "1 HOUR"],
          [240, "4 HOURS"],
        ].map(([horizon, title]) => {
          const item = state?.predictions.horizons[String(horizon)] as
            HorizonPrediction | undefined;
          return (
            <div key={horizon}>
              <PredictionCard
                label={title as string}
                prediction={item}
                predictionTime={state?.decision.decision_time.utc}
                featureTime={operator?.data_freshness.feature_row_utc}
                now={now}
              />
              <button
                className="details-button"
                onClick={() =>
                  setDetails(details === horizon ? null : (horizon as number))
                }
              >
                {details === horizon ? "HIDE DETAILS" : "DETAILS"}
              </button>
              {details === horizon && (
                <div className="detail-drawer">
                  <span>
                    Model ID <b>{item?.model_id ?? "N/A"}</b>
                  </span>
                  <span>
                    Prediction{" "}
                    <b>{utcTime(state?.decision.decision_time.utc)}</b>
                  </span>
                  <span>
                    Feature{" "}
                    <b>{utcTime(operator?.data_freshness.feature_row_utc)}</b>
                  </span>
                  <span>
                    Gate <b>{label(item?.decision_gate)}</b>
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
function Engines({ engines }: { engines: EngineStatus | null }) {
  const [details, setDetails] = useState<string | null>(null);
  const all = [
    ...Object.values(engines?.engines ?? {}),
    ...Object.values(engines?.external_engines ?? {}),
  ];
  const available = all.filter(
    (item) => item.status === "AVAILABLE" && item.direction_score != null,
  );
  const bullish = available.filter(
    (item) => (item.direction_score ?? 0) > 0.1,
  ).length;
  const bearish = available.filter(
    (item) => (item.direction_score ?? 0) < -0.1,
  ).length;
  const neutral = available.length - bullish - bearish;
  return (
    <section className="tab-content">
      <div className="section-intro">
        <div>
          <span className="eyebrow">V0.9A · RESEARCH SIGNALS</span>
          <h2>Engine Intelligence</h2>
        </div>
        <p>
          Scores range from -1 to +1. They are not calibrated trading
          probabilities.
        </p>
      </div>
      <div className="agreement">
        <span>ENGINE AGREEMENT</span>
        <b>{bullish} bullish</b>
        <b>{neutral} neutral</b>
        <b>{bearish} bearish</b>
        <small>
          {available.length
            ? `${Math.round((bullish / available.length) * 100)}% bullish`
            : "INSUFFICIENT DATA"}
        </small>
      </div>
      <div className="engine-grid engine-grid-detail">
        {all.map((engine) => (
          <EngineCard
            key={engine.engine_name}
            engine={engine}
            detailed={details === engine.engine_name}
            onToggle={() =>
              setDetails(
                details === engine.engine_name ? null : engine.engine_name,
              )
            }
          />
        ))}
      </div>
    </section>
  );
}
function Validation({
  monitoring,
  shadowSummary,
  shadowRecent,
}: Pick<Props, "monitoring" | "shadowSummary" | "shadowRecent">) {
  return (
    <section className="tab-content">
      <div className="section-intro">
        <div>
          <span className="eyebrow">V0.8 · TRUE FORWARD SHADOW</span>
          <h2>Validation</h2>
        </div>
        <p>
          No historical predictions are reconstructed and no profitability claim
          is made.
        </p>
      </div>
      <LiveShadowValidation summary={shadowSummary} recent={shadowRecent} />
    </section>
  );
}
function System({
  operator,
  state,
  connection,
  refresh,
}: Pick<Props, "operator" | "state" | "connection" | "refresh">) {
  const services: [string, string | undefined][] = [
    ["V0.5A", state?.health.sources.v05a_market],
    ["V0.5B", state?.health.sources.v05b_intelligence],
    ["V0.6", state?.system.status],
    ["V0.7 API", connection === "LIVE" ? "RUNNING" : connection],
    ["V0.8", "RUNNING"],
    ["V0.9A", "RUNNING"],
  ];
  return (
    <section className="tab-content">
      <div className="section-intro">
        <div>
          <span className="eyebrow">LOCAL RUNTIME PIPELINE</span>
          <h2>System Diagnostics</h2>
        </div>
        <button className="action-button" onClick={refresh}>
          <RefreshCw size={15} /> REFRESH NOW
        </button>
      </div>
      <div className="system-grid">
        {services.map(([name, status]) => (
          <Panel title={name} key={name}>
            <StatusDot
              label="STATUS"
              value={String(status ?? "UNAVAILABLE")}
              tone={statusTone(String(status))}
            />
            <div className="system-line">
              {name === "V0.6"
                ? `Gate: ${reasonLabel(state?.decision.gate)}`
                : "Read-only service"}
            </div>
          </Panel>
        ))}
      </div>
      <Panel title="Operator diagnostics" eyebrow="CURRENT SERVER STATE">
        <div className="diagnostics-grid">
          <Metric
            label="SYSTEM HEALTH"
            value={operator?.system_health.status ?? "UNKNOWN"}
          />
          <Metric label="MARKET" value={operator?.market.status ?? "UNKNOWN"} />
          <Metric
            label="SESSION"
            value={operator?.session.current_session ?? "UNKNOWN"}
          />
          <Metric
            label="FEATURES"
            value={operator?.data_freshness.status ?? "UNKNOWN"}
          />
          <Metric label="MODE" value="SHADOW / MANUAL" />
          <Metric label="EXECUTION" value="DISABLED" />
        </div>
      </Panel>
    </section>
  );
}
export function DashboardShell(props: Props) {
  const [tab, setTab] = useState<Tab>("overview");
  const [why, setWhy] = useState(false);
  const serverOffsetMs = useServerClock(props.operator);
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    const text = `MarketFusion\nMarket ${props.operator?.market.status ?? "UNKNOWN"}\nMT5 ${props.quote?.status?.startsWith("PASS") ? "CONNECTED" : "UNAVAILABLE"}\nQuote ${props.operator?.data_freshness.status ?? "UNKNOWN"}\nFeature ${props.operator?.data_freshness.status ?? "UNKNOWN"}\n15m ${props.state?.predictions.horizons["15"]?.model_status ?? "NONE"}\n60m ${props.state?.predictions.horizons["60"]?.model_status ?? "NONE"}\n240m ${props.state?.predictions.horizons["240"]?.model_status ?? "NONE"}\nDecision ${props.state?.decision.action ?? "WAIT"}\nGate ${props.state?.decision.gate ?? "UNKNOWN"}`;
    await navigator.clipboard?.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };
  const content =
    tab === "market" ? (
      <section className="tab-content">
        <MarketCard {...props} />
      </section>
    ) : tab === "models" ? (
      <Models {...props} />
    ) : tab === "engines" ? (
      <Engines engines={props.engines} />
    ) : tab === "validation" ? (
      <Validation {...props} />
    ) : tab === "system" ? (
      <System {...props} />
    ) : (
      <>
        <div className="primary-grid">
          <MarketCard {...props} />
          <AdvisoryCard
            state={props.state}
            operator={props.operator}
            connection={props.connection}
            why={why}
            setWhy={setWhy}
          />
        </div>
        <SummaryCards {...props} />
        <div className="overview-lower">
          <Panel title="Models" eyebrow="APPROVED HORIZONS">
            <div className="mini-models">
              {[15, 60, 240].map((h) => (
                <StatusDot
                  key={h}
                  label={`${h}M`}
                  value={
                    props.state?.predictions.horizons[String(h)]
                      ?.model_status ?? "UNKNOWN"
                  }
                  tone={statusTone(
                    props.state?.predictions.horizons[String(h)]?.model_status,
                  )}
                />
              ))}
            </div>
            <button className="text-button" onClick={() => setTab("models")}>
              VIEW MODEL DETAILS
            </button>
          </Panel>
          <Panel title="Engine Intelligence" eyebrow="RESEARCH SIGNALS">
            <div className="mini-engines">
              {Object.values(props.engines?.engines ?? {})
                .slice(0, 4)
                .map((engine) => (
                  <Metric
                    key={engine.engine_name}
                    label={engine.engine_name}
                    value={
                      engine.direction_score == null
                        ? "UNAVAILABLE"
                        : engine.direction_score.toFixed(2)
                    }
                    tone={
                      engine.direction_score && engine.direction_score > 0.1
                        ? "good"
                        : "muted"
                    }
                  />
                ))}
            </div>
            <button className="text-button" onClick={() => setTab("engines")}>
              VIEW ALL ENGINES
            </button>
          </Panel>
          <Panel title="Live Validation" eyebrow="V0.8 SHADOW">
            <div className="validation-hero">
              <strong>
                {props.shadowSummary?.performance_observations ?? 0}
              </strong>
              <span>evaluated observations</span>
            </div>
            <StatusDot
              label="RECORDER"
              value={props.shadowSummary?.recorder.status ?? "UNKNOWN"}
              tone="good"
            />
            <button
              className="text-button"
              onClick={() => setTab("validation")}
            >
              VIEW VALIDATION
            </button>
          </Panel>
        </div>
      </>
    );
  return (
    <main className="app-shell redesigned">
      <header className="command-bar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={22} />
          </div>
          <div>
            <h1>
              MARKETFUSION <span>AI</span>
            </h1>
            <p>EUR/USD · Real-Time Research Intelligence</p>
          </div>
        </div>
        <div className="top-health">
          <StatusDot
            label="MT5"
            value={
              props.quote?.status?.startsWith("PASS")
                ? "CONNECTED"
                : "UNAVAILABLE"
            }
            tone={props.quote?.status?.startsWith("PASS") ? "good" : "warn"}
          />
          <StatusDot
            label="API"
            value={props.operator?.system_health.status ?? "UNKNOWN"}
            tone={statusTone(props.operator?.system_health.status)}
          />
          <StatusDot
            label="MARKET"
            value={props.operator?.market.market_open ? "OPEN" : "CLOSED"}
            tone={props.operator?.market.market_open ? "good" : "warn"}
          />
          <StatusDot
            label="QUOTE"
            value={props.operator?.data_freshness.status ?? "UNKNOWN"}
            tone={statusTone(props.operator?.data_freshness.status)}
          />
          <StatusDot
            label="FEATURES"
            value={props.operator?.data_freshness.status ?? "UNKNOWN"}
            tone={statusTone(props.operator?.data_freshness.status)}
          />
          <StatusDot
            label="MODELS"
            value={`${props.state?.predictions.fusion?.valid_horizons?.length ?? 0}/3`}
            tone="warn"
          />
          <StatusDot label="MODE" value="SHADOW" tone="info" />
          <MarketOpenCountdown
            operator={props.operator}
            now={props.now}
            serverOffsetMs={serverOffsetMs}
          />
        </div>
        <div className="mode-block">
          <b>SHADOW / MANUAL</b>
          <span>{localTime(new Date(props.now).toISOString())} LKT</span>
          <small>
            {props.connection === "LIVE"
              ? "SERVER SYNCHRONIZED"
              : "STATE UNAVAILABLE"}
          </small>
        </div>
      </header>
      <nav className="main-nav" aria-label="Dashboard sections">
        {tabs.map(([id, title, Icon]) => (
          <button
            key={id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id as Tab)}
          >
            <Icon size={15} />
            {title}
          </button>
        ))}
        <span className="nav-spacer" />
        <button
          className="icon-button"
          onClick={props.refresh}
          title="Refresh read-only API data"
        >
          <RefreshCw size={16} /> <span>REFRESH</span>
        </button>
        <button
          className="icon-button"
          onClick={copy}
          title="Copy safe diagnostics"
        >
          <Clipboard size={16} />{" "}
          <span>{copied ? "COPIED" : "DIAGNOSTICS"}</span>
        </button>
      </nav>
      {props.connection !== "LIVE" && (
        <div className="connection-banner">
          <Wifi size={16} />
          <b>STATE {props.connection}</b>
          <span>Directional advice remains suppressed and fail-closed.</span>
        </div>
      )}
      <div className="tab-frame">{content}</div>
      <footer className="footer">
        <span>
          <Activity size={14} /> Read-only research terminal
        </span>
        <span>Manual execution in MT5 only</span>
        <span>API {props.operator?.system_health.status ?? "UNKNOWN"}</span>
        <span>V0.6 decisions unchanged</span>
        <BarChart3 size={14} />
      </footer>
    </main>
  );
}
