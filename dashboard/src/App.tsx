import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, BarChart3, BrainCircuit, CalendarClock, Clock3, Database, Radio, ShieldCheck, WifiOff } from "lucide-react";
import { CandleChart } from "./components/CandleChart";
import { Metric, Panel, StatusDot } from "./components/Panel";
import { PredictionCard } from "./components/PredictionCard";
import { useCandles } from "./hooks/useCandles";
import { useMarketFusionState } from "./hooks/useMarketFusionState";
import { useSupplementary } from "./hooks/useSupplementary";
import type { AdvisoryAction, Timeframe } from "./types/marketfusion";
import { countdown, dash, label, localTime, number, probability, utcTime } from "./utils/format";

const timeframes: Timeframe[] = ["M1", "M5", "M15", "H1"];

export default function App() {
  const { state, connection, lastSuccess } = useMarketFusionState();
  const [timeframe, setTimeframe] = useState<Timeframe>("M5");
  const { candles, status: candleStatus } = useCandles(timeframe);
  const { market: quote, intelligence, research, events } = useSupplementary();
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, []);

  const safe = connection === "LIVE";
  const action: AdvisoryAction = safe && state ? state.decision.action : "WAIT";
  const confidence = safe && state ? state.decision.confidence : "VERY_LOW";
  const reason = state?.reasons[0]?.message ?? "Live runtime state is unavailable.";
  const horizons = state?.predictions.horizons ?? {};
  const fusion = state?.predictions.fusion;
  const approved = fusion?.valid_horizons?.length ?? 0;
  const event = state?.event;
  const eventTime = event?.nearest_event?.event_timestamp_utc;
  const auditedEvent = events?.events.find(item =>
    String(item.event_id) === String(event?.nearest_event?.event_id) ||
    (!!item.event_code && item.event_code === event?.nearest_event?.event_code)
  );
  const verifiedForecast = auditedEvent?.consensus_status === "VERIFIED_PRE_RELEASE_CONSENSUS" ? auditedEvent.forecast_value : null;
  const nextReview = state?.decision.next_reassessment.utc;
  const risk = state?.reasons.some(item => item.blocking) ? "BLOCKING" : state?.reasons.length ? "CAUTION" : "NORMAL";
  const healthEntries = Object.entries(state?.health.sources ?? {});
  const session = state?.market.session === "OVERLAP" ? "LONDON + NY" : label(state?.market.session);
  const agreement = approved === 0 ? "NO APPROVED MODELS" : fusion?.status ?? "PARTIAL";
  const currentLocal = useMemo(() => new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Colombo", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(now)), [now]);

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><BrainCircuit size={23} /></div><div><h1>MARKETFUSION <span>AI</span></h1><p>Real-Time EUR/USD Research Intelligence</p></div></div>
      <div className="top-health">
        <StatusDot label="MT5" value={quote?.status?.startsWith("PASS") ? "CONNECTED" : "UNAVAILABLE"} />
        <StatusDot label="V0.5A" value={state?.market.freshness?.status ?? "UNAVAILABLE"} />
        <StatusDot label="V0.5B" value={state?.intelligence?.status ?? "UNAVAILABLE"} />
        <StatusDot label="V0.5C" value={approved ? `${approved}/3 APPROVED` : "NO CHAMPION"} />
        <StatusDot label="V0.6" value={state?.decision.action ?? "UNAVAILABLE"} />
        <StatusDot label="Trading" value="DISABLED" tone="warn" />
      </div>
      <div className="mode-block"><b>SHADOW / MANUAL MODE</b><span>{currentLocal} LKT</span><small>{utcTime(state?.system.generated_time.utc)}</small></div>
    </header>

    {connection !== "LIVE" && <div className="connection-banner"><WifiOff size={18} /><b>{connection === "DISCONNECTED" ? "CONNECTION LOST" : "STATE STALE"}</b><span>Directional advice is suppressed. Displaying WAIT and last known state.</span>{lastSuccess && <small>Last success {lastSuccess.toLocaleTimeString()}</small>}</div>}

    <div className="primary-grid">
      <Panel title="EUR/USD Live Market" eyebrow="COMPLETED LOCAL CANDLES" className="chart-panel" action={<div className="timeframe-tabs">{timeframes.map(item => <button className={item === timeframe ? "active" : ""} onClick={() => setTimeframe(item)} key={item}>{item}</button>)}</div>}>
        <div className="quote-line"><div><span>EUR/USD</span><strong>{number(quote?.mid ?? state?.market.close)}</strong></div><Metric label="BID" value={number(quote?.bid)} /><Metric label="ASK" value={number(quote?.ask)} /><Metric label="SPREAD" value={quote?.spread_points == null ? dash : `${quote.spread_points.toFixed(1)} pts`} /></div>
        <CandleChart candles={candles} />
        <div className="chart-footer"><span><Radio size={13} /> {candleStatus}</span><span>Latest completed: {localTime(candles.at(-1)?.time, true)} LKT</span><span>Decision remains tied to completed causal M5 data</span></div>
      </Panel>

      <Panel title="Current Advisory" eyebrow="V0.6C DECISION" className={`decision-panel decision-${action.toLowerCase()}`}>
        <div className="advisory"><span>{connection === "LIVE" ? "CURRENT ADVISORY" : "SAFETY ADVISORY"}</span><strong>{action.replace("_", " ")}</strong><p>{action === "WAIT" ? "System is collecting live market intelligence. No V0.5C model has yet passed promotion requirements." : state?.decision.action_meaning}</p></div>
        <div className="decision-metrics"><Metric label="CONFIDENCE" value={confidence.replace("_", " ")} tone="warn" /><Metric label="RISK" value={risk} tone={risk === "NORMAL" ? "good" : "warn"} /><Metric label="MANUAL EXECUTION" value="YES" /><Metric label="TRADING EXECUTION" value="DISABLED" tone="warn" /></div>
        <div className="primary-reason"><AlertTriangle size={18} /><div><span>PRIMARY REASON</span><b>{reason}</b></div></div>
      </Panel>
    </div>

    <div className="model-grid">
      <PredictionCard label="15 MIN" prediction={horizons["15"]} />
      <PredictionCard label="1 HOUR" prediction={horizons["60"]} />
      <PredictionCard label="4 HOURS" prediction={horizons["240"]} />
    </div>

    <div className="secondary-grid">
      <Panel title="Fusion State" eyebrow="WEIGHTED HORIZON CONSENSUS">
        <div className="fusion-heading"><span>Horizon agreement</span><b>{label(agreement)}</b><small>{approved}/3 approved horizons</small></div>
        <div className="fusion-probs"><Metric label="DOWN" value={probability(fusion?.probabilities?.down)} tone="down" /><Metric label="NEUTRAL" value={probability(fusion?.probabilities?.neutral)} /><Metric label="UP" value={probability(fusion?.probabilities?.up)} tone="good" /></div>
        <div className="inline-detail"><span>Directional margin</span><b>{probability(fusion?.margin)}</b><span>Fusion action</span><b className="tone-warn">{action.replace("_", " ")}</b></div>
      </Panel>

      <Panel title="Manual Trade Window" eyebrow="OBSERVATION WINDOW ONLY">
        <div className="window-status"><ShieldCheck size={20} /><div><span>STATUS</span><b>{state?.decision.gate ?? "WAIT_SYSTEM_DATA_INVALID"}</b></div></div>
        <div className="window-grid"><Metric label="SUGGESTED START" value={safe ? localTime(state?.trade_window.start_utc) : dash} sub={safe ? utcTime(state?.trade_window.start_utc) : undefined} /><Metric label="SUGGESTED END" value={safe ? localTime(state?.trade_window.end_utc) : dash} sub={safe ? utcTime(state?.trade_window.end_utc) : undefined} /><Metric label="HORIZON" value={state?.trade_window.horizon_minutes ? `${state.trade_window.horizon_minutes} min` : dash} /></div>
        <p className="muted-copy">{state?.trade_window.status === "ADVISORY_WINDOW" && safe ? "Manual confirmation in MT5 remains mandatory." : "No manual trade window is available. No approved model."}</p>
      </Panel>

      <Panel title="Next System Review" eyebrow="CLIENT-SIDE COUNTDOWN" className="countdown-panel">
        <Clock3 size={24} /><strong>{countdown(nextReview, now)}</strong><span>{localTime(nextReview)} LKT</span><small>{utcTime(nextReview)}</small><p>Next completed M5 decision</p>
      </Panel>

      <Panel title="Current Market Context" eyebrow="CAUSAL V0.5A SNAPSHOT">
        <div className="metric-grid compact"><Metric label="SESSION" value={session} /><Metric label="TREND" value={label(state?.market.regime?.trend_regime)} /><Metric label="VOLATILITY" value={label(state?.market.regime?.volatility_regime)} /><Metric label="SPREAD" value={label(state?.market.spread?.status)} /><Metric label="DATA AGE" value={state?.market.freshness?.age_minutes == null ? dash : `${state.market.freshness.age_minutes.toFixed(1)} min`} /><Metric label="M5 DECISION" value={localTime(state?.decision.decision_time.utc)} /></div>
      </Panel>
    </div>

    <div className="detail-grid">
      <Panel title="Event Risk" eyebrow="AUDITED TARGET EVENTS" className="event-panel">
        <div className="event-state"><CalendarClock size={22} /><div><span>RISK STATE</span><b>{label(event?.status ?? "EVENT_DATA_INCOMPLETE")}</b></div></div>
        {event?.nearest_event && eventTime ? <><h3>{event.nearest_event.event_name}</h3><p>{localTime(eventTime, true)} LKT <small>{utcTime(eventTime)}</small></p><div className="event-countdown">{countdown(eventTime, now)}</div><div className="timeline"><i /><span>NOW</span><i /><span>BLOCK −15m</span><i /><span>RELEASE</span><i /><span>POST +60m</span></div></> : <div className="empty-state">AUDITED EVENT DATA UNAVAILABLE</div>}
        <div className="event-values"><Metric label="FORECAST" value={number(verifiedForecast, 1)} /><Metric label="ACTUAL" value={dash} /><Metric label="SURPRISE" value={dash} /></div>
      </Panel>

      <Panel title="News & Macro" eyebrow="V0.5B CAUSAL INTELLIGENCE">
        <div className="news-window">{[["15M","15"],["1H","60"],["4H","240"],["24H","1440"]].map(([name,key]) => <Metric key={key} label={name} value={intelligence?.news_counts[key] ?? dash} />)}</div>
        <h3 className="section-label">MACRO SNAPSHOT</h3>
        <div className="macro-grid">
          <Metric label="FED FUNDS" value={number(intelligence?.macro.macro_us_effective_fed_funds_value, 2)} />
          <Metric label="US 2Y" value={number(intelligence?.macro.macro_us_2y_yield_value, 2)} />
          <Metric label="US 10Y" value={number(intelligence?.macro.macro_us_10y_yield_value, 2)} />
          <Metric label="10Y−2Y" value={number(intelligence?.macro.macro_us_10y_minus_2y_value, 2)} />
          <Metric label="ECB MRR" value={number(intelligence?.macro.macro_ecb_main_refinancing_rate_value, 2)} />
          <Metric label="US−ECB" value={number(intelligence?.macro.macro_policy_rate_spread_us_minus_ecb_pctpt, 2)} />
        </div>
        <div className="provider-row">{Object.entries(intelligence?.provider_health ?? {}).map(([name,value]) => <StatusDot key={name} label={name} value={value} />)}</div>
        <h3 className="section-label">24H TOPICS</h3>
        <div className="news-window">{Object.entries(intelligence?.topic_counts_24h ?? {}).map(([name,value]) => <Metric key={name} label={name.replace("_", " ").toUpperCase()} value={value ?? dash} />)}</div>
      </Panel>

      <Panel title="System Health" eyebrow="LOCAL RUNTIME PIPELINE">
        <div className="health-list">{healthEntries.map(([name,value]) => <StatusDot key={name} label={name.replace("_", " ").toUpperCase()} value={value} />)}</div>
        <div className="provider-row"><StatusDot label="V06C RUNTIME" value={state?.system.status ?? "UNAVAILABLE"} /></div>
        <div className="health-times"><span>Last market update</span><b>{localTime(state?.market.freshness?.observed_at_utc, true)}</b><span>Last intelligence update</span><b>{localTime(intelligence?.captured_at_utc, true)}</b><span>Last model training</span><b>{localTime(research?.last_training_utc, true)}</b><span>Last runtime state</span><b>{localTime(state?.system.generated_time.utc, true)}</b></div>
      </Panel>
    </div>

    <div className="bottom-grid">
      <Panel title="Decision Reasons" eyebrow="DETERMINISTIC BACKEND RECORDS">
        <div className="reason-list">{state?.reasons.length ? state.reasons.map(item => <article key={`${item.code}-${item.priority}`} className={item.blocking ? "blocking" : "diagnostic"}><AlertTriangle size={17} /><div><span>{item.source} · {item.code.replaceAll("_", " ")}</span><p>{item.message}</p></div></article>) : <div className="empty-state">NO REASON RECORDS AVAILABLE</div>}</div>
      </Panel>

      <Panel title="Model Quality" eyebrow="RESEARCH ONLY — NOT APPROVED">
        <div className="research-list">{[15,60,240].map(horizon => { const item = research?.candidates[String(horizon)]; return <article key={horizon}><span>{horizon === 60 ? "1H" : horizon === 240 ? "4H" : "15M"}</span><b>{item?.family?.replaceAll("_", " ") ?? dash}</b><small>Balanced accuracy {item ? item.balanced_accuracy.toFixed(4) : dash}</small><em>{item?.promotion_status ?? "NO PROMOTION"}</em></article>; })}</div>
        <div className="approval-divider"><ShieldCheck size={18} /><b>APPROVED PRODUCTION MODELS: {approved}/3</b></div>
      </Panel>

      <Panel title="Daily Learning" eyebrow="CONTROLLED — NOT CONTINUOUS">
        <div className="metric-grid compact"><Metric label="LAST TRAINING" value={localTime(research?.last_training_utc, true)} /><Metric label="MODEL PROMOTION" value="NONE" tone="warn" /><Metric label="MARKET CORE ROWS" value={research?.market_core_rows ?? dash} sub={research?.history_days == null ? undefined : `${research.history_days} history days`} /><Metric label="V0.5B ELIGIBILITY" value={label(research?.v05b_eligibility)} /><Metric label="LIVE SURPRISES" value={research?.live_surprise_samples ?? dash} /><Metric label="NEXT TRAINING" value={label(research?.next_recommended_training)} /></div>
      </Panel>

      <Panel title="Historical Memory" eyebrow="V0.4B DIAGNOSTIC CONTEXT">
        <div className="memory-empty"><Database size={26} /><b>HISTORICAL CONTEXT NOT ACTIVE</b><p>No current analogue ranking is present in the V0.6C state. No analogue returns or direction balance are inferred.</p></div>
      </Panel>
    </div>

    <footer className="footer"><span><Activity size={14} /> Local-first research dashboard</span><span>Manual execution in MT5 only</span><span>No profitability or predictive-edge claim</span><span>API state: {connection}</span><BarChart3 size={14} /></footer>
  </main>;
}
