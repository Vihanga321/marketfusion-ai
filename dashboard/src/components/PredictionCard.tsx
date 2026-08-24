import type { HorizonPrediction } from "../types/marketfusion";
import { dash, probability } from "../utils/format";

function ProbabilityRow({ label, value, tone }: { label: string; value?: number | null; tone: string }) {
  return <div className="prob-row"><span>{label}</span>{value == null ? <b>{dash}</b> : <><div className="prob-track"><i className={`prob-fill ${tone}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div><b>{probability(value)}</b></>}</div>;
}

export function PredictionCard({ label, prediction }: { label: string; prediction?: HorizonPrediction }) {
  const unavailable = !prediction || prediction.model_status !== "APPROVED_CHAMPION";
  return <article className="prediction-card">
    <div className="prediction-title"><span>{label}</span><b className={unavailable ? "tone-warn" : "tone-good"}>{unavailable ? "NO APPROVED MODEL" : prediction.model_status}</b></div>
    {unavailable ? <div className="no-model">Promotion requirements have not been met.</div> : <div className="model-id">{prediction.model_id}</div>}
    <ProbabilityRow label="DOWN" value={prediction?.prob_down} tone="down" />
    <ProbabilityRow label="NEUTRAL" value={prediction?.prob_neutral} tone="neutral" />
    <ProbabilityRow label="UP" value={prediction?.prob_up} tone="up" />
    <footer><span>Top class</span><b>{prediction?.top_class ?? dash}</b><span>Top probability</span><b>{probability(prediction?.top_probability)}</b><span>Calibration</span><b>{prediction?.calibration_status ?? dash}</b></footer>
  </article>;
}
