import type { HorizonPrediction } from "../types/marketfusion";
import { ageFrom, dash, label as formatLabel, localTime, probability } from "../utils/format";

function ProbabilityRow({ label, value, tone }: { label: string; value?: number | null; tone: string }) {
  return <div className="prob-row"><span>{label}</span>{value == null ? <b>N/A</b> : <><div className="prob-track"><i className={`prob-fill ${tone}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div><b>{probability(value)}</b></>}</div>;
}

export function PredictionCard({ label, prediction, predictionTime, featureTime, now }: { label: string; prediction?: HorizonPrediction; predictionTime?: string | null; featureTime?: string | null; now: number }) {
  const unavailable = !prediction || prediction.model_status !== "APPROVED_CHAMPION";
  return <article className="prediction-card">
    <div className="prediction-title"><span>{label}</span><b className={unavailable ? "tone-warn" : "tone-good"}>{unavailable ? "NO APPROVED MODEL" : prediction.model_status}</b></div>
    <div className="model-id"><span>MODEL ID</span><b>{unavailable ? "N/A" : prediction?.model_id ?? "N/A"}</b></div>
    <ProbabilityRow label="DOWN" value={prediction?.prob_down} tone="down" />
    <ProbabilityRow label="NEUTRAL" value={prediction?.prob_neutral} tone="neutral" />
    <ProbabilityRow label="UP" value={prediction?.prob_up} tone="up" />
    <footer><span>Direction</span><b>{unavailable ? "N/A" : formatLabel(prediction?.shadow_direction)}</b><span>Confidence</span><b>{unavailable ? "N/A" : formatLabel(prediction?.probability_confidence_band)}</b><span>Gate</span><b>{formatLabel(prediction?.decision_gate)}</b><span>Prediction</span><b>{unavailable ? "N/A" : localTime(predictionTime)}</b><span>Feature row</span><b>{localTime(featureTime)}</b><span>Data age</span><b>{ageFrom(featureTime, now)}</b><span>Top class</span><b>{unavailable ? "N/A" : prediction?.top_class ?? dash}</b><span>Top probability</span><b>{unavailable ? "N/A" : probability(prediction?.top_probability)}</b></footer>
  </article>;
}
