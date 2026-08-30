"""Causal, observational V0.9A engines over completed V0.5A feature rows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.engines.contract import ENGINE_CONTRACT_VERSION, EngineOutput, unavailable
from src.marketdata.market_calendar import forex_session_state, market_calendar
from src.marketdata.v05a_contract import FEATURE_FILE


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _base(name: str, row: pd.Series, now: pd.Timestamp, status: str = "AVAILABLE") -> EngineOutput:
    decision = _utc(row["decision_timestamp_utc"])
    age = (now - decision).total_seconds()
    freshness = "FRESH" if 0 <= age <= 360 else "STALE"
    return {
        "engine_name": name, "contract_version": ENGINE_CONTRACT_VERSION,
        "decision_timestamp_utc": decision.isoformat(), "status": status,
        "direction_score": None, "confidence": None, "regime": None,
        "feature_count": 0, "input_freshness": freshness,
        "reason_codes": [], "components": {},
    }


def _bounded(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _market_engine(frame: pd.DataFrame, now: pd.Timestamp) -> dict[str, EngineOutput]:
    row = frame.iloc[-1]
    returns = {name: float(row[name]) for name in ("m5_return_15m", "m5_return_60m", "m5_return_240m") if pd.notna(row.get(name))}
    outputs: dict[str, EngineOutput] = {}

    technical = _base("technical", row, now, "AVAILABLE" if returns else "PARTIAL")
    trend = float(np.mean([np.sign(value) for value in returns.values()])) if returns else 0.0
    technical.update(direction_score=_bounded(trend), confidence=min(1.0, len(returns) / 3), feature_count=len(returns), reason_codes=["RETURN_ALIGNMENT"] if returns else ["TECHNICAL_INPUTS_INCOMPLETE"], components={"return_alignment": trend})
    outputs["technical"] = technical

    price = _base("price_action", row, now, "AVAILABLE" if pd.notna(row.get("m5_return_5m")) else "PARTIAL")
    short_return = float(row.get("m5_return_5m", np.nan))
    if np.isfinite(short_return):
        price.update(direction_score=_bounded(short_return / 0.001), confidence=0.5, feature_count=1, reason_codes=["SHORT_HORIZON_RETURN"], components={"m5_return_5m": short_return})
    else:
        price["reason_codes"] = ["PRICE_ACTION_INPUT_UNAVAILABLE"]
    outputs["price_action"] = price

    structure = _base("structure", row, now, "AVAILABLE" if len(returns) >= 2 else "PARTIAL")
    if returns:
        score = _bounded(float(np.mean(list(returns.values()))) / 0.001)
        structure.update(direction_score=score, confidence=min(1.0, len(returns) / 3), feature_count=len(returns), regime="TREND_UP" if score > 0.25 else "TREND_DOWN" if score < -0.25 else "RANGE", reason_codes=["MULTI_HORIZON_DIRECTION"], components={"return_alignment": score})
    else:
        structure["reason_codes"] = ["STRUCTURE_INPUTS_INCOMPLETE"]
    outputs["structure"] = structure

    liquidity = _base("liquidity", row, now, "PARTIAL")
    spread = row.get("m5_spread_points")
    if pd.notna(spread):
        liquidity.update(direction_score=0.0, confidence=0.25, feature_count=1, reason_codes=["SPREAD_CONTEXT_ONLY"], components={"spread_points": float(spread)})
    else:
        liquidity["reason_codes"] = ["LIQUIDITY_INPUT_UNAVAILABLE"]
    outputs["liquidity"] = liquidity

    volatility = _base("volatility", row, now, "AVAILABLE" if pd.notna(row.get("m5_volatility_60m")) else "PARTIAL")
    vol = row.get("m5_volatility_60m")
    if pd.notna(vol):
        regime = "LOW" if float(vol) < 0.0001 else "HIGH" if float(vol) > 0.0005 else "NORMAL"
        volatility.update(direction_score=0.0, confidence=0.5, regime=regime, feature_count=1, reason_codes=["ROLLING_VOLATILITY"], components={"m5_volatility_60m": float(vol)})
    else:
        volatility["reason_codes"] = ["VOLATILITY_INPUT_UNAVAILABLE"]
    outputs["volatility"] = volatility

    calendar = market_calendar(now)
    session = forex_session_state(now)
    session_output = _base("session", row, now, "AVAILABLE")
    session_output.update(direction_score=0.0, confidence=1.0, feature_count=3, regime=session["current_session"], reason_codes=[calendar["status"], session["current_session"]], components={"market_open": float(bool(calendar["market_open"])), "utc_hour": int(now.hour), "day_of_week": int(now.dayofweek)})
    outputs["session"] = session_output

    regime_output = _base("regime", row, now, "AVAILABLE" if returns else "PARTIAL")
    if returns:
        score = _bounded(float(np.mean(list(returns.values()))) / 0.001)
        regime_output.update(direction_score=score, confidence=min(1.0, len(returns) / 3), feature_count=len(returns), regime="TRENDING_UP" if score > 0.25 else "TRENDING_DOWN" if score < -0.25 else "RANGING", reason_codes=["CAUSAL_RETURN_REGIME"])
    else:
        regime_output["reason_codes"] = ["REGIME_INPUTS_INCOMPLETE"]
    outputs["regime"] = regime_output
    return outputs


def load_latest_frame(path: Path = FEATURE_FILE, rows: int = 256, as_of: object | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    columns = ["decision_timestamp_utc", "m5_return_5m", "m5_return_15m", "m5_return_60m", "m5_return_240m", "m5_volatility_60m", "m5_spread_points"]
    frame = pd.read_parquet(path, columns=columns)
    frame["decision_timestamp_utc"] = pd.to_datetime(frame["decision_timestamp_utc"], utc=True, errors="raise")
    if as_of is not None:
        frame = frame.loc[frame["decision_timestamp_utc"].le(_utc(as_of))]
    return frame.sort_values("decision_timestamp_utc").tail(rows).reset_index(drop=True)


def engine_status(now_utc: object | None = None, path: Path = FEATURE_FILE) -> dict[str, Any]:
    now = _utc(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    frame = load_latest_frame(path, as_of=now)
    if frame.empty:
        engines = {name: unavailable(name) for name in ("technical", "price_action", "structure", "liquidity", "volatility", "session", "regime")}
        return {"contract_version": ENGINE_CONTRACT_VERSION, "status": "UNAVAILABLE", "decision_timestamp_utc": None, "engines": engines, "external_engines": {name: unavailable(name) for name in ("fundamental", "news", "sentiment", "intermarket")}, "observational_only": True, "v06_integration": False}
    engines = _market_engine(frame, now)
    external = {name: unavailable(name, "NO_VERIFIED_PROVIDER") for name in ("fundamental", "news", "sentiment", "intermarket")}
    return {"contract_version": ENGINE_CONTRACT_VERSION, "status": "PASS", "decision_timestamp_utc": frame.iloc[-1]["decision_timestamp_utc"].isoformat(), "engines": engines, "external_engines": external, "observational_only": True, "v06_integration": False}
