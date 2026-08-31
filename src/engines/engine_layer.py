"""Causal V0.9A.1 observational engines over completed local market data."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.engines.contract import ENGINE_CONTRACT_VERSION, EngineOutput, unavailable
from src.engines.liquidity import liquidity_analysis
from src.engines.patterns import detect_patterns
from src.engines.price_action import breakout_events, candle_events
from src.engines.structure import classify_swings, structure_events, structure_state, support_resistance
from src.engines.swings import causal_bars, confirmed_swings, utc
from src.assets.contracts import normalize_asset_id
from src.marketdata.market_calendar import forex_session_state, market_calendar
from src.marketdata.v05a_contract import CONTINUOUS_DIR, FEATURE_FILE, TIMEFRAMES

ANALYSIS_TIMEFRAMES = ("M5", "M15", "H1")
BAR_WINDOW = 240
TIMEFRAME_LOAD_WINDOWS = {"M5": 600, "M15": 600, "H1": 400}


def _base(name: str, decision: object, now: pd.Timestamp, status: str = "AVAILABLE") -> EngineOutput:
    stamp = utc(decision)
    age = (now - stamp).total_seconds()
    freshness = "FRESH" if 0 <= age <= 360 else "STALE"
    return {
        "engine_name": name, "contract_version": ENGINE_CONTRACT_VERSION,
        "decision_timestamp_utc": stamp.isoformat(), "status": status,
        "direction_score": None, "confidence": None, "regime": None,
        "feature_count": 0, "input_freshness": freshness,
        "reason_codes": [], "components": {},
    }


def _bounded(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _direction_score(values: list[str]) -> float:
    numeric = [1.0 if value == "BULLISH" else -1.0 if value == "BEARISH" else 0.0 for value in values]
    return float(np.mean(numeric)) if numeric else 0.0


@lru_cache(maxsize=12)
def _cached_parquet(path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    columns = ["bar_open_utc", "bar_close_utc", "open", "high", "low", "close", "tick_volume", "spread_points", "first_observed_utc"]
    return pd.read_parquet(path_text, columns=columns)


def load_completed_bars(timeframe: str, *, root: Path = CONTINUOUS_DIR, symbol: str = "EURUSD", as_of: object | None = None, rows: int = BAR_WINDOW, observed_as_of: object | None = None) -> pd.DataFrame:
    if timeframe not in ANALYSIS_TIMEFRAMES:
        raise ValueError(f"Unsupported analysis timeframe: {timeframe}")
    asset = normalize_asset_id(symbol)
    path = root / (TIMEFRAMES[timeframe].filename if asset == "EURUSD" else f"{asset}_{timeframe}.parquet")
    if not path.exists():
        return pd.DataFrame()
    frame = _cached_parquet(str(path.resolve()), path.stat().st_mtime_ns).copy()
    return causal_bars(frame, as_of=as_of, limit=rows, observed_as_of=observed_as_of)


def load_latest_frame(path: Path = FEATURE_FILE, rows: int = 256, as_of: object | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    columns = ["decision_timestamp_utc", "m5_return_5m", "m5_return_15m", "m5_return_60m", "m5_return_240m", "m5_volatility_60m", "m5_spread_points"]
    frame = pd.read_parquet(path, columns=columns)
    frame["decision_timestamp_utc"] = pd.to_datetime(frame["decision_timestamp_utc"], utc=True, errors="raise")
    if as_of is not None:
        frame = frame.loc[frame["decision_timestamp_utc"].le(utc(as_of))]
    return frame.sort_values("decision_timestamp_utc").tail(rows).reset_index(drop=True)


def _legacy_outputs(frame: pd.DataFrame, now: pd.Timestamp) -> dict[str, EngineOutput]:
    row = frame.iloc[-1]
    decision = row["decision_timestamp_utc"]
    returns = {name: float(row[name]) for name in ("m5_return_15m", "m5_return_60m", "m5_return_240m") if pd.notna(row.get(name))}
    outputs: dict[str, EngineOutput] = {}
    technical = _base("technical", decision, now, "AVAILABLE" if returns else "PARTIAL")
    trend = float(np.mean([np.sign(value) for value in returns.values()])) if returns else 0.0
    technical.update(direction_score=_bounded(trend), confidence=min(1.0, len(returns) / 3), feature_count=len(returns), reason_codes=["RETURN_ALIGNMENT"] if returns else ["TECHNICAL_INPUTS_INCOMPLETE"], components={"return_alignment": trend})
    outputs["technical"] = technical
    price = _base("price_action", decision, now, "AVAILABLE" if pd.notna(row.get("m5_return_5m")) else "PARTIAL")
    short_return = float(row.get("m5_return_5m", np.nan))
    if np.isfinite(short_return):
        price.update(direction_score=_bounded(short_return / 0.001), confidence=0.5, feature_count=1, reason_codes=["SHORT_HORIZON_RETURN_FALLBACK"], components={"m5_return_5m": short_return})
    else:
        price["reason_codes"] = ["PRICE_ACTION_INPUT_UNAVAILABLE"]
    outputs["price_action"] = price
    structure = _base("structure", decision, now, "AVAILABLE" if len(returns) >= 2 else "PARTIAL")
    if returns:
        score = _bounded(float(np.mean(list(returns.values()))) / 0.001)
        structure.update(direction_score=score, confidence=min(1.0, len(returns) / 3), feature_count=len(returns), regime="TREND_UP" if score > 0.25 else "TREND_DOWN" if score < -0.25 else "RANGE", reason_codes=["MULTI_HORIZON_DIRECTION_FALLBACK"])
    else:
        structure["reason_codes"] = ["STRUCTURE_INPUTS_INCOMPLETE"]
    outputs["structure"] = structure
    liquidity = _base("liquidity", decision, now, "PARTIAL")
    spread = row.get("m5_spread_points")
    if pd.notna(spread):
        liquidity.update(direction_score=0.0, confidence=0.25, feature_count=1, reason_codes=["SPREAD_CONTEXT_FALLBACK"], components={"spread_points": float(spread)})
    else:
        liquidity["reason_codes"] = ["LIQUIDITY_INPUT_UNAVAILABLE"]
    outputs["liquidity"] = liquidity
    volatility = _base("volatility", decision, now, "AVAILABLE" if pd.notna(row.get("m5_volatility_60m")) else "PARTIAL")
    vol = row.get("m5_volatility_60m")
    if pd.notna(vol):
        regime = "LOW" if float(vol) < 0.0001 else "HIGH" if float(vol) > 0.0005 else "NORMAL"
        volatility.update(direction_score=0.0, confidence=0.5, regime=regime, feature_count=1, reason_codes=["ROLLING_VOLATILITY"], components={"m5_volatility_60m": float(vol)})
    else:
        volatility["reason_codes"] = ["VOLATILITY_INPUT_UNAVAILABLE"]
    outputs["volatility"] = volatility
    calendar = market_calendar(now)
    session = forex_session_state(now)
    session_output = _base("session", decision, now)
    session_output.update(direction_score=0.0, confidence=1.0, feature_count=3, regime=session["current_session"], reason_codes=[calendar["status"], session["current_session"]], components={"market_open": bool(calendar["market_open"]), "utc_hour": int(now.hour), "day_of_week": int(now.dayofweek)})
    outputs["session"] = session_output
    regime_output = _base("regime", decision, now, "AVAILABLE" if returns else "PARTIAL")
    if returns:
        score = _bounded(float(np.mean(list(returns.values()))) / 0.001)
        regime_output.update(direction_score=score, confidence=min(1.0, len(returns) / 3), feature_count=len(returns), regime="TRENDING_UP" if score > 0.25 else "TRENDING_DOWN" if score < -0.25 else "RANGING", reason_codes=["CAUSAL_RETURN_REGIME"])
    else:
        regime_output["reason_codes"] = ["REGIME_INPUTS_INCOMPLETE"]
    outputs["regime"] = regime_output
    return outputs


def _analyse_timeframe(frame: pd.DataFrame, timeframe: str, as_of: pd.Timestamp) -> dict[str, Any]:
    swings = confirmed_swings(frame, timeframe, as_of=as_of, limit=BAR_WINDOW)
    levels = support_resistance(frame, timeframe, swings, as_of=as_of, limit=len(frame))
    patterns = detect_patterns(frame, timeframe, swings, as_of=as_of, limit=BAR_WINDOW)
    candles = candle_events(frame, timeframe, as_of=as_of, limit=40)
    market_structure = structure_events(frame, timeframe, swings, as_of=as_of, limit=BAR_WINDOW)
    liquidity = liquidity_analysis(frame, timeframe, swings, as_of=as_of, limit=BAR_WINDOW)
    breakouts = breakout_events(frame, timeframe, levels["levels"], as_of=as_of, limit=160)
    classified = classify_swings(swings, tolerance=float(levels.get("tolerance") or 0.0))
    return {
        "timeframe": timeframe, "bar_count": len(frame),
        "latest_completed_bar_utc": pd.Timestamp(frame.iloc[-1]["bar_close_utc"]).isoformat(),
        "swings": classified[-12:], "structure_state": structure_state(classified),
        "structure_events": market_structure[-10:], "patterns": patterns[-12:],
        "support_resistance": levels, "price_action_events": [*candles, *breakouts][-15:],
        "liquidity": liquidity,
    }


@lru_cache(maxsize=24)
def _cached_timeframe_analysis(timeframe: str, root_text: str, symbol: str, modified_ns: int, decision_iso: str, observed_iso: str) -> dict[str, Any]:
    del modified_ns
    bars = load_completed_bars(timeframe, root=Path(root_text), symbol=symbol, as_of=decision_iso, rows=TIMEFRAME_LOAD_WINDOWS[timeframe], observed_as_of=observed_iso)
    return _analyse_timeframe(bars, timeframe, utc(decision_iso))


def _advanced_outputs(analyses: dict[str, dict[str, Any]], decision: object, now: pd.Timestamp) -> dict[str, EngineOutput]:
    available = list(analyses.values())
    if not available:
        return {name: unavailable(name, "COMPLETED_BAR_STORE_UNAVAILABLE") for name in ("chart_patterns", "support_resistance")}
    states = [item["structure_state"] for item in available]
    state_directions = ["BULLISH" if item == "TREND_UP" else "BEARISH" if item == "TREND_DOWN" else "NEUTRAL" for item in states]
    structure = _base("structure", decision, now)
    structure.update(direction_score=_direction_score(state_directions), confidence=min(1.0, sum(len(item["swings"]) for item in available) / 24), regime=analyses.get("M5", available[0])["structure_state"], feature_count=sum(len(item["swings"]) for item in available), reason_codes=["CONFIRMED_SWING_STRUCTURE", "BOS_CHOCH_CLOSE_CONFIRMATION"], components={tf: {"state": item["structure_state"], "latest_event": item["structure_events"][-1] if item["structure_events"] else None} for tf, item in analyses.items()})
    directions = [event["direction"] for item in available for event in item["price_action_events"][-3:]]
    price = _base("price_action", decision, now)
    price.update(direction_score=_direction_score(directions), confidence=min(1.0, len(directions) / 6), feature_count=sum(len(item["price_action_events"]) for item in available), reason_codes=["NUMERIC_CANDLE_GEOMETRY", "CAUSAL_BREAKOUT_TRACKING"], components={tf: {"latest_events": item["price_action_events"][-5:]} for tf, item in analyses.items()})
    liquidity_directions = [event["direction"] for item in available for event in item["liquidity"]["sweeps"][-3:]]
    liquidity = _base("liquidity", decision, now)
    liquidity.update(direction_score=_direction_score(liquidity_directions), confidence=min(1.0, len(liquidity_directions) / 4), feature_count=sum(len(item["liquidity"]["equal_levels"]) + len(item["liquidity"]["fair_value_gaps"]) for item in available), reason_codes=["EQUAL_EXTREMA_ATR_SPREAD_TOLERANCE", "MECHANICAL_THREE_BAR_FVG"], components={tf: {"nearest": item["liquidity"]["nearest_liquidity"], "latest_sweep": item["liquidity"]["sweeps"][-1] if item["liquidity"]["sweeps"] else None, "open_fvg_count": sum(gap["status"] != "FILLED" for gap in item["liquidity"]["fair_value_gaps"])} for tf, item in analyses.items()})
    pattern_items = [pattern for item in available for pattern in item["patterns"]]
    pattern = _base("chart_patterns", decision, now, "AVAILABLE" if pattern_items else "PARTIAL")
    pattern.update(direction_score=_direction_score([item["direction"] for item in pattern_items[-10:]]), confidence=float(np.mean([item["score"] for item in pattern_items[-10:]])) if pattern_items else 0.0, feature_count=len(pattern_items), reason_codes=["DETERMINISTIC_NUMERIC_GEOMETRY"] if pattern_items else ["NO_OBJECTIVE_PATTERN_DETECTED"], components={"recent": sorted(pattern_items, key=lambda item: item["detected_at_utc"], reverse=True)[:10], "score_semantics": "STRUCTURAL_GEOMETRY_FIT_NOT_TRADING_PROBABILITY"})
    sr = _base("support_resistance", decision, now)
    sr.update(direction_score=0.0, confidence=min(1.0, sum(len(item["support_resistance"]["levels"]) for item in available) / 20), feature_count=sum(len(item["support_resistance"]["levels"]) for item in available), reason_codes=["CONFIRMED_SWINGS_AND_COMPLETED_PERIOD_LEVELS", "EMA_AND_FIBONACCI_OBSERVATIONAL_ONLY"], components={tf: {"nearest_support": item["support_resistance"]["nearest_support"], "nearest_resistance": item["support_resistance"]["nearest_resistance"], "dynamic": item["support_resistance"]["dynamic"], "fibonacci": item["support_resistance"]["fibonacci"]} for tf, item in analyses.items()})
    return {"price_action": price, "structure": structure, "liquidity": liquidity, "chart_patterns": pattern, "support_resistance": sr}


def engine_status(now_utc: object | None = None, path: Path = FEATURE_FILE, bar_root: Path = CONTINUOUS_DIR, decision_as_of: object | None = None, symbol: str = "EURUSD") -> dict[str, Any]:
    asset = normalize_asset_id(symbol)
    now = utc(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    decision_cutoff = utc(decision_as_of) if decision_as_of is not None else now
    feature_frame = load_latest_frame(path, as_of=decision_cutoff)
    if feature_frame.empty:
        names = ("technical", "price_action", "structure", "liquidity", "volatility", "session", "regime", "chart_patterns", "support_resistance")
        engines = {name: unavailable(name) for name in names}
        return {"contract_version": ENGINE_CONTRACT_VERSION, "status": "UNAVAILABLE", "decision_timestamp_utc": None, "engines": engines, "external_engines": {name: unavailable(name) for name in ("fundamental", "news", "sentiment", "intermarket")}, "timeframes": {}, "recent_patterns": [], "event_history": [], "timeframe_availability": {"H4": {"status": "UNAVAILABLE", "reason": "NO_VERIFIED_COMPLETED_H4_STORE"}}, "not_implemented_features": {"order_blocks": {"status": "UNAVAILABLE", "reason": "NO_OBJECTIVE_DETERMINISTIC_DEFINITION"}}, "observational_only": True, "v06_integration": False, "trading_enabled": False}
    decision = utc(feature_frame.iloc[-1]["decision_timestamp_utc"])
    analyses: dict[str, dict[str, Any]] = {}
    # A caller supplying a synthetic feature fixture must explicitly supply its
    # matching bar root; never blend test/research features with live bars.
    use_bars = path.resolve() == FEATURE_FILE.resolve() or bar_root.resolve() != CONTINUOUS_DIR.resolve()
    if use_bars:
        for timeframe in ANALYSIS_TIMEFRAMES:
            bar_path = bar_root / (TIMEFRAMES[timeframe].filename if asset == "EURUSD" else f"{asset}_{timeframe}.parquet")
            if bar_path.exists():
                modified_ns = bar_path.stat().st_mtime_ns
                modified = pd.Timestamp(modified_ns, unit="ns", tz="UTC")
                observed_cutoff = min(now, modified)
                analyses[timeframe] = _cached_timeframe_analysis(timeframe, str(bar_root.resolve()), asset, modified_ns, decision.isoformat(), observed_cutoff.isoformat())
    engines = _legacy_outputs(feature_frame, now)
    engines.update(_advanced_outputs(analyses, decision, now))
    external = {name: unavailable(name, "NO_VERIFIED_PROVIDER") for name in ("fundamental", "news", "sentiment", "intermarket")}
    patterns = sorted([pattern for item in analyses.values() for pattern in item["patterns"]], key=lambda item: item["detected_at_utc"], reverse=True)
    history: list[dict[str, Any]] = []
    for item in analyses.values():
        history.extend(item["structure_events"])
        history.extend(item["price_action_events"])
        history.extend(item["liquidity"]["sweeps"])
        history.extend(item["liquidity"]["fair_value_gaps"])
    history = sorted(history, key=lambda item: item["detected_at_utc"], reverse=True)[:30]
    nonzero = {name: float(item.get("direction_score") or 0.0) for name, item in engines.items() if name not in {"session", "volatility", "support_resistance"} and abs(float(item.get("direction_score") or 0.0)) > 0.1}
    signs = {int(np.sign(value)) for value in nonzero.values()}
    agreement = "MIXED" if len(signs) > 1 else "BULLISH_ALIGNMENT" if signs == {1} else "BEARISH_ALIGNMENT" if signs == {-1} else "NO_DIRECTIONAL_ALIGNMENT"
    calendar = market_calendar(now)
    return {
        "contract_version": ENGINE_CONTRACT_VERSION, "status": "PASS", "asset_id": asset, "symbol": asset,
        "decision_timestamp_utc": decision.isoformat(), "market_status": calendar["status"],
        "data_freshness": engines["technical"]["input_freshness"], "engines": engines,
        "external_engines": external, "timeframes": analyses,
        "recent_patterns": patterns[:20], "event_history": history,
        "timeframe_availability": {**{name: {"status": "AVAILABLE" if name in analyses else "UNAVAILABLE"} for name in ANALYSIS_TIMEFRAMES}, "H4": {"status": "UNAVAILABLE", "reason": "NO_VERIFIED_COMPLETED_H4_STORE"}},
        "not_implemented_features": {"order_blocks": {"status": "UNAVAILABLE", "reason": "NO_OBJECTIVE_DETERMINISTIC_DEFINITION"}},
        "informational_agreement": {"status": agreement, "scores": nonzero, "conflict_resolution": "NONE"},
        "observational_only": True, "v06_integration": False, "trading_enabled": False,
    }


def recent_patterns(limit: int = 20, now_utc: object | None = None, symbol: str = "EURUSD", path: Path = FEATURE_FILE, bar_root: Path = CONTINUOUS_DIR) -> dict[str, Any]:
    bounded = max(1, min(int(limit), 100))
    payload = engine_status(now_utc=now_utc, symbol=symbol, path=path, bar_root=bar_root)
    items = payload.get("recent_patterns", [])[:bounded]
    return {"contract_version": ENGINE_CONTRACT_VERSION, "status": payload["status"], "count": len(items), "limit": bounded, "items": items, "observational_only": True, "trading_enabled": False}


def decision_engine_snapshot(decision_utc: object, observed_at_utc: object | None = None, *, symbol: str = "EURUSD", path: Path = FEATURE_FILE, bar_root: Path = CONTINUOUS_DIR) -> dict[str, Any]:
    """Compact observational context suitable for an immutable V0.8 row."""
    observed = pd.Timestamp.now(tz="UTC") if observed_at_utc is None else utc(observed_at_utc)
    payload = engine_status(now_utc=observed, decision_as_of=decision_utc, symbol=symbol, path=path, bar_root=bar_root)
    compact_engines = {}
    for name in ("technical", "price_action", "structure", "liquidity", "regime"):
        item = payload.get("engines", {}).get(name, {})
        compact_engines[name] = {key: item.get(key) for key in ("status", "direction_score", "confidence", "regime", "reason_codes")}
    pattern = next(iter(payload.get("recent_patterns") or []), None)
    m5 = (payload.get("timeframes") or {}).get("M5") or {}
    levels = m5.get("support_resistance") or {}
    structure_event = next(iter(reversed(m5.get("structure_events") or [])), None)
    liquidity = m5.get("liquidity") or {}
    liquidity_event = next(iter(reversed(liquidity.get("sweeps") or [])), None)
    if liquidity_event is None:
        liquidity_event = next((item for item in reversed(liquidity.get("fair_value_gaps") or []) if item.get("status") != "FILLED"), None)
    support = levels.get("nearest_support") or {}
    resistance = levels.get("nearest_resistance") or {}
    return {
        "contract_version": ENGINE_CONTRACT_VERSION, "decision_timestamp_utc": payload.get("decision_timestamp_utc"),
        "engines": compact_engines,
        "chart_pattern": None if pattern is None else {key: pattern.get(key) for key in ("pattern_id", "pattern_type", "timeframe", "status", "direction", "score", "detected_at_utc")},
        "nearest_support_distance_price": support.get("distance_price"),
        "nearest_resistance_distance_price": resistance.get("distance_price"),
        "structure_event": None if structure_event is None else {key: structure_event.get(key) for key in ("event_id", "event_type", "timeframe", "direction", "detected_at_utc")},
        "liquidity_event": None if liquidity_event is None else {key: liquidity_event.get(key) for key in ("event_id", "event_type", "timeframe", "direction", "status", "detected_at_utc")},
        "observational_only": True, "v06_integration": False,
    }
