"""Stable contracts for causal, observational market-structure analysis."""
from __future__ import annotations

from typing import Any, Literal, TypedDict

ENGINE_CONTRACT_VERSION = "v0.9a1-engine-output-v2"
EngineStatus = Literal["AVAILABLE", "PARTIAL", "UNAVAILABLE", "INVALID"]
PatternStatus = Literal["FORMING", "CONFIRMED", "BROKEN", "INVALIDATED"]
Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class SwingPoint(TypedDict):
    swing_id: str
    timeframe: str
    kind: Literal["HIGH", "LOW"]
    pivot_time_utc: str
    confirmed_at_utc: str
    price: float
    left_bars: int
    right_bars: int


class PatternRecord(TypedDict, total=False):
    pattern_id: str
    pattern_type: str
    timeframe: str
    detected_at_utc: str
    confirmed_at_utc: str | None
    status: PatternStatus
    direction: Direction
    score: float
    confidence: float
    start_time_utc: str
    end_time_utc: str
    upper_boundary: float | None
    lower_boundary: float | None
    breakout_level: float | None
    invalidation_level: float | None
    evidence: dict[str, Any]

class EngineOutput(TypedDict, total=False):
    engine_name: str
    contract_version: str
    decision_timestamp_utc: str | None
    status: EngineStatus
    direction_score: float | None
    confidence: float | None
    regime: str | None
    feature_count: int
    input_freshness: str
    reason_codes: list[str]
    components: dict[str, Any]


def unavailable(engine_name: str, reason: str = "INPUT_UNAVAILABLE") -> EngineOutput:
    return {
        "engine_name": engine_name, "contract_version": ENGINE_CONTRACT_VERSION,
        "decision_timestamp_utc": None, "status": "UNAVAILABLE",
        "direction_score": None, "confidence": None, "regime": None,
        "feature_count": 0, "input_freshness": "UNAVAILABLE",
        "reason_codes": [reason], "components": {},
    }
