"""Stable output contract for observational engine analysis."""
from __future__ import annotations

from typing import Literal, TypedDict

ENGINE_CONTRACT_VERSION = "v0.9a-engine-output-v1"
EngineStatus = Literal["AVAILABLE", "PARTIAL", "UNAVAILABLE", "INVALID"]

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
    components: dict[str, float | str | None]


def unavailable(engine_name: str, reason: str = "INPUT_UNAVAILABLE") -> EngineOutput:
    return {
        "engine_name": engine_name, "contract_version": ENGINE_CONTRACT_VERSION,
        "decision_timestamp_utc": None, "status": "UNAVAILABLE",
        "direction_score": None, "confidence": None, "regime": None,
        "feature_count": 0, "input_freshness": "UNAVAILABLE",
        "reason_codes": [reason], "components": {},
    }
