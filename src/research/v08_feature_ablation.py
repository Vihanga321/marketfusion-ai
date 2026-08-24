"""Deterministic MARKET_CORE feature groups for predefined ablation research."""
from __future__ import annotations

from src.learning.v05c_contract import MARKET_FEATURES

FEATURE_GROUPS = {
    "PRICE_RETURN": tuple(name for name in MARKET_FEATURES if "return" in name),
    "VOLATILITY": tuple(name for name in MARKET_FEATURES if "volatility" in name),
    "SPREAD_LIQUIDITY": tuple(name for name in MARKET_FEATURES if "spread" in name),
    "VOLUME": tuple(name for name in MARKET_FEATURES if "volume" in name),
    "SESSION_TIME": tuple(name for name in MARKET_FEATURES if name.startswith(("utc_", "day_", "is_"))),
}


def ablated_features(group: str) -> tuple[str, ...]:
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown predefined feature group: {group}")
    removed = set(FEATURE_GROUPS[group])
    return tuple(name for name in MARKET_FEATURES if name not in removed)


def validate_groups() -> None:
    if any(not fields for fields in FEATURE_GROUPS.values()):
        raise RuntimeError("Every predefined V0.8 feature-ablation group must be non-empty")
    flattened = [name for fields in FEATURE_GROUPS.values() for name in fields]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("V0.8 feature-ablation groups must be deterministic and non-overlapping")
