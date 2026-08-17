"""Deterministic, time-causal analogue retrieval for V0.4B event memory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from retrieval_feature_registry import (
        FEATURES, RetrievalFeature, approved_feature_names, validate_registry,
    )
    from v04b_contract import ROOT
except ImportError:  # pragma: no cover
    from src.memory.retrieval_feature_registry import (
        FEATURES, RetrievalFeature, approved_feature_names, validate_registry,
    )
    from src.memory.v04b_contract import ROOT


MEMORY_PATH = ROOT / "data" / "processed" / "v04b_historical_event_memory.parquet"
DEFAULT_K = 5
HORIZONS = ("1m", "5m", "15m", "60m", "240m")


@dataclass(frozen=True)
class Scale:
    center: float
    denominator: float


def _finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def candidate_history_scales(
    candidates: pd.DataFrame,
    features: tuple[RetrievalFeature, ...] = FEATURES,
) -> dict[str, Scale]:
    """Fit robust scales on the candidate/history set, never on the query."""
    scales: dict[str, Scale] = {}
    for feature in features:
        if feature.name not in candidates:
            raise ValueError(f"Candidate memory lacks registered feature: {feature.name}")
        values = pd.to_numeric(candidates[feature.name], errors="coerce")
        finite = values[np.isfinite(values)]
        if feature.required and len(finite) != len(candidates):
            raise ValueError(f"Required candidate context is missing/non-finite: {feature.name}")
        if finite.empty:
            if feature.required:
                raise ValueError(f"No finite candidate history for required feature: {feature.name}")
            continue
        center = float(finite.median())
        q1, q3 = finite.quantile([0.25, 0.75])
        denominator = float(q3 - q1)
        if not np.isfinite(denominator) or denominator <= 0:
            denominator = float(finite.std(ddof=0))
        if not np.isfinite(denominator) or denominator <= 0:
            denominator = 1.0
        scales[feature.name] = Scale(center=center, denominator=denominator)
    return scales


def _context_only_ranking(
    query: pd.Series,
    candidates: pd.DataFrame,
    features: tuple[RetrievalFeature, ...] = FEATURES,
) -> pd.DataFrame:
    """Rank with registry-approved context only; no outcome column is read here."""
    validate_registry()
    if tuple(feature.name for feature in features) != approved_feature_names():
        raise ValueError("Ranking features must exactly match the approved retrieval registry")
    if any(name.startswith("outcome_") for name in candidates.columns) or any(
        str(name).startswith("outcome_") for name in query.index
    ):
        raise ValueError("Historical outcomes must not cross the retrieval-engine boundary")
    scales = candidate_history_scales(candidates, features)
    query_values: dict[str, float] = {}
    for feature in features:
        value = query.get(feature.name, np.nan)
        if feature.required and not _finite_number(value):
            raise ValueError(f"Required query context is missing/non-finite: {feature.name}")
        query_values[feature.name] = float(value) if _finite_number(value) else np.nan

    ranking_rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        squared = 0.0
        used_weight = 0.0
        differences: dict[str, dict[str, float]] = {}
        for feature in features:
            scale = scales.get(feature.name)
            query_value = query_values[feature.name]
            candidate_value = getattr(candidate, feature.name)
            if scale is None or not _finite_number(query_value) or not _finite_number(candidate_value):
                if feature.required:
                    raise ValueError(f"Required retrieval value unavailable: {feature.name}")
                continue
            raw_difference = query_value - float(candidate_value)
            normalized_difference = raw_difference / scale.denominator
            squared += feature.weight * normalized_difference * normalized_difference
            used_weight += feature.weight
            differences[feature.name] = {
                "query": query_value,
                "candidate": float(candidate_value),
                "difference": raw_difference,
                "normalized_difference": normalized_difference,
            }
        if used_weight <= 0:
            raise ValueError("No jointly available retrieval context features")
        distance = math.sqrt(squared / used_weight)
        if not np.isfinite(distance):
            raise ValueError("Non-finite analogue distance")
        ranking_rows.append(
            {
                "event_id": candidate.event_id,
                "event_timestamp_utc": candidate.event_timestamp_utc,
                "distance": distance,
                "similarity_score": 1.0 / (1.0 + distance),
                "matched_context": differences,
                "features_used": len(differences),
                "weight_used": used_weight,
            }
        )
    return (
        pd.DataFrame(ranking_rows)
        .sort_values(["distance", "event_timestamp_utc", "event_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def eligible_candidates(
    memory: pd.DataFrame,
    query: pd.Series,
    same_event_type: bool = True,
) -> pd.DataFrame:
    query_time = pd.Timestamp(query["event_timestamp_utc"])
    if query_time.tzinfo is None:
        raise ValueError("Query timestamp must be timezone-aware")
    timestamps = pd.to_datetime(memory["event_timestamp_utc"], utc=True, errors="raise")
    mask = timestamps < query_time.tz_convert("UTC")
    if same_event_type:
        mask &= memory["event_type"].eq(query["event_type"])
    candidates = memory.loc[mask].copy()
    if query["event_id"] in set(candidates["event_id"]):
        raise AssertionError("Query event entered its own candidate set")
    if not (pd.to_datetime(candidates["event_timestamp_utc"], utc=True) < query_time).all():
        raise AssertionError("Future event entered candidate set")
    return candidates


def retrieval_input_view(memory: pd.DataFrame) -> pd.DataFrame:
    """Physically isolate identity plus registry-approved context from outcomes."""
    validate_registry()
    identity = ["event_id", "event_type", "event_timestamp_utc"]
    columns = identity + list(approved_feature_names())
    missing = [column for column in columns if column not in memory]
    if missing:
        raise ValueError(f"Memory lacks retrieval input columns: {missing}")
    view = memory.loc[:, columns].copy()
    if any(column.startswith("outcome_") for column in view):
        raise AssertionError("Outcome field entered retrieval input view")
    return view


def _outcome_payload(row: pd.Series) -> dict[str, Any]:
    """Attach outcomes only after context ranking has been frozen."""
    payload: dict[str, Any] = {}
    for column, value in row.items():
        if not column.startswith("outcome_"):
            continue
        if isinstance(value, pd.Timestamp):
            payload[column.removeprefix("outcome_")] = value.isoformat()
        elif isinstance(value, np.generic):
            payload[column.removeprefix("outcome_")] = value.item()
        else:
            payload[column.removeprefix("outcome_")] = value
    return payload


def retrieve_analogues(
    memory: pd.DataFrame,
    query: pd.Series,
    k: int = DEFAULT_K,
    same_event_type: bool = True,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    retrieval_memory = retrieval_input_view(memory)
    query_context = retrieval_input_view(query.to_frame().T).iloc[0]
    candidates = eligible_candidates(retrieval_memory, query_context, same_event_type=same_event_type)
    if candidates.empty:
        raise ValueError("No strictly earlier candidate events")
    ranking = _context_only_ranking(query_context, candidates).head(k).copy()
    frozen_ids = ranking["event_id"].tolist()
    lookup = memory.set_index("event_id", drop=False)
    analogues = []
    for rank, ranked in enumerate(ranking.itertuples(index=False), start=1):
        source = lookup.loc[ranked.event_id]
        analogues.append(
            {
                "rank": rank,
                "event_id": ranked.event_id,
                "timestamp": pd.Timestamp(ranked.event_timestamp_utc).isoformat(),
                "distance": float(ranked.distance),
                "similarity_score": float(ranked.similarity_score),
                "features_used": int(ranked.features_used),
                "matched_context": ranked.matched_context,
                "historical_outcomes": _outcome_payload(source),
            }
        )
    if frozen_ids != [analogue["event_id"] for analogue in analogues]:
        raise AssertionError("Outcome attachment changed the frozen ranking")

    selected = lookup.loc[frozen_ids]
    aggregate: dict[str, float] = {}
    for horizon in HORIZONS:
        reactions = pd.to_numeric(selected[f"outcome_reaction_{horizon}_pips"], errors="raise")
        aggregate[f"median_{horizon}_pips"] = float(reactions.median())
        aggregate[f"up_fraction_{horizon}"] = float((reactions > 0).mean())
        aggregate[f"dispersion_{horizon}_pips"] = float(reactions.std(ddof=0))

    context_summary = {
        feature.name: (
            float(query_context[feature.name])
            if _finite_number(query_context.get(feature.name)) else None
        )
        for feature in FEATURES
    }
    return {
        "query_event_id": query["event_id"],
        "query_timestamp": pd.Timestamp(query["event_timestamp_utc"]).isoformat(),
        "event_type": query["event_type"],
        "candidate_count": len(candidates),
        "top_k": min(k, len(candidates)),
        "retrieval_context_summary": context_summary,
        "analogues": analogues,
        "aggregate_historical_response": aggregate,
        "interpretation_warning": "Empirical historical analogue statistics; not trade probabilities.",
    }


def load_memory(path: Path = MEMORY_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True)
    return frame.sort_values(["event_timestamp_utc", "event_id"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    memory = load_memory()
    matches = memory[memory["event_id"].eq(args.event_id)]
    if len(matches) != 1:
        raise SystemExit(f"Expected one query event, found {len(matches)}: {args.event_id}")
    print(json.dumps(retrieve_analogues(memory, matches.iloc[0], args.k), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
