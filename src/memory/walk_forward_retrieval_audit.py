"""Offline time-causal audit of the V0.4B analogue-retrieval baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from analogue_retrieval import DEFAULT_K, HORIZONS, load_memory, retrieve_analogues
    from v04b_contract import ROOT
except ImportError:  # pragma: no cover
    from src.memory.analogue_retrieval import DEFAULT_K, HORIZONS, load_memory, retrieve_analogues
    from src.memory.v04b_contract import ROOT


AUDIT_REPORT = ROOT / "reports" / "v04b_walk_forward_retrieval_audit.csv"
EXAMPLES_REPORT = ROOT / "reports" / "v04b_analogue_examples.txt"
DEFAULT_WARMUP = 10


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def run_walk_forward_audit(
    memory: pd.DataFrame,
    k: int = DEFAULT_K,
    warmup: int = DEFAULT_WARMUP,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    ordered = memory.sort_values(["event_timestamp_utc", "event_id"]).reset_index(drop=True)
    seen: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    for _, query in ordered.iterrows():
        event_type = str(query["event_type"])
        prior_count = seen.get(event_type, 0)
        seen[event_type] = prior_count + 1
        if prior_count < warmup:
            continue

        # Retrieval is completed and frozen before this query's outcome is read.
        response = retrieve_analogues(ordered, query, k=k, same_event_type=True)
        prior = ordered[
            (ordered["event_type"] == event_type)
            & (ordered["event_timestamp_utc"] < query["event_timestamp_utc"])
        ].sort_values("event_timestamp_utc")
        if len(prior) != response["candidate_count"]:
            raise AssertionError("Candidate-count audit mismatch")
        analogue_times = [pd.Timestamp(item["timestamp"]) for item in response["analogues"]]
        if not all(timestamp < query["event_timestamp_utc"] for timestamp in analogue_times):
            raise AssertionError("Time-causality failure in walk-forward audit")

        row: dict[str, object] = {
            "query_event_id": query["event_id"],
            "query_event_type": event_type,
            "query_timestamp_utc": query["event_timestamp_utc"],
            "candidate_count": response["candidate_count"],
            "top_k": response["top_k"],
            "analogue_event_ids": json.dumps([item["event_id"] for item in response["analogues"]]),
            "analogue_timestamps_utc": json.dumps([item["timestamp"] for item in response["analogues"]]),
            "analogue_distances": json.dumps([item["distance"] for item in response["analogues"]]),
        }
        for horizon in HORIZONS:
            outcome_column = f"outcome_reaction_{horizon}_pips"
            actual = float(query[outcome_column])
            same_type_median = float(prior[outcome_column].median())
            recent_median = float(prior.tail(k)[outcome_column].median())
            analogue_median = float(
                response["aggregate_historical_response"][f"median_{horizon}_pips"]
            )
            row.update(
                {
                    f"query_actual_{horizon}_pips": actual,
                    f"analogue_median_{horizon}_pips": analogue_median,
                    f"analogue_up_fraction_{horizon}": response["aggregate_historical_response"][f"up_fraction_{horizon}"],
                    f"analogue_dispersion_{horizon}_pips": response["aggregate_historical_response"][f"dispersion_{horizon}_pips"],
                    f"same_type_historical_median_{horizon}_pips": same_type_median,
                    f"same_type_historical_up_fraction_{horizon}": float((prior[outcome_column] > 0).mean()),
                    f"recent_{k}_same_type_median_{horizon}_pips": recent_median,
                    f"analogue_direction_correct_{horizon}": _sign(analogue_median) == _sign(actual),
                    f"same_type_direction_correct_{horizon}": _sign(same_type_median) == _sign(actual),
                    f"recent_direction_correct_{horizon}": _sign(recent_median) == _sign(actual),
                }
            )
        rows.append(row)
        responses.append(response)
    return pd.DataFrame(rows), responses


def _descriptive_summary(audit: pd.DataFrame, k: int, warmup: int) -> list[str]:
    lines = [
        "WALK-FORWARD DESCRIPTIVE COMPARISON",
        f"queries: {len(audit)}",
        f"same_event_type_warmup: {warmup}",
        f"top_k: {k}",
        "Interpretation: directional hit rates are descriptive research diagnostics, not evidence of edge.",
    ]
    for horizon in HORIZONS:
        lines.append(
            f"{horizon}: analogue={audit[f'analogue_direction_correct_{horizon}'].mean():.3f}; "
            f"same-type-unconditional={audit[f'same_type_direction_correct_{horizon}'].mean():.3f}; "
            f"recent-{k}={audit[f'recent_direction_correct_{horizon}'].mean():.3f}"
        )
    return lines


def write_examples(
    audit: pd.DataFrame,
    responses: list[dict[str, object]],
    k: int,
    warmup: int,
) -> None:
    response_by_id = {response["query_event_id"]: response for response in responses}
    candidates = audit.copy()
    candidates["year"] = pd.to_datetime(candidates["query_timestamp_utc"], utc=True).dt.year
    chosen = []
    for event_type, year in (
        ("us_cpi_release", 2018),
        ("us_employment_situation", 2020),
        ("us_cpi_release", 2023),
        ("us_employment_situation", 2026),
    ):
        matches = candidates[(candidates["query_event_type"] == event_type) & (candidates["year"] == year)]
        if not matches.empty:
            chosen.append(matches.iloc[len(matches) // 2])

    lines = [
        "MARKETFUSION V0.4B ANALOGUE EXAMPLES",
        "Method: same-event-type weighted Euclidean distance over the explicit retrieval registry.",
        "Normalization: candidate-history median/IQR only; standard deviation fallback; query excluded.",
        "Ranking explanations below use context dimensions only. Query outcomes are not used.",
        "Historical outcomes are attached only after each ranking is frozen.",
        "Analogue aggregates are empirical statistics, not trade probabilities.",
        "",
        *_descriptive_summary(audit, k, warmup),
        "",
    ]
    for query_row in chosen:
        response = response_by_id[query_row["query_event_id"]]
        lines.extend(
            [
                f"QUERY: {response['query_event_id']}",
                f"timestamp: {response['query_timestamp']}",
                f"event_type: {response['event_type']}",
                f"candidate_count: {response['candidate_count']}",
            ]
        )
        for analogue in response["analogues"]:
            dimensions = sorted(
                analogue["matched_context"].items(),
                key=lambda item: (abs(item[1]["normalized_difference"]), item[0]),
            )[:5]
            why = ", ".join(
                f"{name} delta_z={values['normalized_difference']:.3f}" for name, values in dimensions
            )
            outcomes = analogue["historical_outcomes"]
            reaction_text = ", ".join(
                f"{horizon}={float(outcomes[f'reaction_{horizon}_pips']):.2f}p"
                for horizon in HORIZONS
            )
            lines.append(
                f"  #{analogue['rank']} {analogue['event_id']} | {analogue['timestamp']} | "
                f"distance={analogue['distance']:.4f} | closest context: {why}"
            )
            lines.append(f"     historical outcomes (post-ranking): {reaction_text}")
        lines.append("")
    _atomic_text("\n".join(lines), EXAMPLES_REPORT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args()
    memory = load_memory()
    audit, responses = run_walk_forward_audit(memory, k=args.k, warmup=args.warmup)
    if audit.empty:
        raise RuntimeError("Walk-forward audit produced no eligible queries")
    _atomic_frame(audit, AUDIT_REPORT)
    write_examples(audit, responses, args.k, args.warmup)
    print("V04B_WALK_FORWARD_RETRIEVAL_STATUS: PASS")
    print(f"Queries: {len(audit)}")
    print(f"Failures: 0")
    print(f"Saved: {AUDIT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
