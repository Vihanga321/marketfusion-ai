"""Separated descriptive pattern research; never trains or promotes models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.engines.engine_layer import load_completed_bars
from src.engines.patterns import detect_historical_patterns
from src.engines.swings import confirmed_swings, utc
from src.marketdata.v05a_contract import CONTINUOUS_DIR, ROOT

HORIZONS = (15, 60, 240)
MIN_PATTERN_RESEARCH_SAMPLES = 30
RESEARCH_CSV = ROOT / "reports" / "v09a1_pattern_research.csv"
RESEARCH_SUMMARY = ROOT / "reports" / "v09a1_pattern_research.txt"
ABLATION_GROUPS = {
    "BASE_V09A": ("technical", "volatility", "session", "regime"),
    "PRICE_ACTION": ("candle_geometry", "breakouts", "retests"),
    "STRUCTURE": ("confirmed_swings", "hh_hl_lh_ll", "bos_choch"),
    "LIQUIDITY": ("equal_extrema", "sweeps", "fair_value_gaps"),
    "CHART_PATTERNS": ("reversal_patterns", "continuation_patterns", "consolidation_patterns"),
    "SUPPORT_RESISTANCE": ("swing_levels", "completed_period_levels", "ema", "fibonacci"),
}


def _session(timestamp: pd.Timestamp) -> str:
    if timestamp.dayofweek >= 5:
        return "WEEKEND"
    hour = timestamp.hour
    if 13 <= hour < 16:
        return "LONDON_NEW_YORK_OVERLAP"
    if 7 <= hour < 16:
        return "LONDON"
    if 13 <= hour < 22:
        return "NEW_YORK"
    return "ASIA"


def evaluate_pattern_records(patterns: list[dict[str, Any]], bars: pd.DataFrame, min_samples: int = MIN_PATTERN_RESEARCH_SAMPLES) -> pd.DataFrame:
    """Join detections to exact later completed closes without altering records."""
    if not patterns or bars.empty:
        return pd.DataFrame(columns=["pattern_type", "timeframe", "session", "horizon_minutes", "sample_count", "sample_status", "direction_agreement", "mean_return", "median_return"])
    source = bars.copy().sort_values("bar_close_utc")
    source["bar_close_utc"] = pd.to_datetime(source["bar_close_utc"], utc=True)
    close_map = dict(zip(source["bar_close_utc"], source["close"].astype(float)))
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        detected = utc(pattern["detected_at_utc"])
        eligible = source.loc[source["bar_close_utc"].le(detected)]
        if eligible.empty:
            continue
        entry = float(eligible.iloc[-1]["close"])
        for horizon in HORIZONS:
            future = close_map.get(detected + pd.Timedelta(minutes=horizon))
            if future is None:
                continue
            value = float(future / entry - 1.0)
            expected = 1 if pattern.get("direction") == "BULLISH" else -1 if pattern.get("direction") == "BEARISH" else 0
            rows.append({"pattern_type": pattern["pattern_type"], "timeframe": pattern["timeframe"], "session": _session(detected), "horizon_minutes": horizon, "return": value, "agrees": None if expected == 0 else bool(np.sign(value) == expected)})
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    results = []
    for keys, group in raw.groupby(["pattern_type", "timeframe", "session", "horizon_minutes"], dropna=False):
        count = len(group)
        sufficient = count >= min_samples
        results.append({"pattern_type": keys[0], "timeframe": keys[1], "session": keys[2], "horizon_minutes": int(keys[3]), "sample_count": count, "sample_status": "SUFFICIENT_DESCRIPTIVE_SAMPLE" if sufficient else "INSUFFICIENT_DATA", "direction_agreement": float(group["agrees"].dropna().mean()) if sufficient and group["agrees"].notna().any() else None, "mean_return": float(group["return"].mean()) if sufficient else None, "median_return": float(group["return"].median()) if sufficient else None})
    return pd.DataFrame(results).sort_values(["pattern_type", "timeframe", "session", "horizon_minutes"]).reset_index(drop=True)


def generate_pattern_research(bar_root: Path = CONTINUOUS_DIR, report_csv: Path = RESEARCH_CSV, summary_path: Path = RESEARCH_SUMMARY) -> dict[str, Any]:
    all_patterns: list[dict[str, Any]] = []
    m5 = load_completed_bars("M5", root=bar_root, rows=2000)
    for timeframe in ("M5", "M15"):
        bars = load_completed_bars(timeframe, root=bar_root, rows=2000)
        swings = confirmed_swings(bars, timeframe, limit=2000)
        all_patterns.extend(detect_historical_patterns(bars, timeframe, swings, limit=2000))
    report = evaluate_pattern_records(all_patterns, m5)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_csv, index=False, lineterminator="\n")
    sufficient = int(report["sample_status"].eq("SUFFICIENT_DESCRIPTIVE_SAMPLE").sum()) if not report.empty else 0
    summary_path.write_text("\n".join([
        "MARKETFUSION V0.9A.1 PATTERN RESEARCH", "",
        "mode: RESEARCH_ONLY", "production_retrain: DISABLED", "model_promotion: DISABLED",
        f"detected_patterns: {len(all_patterns)}", f"report_groups: {len(report)}", f"sample_sufficient_groups: {sufficient}",
        "claim: DESCRIPTIVE_ONLY_NO_PREDICTIVE_EDGE_INFERENCE", "",
        "ABLATION_GROUPS", *[f"{name}: {','.join(features)}" for name, features in ABLATION_GROUPS.items()], "",
    ]), encoding="utf-8")
    return {"status": "PASS_RESEARCH_ONLY", "patterns": len(all_patterns), "groups": len(report), "sufficient_groups": sufficient, "production_retrain": False, "model_promotion": False}
