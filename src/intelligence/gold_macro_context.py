"""Verified, research-only macro/intermarket context for XAUUSD.

This module intentionally uses only named public Federal Reserve series and
never fabricates unavailable values.  FRED historical CSV downloads are
current-vintage data, so they are NOT production/vintage-safe evidence for
model promotion.  A conservative availability lag is applied before any
as-of join to XAUUSD decision timestamps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.assets.contracts import ROOT

PROVIDER_ID = "FRED_PUBLIC_CSV"
CONTRACT_VERSION = "v1.0b3-gold-macro-context-v1"
FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
# FRED page timestamps show daily market series are generally published after
# the observation date.  Because fredgraph CSV does not carry historical
# release timestamps/vintages, this research layer waits four full calendar
# days and uses 23:59 UTC before a value may become visible to XAUUSD research.
# This is deliberately conservative, but still not a substitute for ALFRED or
# another vintage-safe production provider.
CONSERVATIVE_AVAILABILITY_LAG_DAYS = 4


@dataclass(frozen=True)
class ContextSeries:
    key: str
    series_id: str
    description: str
    units: str
    source: str


CONTEXT_SERIES: tuple[ContextSeries, ...] = (
    ContextSeries(
        "usd_broad", "DTWEXBGS", "Nominal Broad U.S. Dollar Index",
        "index", "Board of Governors of the Federal Reserve System via FRED",
    ),
    ContextSeries(
        "yield_2y", "DGS2", "2-Year U.S. Treasury Constant Maturity Yield",
        "percent", "Board of Governors of the Federal Reserve System via FRED",
    ),
    ContextSeries(
        "yield_10y", "DGS10", "10-Year U.S. Treasury Constant Maturity Yield",
        "percent", "Board of Governors of the Federal Reserve System via FRED",
    ),
    ContextSeries(
        "real_yield_10y", "DFII10", "10-Year Inflation-Indexed U.S. Treasury Yield",
        "percent", "Board of Governors of the Federal Reserve System via FRED",
    ),
)


def context_root(root: Path = ROOT) -> Path:
    return root / "data" / "context" / "XAUUSD" / "fred"


def context_store(root: Path = ROOT) -> Path:
    return context_root(root) / "gold_macro_context.parquet"


def context_manifest_path(root: Path = ROOT) -> Path:
    return context_root(root) / "manifest.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _parse_fred_csv(text: str, spec: ContextSeries) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text))
    if frame.empty or len(frame.columns) < 2:
        raise ValueError(f"FRED {spec.series_id} returned no usable observations")
    date_column = "DATE" if "DATE" in frame.columns else frame.columns[0]
    value_column = spec.series_id if spec.series_id in frame.columns else frame.columns[1]
    observation_date = pd.to_datetime(frame[date_column], errors="raise", utc=True).dt.normalize()
    values = pd.to_numeric(frame[value_column].replace(".", np.nan), errors="coerce")
    normalized = pd.DataFrame({
        "series_key": spec.key,
        "series_id": spec.series_id,
        "observation_date_utc": observation_date,
        "value": values,
    }).dropna(subset=["value"]).sort_values("observation_date_utc").reset_index(drop=True)
    if normalized.empty:
        raise ValueError(f"FRED {spec.series_id} contained no numeric observations")
    if normalized["observation_date_utc"].duplicated().any():
        raise ValueError(f"FRED {spec.series_id} contains duplicate observation dates")
    normalized["available_at_utc"] = (
        normalized["observation_date_utc"]
        + pd.Timedelta(days=CONSERVATIVE_AVAILABILITY_LAG_DAYS)
        + pd.Timedelta(hours=23, minutes=59)
    )
    normalized["change_1"] = normalized["value"].diff(1)
    normalized["change_5"] = normalized["value"].diff(5)
    normalized["pct_change_1"] = normalized["value"].pct_change(1)
    normalized["pct_change_5"] = normalized["value"].pct_change(5)
    normalized["provider_id"] = PROVIDER_ID
    normalized["vintage_safe"] = False
    return normalized


def fetch_fred_series(
    spec: ContextSeries,
    session: requests.Session | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = session or requests.Session()
    url = FREDGRAPH_URL.format(series_id=spec.series_id)
    response = client.get(url, timeout=timeout_seconds, headers={"User-Agent": "MarketFusion-research/1.0"})
    response.raise_for_status()
    text = response.text
    frame = _parse_fred_csv(text, spec)
    metadata = {
        **asdict(spec),
        "provider_id": PROVIDER_ID,
        "url": url,
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "payload_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "rows": len(frame),
        "first_observation": pd.Timestamp(frame["observation_date_utc"].min()).isoformat(),
        "last_observation": pd.Timestamp(frame["observation_date_utc"].max()).isoformat(),
        "availability_policy": f"observation_date + {CONSERVATIVE_AVAILABILITY_LAG_DAYS} calendar days + 23:59 UTC",
        "vintage_safe": False,
        "research_only": True,
    }
    return frame, metadata


def refresh_gold_macro_context(
    root: Path = ROOT,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    series_metadata: list[dict[str, Any]] = []
    for spec in CONTEXT_SERIES:
        frame, metadata = fetch_fred_series(spec, session=session)
        frames.append(frame)
        series_metadata.append(metadata)
    combined = pd.concat(frames, ignore_index=True).sort_values(["series_key", "available_at_utc"]).reset_index(drop=True)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "provider_id": PROVIDER_ID,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "series": series_metadata,
        "row_count": len(combined),
        "vintage_safe": False,
        "production_model_eligible": False,
        "automatic_execution": "DISABLED",
    }
    _atomic_parquet(context_store(root), combined)
    _atomic_json(context_manifest_path(root), manifest)
    return combined, manifest


def load_gold_macro_context(root: Path = ROOT) -> pd.DataFrame:
    path = context_store(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing verified XAUUSD context store: {path}")
    frame = pd.read_parquet(path)
    required = {"series_key", "series_id", "observation_date_utc", "available_at_utc", "value", "vintage_safe"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Invalid XAUUSD context store; missing: " + ", ".join(missing))
    frame["observation_date_utc"] = pd.to_datetime(frame["observation_date_utc"], utc=True, errors="raise")
    frame["available_at_utc"] = pd.to_datetime(frame["available_at_utc"], utc=True, errors="raise")
    if frame["available_at_utc"].isna().any():
        raise ValueError("Context availability timestamps are required")
    return frame.sort_values(["series_key", "available_at_utc"]).reset_index(drop=True)


def context_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for spec in CONTEXT_SERIES:
        prefix = f"ctx_{spec.key}"
        names.extend((f"{prefix}_value", f"{prefix}_change_1", f"{prefix}_change_5", f"{prefix}_age_hours"))
    names.extend(("ctx_curve_2s10s", "ctx_breakeven_10y"))
    return tuple(names)


def join_context_asof(decisions: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Attach only context values whose conservative available_at <= decision."""
    if "decision_timestamp_utc" not in decisions.columns:
        raise ValueError("decision_timestamp_utc is required for context join")
    result = decisions.copy()
    result["decision_timestamp_utc"] = pd.to_datetime(result["decision_timestamp_utc"], utc=True, errors="raise")
    result = result.sort_values("decision_timestamp_utc").reset_index(drop=True)
    for spec in CONTEXT_SERIES:
        series = context.loc[context["series_key"].eq(spec.key)].copy()
        if series.empty:
            raise ValueError(f"Verified context missing required series {spec.series_id}")
        series = series.sort_values("available_at_utc")
        prefix = f"ctx_{spec.key}"
        right = series.loc[:, ["available_at_utc", "value", "change_1", "change_5"]].rename(columns={
            "available_at_utc": f"{prefix}_available_at_utc",
            "value": f"{prefix}_value",
            "change_1": f"{prefix}_change_1",
            "change_5": f"{prefix}_change_5",
        })
        result = pd.merge_asof(
            result,
            right,
            left_on="decision_timestamp_utc",
            right_on=f"{prefix}_available_at_utc",
            direction="backward",
            allow_exact_matches=True,
        )
        available = pd.to_datetime(result[f"{prefix}_available_at_utc"], utc=True, errors="coerce")
        if ((available > result["decision_timestamp_utc"]) & available.notna()).any():
            raise ValueError(f"Future {spec.series_id} context leaked into XAUUSD decisions")
        result[f"{prefix}_age_hours"] = (
            result["decision_timestamp_utc"] - available
        ).dt.total_seconds() / 3600.0
    result["ctx_curve_2s10s"] = result["ctx_yield_10y_value"] - result["ctx_yield_2y_value"]
    result["ctx_breakeven_10y"] = result["ctx_yield_10y_value"] - result["ctx_real_yield_10y_value"]
    return result
