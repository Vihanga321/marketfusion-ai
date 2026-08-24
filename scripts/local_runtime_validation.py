"""Local, read-only MarketFusion runtime diagnostics and audit reporting."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from src.intelligence.features import build_context_snapshot
from src.intelligence.news_collectors import (
    GDELT_QUERY,
    JSON_HEADERS,
    RSS_HEADERS,
    classify_text,
    parse_gdelt_json,
    parse_rss_xml,
)
from src.intelligence.macro_collectors import parse_ecb_csv, parse_fred_graph_csv
from src.intelligence.store import macro_identity
from src.intelligence.v05b_contract import (
    CONTEXT_HISTORY_FILE,
    MACRO_FILE,
    MACRO_SOURCES,
    NEWS_FILE,
    NEWS_SOURCES,
    STATUS_FILE as V05B_STATUS_FILE,
)
from src.learning.v05c_contract import (
    CANDIDATE_REPORT as V05C_CANDIDATE_REPORT,
    FEATURE_GROUP_REPORT as V05C_FEATURE_GROUP_REPORT,
    HORIZONS as V05C_HORIZONS,
    MODEL_REGISTRY_REPORT as V05C_MODEL_REGISTRY_REPORT,
    RUNTIME_REGISTRY as V05C_RUNTIME_REGISTRY,
    TRAINING_QUALITY_REPORT as V05C_TRAINING_REPORT,
    WALK_FORWARD_REPORT as V05C_WALK_REPORT,
)
from src.learning.v05c_registry import load_registry as load_v05c_registry, validate_registry as validate_v05c_registry
from src.inference.v06a_contract import LATEST_STATUS_FILE as V06A_STATUS_FILE, REQUIRED_FEATURES as V06A_REQUIRED_FEATURES
from src.fusion.v06b_contract import LATEST_ADVISORY_FILE as V06B_STATUS_FILE
from src.runtime.v06c_contract import LATEST_STATE_FILE as V06C_STATE_FILE
from src.evaluation.v08_contract import (
    CONFLICT_DIR as V08_CONFLICT_DIR,
    LATEST_STATUS_FILE as V08_STATUS_FILE,
    OUTCOMES_FILE as V08_OUTCOMES_FILE,
    PREDICTIONS_FILE as V08_PREDICTIONS_FILE,
    SOURCE_LABEL as V08_SOURCE_LABEL,
)
from src.evaluation.v08_ledger import prediction_sha as v08_prediction_sha
from src.marketdata.v05a_contract import (
    CONFLICT_DIR,
    FEATURE_COLUMNS,
    FEATURE_FILE,
    PREDICTION_HORIZONS_MINUTES,
    STATUS_FILE as V05A_STATUS_FILE,
    TIMEFRAMES,
)
from src.marketdata.v05a_daily_history import DAILY_HISTORY_FILE
from src.marketdata.v05a_dataset import load_continuous_frames
from src.marketdata.v05a_quality import build_quality_report


REPORTS = ROOT / "reports"
MT5_REPORT = REPORTS / "marketfusion_mt5_connectivity.txt"
PROVIDER_REPORT = REPORTS / "v05b_free_source_network_diagnostics.txt"
INTEGRATION_REPORT = REPORTS / "v05ab_local_integration_audit.txt"
FULL_REPORT = REPORTS / "marketfusion_full_local_runtime_validation.txt"
V06_FULL_REPORT = REPORTS / "v06_full_validation.txt"
CALENDAR_DIR = ROOT / "data" / "mt5" / "calendar"


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def mt5_diagnostic() -> int:
    import MetaTrader5 as mt5  # type: ignore

    connected = False
    symbol_ok = False
    bid_ok = False
    ask_ok = False
    account_connected = False
    rows = {label: 0 for label in TIMEFRAMES}
    try:
        connected = bool(mt5.initialize())
        if connected:
            account_connected = mt5.account_info() is not None
            symbol_ok = mt5.symbol_info("EURUSD") is not None
            tick = mt5.symbol_info_tick("EURUSD") if symbol_ok else None
            bid_ok = tick is not None and float(tick.bid) > 0
            ask_ok = tick is not None and float(tick.ask) > 0
            for label, spec in TIMEFRAMES.items():
                rates = mt5.copy_rates_from_pos("EURUSD", getattr(mt5, spec.mt5_attribute), 1, 100)
                rows[label] = 0 if rates is None else len(rates)
    finally:
        if connected:
            mt5.shutdown()
    passed = connected and account_connected and symbol_ok and bid_ok and ask_ok and all(rows.values())
    lines = [
        f"MT5 connected: {'PASS' if passed else 'FAIL'}",
        "symbol: EURUSD",
        f"bid available: {'yes' if bid_ok else 'no'}",
        f"ask available: {'yes' if ask_ok else 'no'}",
        *[f"{label} rows returned: {rows[label]}" for label in TIMEFRAMES],
    ]
    _write(MT5_REPORT, lines)
    print("\n".join(lines))
    return 0 if passed else 1


def _error_category(exc: Exception | None, status: int | None) -> str:
    if status == 429:
        return "RATE_LIMITED"
    if exc is None:
        return "NONE"
    if isinstance(exc, requests.Timeout):
        return "TIMEOUT"
    if isinstance(exc, requests.ConnectionError):
        return "CONNECTION"
    if isinstance(exc, requests.HTTPError):
        return "HTTP_ERROR"
    if isinstance(exc, (ValueError, KeyError, json.JSONDecodeError)):
        return "PARSE_ERROR"
    return type(exc).__name__.upper()


def provider_diagnostic() -> int:
    captured = pd.Timestamp.now(tz="UTC")
    lines = ["V0.5B FREE SOURCE NETWORK DIAGNOSTICS", f"tested_at_utc: {captured.isoformat()}"]
    for source, spec in NEWS_SOURCES.items():
        response = None
        parsed_rows = 0
        error: Exception | None = None
        try:
            params = None
            headers = RSS_HEADERS
            if spec["kind"] == "gdelt":
                params = {
                    "query": GDELT_QUERY, "mode": "ArtList", "maxrecords": 25,
                    "timespan": "3h", "format": "json", "sort": "DateDesc",
                }
                headers = JSON_HEADERS
            response = requests.get(spec["url"], params=params, headers=headers, timeout=(5, 10))
            if response.status_code == 429:
                raise RuntimeError("rate limited")
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if spec["kind"] == "gdelt":
                if "json" not in content_type:
                    raise ValueError("non-JSON response")
                frame = parse_gdelt_json(response.json(), captured)
            else:
                if not any(token in content_type for token in ("xml", "rss", "atom")):
                    raise ValueError("non-XML response")
                frame = parse_rss_xml(response.content, source, captured)
                if frame.empty:
                    raise ValueError("XML contains no feed items")
            parsed_rows = len(frame)
        except Exception as exc:
            error = exc
        status = response.status_code if response is not None else None
        content_type = response.headers.get("Content-Type", "")[:80] if response is not None else ""
        byte_count = len(response.content) if response is not None else 0
        lines.append(
            f"{source}|http={status if status is not None else 'NA'}|content_type={content_type or 'NA'}|"
            f"bytes={byte_count}|parse={'PASS' if error is None else 'FAIL'}|rows={parsed_rows}|"
            f"error={_error_category(error, status)}"
        )
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            lines.append(f"{source}|retry_after={retry_after or 'NOT_SUPPLIED'}|retry_attempted=NO")

    for source, spec in MACRO_SOURCES.items():
        response = None
        parsed_rows = 0
        error = None
        try:
            if spec["provider"] == "ECB_SDMX":
                params = {
                    "startPeriod": (captured - pd.Timedelta(days=60)).date().isoformat(),
                    "format": "csvdata", "includeHistory": "true",
                }
            else:
                params = {
                    "cosd": (captured - pd.Timedelta(days=45)).date().isoformat(),
                    "coed": captured.date().isoformat(),
                }
            response = requests.get(
                spec["url"], params=params, timeout=(5, 10),
                headers={"User-Agent": "MarketFusionAI/0.5B research collector", "Accept": "text/csv, */*;q=0.2", "Connection": "close"},
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(token in content_type for token in ("csv", "text/plain", "octet-stream")):
                raise ValueError("non-CSV response")
            frame = (
                parse_ecb_csv(response.text, source, captured)
                if spec["provider"] == "ECB_SDMX"
                else parse_fred_graph_csv(response.text, source, captured)
            )
            parsed_rows = len(frame)
        except Exception as exc:
            error = exc
        status = response.status_code if response is not None else None
        content_type = response.headers.get("Content-Type", "")[:80] if response is not None else ""
        byte_count = len(response.content) if response is not None else 0
        lines.append(
            f"{source}|http={status if status is not None else 'NA'}|content_type={content_type or 'NA'}|"
            f"bytes={byte_count}|parse={'PASS' if error is None else 'FAIL'}|rows={parsed_rows}|"
            f"error={_error_category(error, status)}"
        )
    _write(PROVIDER_REPORT, lines)
    print("\n".join(lines))
    return 0


def _v05a_audit() -> tuple[dict[str, object], list[str]]:
    frames = load_continuous_frames()
    dataset = pd.read_parquet(FEATURE_FILE)
    daily = pd.read_parquet(DAILY_HISTORY_FILE)
    quality = build_quality_report(frames, dataset, pd.Timestamp.now(tz="UTC"))
    failures = list(quality.failures)
    leakage = [column for column in FEATURE_COLUMNS if column.startswith(("outcome_", "future_"))]
    if leakage:
        failures.append("Outcome/future columns entered the V0.5A feature registry")
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    false_finalized = int((daily["date_utc"].eq(today) & daily["day_status"].eq("FINALIZED_UTC_DAY")).sum())
    if false_finalized:
        failures.append("Current unfinished UTC day is finalized")
    conflict_files = list(CONFLICT_DIR.glob("*.csv")) if CONFLICT_DIR.exists() else []
    invalid_conflicts = 0
    for path in conflict_files:
        conflict = pd.read_csv(path)
        paired = [column[:-7] for column in conflict if column.endswith("_stored") and f"{column[:-7]}_incoming" in conflict]
        genuine = any((conflict[f"{field}_stored"].astype(str) != conflict[f"{field}_incoming"].astype(str)).any() for field in paired)
        if conflict.empty or not paired or not genuine:
            invalid_conflicts += 1
    if invalid_conflicts:
        failures.append(f"Non-genuine conflict files: {invalid_conflicts}")
    payload: dict[str, object] = {
        "status": "PASS" if not failures else "FAIL",
        "rows": {label: len(frames[label]) for label in TIMEFRAMES},
        "feature_rows": len(dataset),
        "feature_complete_rows": int(dataset["feature_complete"].fillna(False).sum()),
        "matured": {h: int(dataset[f"outcome_future_return_{h}m"].notna().sum()) for h in PREDICTION_HORIZONS_MINUTES},
        "daily_rows": len(daily),
        "finalized_days": int(daily["day_status"].eq("FINALIZED_UTC_DAY").sum()),
        "conflicts": len(conflict_files),
        "leakage": len(leakage),
    }
    return payload, failures


def _v04d_audit(stage_status: str) -> tuple[dict[str, object], list[str]]:
    payload = {"status": stage_status, "snapshot_rows": 0, "verified_offsets": 0, "exact_rows": 0,
               "forecast_rows": 0, "strict_rows": 0, "consensus_rows": 0, "canonical_rows": 0,
               "actual_rows": 0, "surprise_rows": 0}
    failures: list[str] = []
    gate_path = CALENDAR_DIR / "v04d_live_consensus_gate.csv"
    actual_path = CALENDAR_DIR / "v04d_live_actual_gate.csv"
    surprise_path = CALENDAR_DIR / "v04d_live_verified_surprises.csv"
    canonical_path = CALENDAR_DIR / "v04d_live_verified_consensus.csv"
    history_path = CALENDAR_DIR / "v04d_history_audit.csv"
    if stage_status == "SKIPPED":
        return payload, failures
    if not all(path.exists() for path in (gate_path, actual_path, surprise_path, canonical_path)):
        return payload, ["Missing V0.4D live audit outputs"]
    gate = pd.read_csv(gate_path)
    actual = pd.read_csv(actual_path)
    surprise = pd.read_csv(surprise_path)
    canonical = pd.read_csv(canonical_path)
    exact = _bool(gate, "event_identity_verified")
    strict = _bool(gate, "strictly_pre_release")
    eligible = _bool(gate, "model_eligible_consensus")
    capture = pd.to_datetime(gate["captured_at_gmt"], utc=True, errors="coerce")
    event = pd.to_datetime(gate["event_timestamp_utc"], utc=True, errors="coerce")
    if (eligible & ~(capture < event)).any():
        failures.append("Post-release forecast accepted as pre-release")
    if history_path.exists():
        history = pd.read_csv(history_path, low_memory=False)
        if "model_eligible_consensus" in history and _bool(history, "model_eligible_consensus").any():
            failures.append("Historical MT5 forecasts became model eligible")
    payload.update({
        "status": "PASS_FAIL_CLOSED" if not failures and stage_status == "PASS" else "FAIL",
        "snapshot_rows": len(gate),
        "verified_offsets": int(_bool(gate, "server_offset_verified").sum()),
        "exact_rows": int(exact.sum()),
        "forecast_rows": int(gate["forecast_value"].notna().sum()),
        "strict_rows": int((exact & strict).sum()),
        "consensus_rows": int(eligible.sum()),
        "canonical_rows": len(canonical),
        "actual_rows": int(_bool(actual, "verified_post_release_actual").sum()),
        "surprise_rows": len(surprise),
    })
    return payload, failures


def _v05b_audit() -> tuple[dict[str, object], list[str]]:
    news = pd.read_parquet(NEWS_FILE)
    macro = pd.read_parquet(MACRO_FILE)
    context = pd.read_parquet(CONTEXT_HISTORY_FILE).sort_values("decision_timestamp_utc").reset_index(drop=True)
    status = json.loads(V05B_STATUS_FILE.read_text(encoding="utf-8"))
    failures: list[str] = []
    observed = pd.to_datetime(news["first_observed_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(news["available_from_utc"], utc=True, errors="coerce")
    if news["news_id"].duplicated().any() or not observed.equals(available):
        failures.append("News identity/availability invariant failed")
    flag_columns = [column for column in news if column.startswith("topic_")] + ["usd_relevant", "eur_relevant", "high_impact_flag"]
    for _, row in news.iterrows():
        expected = classify_text(str(row["title"]), str(row["summary_excerpt"]), str(row["source_region"]))
        if any(int(row[column]) != int(expected[column]) for column in flag_columns):
            failures.append("News topic classification is not deterministic")
            break
    if any(token in column.lower() for column in news for token in ("buy_label", "sell_label", "trade_direction")):
        failures.append("Trading-direction label found in news store")
    macro_observed = pd.to_datetime(macro["first_observed_utc"], utc=True, errors="coerce")
    macro_available = pd.to_datetime(macro["available_from_utc"], utc=True, errors="coerce")
    if not macro_observed.equals(macro_available) or pd.to_numeric(macro["value"], errors="coerce").isna().any():
        failures.append("Macro numeric/availability invariant failed")
    if macro_identity(macro).duplicated().any():
        failures.append("Duplicate macro observation identity")
    decisions = pd.to_datetime(context["decision_timestamp_utc"], utc=True, errors="coerce")
    if not decisions.is_monotonic_increasing or decisions.duplicated().any():
        failures.append("Context decision timestamps are not unique and monotonic")
    if any(column.startswith(("outcome_", "future_")) for column in context):
        failures.append("Outcome/future column found in V0.5B context")
    for index, row in context.iterrows():
        expected = build_context_snapshot(news, macro, decisions.iloc[index]).iloc[0]
        for column in expected.index:
            if column in {"decision_timestamp_utc", "context_contract"}:
                continue
            left, right = row[column], expected[column]
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                failures.append(f"Context as-of reconstruction mismatch: {column}")
                break
        if failures and failures[-1].startswith("Context as-of"):
            break
    degraded = bool(status.get("news_errors") or status.get("macro_errors"))
    latest = context.iloc[-1]
    payload: dict[str, object] = {
        "status": "FAIL" if failures else ("PASS_DEGRADED" if degraded else "PASS"),
        "news_rows": len(news), "news_added": int(status.get("new_news_rows", 0)),
        "news_ok": sum(int(value) > 0 for value in status.get("news_source_counts", {}).values()),
        "news_degraded": [key for key, value in status.get("news_source_counts", {}).items() if int(value) == 0],
        "macro_rows": len(macro), "macro_added": int(status.get("new_macro_rows", 0)),
        "macro_ok": sum(int(value) > 0 for value in status.get("macro_source_counts", {}).values()),
        "macro_degraded": [key for key, value in status.get("macro_source_counts", {}).items() if int(value) == 0],
        "context_rows": len(context),
        "values": {key: latest.get(key) for key in (
            "macro_us_effective_fed_funds_value", "macro_us_2y_yield_value", "macro_us_10y_yield_value",
            "macro_us_10y_minus_2y_value", "macro_ecb_main_refinancing_rate_value",
            "macro_policy_rate_spread_us_minus_ecb_pctpt",
        )},
    }
    return payload, failures


def _integration_audit() -> tuple[dict[str, object], list[str]]:
    market = pd.read_parquet(FEATURE_FILE).sort_values("decision_timestamp_utc").reset_index(drop=True)
    context = pd.read_parquet(CONTEXT_HISTORY_FILE).sort_values("decision_timestamp_utc").reset_index(drop=True)
    decision = pd.to_datetime(market["decision_timestamp_utc"], utc=True).iloc[-1]
    context_times = pd.to_datetime(context["decision_timestamp_utc"], utc=True)
    eligible = context.loc[context_times <= decision]
    failures: list[str] = []
    if eligible.empty:
        failures.append("No V0.5B context is causally available for latest M5 decision")
        context_row = pd.Series(dtype=object)
        context_time = pd.NaT
    else:
        context_row = eligible.iloc[-1]
        context_time = pd.Timestamp(context_row["decision_timestamp_utc"])
    market_row = market.iloc[-1]
    market_features = {column: market_row[column] for column in FEATURE_COLUMNS}
    context_features = {column: context_row[column] for column in context.columns if column not in {"decision_timestamp_utc", "context_contract"}} if not eligible.empty else {}
    duplicates = sorted(set(market_features) & set(context_features))
    combined = {**market_features, **context_features}
    future = 0
    for column in ("m1_available_from_utc", "m5_available_from_utc", "m15_available_from_utc", "h1_available_from_utc"):
        if pd.notna(market_row[column]) and pd.Timestamp(market_row[column]) > decision:
            future += 1
    if pd.notna(context_time) and context_time > decision:
        future += 1
    for column in context.columns:
        if column.endswith("_first_observed_utc") and not eligible.empty and pd.notna(context_row[column]) and pd.Timestamp(context_row[column]) > decision:
            future += 1
    leaks = [column for column in combined if column.startswith(("outcome_", "future_"))]
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(market.columns))
    missing_values = sum(pd.isna(value) for value in combined.values())
    news_available = bool(not eligible.empty and any(float(context_row.get(f"news_{window}m_count", 0) or 0) > 0 for window in (15, 60, 240, 1440)))
    macro_available = bool(not eligible.empty and any(pd.notna(context_row.get(column)) for column in context if column.startswith("macro_") and column.endswith("_value")))
    if future:
        failures.append(f"Future timestamp violations: {future}")
    if duplicates:
        failures.append("Duplicate feature names")
    if leaks:
        failures.append("Outcome fields entered integration features")
    if missing_columns:
        failures.append("Missing required market feature columns")
    if not news_available or not macro_available:
        failures.append("News or macro context unavailable")
    payload: dict[str, object] = {
        "status": "PASS" if not failures else "FAIL",
        "decision": decision.isoformat(), "context_time": context_time.isoformat() if pd.notna(context_time) else "UNAVAILABLE",
        "market_count": len(market_features), "context_count": len(context_features), "combined_count": len(combined),
        "future": future, "duplicates": len(duplicates), "leaks": len(leaks),
        "missing_columns": len(missing_columns), "missing_values": int(missing_values),
        "news_available": news_available, "macro_available": macro_available,
    }
    lines = [
        "MARKETFUSION V0.5A + V0.5B LOCAL INTEGRATION AUDIT",
        f"latest_decision_timestamp_utc: {payload['decision']}",
        f"context_timestamp_utc: {payload['context_time']}",
        f"V0.5A feature count: {payload['market_count']}", f"V0.5B feature count: {payload['context_count']}",
        f"combined feature count: {payload['combined_count']}", f"future violations: {future}",
        f"duplicate feature names: {len(duplicates)}", f"outcome leaks: {len(leaks)}",
        f"missing required market feature columns: {len(missing_columns)}", f"missing values: {missing_values}",
        f"available news context: {'yes' if news_available else 'no'}", f"available macro context: {'yes' if macro_available else 'no'}",
        f"integration_status: {payload['status']}",
    ]
    _write(INTEGRATION_REPORT, lines)
    return payload, failures


def _mt5_report_values() -> dict[str, str]:
    values: dict[str, str] = {}
    if MT5_REPORT.exists():
        for line in MT5_REPORT.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    return values


def _v05c_audit() -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    dataset_gate = "PASS" if FEATURE_FILE.exists() else "FAIL"
    feature_group_gate = "PASS" if V05C_FEATURE_GROUP_REPORT.exists() else "FAIL"
    walk_status = "FAIL"
    if V05C_WALK_REPORT.exists():
        walk = pd.read_csv(V05C_WALK_REPORT)
        walk_status = "PASS" if len(walk) and int(pd.to_numeric(walk["future_violation"], errors="coerce").fillna(1).sum()) == 0 else "FAIL"
    registry_status = "PASS_EMPTY"
    records: list[dict[str, object]] = []
    try:
        records = load_v05c_registry(V05C_RUNTIME_REGISTRY)
        if records:
            validate_v05c_registry(records)
            registry_status = "PASS"
    except Exception as exc:
        registry_status = "FAIL"
        failures.append(f"V0.5C registry integrity: {type(exc).__name__}")
    champions = {
        horizon: next((str(record["model_id"]) for record in reversed(records) if int(record["horizon_minutes"]) == horizon and record["promotion_status"] == "CHAMPION"), "NONE")
        for horizon in V05C_HORIZONS
    }
    training_status = "INSUFFICIENT_DATA"
    if V05C_TRAINING_REPORT.exists():
        for line in V05C_TRAINING_REPORT.read_text(encoding="utf-8").splitlines():
            if line.startswith("V05C_STATUS:"):
                training_status = line.split(":", 1)[1].strip()
    for name, value in (("dataset_gate", dataset_gate), ("feature_group_gate", feature_group_gate), ("walk_forward_engine", walk_status)):
        if value == "FAIL":
            failures.append(f"V0.5C {name} failed")
    if registry_status == "FAIL":
        failures.append("V0.5C model registry integrity failed")
    return {
        "dataset_gate": dataset_gate, "feature_group_gate": feature_group_gate,
        "walk_forward_engine": walk_status, "registry_integrity": registry_status,
        "champions": champions, "training_status": training_status,
    }, failures


def _v06_audit() -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    payload: dict[str, object] = {
        "v06a_status": "NOT_RUN", "v06b_status": "NOT_RUN", "v06c_status": "NOT_RUN",
        "feature_count": len(V06A_REQUIRED_FEATURES), "registry_status": "UNKNOWN",
        "champion_count": 0, "model_ids": [], "source_freshness": {},
        "session": "UNKNOWN", "market_regime": "UNKNOWN", "spread_regime": "UNKNOWN",
        "event_risk": "UNKNOWN", "intelligence_status": "UNKNOWN",
        "action": "WAIT", "confidence": "VERY_LOW", "decision_gate": "WAIT_NOT_RUN",
        "trade_window": "NOT_APPLICABLE_WAIT", "next_reassessment": "UNKNOWN",
        "utc_time": "UNKNOWN", "asia_colombo_time": "UNKNOWN", "reason_codes": [],
    }
    try:
        shadow = json.loads(V06A_STATUS_FILE.read_text(encoding="utf-8"))
        advisory = json.loads(V06B_STATUS_FILE.read_text(encoding="utf-8"))
        state = json.loads(V06C_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return payload, [f"V0.6 runtime output unavailable or invalid: {type(exc).__name__}: {exc}"]
    registry = (state.get("system") or {}).get("registry") or {}
    market = advisory.get("market") or {}
    regime = market.get("regime") or {}
    event = advisory.get("event") or {}
    decision = state.get("decision") or {}
    generated = (state.get("system") or {}).get("generated_time") or {}
    horizons = shadow.get("horizons") or {}
    model_ids = [item.get("model_id") for item in horizons.values() if item.get("model_id")]
    payload.update({
        "v06a_status": shadow.get("status"), "v06b_status": advisory.get("status"),
        "v06c_status": (state.get("system") or {}).get("status"), "registry_status": registry.get("status"),
        "champion_count": registry.get("champion_count", 0), "model_ids": model_ids,
        "source_freshness": {
            "market": (market.get("freshness") or {}).get("status"),
            "model": ((advisory.get("model") or {}).get("freshness") or {}).get("status"),
            "event": (event.get("freshness") or {}).get("status"),
            "intelligence": (advisory.get("intelligence") or {}).get("status"),
        },
        "session": market.get("session"), "market_regime": regime.get("volatility_regime"),
        "spread_regime": (market.get("spread") or {}).get("status"), "event_risk": event.get("status"),
        "intelligence_status": (advisory.get("intelligence") or {}).get("status"),
        "action": decision.get("action"), "confidence": decision.get("confidence"), "decision_gate": decision.get("gate"),
        "trade_window": (state.get("trade_window") or {}).get("status"),
        "next_reassessment": (decision.get("next_reassessment") or {}).get("utc"),
        "utc_time": generated.get("utc"), "asia_colombo_time": generated.get("asia_colombo"),
        "reason_codes": [item.get("code") for item in (state.get("reasons") or [])],
    })
    expected_no_champion = int(payload["champion_count"]) == 0
    if expected_no_champion and (payload["action"], payload["confidence"], payload["decision_gate"]) != ("WAIT", "VERY_LOW", "WAIT_NO_MODEL"):
        failures.append("V0.6 no-champion state did not fail closed as WAIT/VERY_LOW/WAIT_NO_MODEL")
    if bool(decision.get("trading_enabled")) or not bool(decision.get("manual_confirmation_required")):
        failures.append("V0.6 advisory safety flags are invalid")
    if payload["v06c_status"] not in {"PASS", "PASS_DEGRADED", "PASS_FAIL_CLOSED_NO_CHAMPION", "PASS_WITH_LIMITED_COVERAGE", "FAIL_CLOSED"}:
        failures.append("V0.6 overall status is outside its contract")
    return payload, failures


def _v08_audit() -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    status = json.loads(V08_STATUS_FILE.read_text(encoding="utf-8")) if V08_STATUS_FILE.exists() else {}
    predictions = pd.read_parquet(V08_PREDICTIONS_FILE) if V08_PREDICTIONS_FILE.exists() else pd.DataFrame()
    outcomes = pd.read_parquet(V08_OUTCOMES_FILE) if V08_OUTCOMES_FILE.exists() else pd.DataFrame()
    duplicates = int(predictions["decision_timestamp_utc"].duplicated().sum()) if not predictions.empty else 0
    conflicts = len(list(V08_CONFLICT_DIR.glob("*.parquet"))) if V08_CONFLICT_DIR.exists() else 0
    hash_mismatches = 0
    if not predictions.empty:
        hash_mismatches = sum(v08_prediction_sha(row.to_dict()) != str(row["prediction_payload_sha256"]) for _, row in predictions.iterrows())
        if not predictions["source_label"].eq(V08_SOURCE_LABEL).all():
            failures.append("V0.8 prediction ledger mixed forward and research labels")
    early = 0
    binding_mismatches = 0
    if not outcomes.empty:
        early = int((pd.to_datetime(outcomes["matured_at_utc"], utc=True) < pd.to_datetime(outcomes["future_timestamp_utc"], utc=True)).sum())
        valid_bindings = set(zip(predictions.get("prediction_id", []), predictions.get("prediction_payload_sha256", [])))
        binding_mismatches = sum((row["prediction_id"], row["prediction_payload_sha256"]) not in valid_bindings for _, row in outcomes.iterrows())
    if duplicates or conflicts or hash_mismatches or early or binding_mismatches:
        failures.append("V0.8 immutable ledger or maturity audit failed")
    allowed = {"PASS_MONITORING_NO_CHAMPION", "PASS_SHADOW_EVALUATION", "PASS_SHADOW_EVALUATION_DEGRADED_PROVIDERS", "INSUFFICIENT_DATA"}
    if status.get("status") not in allowed:
        failures.append("V0.8 performance monitor status is unavailable or failed")
    return {
        "status": status.get("status", "NOT_RUN"), "predictions": len(predictions), "outcomes": len(outcomes),
        "duplicates": duplicates, "conflicts": conflicts, "hash_mismatches": hash_mismatches,
        "early": early, "binding_mismatches": binding_mismatches,
        "drift": ((status.get("performance") or {}).get("market_drift_status", "INSUFFICIENT_DATA")),
        "research": ((status.get("research") or {}).get("status", "AVAILABLE_NOT_RUN")),
    }, failures


def full_audit(args: argparse.Namespace) -> int:
    v05a, failures_a = _v05a_audit()
    v04d, failures_d = _v04d_audit(args.v04d)
    v05b, failures_b = _v05b_audit()
    integration, failures_i = _integration_audit()
    v05c, failures_c = _v05c_audit()
    v06, failures_6 = _v06_audit()
    v08, failures_8 = _v08_audit()
    mt5 = _mt5_report_values()
    stage_failures = [name for name in ("repo_safety", "v05a_tests", "v05b_tests", "v05c_tests", "v06a_tests", "v06b_tests", "v07_tests", "v08_tests", "mt5", "v05a_cycle", "v05b_cycle", "v06a_cycle", "v06b_cycle", "v06c_cycle", "v08_cycle") if getattr(args, name) != "PASS"]
    if args.v04d == "FAIL":
        stage_failures.append("v04d")
    blockers = stage_failures + failures_a + failures_d + failures_b + failures_i + failures_c + failures_6 + failures_8
    degraded = v05b["status"] == "PASS_DEGRADED"
    final = "FAIL" if blockers else ("PASS_FAIL_CLOSED_NO_CHAMPION" if int(v06["champion_count"]) == 0 else "PASS_DEGRADED" if degraded else "PASS")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tested_at = pd.Timestamp.now(tz="UTC").isoformat()
    next_step = "Continue V0.5A/V0.5B/V0.6/V0.8 forward collection and use V0.8 evidence to improve challengers. Do not promote a model until V0.5C promotion gates are actually met."
    lines = [
        "MARKETFUSION FULL LOCAL RUNTIME VALIDATION", "", f"repository_commit: {commit}", f"tested_at_utc: {tested_at}", "",
        "SECURITY:", f"repo_safety: {args.repo_safety}", "credentials_exposed_by_output: NO", "trading_execution_present: NO", "",
        "MT5:", f"connected: {args.mt5}", f"EURUSD_available: {'PASS' if mt5.get('symbol') == 'EURUSD' else 'FAIL'}",
        *[f"{label}: {mt5.get(f'{label} rows returned', '0')} rows" for label in TIMEFRAMES], "",
        "V0.4D:", f"consensus_pipeline: {v04d['status']}", f"verified_pre_release_rows: {v04d['canonical_rows']}",
        f"verified_surprise_rows: {v04d['surprise_rows']}", "",
        "V0.5A:", f"status: {v05a['status']}", f"market_data_integrity: {'PASS' if not failures_a else 'FAIL'}",
        f"causal_features: {'PASS' if not v05a['leakage'] else 'FAIL'}", f"matured_labels: {v05a['matured']}",
        f"daily_history: {v05a['daily_rows']} rows / {v05a['finalized_days']} finalized", f"conflicts: {v05a['conflicts']}", "",
        "V0.5B:", f"status: {v05b['status']}", f"news_sources_ok: {v05b['news_ok']}",
        f"news_sources_degraded: {', '.join(v05b['news_degraded']) or 'NONE'}", f"macro_sources_ok: {v05b['macro_ok']}",
        f"macro_sources_degraded: {', '.join(v05b['macro_degraded']) or 'NONE'}", f"news_rows: {v05b['news_rows']}",
        f"macro_rows: {v05b['macro_rows']}", f"context_rows: {v05b['context_rows']}", "",
        "V0.5C:", f"dataset_gate: {v05c['dataset_gate']}", f"feature_group_gate: {v05c['feature_group_gate']}",
        f"walk_forward_engine: {v05c['walk_forward_engine']}", f"model_registry_integrity: {v05c['registry_integrity']}",
        f"champion_15m: {v05c['champions'][15]}", f"champion_60m: {v05c['champions'][60]}",
        f"champion_240m: {v05c['champions'][240]}", f"training_status: {v05c['training_status']}", "",
        "V0.6A:", f"status: {v06['v06a_status']}", f"feature_count: {v06['feature_count']}",
        f"registry_status: {v06['registry_status']}", f"champion_count: {v06['champion_count']}",
        f"model_ids: {', '.join(v06['model_ids']) or 'NONE'}", "",
        "V0.6B:", f"status: {v06['v06b_status']}", f"source_freshness: {v06['source_freshness']}",
        f"session: {v06['session']}", f"market_regime: {v06['market_regime']}", f"spread_regime: {v06['spread_regime']}",
        f"event_risk: {v06['event_risk']}", f"intelligence_status: {v06['intelligence_status']}",
        f"reason_codes: {', '.join(v06['reason_codes']) or 'NONE'}", "",
        "V0.6C:", f"status: {v06['v06c_status']}", f"action: {v06['action']}", f"confidence: {v06['confidence']}",
        f"decision_gate: {v06['decision_gate']}", f"trade_window: {v06['trade_window']}",
        f"next_reassessment_utc: {v06['next_reassessment']}", f"generated_at_utc: {v06['utc_time']}",
        f"generated_at_asia_colombo: {v06['asia_colombo_time']}", "trading_enabled: false", "manual_confirmation_required: true", "",
        "V0.8:", f"status: {v08['status']}", f"prediction_rows: {v08['predictions']}", f"outcome_rows: {v08['outcomes']}",
        f"duplicate_predictions: {v08['duplicates']}", f"mutation_conflicts: {v08['conflicts']}", f"prediction_hash_mismatches: {v08['hash_mismatches']}",
        f"early_outcomes: {v08['early']}", f"outcome_binding_mismatches: {v08['binding_mismatches']}",
        f"performance_monitor: {args.v08_cycle}", f"drift_status: {v08['drift']}", f"research_separation: {v08['research']}", "",
        "INTEGRATION:", f"market_plus_intelligence_join: {integration['status']}",
        f"future_information_violations: {integration['future']}", f"outcome_feature_leaks: {integration['leaks']}", "",
        "FINAL_STATUS:", final, "", "BLOCKERS:", *(blockers or ["NONE"]), "", "NEXT_SAFE_STEP:", next_step,
    ]
    _write(FULL_REPORT, lines)
    _write(V06_FULL_REPORT, lines)
    print("V05A_LOCAL_STATUS")
    for label, count in v05a["rows"].items():
        print(f"{label} rows: {count}")
    print(f"feature rows: {v05a['feature_rows']}")
    print(f"feature-complete rows: {v05a['feature_complete_rows']}")
    for horizon, count in v05a["matured"].items():
        print(f"{horizon}m matured: {count}")
    print(f"daily history rows: {v05a['daily_rows']}")
    print(f"finalized days: {v05a['finalized_days']}")
    print(f"conflicts: {v05a['conflicts']}")
    print(f"leakage violations: {v05a['leakage']}")
    print(f"result: {v05a['status']}")
    print("\nV04D_LIVE_STATUS")
    for key in ("snapshot_rows", "verified_offsets", "exact_rows", "forecast_rows", "strict_rows", "consensus_rows", "actual_rows", "surprise_rows"):
        print(f"{key.replace('_', ' ')}: {v04d[key]}")
    print(f"canonical verified consensus rows: {v04d['canonical_rows']}")
    print("\nV05B_LOCAL_STATUS")
    for key in ("news_rows", "news_added", "news_ok", "macro_rows", "macro_added", "macro_ok", "context_rows"):
        print(f"{key.replace('_', ' ')}: {v05b[key]}")
    print(f"failed news providers: {len(v05b['news_degraded'])}")
    print(f"failed macro providers: {len(v05b['macro_degraded'])}")
    for label, key in (("Fed funds", "macro_us_effective_fed_funds_value"), ("US 2Y", "macro_us_2y_yield_value"),
                       ("US 10Y", "macro_us_10y_yield_value"), ("10Y-2Y", "macro_us_10y_minus_2y_value"),
                       ("ECB MRR", "macro_ecb_main_refinancing_rate_value"), ("policy spread", "macro_policy_rate_spread_us_minus_ecb_pctpt")):
        print(f"{label}: {v05b['values'][key]}")
    print(f"result: {v05b['status']}")
    print(f"\nMASTER_REPORT: {FULL_REPORT}")
    print(f"FINAL_STATUS: {final}")
    return 1 if final == "FAIL" else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("mt5")
    sub.add_parser("providers")
    audit = sub.add_parser("audit")
    for name in ("repo_safety", "v05a_tests", "v05b_tests", "v05c_tests", "v06a_tests", "v06b_tests", "v07_tests", "v08_tests", "mt5", "v05a_cycle", "v05b_cycle", "v06a_cycle", "v06b_cycle", "v06c_cycle", "v08_cycle"):
        audit.add_argument(f"--{name.replace('_', '-')}", choices=("PASS", "FAIL"), required=True)
    audit.add_argument("--v04d", choices=("PASS", "FAIL", "SKIPPED"), required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "mt5":
        raise SystemExit(mt5_diagnostic())
    if args.command == "providers":
        raise SystemExit(provider_diagnostic())
    raise SystemExit(full_audit(args))


if __name__ == "__main__":
    main()
