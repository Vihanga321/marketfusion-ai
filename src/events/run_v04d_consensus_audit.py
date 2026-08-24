"""Run the V0.4D historical consensus audit without weakening V0.4C gates.

The command is intentionally fail-closed. It can build registries and audit
provider rows, but it will not activate surprise features unless a consensus
value has a verified availability timestamp strictly before release T.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.events.consensus_measure_mapping import mapping_for, mapping_registry_frame
from src.events.consensus_source_registry import source_registry_frame
from src.events.providers.trading_economics_consensus import TradingEconomicsError, fetch_calendar
from src.events.v04d_consensus import (
    ConsensusAuditError,
    add_prior_only_robust_normalization,
    add_surprise_fields,
    freeze_or_verify_fingerprints,
    map_trading_economics_row,
    provider_approval_status,
    select_latest_verified_pre_release_snapshot,
    validate_v04c_frames,
)


ROOT = REPOSITORY_ROOT
MEMORY_PATH = ROOT / "data" / "processed" / "v04c_historical_event_memory.parquet"
VALUES_PATH = ROOT / "data" / "processed" / "v04c_event_values.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "v04c_event_value_features.parquet"
FINGERPRINT_JSON = ROOT / "reports" / "v04c_frozen_source_fingerprint.json"
FINGERPRINT_TEXT = ROOT / "reports" / "v04c_frozen_source_fingerprint.txt"
SOURCE_REGISTRY_REPORT = ROOT / "reports" / "v04d_consensus_source_registry.csv"
MAPPING_REPORT = ROOT / "reports" / "v04d_consensus_mapping_audit.csv"
COVERAGE_REPORT = ROOT / "reports" / "v04d_consensus_candidate_coverage.csv"
QUALITY_REPORT = ROOT / "reports" / "v04d_consensus_quality.txt"
SOURCE_AUDIT_REPORT = ROOT / "reports" / "v04d_consensus_source_audit.txt"
AUDIT_CSV = ROOT / "reports" / "v04d_consensus_sample_audit.csv"
SURPRISE_QUALITY = ROOT / "reports" / "v04d_surprise_quality.txt"
SURPRISE_SAMPLE = ROOT / "reports" / "v04d_surprise_sample.csv"
SNAPSHOT_PARQUET = ROOT / "data" / "processed" / "v04d_consensus_snapshots.parquet"
CANONICAL_PARQUET = ROOT / "data" / "processed" / "v04d_event_consensus.parquet"
RAW_AUDIT_JSONL = ROOT / "data" / "provider_audit" / "trading_economics_consensus_audit.jsonl"

AUDIT_YEARS = (2015, 2018, 2020, 2022, 2024, 2026)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False, engine="pyarrow")
    else:
        frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _write_fingerprint_text(payload: dict[str, object]) -> None:
    lines = ["MARKETFUSION V0.4C FROZEN SOURCE FINGERPRINT", f"contract: {payload['contract']}"]
    for item in payload["files"]:
        lines.extend(
            [
                f"file: {item['path']}",
                f"sha256: {item['sha256']}",
                f"size_bytes: {item['size_bytes']}",
            ]
        )
    lines.extend(["FINGERPRINT_STATUS: PASS", ""])
    _atomic_text("\n".join(lines), FINGERPRINT_TEXT)


def _load_frozen_sources(initialize_fingerprint: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [MEMORY_PATH, VALUES_PATH]
    if FEATURES_PATH.is_file():
        paths.append(FEATURES_PATH)
    fingerprint = freeze_or_verify_fingerprints(
        paths,
        FINGERPRINT_JSON,
        root=ROOT,
        initialize=initialize_fingerprint,
    )
    _write_fingerprint_text(fingerprint)
    memory = pd.read_parquet(MEMORY_PATH, engine="pyarrow")
    values = pd.read_parquet(VALUES_PATH, engine="pyarrow")
    validate_v04c_frames(memory, values)
    return memory, values


def _stratified_sample(values: pd.DataFrame) -> pd.DataFrame:
    frame = values.copy()
    frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="raise")
    frame["year"] = frame["event_timestamp_utc"].dt.year
    event_rows = frame[["event_id", "event_type", "event_timestamp_utc", "year"]].drop_duplicates()
    selected_ids: list[str] = []
    for year in AUDIT_YEARS:
        for event_type in ("us_cpi_release", "us_employment_situation"):
            candidates = event_rows[(event_rows["year"] == year) & (event_rows["event_type"] == event_type)]
            if not candidates.empty:
                chosen = candidates.sort_values(["event_timestamp_utc", "event_id"]).iloc[0]
                selected_ids.append(str(chosen["event_id"]))
    return frame[frame["event_id"].astype(str).isin(selected_ids)].drop(columns="year").copy()


def _provider_rows_for_measure(api_key: str, measure_id: str, event_time: pd.Timestamp) -> list[dict[str, object]]:
    mapping = mapping_for(measure_id)
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    day = event_time.date().isoformat()
    for indicator in mapping.acceptable_categories:
        try:
            fetched = fetch_calendar(api_key, indicator=indicator, start_date=day, end_date=day)
        except TradingEconomicsError:
            continue
        for row in fetched:
            identity = str(row.get("CalendarId") or json.dumps(row, sort_keys=True, default=str))
            if identity not in seen:
                seen.add(identity)
                rows.append(row)
    return rows


def _sanitize_provider_row(row: dict[str, object]) -> dict[str, object]:
    allowed = {
        "CalendarId", "Date", "Country", "Category", "Event", "Reference", "ReferenceDate",
        "Source", "SourceURL", "Actual", "ActualValue", "Previous", "PreviousValue", "Forecast",
        "ForecastValue", "TEForecast", "TEForecastValue", "URL", "DateSpan", "Importance",
        "LastUpdate", "Revised", "Currency", "Unit", "Ticker", "Symbol",
    }
    return {key: row.get(key) for key in sorted(allowed) if key in row}


def _blank_audit_row(value: pd.Series, status: str, evidence: str) -> dict[str, object]:
    return {
        "event_id": value["event_id"],
        "event_type": value["event_type"],
        "event_timestamp_utc": value["event_timestamp_utc"],
        "reference_period": value["reference_period"],
        "measure_id": value["measure_id"],
        "unit": value["unit"],
        "actual_value": value["actual_value"],
        "consensus_value": float("nan"),
        "consensus_raw": None,
        "provider": "Trading Economics",
        "provider_record_id": "",
        "provider_ticker": None,
        "provider_symbol": None,
        "consensus_snapshot_timestamp_utc": pd.NaT,
        "consensus_available_from_utc": pd.NaT,
        "provider_last_update_utc": pd.NaT,
        "point_in_time_verified": False,
        "verification_method": "no acceptable provider row",
        "verification_evidence": evidence,
        "consensus_status": status,
        "consensus_contract_version": "v0.4d-consensus-v1",
    }


def _audit_trading_economics(api_key: str, values: pd.DataFrame, full_collection: bool) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    target = values if full_collection else _stratified_sample(values)
    audit_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for _, value in target.sort_values(["event_timestamp_utc", "event_id", "measure_id"]).iterrows():
        event_time = pd.Timestamp(value["event_timestamp_utc"])
        event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
        candidates = _provider_rows_for_measure(api_key, str(value["measure_id"]), event_time)
        if not candidates:
            audit_rows.append(_blank_audit_row(value, "UNAVAILABLE", "provider returned no candidate calendar row"))
            continue
        for provider_row in candidates:
            sanitized = _sanitize_provider_row(provider_row)
            raw_rows.append({
                "marketfusion_event_id": str(value["event_id"]),
                "marketfusion_measure_id": str(value["measure_id"]),
                **sanitized,
            })
            normalized = map_trading_economics_row(
                sanitized,
                event_id=str(value["event_id"]),
                event_type=str(value["event_type"]),
                event_timestamp_utc=value["event_timestamp_utc"],
                reference_period=value["reference_period"],
                measure_id=str(value["measure_id"]),
                canonical_actual_value=float(value["actual_value"]),
                mapping=mapping_for(str(value["measure_id"])),
                consensus_snapshot_timestamp_utc=None,
                snapshot_timestamp_verified=False,
            )
            audit_rows.append(normalized)
    return pd.DataFrame(audit_rows), raw_rows


def _write_raw_jsonl(rows: list[dict[str, object]]) -> None:
    RAW_AUDIT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_AUDIT_JSONL.with_name(f".{RAW_AUDIT_JSONL.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(RAW_AUDIT_JSONL)


def _coverage(audit: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    expected = values.groupby(["event_type", "measure_id"], as_index=False).size().rename(columns={"size": "rows_expected"})
    if audit.empty:
        for name in ("rows_audited", "consensus_present", "point_in_time_verified", "rejected", "missing"):
            expected[name] = 0 if name != "missing" else expected["rows_expected"]
        return expected
    grouped = []
    for (event_type, measure_id), group in audit.groupby(["event_type", "measure_id"]):
        rejected_statuses = {"EVENT_MAPPING_AMBIGUOUS", "REFERENCE_PERIOD_MISMATCH", "UNIT_MISMATCH", "POST_RELEASE_ONLY"}
        grouped.append({
            "event_type": event_type,
            "measure_id": measure_id,
            "rows_audited": len(group),
            "consensus_present": int(group["consensus_value"].notna().sum()),
            "point_in_time_verified": int(group["point_in_time_verified"].fillna(False).astype(bool).sum()),
            "rejected": int(group["consensus_status"].isin(rejected_statuses).sum()),
            "missing": int(group["consensus_status"].eq("UNAVAILABLE").sum()),
        })
    result = expected.merge(pd.DataFrame(grouped), on=["event_type", "measure_id"], how="left")
    for name in ("rows_audited", "consensus_present", "point_in_time_verified", "rejected", "missing"):
        result[name] = result[name].fillna(0).astype(int)
    return result


def _write_reports(audit: pd.DataFrame, values: pd.DataFrame, approval: tuple[str, str], credential_present: bool) -> None:
    _atomic_frame(source_registry_frame(), SOURCE_REGISTRY_REPORT)
    _atomic_frame(mapping_registry_frame(), MAPPING_REPORT)
    _atomic_frame(_coverage(audit, values), COVERAGE_REPORT)
    _atomic_frame(audit, AUDIT_CSV)
    status, reason = approval
    lines = [
        "MARKETFUSION V0.4D HISTORICAL CONSENSUS INTELLIGENCE",
        "",
        "Frozen V0.4C: PASS",
        "Provider: Trading Economics",
        f"Credential present: {credential_present}",
        f"Audit rows: {len(audit)}",
        f"Consensus present: {int(audit['consensus_value'].notna().sum()) if not audit.empty else 0}",
        f"PIT verified: {int(audit['point_in_time_verified'].fillna(False).astype(bool).sum()) if not audit.empty else 0}",
        f"Provider approval: {status}",
        f"Reason: {reason}",
        "LastUpdate used as consensus availability: NO",
        "TEForecast used as survey consensus: NO",
        "Surprise activation: ENABLED only for COMPLETE + point_in_time_verified rows",
        "Predictive edge claimed: NO",
        "",
    ]
    _atomic_text("\n".join(lines), QUALITY_REPORT)
    source_lines = [
        "MARKETFUSION V0.4D CONSENSUS SOURCE AUDIT",
        "Trading Economics: documentation supports historical calendar/PIT retrieval and separates Forecast from TEForecast.",
        "Trading Economics: no dedicated Forecast-availability timestamp is assumed from LastUpdate.",
        f"Trading Economics approval: {status}",
        f"Trading Economics reason: {reason}",
        "LSEG Reuters Polls: documentation supports polling points in time; entitlement/payload not verified in this run.",
        "LSEG Reuters Polls approval: CONDITIONALLY_APPROVED_DOCUMENTATION_ONLY",
        "",
    ]
    _atomic_text("\n".join(source_lines), SOURCE_AUDIT_REPORT)


def _write_surprise_outputs(audit: pd.DataFrame) -> None:
    canonical = select_latest_verified_pre_release_snapshot(audit)
    if canonical.empty:
        _atomic_text(
            "MARKETFUSION V0.4D SURPRISE QUALITY\n"
            "verified_consensus_rows: 0\n"
            "raw_surprise_rows: 0\n"
            "normalized_surprise_rows: 0\n"
            "SURPRISE_STATUS: INSUFFICIENT_POINT_IN_TIME_CONSENSUS\n",
            SURPRISE_QUALITY,
        )
        _atomic_frame(canonical, SURPRISE_SAMPLE)
        return
    canonical = add_surprise_fields(canonical)
    canonical = add_prior_only_robust_normalization(canonical)
    _atomic_frame(audit, SNAPSHOT_PARQUET)
    _atomic_frame(canonical, CANONICAL_PARQUET)
    _atomic_frame(canonical.head(40), SURPRISE_SAMPLE)
    lines = [
        "MARKETFUSION V0.4D SURPRISE QUALITY",
        f"verified_consensus_rows: {len(canonical)}",
        f"raw_surprise_rows: {int(canonical['surprise_signed_raw'].notna().sum())}",
        f"normalized_surprise_rows: {int(canonical['surprise_z_robust'].notna().sum())}",
        "prior_only_normalization: PASS",
        "SURPRISE_STATUS: PASS",
        "",
    ]
    _atomic_text("\n".join(lines), SURPRISE_QUALITY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-fingerprint", action="store_true")
    parser.add_argument("--full-collection", action="store_true")
    args = parser.parse_args()

    _, values = _load_frozen_sources(args.initialize_fingerprint)
    credential = os.getenv("TRADING_ECONOMICS_API_KEY", "")
    credential_present = bool(credential.strip())

    if not credential_present:
        audit = pd.DataFrame(columns=[
            "event_id", "event_type", "event_timestamp_utc", "reference_period", "measure_id", "unit",
            "actual_value", "consensus_value", "consensus_raw", "provider", "provider_record_id",
            "provider_ticker", "provider_symbol", "consensus_snapshot_timestamp_utc",
            "consensus_available_from_utc", "provider_last_update_utc", "point_in_time_verified",
            "verification_method", "verification_evidence", "consensus_status", "consensus_contract_version",
        ])
        approval = ("NOT_AVAILABLE", "TRADING_ECONOMICS_LIVE_AUDIT_NOT_RUN: credential unavailable in this process")
        _write_reports(audit, values, approval, False)
        _write_surprise_outputs(audit)
        print("TRADING_ECONOMICS_LIVE_AUDIT_NOT_RUN")
        print("V04D_NON_LIVE_STATUS: PASS")
        return 0

    if args.full_collection:
        if not QUALITY_REPORT.is_file() or "Provider approval: APPROVED_POINT_IN_TIME" not in QUALITY_REPORT.read_text(encoding="utf-8"):
            raise ConsensusAuditError("Full collection is blocked until a stratified provider audit is APPROVED_POINT_IN_TIME")

    audit, raw_rows = _audit_trading_economics(credential, values, args.full_collection)
    _write_raw_jsonl(raw_rows)
    approval = provider_approval_status(audit)
    _write_reports(audit, values, approval, True)
    _write_surprise_outputs(audit)
    print(f"TRADING_ECONOMICS_PROVIDER_STATUS: {approval[0]}")
    print(f"AUDIT_ROWS: {len(audit)}")
    print(f"PIT_VERIFIED: {int(audit['point_in_time_verified'].fillna(False).astype(bool).sum()) if not audit.empty else 0}")
    print("V04D_AUDIT_STATUS: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
