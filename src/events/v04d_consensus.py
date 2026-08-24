"""Pure, fail-closed logic for V0.4D historical consensus audits."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.events.consensus_measure_mapping import ConsensusMeasureMapping


V04D_CONSENSUS_CONTRACT = "v0.4d-consensus-v1"
V04D_SNAPSHOT_CONTRACT = "v0.4d-consensus-snapshots-v1"
V04C_MEMORY_CONTRACT = "v0.4c-historical-event-memory-v1"
EXPECTED_MEMORY_ROWS = 242
EXPECTED_MEMORY_COLUMNS = 287
EXPECTED_EVENT_VALUE_ROWS = 484
EXPECTED_EVENT_TYPE_COUNTS = {"us_cpi_release": 118, "us_employment_situation": 124}
EXPECTED_MEASURES = {
    "headline_cpi_yoy_sa",
    "core_cpi_yoy_sa",
    "nonfarm_payroll_change",
    "unemployment_rate",
}


class ConsensusAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fingerprint_file(path: Path, root: Path | None = None) -> SourceFingerprint:
    if not path.is_file():
        raise ConsensusAuditError(f"Missing frozen V0.4C source: {path}")
    display = path.relative_to(root).as_posix() if root is not None else path.as_posix()
    return SourceFingerprint(display, sha256_file(path), path.stat().st_size)


def freeze_or_verify_fingerprints(
    paths: Sequence[Path],
    manifest_path: Path,
    *,
    root: Path | None = None,
    initialize: bool = False,
) -> dict[str, object]:
    current = [fingerprint_file(path, root=root).__dict__ for path in paths]
    payload = {"contract": V04D_CONSENSUS_CONTRACT, "files": current}
    if initialize:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    if not manifest_path.is_file():
        raise ConsensusAuditError(
            f"Missing frozen V0.4C fingerprint manifest: {manifest_path}. "
            "Initialize it explicitly from the already-audited V0.4C outputs before running V0.4D."
        )
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected != payload:
        raise ConsensusAuditError("Frozen V0.4C source fingerprint changed; V0.4D refuses to continue")
    return payload


def validate_v04c_frames(memory: pd.DataFrame, values: pd.DataFrame) -> None:
    errors: list[str] = []
    if memory.shape != (EXPECTED_MEMORY_ROWS, EXPECTED_MEMORY_COLUMNS):
        errors.append(f"memory_shape={memory.shape}")
    if memory["event_id"].nunique() != EXPECTED_MEMORY_ROWS:
        errors.append("memory event IDs are not unique")
    if memory["event_type"].value_counts().to_dict() != EXPECTED_EVENT_TYPE_COUNTS:
        errors.append(f"memory event type counts={memory['event_type'].value_counts().to_dict()}")
    contract_column = "v04c_memory_contract_version"
    if contract_column not in memory or not memory[contract_column].eq(V04C_MEMORY_CONTRACT).all():
        errors.append("V0.4C memory contract changed")
    if len(values) != EXPECTED_EVENT_VALUE_ROWS:
        errors.append(f"event_value_rows={len(values)}")
    if set(values["measure_id"].astype(str).unique()) != EXPECTED_MEASURES:
        errors.append(f"measure_ids={sorted(values['measure_id'].astype(str).unique())}")
    if values[["event_id", "measure_id"]].duplicated().any():
        errors.append("event-value measure identities are not unique")
    if values["actual_value"].isna().any():
        errors.append("V0.4C actual-value coverage changed")
    if errors:
        raise ConsensusAuditError("Frozen V0.4C metadata changed: " + "; ".join(errors))


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _utc(value: object, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise ConsensusAuditError(f"Invalid timestamp in {field}")
    return timestamp


def parse_provider_forecast(
    raw_forecast: object,
    forecast_value: object,
    mapping: ConsensusMeasureMapping,
) -> float | None:
    """Convert TE Forecast/ForecastValue to the canonical V0.4C unit.

    The provider's own ``TEForecast`` fields are intentionally absent from the
    arguments so they cannot be accidentally substituted for survey consensus.
    """

    raw = str(raw_forecast or "").strip()
    if raw:
        match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([KkMmBb%]?)", raw.replace(",", ""))
        if not match:
            raise ConsensusAuditError(f"Unsupported provider Forecast format: {raw!r}")
        number = float(match.group(1))
        suffix = match.group(2).upper()
        if mapping.canonical_unit == "thousands_of_jobs_change":
            if suffix == "K":
                return number
            if suffix == "M":
                return number * 1000.0
            if suffix == "B":
                return number * 1_000_000.0
            if suffix == "":
                return number * mapping.forecast_value_scale
            raise ConsensusAuditError("Payroll consensus cannot use percent units")
        if suffix not in ("", "%"):
            raise ConsensusAuditError(f"Unexpected suffix {suffix!r} for {mapping.measure_id}")
        return number

    if forecast_value is None or pd.isna(forecast_value):
        return None
    number = float(forecast_value)
    if not math.isfinite(number):
        raise ConsensusAuditError("Non-finite ForecastValue")
    return number * mapping.forecast_value_scale


def reference_period_matches(provider_reference: object, marketfusion_reference: object) -> bool:
    if provider_reference is None or str(provider_reference).strip() == "":
        return False
    provider = _utc(provider_reference, "ReferenceDate")
    expected = _utc(marketfusion_reference, "reference_period")
    return (provider.year, provider.month) == (expected.year, expected.month)


def map_trading_economics_row(
    provider_row: Mapping[str, object],
    *,
    event_id: str,
    event_type: str,
    event_timestamp_utc: object,
    reference_period: object,
    measure_id: str,
    canonical_actual_value: float,
    mapping: ConsensusMeasureMapping,
    consensus_snapshot_timestamp_utc: object | None = None,
    snapshot_timestamp_verified: bool = False,
) -> dict[str, object]:
    """Validate and normalize one provider row without inventing PIT metadata."""

    event_time = _utc(event_timestamp_utc, "event_timestamp_utc")
    row_time = _utc(provider_row.get("Date"), "Date")
    status = "COMPLETE"
    reasons: list[str] = []

    if _norm(provider_row.get("Country")) != _norm(mapping.country):
        status = "EVENT_MAPPING_AMBIGUOUS"
        reasons.append("country mismatch")
    if event_type != mapping.event_type or measure_id != mapping.measure_id:
        status = "EVENT_MAPPING_AMBIGUOUS"
        reasons.append("MarketFusion measure/event identity mismatch")
    categories = {_norm(value) for value in mapping.acceptable_categories}
    events = {_norm(value) for value in mapping.acceptable_events}
    if _norm(provider_row.get("Category")) not in categories or _norm(provider_row.get("Event")) not in events:
        status = "EVENT_MAPPING_AMBIGUOUS"
        reasons.append("provider category/event not on exact candidate whitelist")
    if row_time != event_time:
        status = "EVENT_MAPPING_AMBIGUOUS"
        reasons.append(f"provider release time {row_time} != frozen MarketFusion time {event_time}")
    date_span = provider_row.get("DateSpan")
    if date_span not in (None, "", 0, "0"):
        status = "EVENT_MAPPING_AMBIGUOUS"
        reasons.append("provider event time is estimated rather than exact")
    if not reference_period_matches(provider_row.get("ReferenceDate"), reference_period):
        status = "REFERENCE_PERIOD_MISMATCH"
        reasons.append("reference month mismatch or missing ReferenceDate")

    provider_unit = _norm(provider_row.get("Unit"))
    allowed_units = {_norm(value) for value in mapping.acceptable_provider_units}
    if provider_unit not in allowed_units:
        status = "UNIT_MISMATCH"
        reasons.append(f"provider unit {provider_row.get('Unit')!r} not accepted")

    consensus = parse_provider_forecast(
        provider_row.get("Forecast"), provider_row.get("ForecastValue"), mapping
    )
    if consensus is None and status == "COMPLETE":
        status = "UNAVAILABLE"
        reasons.append("Forecast is missing")

    available = pd.NaT
    point_in_time_verified = False
    if snapshot_timestamp_verified and consensus_snapshot_timestamp_utc is not None:
        available = _utc(consensus_snapshot_timestamp_utc, "consensus_snapshot_timestamp_utc")
        if available < event_time:
            point_in_time_verified = True
        else:
            status = "POST_RELEASE_ONLY"
            reasons.append("consensus snapshot is not strictly before release T")
    elif consensus is not None and status == "COMPLETE":
        status = "POINT_IN_TIME_UNVERIFIED"
        reasons.append("no dedicated pre-release consensus snapshot timestamp was verified")

    last_update = provider_row.get("LastUpdate")
    last_update_utc = pd.NaT if last_update in (None, "") else _utc(last_update, "LastUpdate")

    return {
        "event_id": event_id,
        "event_type": event_type,
        "event_timestamp_utc": event_time,
        "reference_period": _utc(reference_period, "reference_period"),
        "measure_id": measure_id,
        "unit": mapping.canonical_unit,
        "actual_value": float(canonical_actual_value),
        "consensus_value": consensus,
        "consensus_raw": provider_row.get("Forecast"),
        "provider": mapping.provider,
        "provider_record_id": str(provider_row.get("CalendarId") or ""),
        "provider_ticker": provider_row.get("Ticker"),
        "provider_symbol": provider_row.get("Symbol"),
        "consensus_snapshot_timestamp_utc": available,
        "consensus_available_from_utc": available,
        "provider_last_update_utc": last_update_utc,
        "point_in_time_verified": point_in_time_verified,
        "verification_method": (
            "verified provider snapshot timestamp" if point_in_time_verified else "provider row only; LastUpdate not used as availability"
        ),
        "verification_evidence": "; ".join(reasons) if reasons else "strict mapping and timestamp checks passed",
        "consensus_status": status,
        "consensus_contract_version": V04D_CONSENSUS_CONTRACT,
    }


def select_latest_verified_pre_release_snapshot(snapshots: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id", "measure_id", "event_timestamp_utc", "consensus_available_from_utc",
        "consensus_value", "point_in_time_verified", "consensus_status", "provider_record_id",
    }
    missing = required - set(snapshots.columns)
    if missing:
        raise ConsensusAuditError(f"Consensus snapshot table lacks fields: {sorted(missing)}")
    frame = snapshots.copy()
    frame["event_timestamp_utc"] = pd.to_datetime(frame["event_timestamp_utc"], utc=True, errors="raise")
    frame["consensus_available_from_utc"] = pd.to_datetime(
        frame["consensus_available_from_utc"], utc=True, errors="coerce"
    )
    eligible = frame[
        frame["point_in_time_verified"].fillna(False).astype(bool)
        & frame["consensus_status"].eq("COMPLETE")
        & frame["consensus_value"].notna()
        & frame["consensus_available_from_utc"].notna()
        & (frame["consensus_available_from_utc"] < frame["event_timestamp_utc"])
    ].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        ["event_id", "measure_id", "consensus_available_from_utc", "provider_record_id"],
        kind="mergesort",
    )
    return eligible.groupby(["event_id", "measure_id"], as_index=False, sort=False).tail(1).reset_index(drop=True)


def add_surprise_fields(canonical: pd.DataFrame) -> pd.DataFrame:
    result = canonical.copy()
    valid = (
        result["point_in_time_verified"].fillna(False).astype(bool)
        & result["consensus_status"].eq("COMPLETE")
        & result["consensus_value"].notna()
        & result["actual_value"].notna()
    )
    result["surprise_signed_raw"] = np.where(
        valid,
        pd.to_numeric(result["actual_value"]) - pd.to_numeric(result["consensus_value"]),
        np.nan,
    )
    result["surprise_abs_raw"] = result["surprise_signed_raw"].abs()
    result["surprise_status"] = np.where(valid, "AVAILABLE", "CONSENSUS_UNAVAILABLE_OR_UNVERIFIED")
    return result


def add_prior_only_robust_normalization(
    frame: pd.DataFrame,
    minimum_history: int = 10,
) -> pd.DataFrame:
    result = frame.sort_values(["event_timestamp_utc", "event_id", "measure_id"], kind="mergesort").copy()
    z = pd.Series(np.nan, index=result.index, dtype=float)
    count = pd.Series(0, index=result.index, dtype="int64")
    method = pd.Series("UNAVAILABLE_RAW_SURPRISE", index=result.index, dtype="object")
    cutoff = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    histories: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for idx, row in result.iterrows():
        measure = str(row["measure_id"])
        history = histories.setdefault(measure, [])
        count.at[idx] = len(history)
        if history:
            cutoff.at[idx] = history[-1][0]
        raw = row.get("surprise_signed_raw")
        if raw is None or pd.isna(raw):
            continue
        if len(history) < minimum_history:
            method.at[idx] = f"INSUFFICIENT_HISTORY_MIN_{minimum_history}"
        else:
            prior = np.asarray([value for _, value in history], dtype=float)
            median = float(np.median(prior))
            q1, q3 = np.quantile(prior, [0.25, 0.75])
            scale = float(q3 - q1)
            used = "prior_only_median_iqr"
            if not math.isfinite(scale) or scale <= 0:
                scale = float(np.std(prior, ddof=0))
                used = "prior_only_median_std_fallback"
            if math.isfinite(scale) and scale > 0:
                z.at[idx] = (float(raw) - median) / scale
                method.at[idx] = used
            else:
                method.at[idx] = "INSUFFICIENT_VARIATION"
        history.append((_utc(row["event_timestamp_utc"], "event_timestamp_utc"), float(raw)))
    result["surprise_z_robust"] = z
    result["normalization_history_count"] = count
    result["normalization_method"] = method
    result["normalization_timestamp_cutoff"] = cutoff
    return result.reset_index(drop=True)


def provider_approval_status(audit_rows: pd.DataFrame) -> tuple[str, str]:
    if audit_rows.empty:
        return "NOT_AVAILABLE", "No provider audit rows were collected"
    mapped = audit_rows[~audit_rows["consensus_status"].isin(["EVENT_MAPPING_AMBIGUOUS", "UNIT_MISMATCH", "REFERENCE_PERIOD_MISMATCH"])]
    if mapped.empty:
        return "INSUFFICIENT_METADATA", "No audit row passed strict event/measure mapping"
    forecasts = mapped[mapped["consensus_value"].notna()]
    if forecasts.empty:
        return "INSUFFICIENT_METADATA", "Mapped rows contain no historical survey consensus"
    verified = forecasts[
        forecasts["point_in_time_verified"].fillna(False).astype(bool)
        & forecasts["consensus_status"].eq("COMPLETE")
    ]
    if verified.empty:
        return (
            "INSUFFICIENT_METADATA",
            "Historical Forecast values exist, but no dedicated pre-release consensus availability timestamp was verified",
        )
    if len(verified) != len(forecasts):
        return "CONDITIONALLY_APPROVED", "Only a subset of historical consensus rows has verified pre-release availability"
    return "APPROVED_POINT_IN_TIME", "All audited consensus rows have strict pre-release availability evidence"
