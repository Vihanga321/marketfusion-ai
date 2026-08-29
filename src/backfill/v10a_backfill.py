"""Audited, read-only MT5 historical MARKET_CORE backfill for MarketFusion V1.0A."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.backfill.v10a_contract import (
    BACKUP_DIR,
    CANDIDATE_FEATURE_FILE,
    CONFLICT_DIR,
    FETCH_LIMITS,
    OVERLAP_IMMUTABLE_FIELDS,
    PRICE_TOLERANCE,
    SOURCE_LABEL,
    STAGING_DIR,
    TARGET_PROMOTION_HISTORY_DAYS,
)
from src.learning.v05c_contract import HORIZONS
from src.marketdata.mt5_continuous_store import (
    Mt5ReadOnlySession,
    atomic_write_parquet,
    merge_immutable_bars,
    rates_to_completed_frame,
    read_bar_store,
)
from src.marketdata.v05a_contract import (
    CONTINUOUS_DIR,
    FEATURE_COLUMNS,
    FEATURE_FILE,
    PREDICTION_HORIZONS_MINUTES,
    TIMEFRAMES,
)
from src.marketdata.v05a_dataset import build_dataset_from_frames
from src.marketdata.v05a_quality import audit_dataset, build_quality_report


@dataclass
class BackfillAudit:
    captured_at_utc: pd.Timestamp
    fetched_frames: dict[str, pd.DataFrame]
    existing_frames: dict[str, pd.DataFrame]
    merged_frames: dict[str, pd.DataFrame]
    candidate_dataset: pd.DataFrame
    timeframe_rows: list[dict[str, object]]
    bar_overlap: pd.DataFrame
    dataset_overlap: dict[str, int]
    market_core: dict[str, object]
    failures: list[str]
    warnings: list[str]


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Deterministic content digest used for local audit evidence."""
    canonical = frame.copy()
    for column in canonical.columns:
        if isinstance(canonical[column].dtype, pd.DatetimeTZDtype):
            canonical[column] = canonical[column].astype("int64")
    hashed = pd.util.hash_pandas_object(canonical, index=False, categorize=False).to_numpy(dtype="uint64")
    digest = sha256()
    digest.update("|".join(map(str, canonical.columns)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def fetch_historical_frames(session: Mt5ReadOnlySession, captured_at_utc: object) -> dict[str, pd.DataFrame]:
    """Fetch historical rates without exposing any trading method."""
    captured = _utc(captured_at_utc)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for label, spec in TIMEFRAMES.items():
        timeframe = getattr(session.mt5, spec.mt5_attribute)
        rates = session.mt5.copy_rates_from_pos(session.symbol, timeframe, 0, int(FETCH_LIMITS[label]))
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed for {label}: {session.mt5.last_error()}")
        frame = rates_to_completed_frame(rates, spec, captured.to_pydatetime())
        if frame.empty:
            raise RuntimeError(f"MT5 returned no completed {label} bars")
        frame["source"] = SOURCE_LABEL
        frame = frame.sort_values("bar_open_utc").reset_index(drop=True)
        atomic_write_parquet(frame, STAGING_DIR / f"{session.symbol}_{label}_historical.parquet")
        frames[label] = frame
    return frames


def _numeric_different(left: pd.Series, right: pd.Series, field: str) -> pd.Series:
    if field in {"open", "high", "low", "close"}:
        return (pd.to_numeric(left, errors="raise") - pd.to_numeric(right, errors="raise")).abs().gt(PRICE_TOLERANCE)
    if field.endswith("_utc"):
        return pd.to_datetime(left, utc=True, errors="raise").ne(pd.to_datetime(right, utc=True, errors="raise"))
    return pd.to_numeric(left, errors="raise").ne(pd.to_numeric(right, errors="raise"))


def audit_bar_overlap(existing: pd.DataFrame, fetched: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-field overlap audit and conflicting overlapping bars."""
    if existing.empty or fetched.empty:
        return pd.DataFrame(columns=["timeframe", "field", "overlap_rows", "mismatch_rows"]), pd.DataFrame()
    left = existing.copy()
    right = fetched.copy()
    left["bar_open_utc"] = pd.to_datetime(left["bar_open_utc"], utc=True)
    right["bar_open_utc"] = pd.to_datetime(right["bar_open_utc"], utc=True)
    overlap = left.merge(right, on="bar_open_utc", how="inner", suffixes=("_stored", "_fetched"))
    conflict_mask = pd.Series(False, index=overlap.index)
    rows: list[dict[str, object]] = []
    for field in OVERLAP_IMMUTABLE_FIELDS:
        mismatch = _numeric_different(overlap[f"{field}_stored"], overlap[f"{field}_fetched"], field)
        conflict_mask |= mismatch
        rows.append({"timeframe": label, "field": field, "overlap_rows": len(overlap), "mismatch_rows": int(mismatch.sum())})
    return pd.DataFrame(rows), overlap.loc[conflict_mask].copy()


def _series_mismatch(left: pd.Series, right: pd.Series, field: str) -> pd.Series:
    if field.endswith("_utc"):
        return pd.to_datetime(left, utc=True, errors="coerce").ne(pd.to_datetime(right, utc=True, errors="coerce"))
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    numeric = left_num.notna() | right_num.notna()
    result = pd.Series(False, index=left.index)
    result.loc[numeric] = ~np.isclose(
        left_num.loc[numeric].to_numpy(dtype=float),
        right_num.loc[numeric].to_numpy(dtype=float),
        rtol=0.0, atol=1e-12, equal_nan=True,
    )
    if (~numeric).any():
        result.loc[~numeric] = left.loc[~numeric].astype(str).ne(right.loc[~numeric].astype(str))
    return result


def audit_dataset_overlap(existing: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, int]:
    """Protect previously materialized non-null V0.5A values from mutation.

    Adding warm-up history may turn previously-null features into valid values.
    Existing non-null decision-time values/outcomes may not change.
    """
    if existing.empty:
        return {"overlap_rows": 0, "compared_values": 0, "mismatches": 0, "filled_previous_nulls": 0}
    joined = existing.merge(candidate, on="decision_timestamp_utc", how="inner", suffixes=("_stored", "_candidate"))
    columns = [
        "m5_close", "m1_available_from_utc", "m5_available_from_utc",
        "m15_available_from_utc", "h1_available_from_utc", *FEATURE_COLUMNS,
    ]
    for horizon in PREDICTION_HORIZONS_MINUTES:
        columns.extend([
            f"outcome_future_timestamp_{horizon}m", f"outcome_future_return_{horizon}m",
            f"outcome_direction_{horizon}m", f"outcome_matured_at_utc_{horizon}m",
        ])
    compared = mismatches = filled = 0
    for column in columns:
        left = joined[f"{column}_stored"]
        right = joined[f"{column}_candidate"]
        left_nonnull = left.notna()
        right_nonnull = right.notna()
        both = left_nonnull & right_nonnull
        filled += int((~left_nonnull & right_nonnull).sum())
        if both.any():
            mismatch = _series_mismatch(left.loc[both], right.loc[both], column)
            compared += int(both.sum())
            mismatches += int(mismatch.sum())
    return {"overlap_rows": len(joined), "compared_values": compared, "mismatches": mismatches, "filled_previous_nulls": filled}


def eligible_market_core_summary(dataset: pd.DataFrame) -> dict[str, object]:
    """Use exactly the complete-feature/matured rows V0.5C may train on."""
    complete = dataset["feature_complete"].fillna(False).astype(bool)
    matured = pd.Series(True, index=dataset.index)
    for horizon in HORIZONS:
        matured &= dataset[f"outcome_future_return_{horizon}m"].notna()
        matured &= pd.to_datetime(dataset[f"outcome_matured_at_utc_{horizon}m"], utc=True, errors="coerce").notna()
    eligible = dataset.loc[complete & matured].copy()
    if eligible.empty:
        return {"eligible_rows": 0, "eligible_trading_days": 0, "calendar_span_days": 0, "first_decision_utc": None, "last_decision_utc": None, "promotion_history_gate": "NOT_MET"}
    decision = pd.to_datetime(eligible["decision_timestamp_utc"], utc=True, errors="raise")
    unique_days = int(decision.dt.floor("D").nunique())
    calendar_span = int((decision.max().floor("D") - decision.min().floor("D")).days + 1)
    return {
        "eligible_rows": len(eligible),
        "eligible_trading_days": unique_days,
        "calendar_span_days": calendar_span,
        "first_decision_utc": decision.min().isoformat(),
        "last_decision_utc": decision.max().isoformat(),
        "promotion_history_gate": "MET" if unique_days >= TARGET_PROMOTION_HISTORY_DAYS else "NOT_MET",
    }


def build_backfill_audit(captured_at_utc: object | None = None) -> BackfillAudit:
    captured = pd.Timestamp.now(tz="UTC") if captured_at_utc is None else _utc(captured_at_utc)
    existing_frames = {label: read_bar_store(CONTINUOUS_DIR / spec.filename) for label, spec in TIMEFRAMES.items()}
    failures: list[str] = []
    warnings: list[str] = []
    overlap_rows: list[pd.DataFrame] = []
    timeframe_rows: list[dict[str, object]] = []
    merged_frames: dict[str, pd.DataFrame] = {}

    with Mt5ReadOnlySession() as session:
        fetched_frames = fetch_historical_frames(session, captured)

    for label, spec in TIMEFRAMES.items():
        existing = existing_frames[label]
        fetched = fetched_frames[label]
        field_audit, conflicts = audit_bar_overlap(existing, fetched, label)
        overlap_rows.append(field_audit)
        if not conflicts.empty:
            failures.append(f"{label} immutable overlap conflicts: {len(conflicts)}")
            CONFLICT_DIR.mkdir(parents=True, exist_ok=True)
            conflicts.to_csv(CONFLICT_DIR / f"{label}_overlap_conflicts.csv", index=False, lineterminator="\n")
        merged, merge_conflicts = merge_immutable_bars(existing, fetched, spec)
        if not merge_conflicts.empty:
            failures.append(f"{label} merge conflict rows: {len(merge_conflicts)}")
        merged_frames[label] = merged
        timeframe_rows.append({
            "timeframe": label,
            "requested_rows": FETCH_LIMITS[label],
            "fetched_rows": len(fetched),
            "stored_rows_before": len(existing),
            "candidate_rows_after": len(merged),
            "candidate_added_rows": max(len(merged) - len(existing), 0),
            "fetched_first_open_utc": pd.Timestamp(fetched["bar_open_utc"].iloc[0]).isoformat(),
            "fetched_last_close_utc": pd.Timestamp(fetched["bar_close_utc"].iloc[-1]).isoformat(),
            "candidate_first_open_utc": pd.Timestamp(merged["bar_open_utc"].iloc[0]).isoformat(),
            "candidate_last_close_utc": pd.Timestamp(merged["bar_close_utc"].iloc[-1]).isoformat(),
            "overlap_conflicts": len(conflicts),
            "candidate_sha256": dataframe_sha256(merged),
        })

    candidate = build_dataset_from_frames(merged_frames)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(candidate, CANDIDATE_FEATURE_FILE)

    quality = build_quality_report(merged_frames, candidate, captured)
    failures.extend(quality.failures)
    warnings.extend(quality.warnings)
    causal_failures, causal_warnings, _ = audit_dataset(candidate)
    failures.extend(causal_failures)
    warnings.extend(causal_warnings)

    existing_dataset = pd.read_parquet(FEATURE_FILE, engine="pyarrow") if FEATURE_FILE.exists() else pd.DataFrame()
    dataset_overlap = audit_dataset_overlap(existing_dataset, candidate)
    if dataset_overlap["mismatches"]:
        failures.append(f"V0.5A previously-materialized non-null value mismatches: {dataset_overlap['mismatches']}")

    market_core = eligible_market_core_summary(candidate)
    if int(market_core["eligible_rows"]) == 0:
        failures.append("No fully-featured matured MARKET_CORE rows after backfill")

    return BackfillAudit(
        captured_at_utc=captured, fetched_frames=fetched_frames, existing_frames=existing_frames,
        merged_frames=merged_frames, candidate_dataset=candidate, timeframe_rows=timeframe_rows,
        bar_overlap=pd.concat(overlap_rows, ignore_index=True) if overlap_rows else pd.DataFrame(),
        dataset_overlap=dataset_overlap, market_core=market_core,
        failures=list(dict.fromkeys(failures)), warnings=list(dict.fromkeys(warnings)),
    )


def create_backup(audit: BackfillAudit) -> Path:
    stamp = audit.captured_at_utc.strftime("%Y%m%dT%H%M%SZ")
    destination = BACKUP_DIR / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for _, spec in TIMEFRAMES.items():
        source = CONTINUOUS_DIR / spec.filename
        if source.exists():
            shutil.copy2(source, destination / spec.filename)
    if FEATURE_FILE.exists():
        shutil.copy2(FEATURE_FILE, destination / FEATURE_FILE.name)
    return destination


def restore_backup(backup: Path) -> None:
    for _, spec in TIMEFRAMES.items():
        saved = backup / spec.filename
        target = CONTINUOUS_DIR / spec.filename
        if saved.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, target)
    saved_feature = backup / FEATURE_FILE.name
    if saved_feature.exists():
        FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_feature, FEATURE_FILE)


def apply_backfill(audit: BackfillAudit) -> tuple[Path, dict[str, object]]:
    """Replace local V0.5A stores only after the dry audit passes."""
    if audit.failures:
        raise RuntimeError("V1.0A audit has blocking failures; apply refused")
    backup = create_backup(audit)
    try:
        for label, spec in TIMEFRAMES.items():
            atomic_write_parquet(audit.merged_frames[label], CONTINUOUS_DIR / spec.filename)
        atomic_write_parquet(audit.candidate_dataset, FEATURE_FILE)
        reread = {label: read_bar_store(CONTINUOUS_DIR / spec.filename) for label, spec in TIMEFRAMES.items()}
        dataset = pd.read_parquet(FEATURE_FILE, engine="pyarrow")
        quality = build_quality_report(reread, dataset, pd.Timestamp.now(tz="UTC"))
        if not quality.passed:
            raise RuntimeError("post-apply V0.5A quality failed: " + "; ".join(quality.failures))
        summary = eligible_market_core_summary(dataset)
        summary["feature_file_sha256"] = dataframe_sha256(dataset)
        return backup, summary
    except Exception:
        restore_backup(backup)
        raise
