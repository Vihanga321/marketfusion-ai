"""Fail-closed V0.5A quality and leakage audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.marketdata.v05a_contract import (
    FEATURE_COLUMNS,
    PREDICTION_HORIZONS_MINUTES,
    QUALITY_REPORT,
    TIMEFRAMES,
)


@dataclass
class V05AQuality:
    passed: bool
    failures: list[str]
    warnings: list[str]
    report: str


def _frame_audit(label: str, frame: pd.DataFrame, now_utc: pd.Timestamp) -> tuple[list[str], list[str], list[str]]:
    spec = TIMEFRAMES[label]
    failures: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []
    if frame.empty:
        return [f"{label} store is empty"], warnings, [f"{label}: EMPTY"]
    opens = pd.to_datetime(frame["bar_open_utc"], utc=True, errors="coerce")
    closes = pd.to_datetime(frame["bar_close_utc"], utc=True, errors="coerce")
    observed = pd.to_datetime(frame["first_observed_utc"], utc=True, errors="coerce")
    numeric = frame[["open", "high", "low", "close", "tick_volume", "spread_points", "real_volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    duplicate = opens.duplicated(keep=False)
    invalid_time = opens.isna() | closes.isna() | observed.isna()
    wrong_duration = closes.sub(opens).ne(pd.Timedelta(minutes=spec.minutes))
    observed_too_early = observed.lt(closes)
    non_positive = (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
    high_bad = numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
    low_bad = numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
    negative_spread = numeric["spread_points"].lt(0)
    for name, mask in (
        ("invalid timestamps", invalid_time),
        ("duplicate bar opens", duplicate),
        ("wrong bar duration", wrong_duration),
        ("observed before close", observed_too_early),
        ("non-positive prices", non_positive),
        ("high enclosure violations", high_bad),
        ("low enclosure violations", low_bad),
        ("negative spread", negative_spread),
    ):
        count = int(mask.fillna(True).sum())
        if count:
            failures.append(f"{label} {name}: {count}")
    if not opens.is_monotonic_increasing:
        failures.append(f"{label} timestamps are not sorted")

    valid = closes.dropna().sort_values()
    last = valid.iloc[-1]
    age_minutes = max((now_utc - last).total_seconds() / 60.0, 0.0)
    if now_utc.weekday() < 5 and age_minutes > max(30, spec.minutes * 6):
        warnings.append(f"{label} latest completed bar is {age_minutes:.1f} minutes old")
    gaps = valid.diff().dropna()
    unexpected_gap_count = int((gaps > pd.Timedelta(minutes=spec.minutes * 3)).sum())
    if unexpected_gap_count:
        warnings.append(f"{label} has {unexpected_gap_count} gaps larger than 3 bars (weekends/closures may explain them)")
    lines.extend([
        f"{label}_rows: {len(frame)}",
        f"{label}_first_bar_open_utc: {opens.iloc[0]}",
        f"{label}_last_bar_close_utc: {last}",
        f"{label}_latest_age_minutes: {age_minutes:.3f}",
        f"{label}_duplicates: {int(duplicate.sum())}",
        f"{label}_negative_spread: {int(negative_spread.sum())}",
        f"{label}_large_gap_count: {unexpected_gap_count}",
    ])
    return failures, warnings, lines


def audit_dataset(dataset: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []
    if dataset.empty:
        return ["Feature dataset is empty"], warnings, ["dataset_rows: 0"]
    decision = pd.to_datetime(dataset["decision_timestamp_utc"], utc=True, errors="raise")
    if decision.duplicated().any():
        failures.append(f"Feature dataset duplicate decision timestamps: {int(decision.duplicated(keep=False).sum())}")
    if not decision.is_monotonic_increasing:
        failures.append("Feature dataset decisions are not sorted")
    for label in TIMEFRAMES:
        column = f"{label.lower()}_available_from_utc"
        available = pd.to_datetime(dataset[column], utc=True, errors="coerce")
        future = available.notna() & available.gt(decision)
        if future.any():
            failures.append(f"Future {label} feature availability leaked into {int(future.sum())} rows")
    forbidden = [column for column in FEATURE_COLUMNS if column.startswith("outcome_") or column.startswith("future_")]
    if forbidden:
        failures.append("Outcome/future fields present in feature registry: " + ", ".join(forbidden))

    complete = dataset["feature_complete"].fillna(False).astype(bool)
    lines.extend([
        f"dataset_rows: {len(dataset)}",
        f"feature_complete_rows: {int(complete.sum())}",
        f"feature_incomplete_rows: {int((~complete).sum())}",
        f"feature_count: {len(FEATURE_COLUMNS)}",
    ])
    for minutes in PREDICTION_HORIZONS_MINUTES:
        future_col = f"outcome_future_timestamp_{minutes}m"
        return_col = f"outcome_future_return_{minutes}m"
        matured_col = f"outcome_matured_at_utc_{minutes}m"
        future = pd.to_datetime(dataset[future_col], utc=True, errors="coerce")
        matured = pd.to_datetime(dataset[matured_col], utc=True, errors="coerce")
        labeled = dataset[return_col].notna()
        expected = decision + pd.Timedelta(minutes=minutes)
        wrong_time = labeled & future.ne(expected)
        wrong_maturity = labeled & matured.ne(expected)
        unlabeled_with_time = (~labeled) & (future.notna() | matured.notna())
        if wrong_time.any():
            failures.append(f"{minutes}m labels use wrong future timestamp: {int(wrong_time.sum())}")
        if wrong_maturity.any():
            failures.append(f"{minutes}m labels use wrong maturity timestamp: {int(wrong_maturity.sum())}")
        if unlabeled_with_time.any():
            failures.append(f"{minutes}m unlabeled rows carry future/maturity timestamps: {int(unlabeled_with_time.sum())}")
        lines.append(f"outcome_labeled_rows_{minutes}m: {int(labeled.sum())}")
    return failures, warnings, lines


def build_quality_report(frames: dict[str, pd.DataFrame], dataset: pd.DataFrame, now_utc: object | None = None) -> V05AQuality:
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    failures: list[str] = []
    warnings: list[str] = []
    lines = [
        "MARKETFUSION V0.5A CONTINUOUS MARKET DATA QUALITY",
        "=" * 78,
        "policy: completed bars only; immutable first-observed history; point-in-time feature joins; labels mature only after horizon",
        "trading_policy: research/data infrastructure only; no order execution",
        "",
    ]
    for label in TIMEFRAMES:
        if label not in frames:
            failures.append(f"Missing timeframe frame: {label}")
            continue
        f, w, detail = _frame_audit(label, frames[label], now)
        failures.extend(f)
        warnings.extend(w)
        lines.extend(detail)
    f, w, detail = audit_dataset(dataset)
    failures.extend(f)
    warnings.extend(w)
    lines.extend(["", *detail, ""])
    lines.append(f"V05A_QUALITY_STATUS: {'PASS' if not failures else 'FAIL'}")
    if failures:
        lines.append("FAILURES:")
        lines.extend(f"- {item}" for item in dict.fromkeys(failures))
    if warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {item}" for item in dict.fromkeys(warnings))
    if not failures:
        lines.append("No blocking bar-integrity, time-causality, or label-maturity violation detected.")
    report = "\n".join(lines) + "\n"
    return V05AQuality(not failures, failures, warnings, report)


def write_quality_report(result: V05AQuality, path: Path = QUALITY_REPORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.report, encoding="utf-8", newline="\n")
