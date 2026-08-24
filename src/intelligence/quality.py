"""Fail-closed quality checks for V0.5B continuous intelligence."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .v05b_contract import MACRO_FEATURE_SERIES, QUALITY_REPORT


@dataclass
class QualityResult:
    passed: bool
    status: str
    failures: list[str]
    warnings: list[str]
    report: str


def build_quality_report(
    news: pd.DataFrame,
    macro: pd.DataFrame,
    context: pd.DataFrame,
    captured_at: object,
    news_errors: list[str] | None = None,
    macro_errors: list[str] | None = None,
) -> QualityResult:
    now = pd.Timestamp(captured_at)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    failures: list[str] = []
    warnings: list[str] = []
    news_errors = news_errors or []
    macro_errors = macro_errors or []

    if news.empty:
        failures.append("No news metadata available from any configured source")
    else:
        observed = pd.to_datetime(news["first_observed_utc"], utc=True, errors="coerce")
        available = pd.to_datetime(news["available_from_utc"], utc=True, errors="coerce")
        if observed.isna().any() or available.isna().any():
            failures.append("News contains invalid first-observed/availability timestamps")
        if (observed > now).any():
            failures.append("News contains future first-observed timestamps")
        if not available.equals(observed):
            failures.append("News availability must equal conservative MarketFusion first-observed time")
        if news["news_id"].duplicated().any():
            failures.append("Duplicate canonical news_id rows")

    if macro.empty:
        failures.append("No macro observations available from any configured source")
    else:
        observed = pd.to_datetime(macro["first_observed_utc"], utc=True, errors="coerce")
        available = pd.to_datetime(macro["available_from_utc"], utc=True, errors="coerce")
        values = pd.to_numeric(macro["value"], errors="coerce")
        if observed.isna().any() or available.isna().any():
            failures.append("Macro contains invalid first-observed/availability timestamps")
        if (observed > now).any():
            failures.append("Macro contains future first-observed timestamps")
        if not available.equals(observed):
            failures.append("Macro availability must equal conservative MarketFusion first-observed time")
        if values.isna().any():
            failures.append("Macro contains non-numeric values")

    if context.empty or len(context) != 1:
        failures.append("Current context snapshot must contain exactly one row")
    else:
        decision = pd.to_datetime(context["decision_timestamp_utc"], utc=True, errors="coerce")
        if decision.isna().any() or decision.iloc[0] != now:
            failures.append("Context decision timestamp does not equal capture time")
        if any(column.startswith("outcome_") for column in context.columns):
            failures.append("Outcome columns are forbidden from V0.5B context")
        for series_id in MACRO_FEATURE_SERIES:
            seen_col = f"macro_{series_id}_first_observed_utc"
            if seen_col in context:
                seen = pd.to_datetime(context[seen_col], utc=True, errors="coerce")
                if (seen.notna() & (seen > now)).any():
                    failures.append(f"Future macro feature availability: {series_id}")

    if news_errors:
        warnings.append(f"News provider errors: {len(news_errors)}")
    if macro_errors:
        warnings.append(f"Macro provider errors: {len(macro_errors)}")
    status = "PASS" if not failures else "FAIL"
    lines = [
        "MARKETFUSION V0.5B CONTINUOUS NEWS + MACRO QUALITY",
        "=" * 72,
        f"captured_at_utc: {now.isoformat()}",
        f"news_rows_total: {len(news):,}",
        f"macro_rows_total: {len(macro):,}",
        f"context_rows_current: {len(context):,}",
        f"news_provider_errors: {len(news_errors)}",
        f"macro_provider_errors: {len(macro_errors)}",
        "availability_policy: MarketFusion first-observed capture time is the causal boundary",
        "historical_backdating_policy: forbidden",
        "trading_policy: research/data only; no order execution",
        f"V05B_QUALITY_STATUS: {status}",
    ]
    if failures:
        lines.append("FAILURES:")
        lines.extend(f"- {item}" for item in failures)
    if warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {item}" for item in warnings)
    report = "\n".join(lines) + "\n"
    return QualityResult(not failures, status, failures, warnings, report)


def write_quality_report(result: QualityResult) -> None:
    QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT.write_text(result.report, encoding="utf-8")
