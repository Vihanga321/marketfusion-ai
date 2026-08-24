"""Causal V0.5B live-context feature snapshots."""
from __future__ import annotations

import pandas as pd

from .v05b_contract import MACRO_FEATURE_SERIES, NEWS_WINDOWS_MINUTES

NEWS_FLAG_COLUMNS = (
    "high_impact_flag", "usd_relevant", "eur_relevant",
    "topic_central_bank", "topic_rates", "topic_inflation",
    "topic_labor", "topic_growth", "topic_risk",
)


def _as_utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def build_context_snapshot(news: pd.DataFrame, macro: pd.DataFrame, decision_timestamp: object) -> pd.DataFrame:
    decision = _as_utc(decision_timestamp)
    row: dict[str, object] = {
        "decision_timestamp_utc": decision,
        "context_contract": "v0.5b-live-context-v1",
    }

    if news.empty:
        eligible_news = pd.DataFrame()
    else:
        eligible_news = news.copy()
        eligible_news["first_observed_utc"] = pd.to_datetime(eligible_news["first_observed_utc"], utc=True, errors="raise")
        eligible_news = eligible_news.loc[eligible_news["first_observed_utc"] <= decision].copy()

    for minutes in NEWS_WINDOWS_MINUTES:
        start = decision - pd.Timedelta(minutes=minutes)
        window = eligible_news.loc[eligible_news["first_observed_utc"] > start] if not eligible_news.empty else eligible_news
        prefix = f"news_{minutes}m"
        row[f"{prefix}_count"] = int(len(window))
        if window.empty:
            row[f"{prefix}_official_count"] = 0
            row[f"{prefix}_gdelt_count"] = 0
            for flag in NEWS_FLAG_COLUMNS:
                row[f"{prefix}_{flag}_count"] = 0
            continue
        row[f"{prefix}_official_count"] = int(window["authority"].eq("OFFICIAL").sum())
        row[f"{prefix}_gdelt_count"] = int(window["source_id"].eq("GDELT_EURUSD").sum())
        for flag in NEWS_FLAG_COLUMNS:
            row[f"{prefix}_{flag}_count"] = int(pd.to_numeric(window[flag], errors="coerce").fillna(0).sum())

    if macro.empty:
        eligible_macro = pd.DataFrame()
    else:
        eligible_macro = macro.copy()
        eligible_macro["first_observed_utc"] = pd.to_datetime(eligible_macro["first_observed_utc"], utc=True, errors="raise")
        eligible_macro["observation_period"] = pd.to_datetime(eligible_macro["observation_period"], utc=True, errors="raise")
        eligible_macro = eligible_macro.loc[eligible_macro["first_observed_utc"] <= decision].copy()

    for series_id in MACRO_FEATURE_SERIES:
        feature = f"macro_{series_id}"
        candidates = eligible_macro.loc[eligible_macro["series_id"].eq(series_id)].copy() if not eligible_macro.empty else eligible_macro
        if candidates.empty:
            row[f"{feature}_value"] = None
            row[f"{feature}_observation_utc"] = pd.NaT
            row[f"{feature}_first_observed_utc"] = pd.NaT
            row[f"{feature}_age_hours"] = None
            continue
        latest = candidates.sort_values(["first_observed_utc", "observation_period"]).iloc[-1]
        first_observed = _as_utc(latest["first_observed_utc"])
        row[f"{feature}_value"] = float(latest["value"])
        row[f"{feature}_observation_utc"] = _as_utc(latest["observation_period"])
        row[f"{feature}_first_observed_utc"] = first_observed
        row[f"{feature}_age_hours"] = float((decision - first_observed).total_seconds() / 3600.0)

    fed = row.get("macro_us_effective_fed_funds_value")
    ecb = row.get("macro_ecb_main_refinancing_rate_value")
    row["macro_policy_rate_spread_us_minus_ecb_pctpt"] = (
        float(fed) - float(ecb) if fed is not None and ecb is not None else None
    )
    return pd.DataFrame([row])


def model_feature_view(context: pd.DataFrame) -> pd.DataFrame:
    forbidden = [column for column in context.columns if column.startswith("outcome_")]
    if forbidden:
        raise ValueError(f"V0.5B context contains forbidden outcome columns: {forbidden}")
    return context.copy()
