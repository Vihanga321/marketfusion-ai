"""Atomic append-only stores for V0.5B point-in-time intelligence."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .v05b_contract import CONTEXT_HISTORY_FILE, MACRO_FILE, NEWS_FILE, STATUS_FILE


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def atomic_write_json(payload: dict, path: Path = STATUS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow") if path.exists() else pd.DataFrame()


def merge_news(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if incoming.empty:
        return existing.copy(), 0
    inc = incoming.copy()
    inc["first_observed_utc"] = pd.to_datetime(inc["first_observed_utc"], utc=True, errors="raise")
    inc["available_from_utc"] = pd.to_datetime(inc["available_from_utc"], utc=True, errors="raise")
    if existing.empty:
        result = inc.drop_duplicates("news_id", keep="first").sort_values("first_observed_utc").reset_index(drop=True)
        return result, len(result)
    old = existing.copy()
    old["first_observed_utc"] = pd.to_datetime(old["first_observed_utc"], utc=True, errors="raise")
    old_ids = set(old["news_id"].astype(str))
    new_rows = inc.loc[~inc["news_id"].astype(str).isin(old_ids)].copy()
    result = pd.concat([old, new_rows], ignore_index=True)
    result = result.drop_duplicates("news_id", keep="first").sort_values("first_observed_utc").reset_index(drop=True)
    return result, len(new_rows)


def macro_identity(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["series_id"].astype(str) + "|" +
        pd.to_datetime(frame["observation_period"], utc=True, errors="raise").astype(str) + "|" +
        pd.to_numeric(frame["value"], errors="raise").map(lambda value: f"{float(value):.12g}")
    )


def merge_macro(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if incoming.empty:
        return existing.copy(), 0
    inc = incoming.copy()
    inc["observation_period"] = pd.to_datetime(inc["observation_period"], utc=True, errors="raise")
    inc["first_observed_utc"] = pd.to_datetime(inc["first_observed_utc"], utc=True, errors="raise")
    inc["available_from_utc"] = pd.to_datetime(inc["available_from_utc"], utc=True, errors="raise")
    inc["_identity"] = macro_identity(inc)
    if existing.empty:
        result = inc.drop_duplicates("_identity", keep="first").drop(columns="_identity").sort_values(["first_observed_utc", "series_id"]).reset_index(drop=True)
        return result, len(result)
    old = existing.copy()
    old["observation_period"] = pd.to_datetime(old["observation_period"], utc=True, errors="raise")
    old["first_observed_utc"] = pd.to_datetime(old["first_observed_utc"], utc=True, errors="raise")
    old["_identity"] = macro_identity(old)
    old_ids = set(old["_identity"])
    new_rows = inc.loc[~inc["_identity"].isin(old_ids)].copy()
    result = pd.concat([old, new_rows], ignore_index=True)
    result = result.drop_duplicates("_identity", keep="first").drop(columns="_identity")
    result = result.sort_values(["first_observed_utc", "series_id"]).reset_index(drop=True)
    return result, len(new_rows)


def append_context(existing: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    if row.empty:
        return existing.copy()
    current = row.copy()
    current["decision_timestamp_utc"] = pd.to_datetime(current["decision_timestamp_utc"], utc=True, errors="raise")
    if existing.empty:
        return current.reset_index(drop=True)
    old = existing.copy()
    old["decision_timestamp_utc"] = pd.to_datetime(old["decision_timestamp_utc"], utc=True, errors="raise")
    result = pd.concat([old, current], ignore_index=True)
    return result.drop_duplicates("decision_timestamp_utc", keep="last").sort_values("decision_timestamp_utc").reset_index(drop=True)


def load_stores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_parquet_or_empty(NEWS_FILE),
        read_parquet_or_empty(MACRO_FILE),
        read_parquet_or_empty(CONTEXT_HISTORY_FILE),
    )


def save_stores(news: pd.DataFrame, macro: pd.DataFrame, context: pd.DataFrame) -> None:
    atomic_write_parquet(news, NEWS_FILE)
    atomic_write_parquet(macro, MACRO_FILE)
    atomic_write_parquet(context, CONTEXT_HISTORY_FILE)
