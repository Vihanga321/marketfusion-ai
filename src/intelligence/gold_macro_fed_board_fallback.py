"""Official Federal Reserve Board HTML fallback for XAUUSD macro context.

FRED is the preferred V1.0B.3 source, but some networks can reset or stall
connections to fred.stlouisfed.org.  This fallback uses only official
federalreserve.gov H.10/H.15 pages and writes the same research-only context
contract consumed by MarketFusion.

The fallback is deliberately conservative and remains current-vintage,
research-only evidence.  It never makes a model production-eligible.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.assets.contracts import ROOT
from src.intelligence.gold_macro_context import (
    CONSERVATIVE_AVAILABILITY_LAG_DAYS,
    CONTEXT_SERIES,
    ContextSeries,
    _atomic_json,
    _atomic_parquet,
    context_manifest_path,
    context_store,
    series_cache_path,
    series_manifest_path,
)

PROVIDER_ID = "FEDERAL_RESERVE_BOARD_HTML"
CONTRACT_VERSION = "v1.0b3-gold-macro-fed-board-fallback-v1"
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 90.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0.0, 2.0, 6.0)

# These are official Board of Governors pages.  H.10 exposes the complete daily
# broad-dollar history on one summary page. H.15 preview pages expose a large
# recent daily window, which is sufficient for the controlled V1.0B.3 evidence
# experiment while keeping the source independent of fred.stlouisfed.org.
FED_BOARD_SOURCES: dict[str, dict[str, str]] = {
    "usd_broad": {
        "url": "https://www.federalreserve.gov/releases/h10/summary/jrxwtfb_nb.htm",
        "mode": "h10_summary",
        "source_series_id": "H10/H10/JRXWTFB_N.B",
    },
    "yield_2y": {
        "url": "https://www.federalreserve.gov/datadownload/Preview.aspx?pi=400&preview=H15%2FH15%2FRIFLGFCY02_N.B&rel=H15",
        "mode": "h15_preview",
        "source_series_id": "H15/H15/RIFLGFCY02_N.B",
    },
    "yield_10y": {
        "url": "https://www.federalreserve.gov/datadownload/Preview.aspx?pi=400&preview=H15%2FH15%2FRIFLGFCY10_N.B&rel=H15",
        "mode": "h15_preview",
        "source_series_id": "H15/H15/RIFLGFCY10_N.B",
    },
    "real_yield_10y": {
        "url": "https://www.federalreserve.gov/datadownload/Preview.aspx?pi=400&preview=H15%2FH15%2FRIFLGFCY10_XII_N.B&rel=H15",
        "mode": "h15_preview",
        "source_series_id": "H15/H15/RIFLGFCY10_XII_N.B",
    },
}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower == "tr":
            self._row = []
        elif lower in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"td", "th"} and self._cell is not None and self._row is not None:
            text = " ".join(" ".join(self._cell).split())
            self._row.append(text)
            self._cell = None
        elif lower == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _html_rows(text: str) -> list[list[str]]:
    parser = _TableParser()
    parser.feed(text)
    return parser.rows


def _to_numeric(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if cleaned.upper() in {"", "ND", "N/A", "NA", "."}:
        return None
    try:
        result = float(cleaned)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def _to_date(value: str) -> pd.Timestamp | None:
    try:
        parsed = pd.to_datetime(value.strip(), utc=True, errors="raise")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def parse_fed_board_html(text: str, spec: ContextSeries, mode: str) -> pd.DataFrame:
    observations: list[tuple[pd.Timestamp, float]] = []
    for row in _html_rows(text):
        if mode == "h10_summary":
            if len(row) < 2:
                continue
            date = _to_date(row[0])
            value = _to_numeric(row[1])
        elif mode == "h15_preview":
            if len(row) < 3:
                continue
            date = _to_date(row[-2])
            value = _to_numeric(row[-1])
        else:
            raise ValueError(f"Unsupported Federal Reserve Board parser mode: {mode}")
        if date is not None and value is not None:
            observations.append((date, value))

    if not observations:
        raise ValueError(f"Federal Reserve Board fallback returned no observations for {spec.series_id}")

    frame = pd.DataFrame(observations, columns=["observation_date_utc", "value"])
    frame = frame.drop_duplicates("observation_date_utc", keep="last").sort_values("observation_date_utc").reset_index(drop=True)
    frame["series_key"] = spec.key
    frame["series_id"] = spec.series_id
    frame["available_at_utc"] = (
        frame["observation_date_utc"]
        + pd.Timedelta(days=CONSERVATIVE_AVAILABILITY_LAG_DAYS)
        + pd.Timedelta(hours=23, minutes=59)
    )
    frame["change_1"] = frame["value"].diff(1)
    frame["change_5"] = frame["value"].diff(5)
    frame["pct_change_1"] = frame["value"].pct_change(1)
    frame["pct_change_5"] = frame["value"].pct_change(5)
    frame["provider_id"] = PROVIDER_ID
    frame["vintage_safe"] = False
    return frame.loc[:, [
        "series_key", "series_id", "observation_date_utc", "value",
        "available_at_utc", "change_1", "change_5", "pct_change_1",
        "pct_change_5", "provider_id", "vintage_safe",
    ]]


def _fetch_text(url: str, session: requests.Session | None = None) -> tuple[str, int]:
    client = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        delay = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        if delay:
            time.sleep(delay)
        try:
            response = client.get(
                url,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                headers={
                    "User-Agent": "MarketFusion-research/1.0",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("empty response")
            return response.text, attempt
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise RuntimeError(f"Federal Reserve Board fallback failed after {MAX_ATTEMPTS} attempts: {type(last_error).__name__}: {last_error}")


def _metadata(spec: ContextSeries, frame: pd.DataFrame, text: str, source: dict[str, str], attempts: int) -> dict[str, Any]:
    return {
        **asdict(spec),
        "provider_id": PROVIDER_ID,
        "provider_url": source["url"],
        "source_series_id": source["source_series_id"],
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "payload_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "rows": len(frame),
        "first_observation": pd.Timestamp(frame["observation_date_utc"].min()).isoformat(),
        "last_observation": pd.Timestamp(frame["observation_date_utc"].max()).isoformat(),
        "availability_policy": f"observation_date + {CONSERVATIVE_AVAILABILITY_LAG_DAYS} calendar days + 23:59 UTC",
        "attempts": attempts,
        "vintage_safe": False,
        "research_only": True,
        "production_model_eligible": False,
    }


def bootstrap_fed_board_context(root: Path = ROOT, session: requests.Session | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in CONTEXT_SERIES:
        source = FED_BOARD_SOURCES[spec.key]
        try:
            text, attempts = _fetch_text(source["url"], session=session)
            frame = parse_fed_board_html(text, spec, source["mode"])
            # A tiny HTML fragment is not enough for serious walk-forward research.
            if len(frame) < 100:
                raise ValueError(f"only {len(frame)} daily observations were parsed")
            metadata = _metadata(spec, frame, text, source, attempts)
            _atomic_parquet(series_cache_path(spec, root), frame)
            _atomic_json(series_manifest_path(spec, root), metadata)
            frames.append(frame)
            metadata_rows.append(metadata)
        except Exception as exc:
            failures.append(f"{spec.series_id}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("Federal Reserve Board fallback incomplete. Missing: " + " | ".join(failures))

    combined = pd.concat(frames, ignore_index=True).sort_values(["series_key", "available_at_utc"]).reset_index(drop=True)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "provider_id": PROVIDER_ID,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "series": metadata_rows,
        "row_count": len(combined),
        "vintage_safe": False,
        "production_model_eligible": False,
        "research_only": True,
        "automatic_execution": "DISABLED",
    }
    _atomic_parquet(context_store(root), combined)
    _atomic_json(context_manifest_path(root), manifest)
    return combined, manifest


def main() -> int:
    frame, manifest = bootstrap_fed_board_context(ROOT)
    print(json.dumps({
        "status": "PASS",
        "provider_id": PROVIDER_ID,
        "rows": len(frame),
        "series": [row["series_id"] for row in manifest["series"]],
        "vintage_safe": False,
        "production_model_eligible": False,
        "automatic_execution": "DISABLED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
