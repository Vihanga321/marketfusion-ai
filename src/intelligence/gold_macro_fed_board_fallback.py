"""Official Federal Reserve Board fallback for XAUUSD macro context.

FRED is the preferred V1.0B.3 source, but some networks can reset or stall
connections to fred.stlouisfed.org. This fallback uses only official
federalreserve.gov sources and writes the same research-only context contract
consumed by MarketFusion.

Important: H.15 preview pages intentionally expose only a small recent sample.
For research history this module uses the Federal Reserve Data Download
Program's direct CSV output for automated systems instead of the preview page.
The fallback remains current-vintage, research-only evidence and never makes a
model production-eligible.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
from hashlib import sha256
from html.parser import HTMLParser
from io import StringIO
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

PROVIDER_ID = "FEDERAL_RESERVE_BOARD_OFFICIAL"
CONTRACT_VERSION = "v1.0b3-gold-macro-fed-board-fallback-v2"
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 120.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (0.0, 2.0, 6.0)

# Official Board of Governors sources.
#
# H.10 broad-dollar history is exposed on the Board summary page.
# H.15 research history is downloaded through the DDP direct CSV endpoint for
# automated systems. The nominal Treasury package contains the 2Y and 10Y
# constant-maturity series. A second official H.15 package contains nominal and
# inflation-indexed long-maturity series, including the 10Y real yield.
H15_NOMINAL_PACKAGE_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=H15&series=bf17364827e38702b42a58cf8eaa3f78&lastobs=&from=&to=&"
    "filetype=csv&label=include&layout=seriescolumn&type=package"
)
H15_REAL_PACKAGE_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=H15&series=0b98a66d3ff5e1ea0fbf88adc59b387f&lastobs=&from=&to=&"
    "filetype=csv&label=include&layout=seriescolumn&type=package"
)

FED_BOARD_SOURCES: dict[str, dict[str, str]] = {
    "usd_broad": {
        "url": "https://www.federalreserve.gov/releases/h10/summary/jrxwtfb_nb.htm",
        "mode": "h10_summary",
        "source_series_id": "H10/H10/JRXWTFB_N.B",
        "source_column": "JRXWTFB_N.B",
    },
    "yield_2y": {
        "url": H15_NOMINAL_PACKAGE_URL,
        "mode": "h15_ddp_csv",
        "source_series_id": "H15/H15/RIFLGFCY02_N.B",
        "source_column": "RIFLGFCY02_N.B",
    },
    "yield_10y": {
        "url": H15_NOMINAL_PACKAGE_URL,
        "mode": "h15_ddp_csv",
        "source_series_id": "H15/H15/RIFLGFCY10_N.B",
        "source_column": "RIFLGFCY10_N.B",
    },
    "real_yield_10y": {
        "url": H15_REAL_PACKAGE_URL,
        "mode": "h15_ddp_csv",
        "source_series_id": "H15/H15/RIFLGFCY10_XII_N.B",
        "source_column": "RIFLGFCY10_XII_N.B",
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


def _normalize_observations(
    observations: list[tuple[pd.Timestamp, float]], spec: ContextSeries
) -> pd.DataFrame:
    if not observations:
        raise ValueError(f"Federal Reserve Board fallback returned no observations for {spec.series_id}")
    frame = pd.DataFrame(observations, columns=["observation_date_utc", "value"])
    frame = (
        frame.drop_duplicates("observation_date_utc", keep="last")
        .sort_values("observation_date_utc")
        .reset_index(drop=True)
    )
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


def _parse_h10_summary(text: str, spec: ContextSeries) -> pd.DataFrame:
    observations: list[tuple[pd.Timestamp, float]] = []
    for row in _html_rows(text):
        if len(row) < 2:
            continue
        date = _to_date(row[0])
        value = _to_numeric(row[1])
        if date is not None and value is not None:
            observations.append((date, value))
    return _normalize_observations(observations, spec)


def _parse_h15_ddp_csv(text: str, spec: ContextSeries, source_column: str) -> pd.DataFrame:
    """Parse a full H.15 Data Download Program CSV package.

    DDP CSV files include several metadata lines before the row whose first
    field is ``Time Period``. We locate that row instead of depending on a
    fixed header offset, then read only the requested official series column.
    """
    rows = list(csv.reader(StringIO(text)))
    header_index: int | None = None
    headers: list[str] = []
    for index, row in enumerate(rows):
        cleaned = [cell.strip().lstrip("\ufeff") for cell in row]
        if cleaned and cleaned[0].lower() == "time period":
            header_index = index
            headers = cleaned
            break
    if header_index is None:
        raise ValueError("Federal Reserve H.15 CSV is missing the Time Period header")

    # The DDP normally uses the short series code in the CSV header. Be
    # tolerant of a fully qualified identifier if the provider changes labels.
    candidates = {
        source_column,
        f"H15/H15/{source_column}",
        source_column.replace("H15/H15/", ""),
    }
    value_index: int | None = None
    for idx, header in enumerate(headers):
        if header in candidates or header.rsplit("/", 1)[-1] == source_column:
            value_index = idx
            break
    if value_index is None:
        raise ValueError(
            f"Federal Reserve H.15 CSV is missing requested series column {source_column}; "
            f"available columns: {', '.join(headers[:20])}"
        )

    observations: list[tuple[pd.Timestamp, float]] = []
    for row in rows[header_index + 1:]:
        if not row or len(row) <= value_index:
            continue
        date = _to_date(row[0])
        value = _to_numeric(row[value_index])
        if date is not None and value is not None:
            observations.append((date, value))
    return _normalize_observations(observations, spec)


def parse_fed_board_payload(
    text: str, spec: ContextSeries, mode: str, source_column: str
) -> pd.DataFrame:
    if mode == "h10_summary":
        return _parse_h10_summary(text, spec)
    if mode == "h15_ddp_csv":
        return _parse_h15_ddp_csv(text, spec, source_column)
    raise ValueError(f"Unsupported Federal Reserve Board parser mode: {mode}")


def _fetch_text(url: str, session: requests.Session | None = None) -> tuple[str, int]:
    client = session or requests.Session()
    last_error: Exception | None = None
    is_csv = "Output.aspx" in url
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
                    "Accept": "text/csv,*/*;q=0.8" if is_csv else "text/html,application/xhtml+xml",
                },
            )
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("empty response")
            return response.text, attempt
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise RuntimeError(
        f"Federal Reserve Board fallback failed after {MAX_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _metadata(
    spec: ContextSeries,
    frame: pd.DataFrame,
    text: str,
    source: dict[str, str],
    attempts: int,
) -> dict[str, Any]:
    return {
        **asdict(spec),
        "provider_id": PROVIDER_ID,
        "provider_url": source["url"],
        "source_series_id": source["source_series_id"],
        "source_mode": source["mode"],
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "payload_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "rows": len(frame),
        "first_observation": pd.Timestamp(frame["observation_date_utc"].min()).isoformat(),
        "last_observation": pd.Timestamp(frame["observation_date_utc"].max()).isoformat(),
        "availability_policy": (
            f"observation_date + {CONSERVATIVE_AVAILABILITY_LAG_DAYS} "
            "calendar days + 23:59 UTC"
        ),
        "attempts": attempts,
        "vintage_safe": False,
        "research_only": True,
        "production_model_eligible": False,
    }


def bootstrap_fed_board_context(
    root: Path = ROOT, session: requests.Session | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    # Shared H.15 packages are fetched once and parsed into multiple series.
    payload_cache: dict[str, tuple[str, int]] = {}

    for spec in CONTEXT_SERIES:
        source = FED_BOARD_SOURCES[spec.key]
        try:
            url = source["url"]
            if url not in payload_cache:
                payload_cache[url] = _fetch_text(url, session=session)
            text, attempts = payload_cache[url]
            frame = parse_fed_board_payload(
                text, spec, source["mode"], source["source_column"]
            )
            # A tiny preview/sample is not enough for walk-forward research.
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
        raise RuntimeError(
            "Federal Reserve Board fallback incomplete. Missing: " + " | ".join(failures)
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["series_key", "available_at_utc"])
        .reset_index(drop=True)
    )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "provider_id": PROVIDER_ID,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "series": metadata_rows,
        "row_count": len(combined),
        "unique_provider_requests": len(payload_cache),
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
        "unique_provider_requests": manifest["unique_provider_requests"],
        "vintage_safe": False,
        "production_model_eligible": False,
        "automatic_execution": "DISABLED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
