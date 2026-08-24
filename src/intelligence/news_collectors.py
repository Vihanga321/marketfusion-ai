"""Free, read-only V0.5B news collectors.

Only metadata/headlines are stored. MarketFusion uses its own first-observed UTC
as the conservative availability time; provider publication timestamps are kept
for reference but never used to move information earlier than our capture.
"""
from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import html
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .v05b_contract import (
    EUR_KEYWORDS, GDELT_QUERY, HIGH_IMPACT_KEYWORDS, NEWS_SOURCES,
    TOPIC_KEYWORDS, USD_KEYWORDS,
)

USER_AGENT = "MarketFusionAI/0.5B research collector"
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
RSS_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.2",
    "Cache-Control": "no-cache",
}
JSON_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, */*;q=0.2",
}


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def clean_text(value: object, limit: int = 600) -> str:
    text = html.unescape(TAG_RE.sub(" ", str(value or "")))
    text = SPACE_RE.sub(" ", text).strip()
    return text[:limit]


def parse_provider_time(value: object) -> pd.Timestamp | pd.NaT:
    raw = str(value or "").strip()
    if not raw:
        return pd.NaT
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return pd.Timestamp(dt).tz_convert("UTC")
    except Exception:
        return pd.to_datetime(raw, utc=True, errors="coerce")


def stable_news_id(source_id: str, provider_id: str, url: str, title: str) -> str:
    basis = "|".join((source_id, provider_id.strip(), url.strip(), title.strip())).encode("utf-8")
    return sha256(basis).hexdigest()


def classify_text(title: str, summary: str, source_region: str) -> dict[str, object]:
    text = f"{title} {summary}".lower()
    topic_flags = {
        f"topic_{topic}": int(any(keyword in text for keyword in keywords))
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    usd = int(source_region == "USD" or any(keyword in text for keyword in USD_KEYWORDS))
    eur = int(source_region == "EUR" or any(keyword in text for keyword in EUR_KEYWORDS))
    high = int(any(keyword in text for keyword in HIGH_IMPACT_KEYWORDS))
    return {"usd_relevant": usd, "eur_relevant": eur, "high_impact_flag": high, **topic_flags}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_text(element: ET.Element, names: tuple[str, ...]) -> str:
    for child in element.iter():
        if _local_name(child.tag) in names and child.text:
            return child.text.strip()
    return ""


def _link(element: ET.Element) -> str:
    for child in element.iter():
        if _local_name(child.tag) == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
            if child.text:
                return child.text.strip()
    return ""


def _parse_xml_root(xml_data: str | bytes) -> ET.Element:
    """Parse RSS/Atom safely without forcing requests' guessed text encoding.

    Federal Reserve feeds are standards-compliant XML but can be served with an
    encoding declaration/BOM that is safer to let ElementTree decode from bytes.
    Tests and local fixtures may still pass ordinary strings.
    """
    if isinstance(xml_data, bytes):
        payload = xml_data.lstrip(b"\xef\xbb\xbf\x00\x20\x09\x0d\x0a")
        if not payload.startswith(b"<"):
            raise ValueError("RSS endpoint returned a non-XML payload")
        return ET.fromstring(payload)
    text = xml_data.lstrip("\ufeff\x00 \t\r\n")
    if not text.startswith("<"):
        raise ValueError("RSS endpoint returned a non-XML payload")
    return ET.fromstring(text)


def parse_rss_xml(xml_text: str | bytes, source_id: str, captured_at: pd.Timestamp) -> pd.DataFrame:
    root = _parse_xml_root(xml_text)
    spec = NEWS_SOURCES[source_id]
    rows: list[dict[str, object]] = []
    candidates = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
    for item in candidates:
        title = clean_text(_first_text(item, ("title",)), 300)
        url = _link(item)
        provider_id = _first_text(item, ("guid", "id")) or url
        summary = clean_text(_first_text(item, ("description", "summary", "content")), 600)
        provider_published = parse_provider_time(_first_text(item, ("pubdate", "published", "updated", "date")))
        if not title and not url:
            continue
        row = {
            "news_id": stable_news_id(source_id, provider_id, url, title),
            "source_id": source_id,
            "authority": spec["authority"],
            "source_region": spec["region"],
            "provider_id": provider_id,
            "title": title,
            "summary_excerpt": summary,
            "url": url,
            "domain": re.sub(r"^www\.", "", requests.utils.urlparse(url).netloc.lower()),
            "provider_published_utc": provider_published,
            "first_observed_utc": captured_at,
            "available_from_utc": captured_at,
            "availability_method": "marketfusion_first_observed_capture",
            "source_kind": "OFFICIAL_RSS",
        }
        row.update(classify_text(title, summary, spec["region"]))
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_rss(source_id: str, captured_at: pd.Timestamp, session: requests.Session | None = None) -> pd.DataFrame:
    spec = NEWS_SOURCES[source_id]
    client = session or requests.Session()
    response = client.get(spec["url"], timeout=(10, 30), headers=RSS_HEADERS)
    response.raise_for_status()
    # Parse bytes so XML encoding declarations/BOMs remain authoritative.
    return parse_rss_xml(response.content, source_id, captured_at)


def parse_gdelt_json(payload: dict, captured_at: pd.Timestamp) -> pd.DataFrame:
    spec = NEWS_SOURCES["GDELT_EURUSD"]
    rows: list[dict[str, object]] = []
    for article in payload.get("articles", []) or []:
        title = clean_text(article.get("title"), 300)
        url = str(article.get("url") or "").strip()
        if not title and not url:
            continue
        provider_id = str(article.get("url") or article.get("socialimage") or title)
        seen = pd.to_datetime(article.get("seendate"), utc=True, errors="coerce")
        row = {
            "news_id": stable_news_id("GDELT_EURUSD", provider_id, url, title),
            "source_id": "GDELT_EURUSD",
            "authority": spec["authority"],
            "source_region": spec["region"],
            "provider_id": provider_id,
            "title": title,
            "summary_excerpt": "",
            "url": url,
            "domain": str(article.get("domain") or "").lower(),
            "provider_published_utc": seen,
            "first_observed_utc": captured_at,
            "available_from_utc": captured_at,
            "availability_method": "marketfusion_first_observed_capture",
            "source_kind": "GDELT_ARTICLE_METADATA",
        }
        row.update(classify_text(title, "", spec["region"]))
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_gdelt(captured_at: pd.Timestamp, session: requests.Session | None = None) -> pd.DataFrame:
    client = session or requests.Session()
    response = client.get(
        NEWS_SOURCES["GDELT_EURUSD"]["url"],
        params={
            "query": GDELT_QUERY,
            "mode": "ArtList",
            "maxrecords": 25,
            "timespan": "3h",
            "format": "json",
            "sort": "DateDesc",
        },
        timeout=(10, 30),
        headers=JSON_HEADERS,
    )
    if response.status_code == 429:
        raise RuntimeError("GDELT rate limited this cycle; collector will retry later")
    response.raise_for_status()
    return parse_gdelt_json(response.json(), captured_at)


def collect_news(captured_at: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    capture = captured_at or utc_now()
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    counts: dict[str, int] = {}
    with requests.Session() as session:
        for source_id, spec in NEWS_SOURCES.items():
            try:
                frame = fetch_gdelt(capture, session) if spec["kind"] == "gdelt" else fetch_rss(source_id, capture, session)
                counts[source_id] = len(frame)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                counts[source_id] = 0
                errors.append(f"{source_id}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors, counts
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates("news_id", keep="first").reset_index(drop=True)
    return result, errors, counts
