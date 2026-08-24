"""Read-only Trading Economics calendar adapter for V0.4D consensus audits."""

from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import quote

import requests


BASE_URL = "https://api.tradingeconomics.com"


class TradingEconomicsError(RuntimeError):
    pass


def fetch_calendar(
    api_key: str,
    *,
    indicator: str,
    start_date: str | date,
    end_date: str | date,
    timeout_seconds: float = 30.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch a historical calendar range without logging or returning the credential."""

    if not api_key or not api_key.strip():
        raise TradingEconomicsError("Trading Economics API credential is missing")
    start = str(start_date)
    end = str(end_date)
    encoded_indicator = quote(indicator.strip(), safe="")
    url = (
        f"{BASE_URL}/calendar/country/united%20states/indicator/"
        f"{encoded_indicator}/{start}/{end}"
    )
    client = session or requests.Session()
    try:
        response = client.get(
            url,
            params={"c": api_key, "f": "json", "values": "true"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise TradingEconomicsError(f"Trading Economics request failed: {type(exc).__name__}") from None
    if response.status_code != 200:
        raise TradingEconomicsError(f"Trading Economics HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        raise TradingEconomicsError("Trading Economics returned non-JSON data") from None
    if not isinstance(payload, list):
        raise TradingEconomicsError("Trading Economics calendar response is not a list")
    return [row for row in payload if isinstance(row, dict)]
