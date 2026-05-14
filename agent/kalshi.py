"""Read-only Kalshi market-data client.

Kalshi's `GET /trade-api/v2/markets/{ticker}` is public and unauthenticated
(verified 2026-05-14). Responses are CloudFront-cached (15s public cache).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_KALSHI_BASE_URL = "https://api.elections.kalshi.com"


def kalshi_base_url() -> str:
    return os.environ.get("KALSHI_BASE_URL", DEFAULT_KALSHI_BASE_URL)


def get_market(ticker: str, *, timeout: float = 10.0) -> dict[str, Any] | None:
    """Fetch a single Kalshi market by ticker. Returns the market dict, or None on failure."""
    url = f"{kalshi_base_url()}/trade-api/v2/markets/{ticker}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Kalshi fetch failed for %s: %s", ticker, e)
        return None
    data = resp.json()
    return data.get("market", data)
