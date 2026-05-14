"""Manifold-Markets-backed prior for Politics, Sports, and similar categories.

Manifold has wide coverage of political and sporting events with active,
calibrated probabilities (the same property that makes Kalshi prices
useful). We search Manifold for an open market that matches the Kalshi
event title and use its current probability as our forecast.

Why this works: Manifold markets are independent of Kalshi markets — they
trade against different traders with different information — so using a
Manifold price as our independent forecast IS a real second opinion, not
a copy of the Kalshi book.

Why this fails gracefully: many Kalshi events have no Manifold equivalent
(niche tournaments, very-short-dated markets). In that case we return None
and the agent falls through to the LLM ensemble.

Manifold API: https://docs.manifold.markets/api (no auth required for
read endpoints; modest rate limits).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE = "https://api.manifold.markets/v0"
TIMEOUT = 10.0
MIN_MARKET_VOLUME = 50.0  # mana — Manifold's internal currency. Low bar.
MAX_QUERY_LEN = 200       # Manifold doesn't like very long search terms.


def _clean_query(title: str) -> str:
    """Strip Kalshi-isms from the title to give Manifold a cleaner search term."""
    s = title.strip()
    # Common openers we don't need in the search.
    s = re.sub(r"^will\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^how\s+\w+\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^who\s+will\s+", "", s, flags=re.IGNORECASE)
    s = s.rstrip("?").strip()
    return s[:MAX_QUERY_LEN]


def _search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    if not query:
        return []
    try:
        r = requests.get(
            f"{BASE}/search-markets",
            params={"term": query, "limit": limit, "filter": "open", "sort": "score"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        logger.warning("Manifold search failed for %r: %s", query, e)
        return []


def _llm_search_query(title: str) -> str | None:
    """Use a small LLM to extract a better search query from a Kalshi title.

    Kalshi titles are often verbose ("Will the temperature in...") whereas
    Manifold's full-text search rewards concise topic phrases. A 3-5 word
    query like "Trump pardon Hunter Biden" beats the full sentence form.

    Returns None on any failure — caller falls back to the rule-based
    `_clean_query`.
    """
    try:
        # Lazy import to keep manifold.py's import cost low for callers
        # that don't trigger the LLM rephrase path.
        from agent.llm import _get_anthropic_client

        client = _get_anthropic_client()
        if client is None:
            return None
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract a 3-5 word search query from this prediction "
                        "market title. Use the most distinctive nouns and proper "
                        "names. Output ONLY the query — no quotes, no "
                        "explanation.\n\nTitle: " + title
                    ),
                }
            ],
        )
        text = ""
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text" or hasattr(block, "text"):
                t = getattr(block, "text", None)
                if t:
                    text += t
        text = text.strip().strip('"').strip("'")
        if not text or len(text) > 200:
            return None
        return text
    except Exception as e:
        logger.warning("LLM query rephrase failed: %s", e)
        return None


def _is_usable(market: dict[str, Any]) -> bool:
    if market.get("outcomeType") != "BINARY":
        return False
    if market.get("isResolved"):
        return False
    try:
        volume = float(market.get("volume", 0) or 0)
    except (ValueError, TypeError):
        return False
    if volume < MIN_MARKET_VOLUME:
        return False
    p = market.get("probability")
    if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
        return False
    return True


def manifold_prior(event: dict) -> tuple[float, str] | None:
    """Search Manifold for a matching binary market; return its probability.

    Two-pass search: first uses a rule-based cleaned query; if that
    returns no usable matches, asks a small LLM to rephrase the title
    as a concise topic query and tries again. This roughly doubles the
    match rate on niche or verbosely-titled markets.

    Returns (p_yes, rationale) on success, or None if no usable match.
    """
    title = event.get("title", "")
    if not title:
        return None

    query = _clean_query(title)
    results = _search(query) if query else []
    usable = [m for m in results if _is_usable(m)]

    rephrase_note = ""
    if not usable:
        rephrased = _llm_search_query(title)
        if rephrased and rephrased.lower() != query.lower():
            results = _search(rephrased)
            usable = [m for m in results if _is_usable(m)]
            if usable:
                rephrase_note = f" (LLM-rephrased to '{rephrased}')"

    if not usable:
        return None

    best = usable[0]
    p = float(best["probability"])
    p = max(0.01, min(0.99, p))

    question = (best.get("question") or "")[:80]
    volume = best.get("volume", 0)
    return p, (
        f"Manifold market '{question}' has probability {best['probability']:.3f} "
        f"(volume {volume:.0f} mana){rephrase_note}"
    )
