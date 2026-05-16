"""Polymarket cross-reference client.

Polymarket runs the largest US prediction-market book for politics and
news events. Many Prophet Hacks markets have Polymarket siblings —
trading against different counterparties with different information,
so blending the two books is a real independent signal, not a copy.

Strategy:
  1. Search Polymarket via the gamma-api for the Kalshi event title.
  2. Score candidate matches by token overlap (proper nouns + numbers
     dominate).
  3. Return (p_yes, depth_proxy, rationale) for the best match if any
     candidate clears MATCH_THRESHOLD.

Anything goes wrong → return None and the caller falls through to the
Kalshi-only path. Polymarket is a bonus signal, never a hard dep.

API: https://gamma-api.polymarket.com — public read endpoints, no auth.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 8.0
SEARCH_LIMIT = 20
MATCH_THRESHOLD = 0.55     # token-overlap score required to call it a match
MIN_VOLUME_24H = 100.0     # USD — below this Polymarket book is too thin to trust

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "will", "be", "is", "are", "was", "were", "by",
        "on", "in", "at", "to", "for", "of", "and", "or", "from", "with",
        "this", "that", "it", "do", "does", "did", "have", "has", "had",
        "any", "before", "after", "more", "less", "than", "many", "much",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with stopwords removed.

    Numbers and proper-noun-like tokens (kept by being alphanumeric) carry
    most of the matching signal, so we don't try to be cleverer than this.
    """
    raw = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {t for t in raw if len(t) > 1 and t not in _STOPWORDS}


def _overlap(a: set[str], b: set[str]) -> float:
    """Jaccard-like overlap, biased toward the smaller (query) side.

    Polymarket questions tend to be longer than Kalshi titles, so we use
    `min(|a|, |b|)` in the denominator instead of `|a ∪ b|`. Otherwise a
    perfect-substring match scores low purely because the Poly question
    has extra context words.
    """
    if not a or not b:
        return 0.0
    common = a & b
    return len(common) / min(len(a), len(b))


def _search(query: str, limit: int = SEARCH_LIMIT) -> list[dict[str, Any]]:
    """Fetch open Polymarket markets matching `query`. Empty list on failure."""
    if not query:
        return []
    try:
        r = requests.get(
            f"{BASE}/markets",
            params={
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false",
                "search": query,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("Polymarket search failed for %r: %s", query, e)
        return []
    except ValueError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data") or data.get("markets") or []
    return []


def _parse_outcome_prices(market: dict[str, Any]) -> tuple[float, float] | None:
    """Polymarket returns outcomePrices as a JSON-stringified list of strings."""
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    try:
        yes = float(raw[0])
        no = float(raw[1])
    except (ValueError, TypeError):
        return None
    if not (0.0 <= yes <= 1.0 and 0.0 <= no <= 1.0):
        return None
    return yes, no


def _f(market: dict[str, Any], *keys: str) -> float:
    """First numeric field present, else 0.0."""
    for k in keys:
        v = market.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (ValueError, TypeError):
            continue
    return 0.0


def _market_p_yes(market: dict[str, Any]) -> float | None:
    """Best-effort YES probability from a Polymarket market dict.

    Preference order:
      1. Midpoint of bestBid / bestAsk if both look usable.
      2. Stringified outcomePrices [yes, no].
      3. lastTradePrice.
    """
    bid = _f(market, "bestBid")
    ask = _f(market, "bestAsk")
    if 0.0 < bid < 1.0 and 0.0 < ask < 1.0 and bid <= ask:
        return (bid + ask) / 2

    prices = _parse_outcome_prices(market)
    if prices is not None:
        yes, _ = prices
        if 0.0 < yes < 1.0:
            return yes

    last = _f(market, "lastTradePrice")
    if 0.0 < last < 1.0:
        return last
    return None


def _is_usable(market: dict[str, Any]) -> bool:
    if market.get("closed") is True or market.get("archived") is True:
        return False
    if market.get("active") is False:
        return False
    # Binary-only. Polymarket multi-outcome markets break the YES/NO contract.
    raw_outcomes = market.get("outcomes")
    if isinstance(raw_outcomes, str):
        try:
            outcomes = json.loads(raw_outcomes)
        except (json.JSONDecodeError, ValueError):
            outcomes = None
    else:
        outcomes = raw_outcomes
    if isinstance(outcomes, list) and len(outcomes) != 2:
        return False
    vol_24h = _f(market, "volume24hr", "volume_24hr")
    return vol_24h >= MIN_VOLUME_24H


def find_match(title: str) -> tuple[dict[str, Any], float] | None:
    """Find the best Polymarket match for `title`. Returns (market, score) or None."""
    if not title:
        return None
    query_tokens = _tokens(title)
    if not query_tokens:
        return None
    candidates = _search(title)
    best: tuple[dict[str, Any], float] | None = None
    for m in candidates:
        if not _is_usable(m):
            continue
        question = m.get("question") or m.get("title") or ""
        score = _overlap(query_tokens, _tokens(question))
        if score < MATCH_THRESHOLD:
            continue
        if best is None or score > best[1]:
            best = (m, score)
    return best


def polymarket_quote(event: dict) -> tuple[float, float, str] | None:
    """Return (p_yes, weight, rationale) from the best Polymarket match.

    `weight` is a depth proxy (24h volume) for use in the cross-market
    blend. None when no usable sibling exists.
    """
    title = event.get("title") or ""
    match = find_match(title)
    if match is None:
        return None
    market, score = match
    p = _market_p_yes(market)
    if p is None:
        return None
    p = max(0.01, min(0.99, p))
    vol_24h = _f(market, "volume24hr", "volume_24hr")
    question = (market.get("question") or "")[:80]
    rationale = (
        f"poly '{question}' p={p:.3f} vol24h=${vol_24h:.0f} (match={score:.2f})"
    )
    return p, vol_24h, rationale


# ---------------------------------------------------------------------------
# Multi-outcome event lookup
# ---------------------------------------------------------------------------
#
# For 3+ outcome Kalshi events, Polymarket often has the same question
# structured as ONE "event" with N child binary markets (one per outcome).
# E.g., Kalshi "Who wins Eurovision 2026?" with 35 outcomes corresponds to
# a Polymarket event "Eurovision 2026 Winner" containing 35 markets like
# "Will Albania win Eurovision 2026?" → Yes/No.
#
# To use Polymarket as a multi-outcome prior we pull the event, extract
# each child market's YES probability, and map child-market titles to our
# event's `outcomes` list by token overlap. The result is a distribution
# we can use as a much stronger prior than naked LLM speculation.

EVENT_SEARCH_LIMIT = 10
MULTI_MATCH_THRESHOLD = 0.45     # looser than binary — event titles differ more
MIN_OUTCOMES_COVERED = 0.60      # need at least 60% of event outcomes mapped to a child


def _search_events(query: str, limit: int = EVENT_SEARCH_LIMIT) -> list[dict[str, Any]]:
    """Fetch open Polymarket events. Empty list on failure."""
    if not query:
        return []
    try:
        r = requests.get(
            f"{BASE}/events",
            params={
                "active": "true",
                "closed": "false",
                "archived": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false",
                "search": query,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("Polymarket event search failed for %r: %s", query, e)
        return []
    except ValueError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data") or data.get("events") or []
    return []


def _find_event_match(title: str) -> tuple[dict[str, Any], float] | None:
    """Best-matching Polymarket event for `title`, or None."""
    query_tokens = _tokens(title)
    if not query_tokens:
        return None
    candidates = _search_events(title)
    best: tuple[dict[str, Any], float] | None = None
    for ev in candidates:
        if ev.get("closed") is True or ev.get("archived") is True:
            continue
        ev_title = ev.get("title") or ev.get("ticker") or ""
        score = _overlap(query_tokens, _tokens(ev_title))
        if score < MULTI_MATCH_THRESHOLD:
            continue
        markets = ev.get("markets")
        if not isinstance(markets, list) or len(markets) < 3:
            continue
        if best is None or score > best[1]:
            best = (ev, score)
    return best


def _map_child_to_outcome(child_question: str, outcomes: list[str]) -> str | None:
    """Match a child market title to one of our event's outcomes by tokens."""
    child_tokens = _tokens(child_question)
    if not child_tokens:
        return None
    best_outcome: str | None = None
    best_score = 0.0
    for outcome in outcomes:
        outcome_tokens = _tokens(outcome)
        if not outcome_tokens:
            continue
        # All outcome tokens must appear in child question — strict match
        # ensures e.g. "Albania" → "Will Albania win Eurovision?" not
        # "Will Albanian-Greek border close?".
        if outcome_tokens.issubset(child_tokens):
            score = len(outcome_tokens) / len(child_tokens)
            if score > best_score:
                best_score = score
                best_outcome = outcome
    return best_outcome


def polymarket_event_distribution(
    event: dict,
) -> tuple[list[dict[str, float]], float, str] | None:
    """Pull a multi-outcome probability distribution from Polymarket.

    Returns (probabilities_list, total_volume_24h, rationale) where the
    list is [{market, probability}, ...] covering as many of our event's
    outcomes as we could map. Caller is responsible for normalizing to
    sum=1 (we leave that to _normalize_distribution upstream).

    None when no usable event match or coverage is too sparse.
    """
    title = event.get("title") or ""
    outcomes = event.get("outcomes") or []
    if not title or len(outcomes) < 3:
        return None

    match = _find_event_match(title)
    if match is None:
        return None
    ev, event_score = match
    markets = ev.get("markets") or []

    # Build {outcome → P(Yes)} from child markets.
    by_outcome: dict[str, float] = {}
    total_vol = 0.0
    for child in markets:
        if not isinstance(child, dict):
            continue
        if child.get("closed") is True or child.get("archived") is True:
            continue
        question = child.get("question") or ""
        mapped = _map_child_to_outcome(question, outcomes)
        if mapped is None or mapped in by_outcome:
            continue
        p = _market_p_yes(child)
        if p is None:
            continue
        by_outcome[mapped] = max(0.0, min(1.0, p))
        total_vol += _f(child, "volume24hr", "volume_24hr")

    coverage = len(by_outcome) / len(outcomes)
    if coverage < MIN_OUTCOMES_COVERED:
        return None

    probs: list[dict[str, float]] = [
        {"market": o, "probability": by_outcome.get(o, 0.0)} for o in outcomes
    ]
    ev_title = (ev.get("title") or "?")[:80]
    rationale = (
        f"poly event '{ev_title}' "
        f"covered {len(by_outcome)}/{len(outcomes)} outcomes "
        f"(match={event_score:.2f}, vol24h=${total_vol:.0f})"
    )
    return probs, total_vol, rationale
