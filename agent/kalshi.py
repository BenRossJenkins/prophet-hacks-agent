"""Read-only Kalshi market-data client.

Kalshi's `/trade-api/v2/markets/{ticker}` and `/trade-api/v2/events/{event_ticker}`
endpoints are public and unauthenticated. Responses are CloudFront-cached
(15s public cache).

Two surfaces:

- `get_market(ticker)` — single market lookup, used by the binary
  prediction path (depth-mid, last-price, volume-weighted shrinkage).
- `get_event(event_ticker)` and `kalshi_event_distribution(event)` —
  multi-outcome events with `mutually_exclusive=True` are structured as
  N child binary markets whose YES prices form a joint distribution.
  Used by the multi-outcome forecast path in agent/predict.py.

Defensive about field naming: Kalshi has two conventions across
endpoints — cents (integers, `yes_bid: 28`) and dollars (decimals,
`yes_bid_dollars: 0.28`). Helpers try both before giving up.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_KALSHI_BASE_URL = "https://api.elections.kalshi.com"

# Outcome-coverage threshold mirrored from agent/polymarket.py.
MIN_OUTCOMES_COVERED = 0.60

# Liquidity floor below which we fall back to last-price instead of mid.
MIN_LIQUID_VOL_FOR_MID = 100.0


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


def get_event(event_ticker: str, *, timeout: float = 10.0) -> dict[str, Any] | None:
    """Fetch a Kalshi event with nested child markets. None on failure.

    Hits `/trade-api/v2/events/{event_ticker}?with_nested_markets=true`,
    which returns event metadata (mutually_exclusive flag, title, etc.)
    plus the full array of child markets in a single round-trip.
    """
    url = f"{kalshi_base_url()}/trade-api/v2/events/{event_ticker}"
    try:
        resp = requests.get(
            url, params={"with_nested_markets": "true"}, timeout=timeout
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Kalshi event fetch failed for %s: %s", event_ticker, e)
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    # Some Kalshi responses wrap under "event"; others return the object directly.
    return data.get("event") or data


# ---------------------------------------------------------------------------
# Multi-outcome distribution
# ---------------------------------------------------------------------------


def _read_price(child: dict, *cents_keys: str, dollars_keys: tuple[str, ...] = ()) -> float | None:
    """Read a Kalshi price field, trying cents (int 0-100) then dollars (float 0-1).

    Returns the price as a probability in [0, 1], or None when neither
    representation is present or parsable.
    """
    for key in cents_keys:
        v = child.get(key)
        if v is None:
            continue
        try:
            cents = float(v)
        except (TypeError, ValueError):
            continue
        # Heuristic: if the value looks like cents (>1), treat as cents.
        if cents > 1.0:
            p = cents / 100.0
        else:
            p = cents
        if 0.0 <= p <= 1.0:
            return p
    for key in dollars_keys:
        v = child.get(key)
        if v is None:
            continue
        try:
            p = float(v)
        except (TypeError, ValueError):
            continue
        if 0.0 <= p <= 1.0:
            return p
    return None


def _read_volume(child: dict) -> float:
    for key in ("volume_24h", "volume_24h_fp", "volume24hr", "volume"):
        v = child.get(key)
        if v is None:
            continue
        try:
            f = float(v)
            if f >= 0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


def _kalshi_child_p_yes(child: dict, *, min_liquid_vol: float = MIN_LIQUID_VOL_FOR_MID) -> float | None:
    """Derive a p_yes from a Kalshi child market.

    Preference:
      1. Depth-mid (yes_bid + yes_ask) / 2 if both present AND book is
         either tight-spread (≤ 0.10) or liquid (vol ≥ min_liquid_vol).
      2. Last trade price if available.
      3. Mid of (yes_bid, yes_ask) even on thin book.
      4. Single side if only one is present.
      5. None — caller excludes this outcome from coverage.
    """
    status = child.get("status")
    if status not in (None, "active", "open"):
        return None

    p_bid = _read_price(child, "yes_bid", dollars_keys=("yes_bid_dollars",))
    p_ask = _read_price(child, "yes_ask", dollars_keys=("yes_ask_dollars",))
    p_last = _read_price(child, "last_price", dollars_keys=("last_price_dollars",))
    vol_24h = _read_volume(child)

    if p_bid is not None and p_ask is not None:
        spread = p_ask - p_bid
        if spread <= 0.10 or vol_24h >= min_liquid_vol:
            mid = (p_bid + p_ask) / 2.0
            # Wide spread but liquid: blend mid with last (weighted toward last)
            # to discount the inflated bid-ask range.
            if spread > 0.10 and p_last is not None:
                return 0.4 * mid + 0.6 * p_last
            return mid

    if p_last is not None:
        return p_last

    if p_bid is not None and p_ask is not None:
        return (p_bid + p_ask) / 2.0
    if p_bid is not None:
        return p_bid
    if p_ask is not None:
        return p_ask
    return None


def _map_kalshi_child_to_outcome(child: dict, outcomes: list[str]) -> str | None:
    """Match a Kalshi child market to one of our event's outcomes.

    Kalshi child markets typically carry the outcome name in `subtitle`
    or `yes_sub_title` (e.g. "Boston Celtics"). Try exact case-insensitive
    match first; fall back to token-subset matching for minor phrasing
    differences ("Celtics" ↔ "Boston Celtics").
    """
    subtitle = (
        (child.get("subtitle") or child.get("yes_sub_title") or "")
        .strip()
    )
    if not subtitle:
        return None
    sub_lower = subtitle.lower()
    # Exact case-insensitive match.
    for o in outcomes:
        if o.strip().lower() == sub_lower:
            return o
    # Token-subset match (e.g. "Celtics" vs "Boston Celtics").
    sub_tokens = set(sub_lower.split())
    if not sub_tokens:
        return None
    best_outcome: str | None = None
    best_score = 0.0
    for o in outcomes:
        o_tokens = set(o.lower().split())
        if not o_tokens:
            continue
        if sub_tokens.issubset(o_tokens) or o_tokens.issubset(sub_tokens):
            score = len(sub_tokens & o_tokens) / max(len(sub_tokens), len(o_tokens))
            if score > best_score:
                best_score = score
                best_outcome = o
    return best_outcome


def _derive_event_ticker(market_ticker: str | None) -> str | None:
    """Derive an event_ticker from a Kalshi market_ticker.

    Kalshi convention: `EVENT-CHILD`, e.g. `KXNBACHAMP-26-BOS` → event
    `KXNBACHAMP-26`. Strip the trailing segment after the last hyphen.
    Returns None if the ticker doesn't have an obvious child suffix.
    """
    if not market_ticker:
        return None
    parts = market_ticker.rsplit("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0]


def kalshi_event_distribution(
    event: dict,
) -> tuple[list[dict[str, float]], float, str] | None:
    """Pull a multi-outcome probability distribution from a Kalshi event.

    Returns (probabilities_list, total_volume_24h, rationale). None if:
      - event_ticker can't be determined
      - event has fewer than 3 outcomes (use the binary path)
      - Kalshi event fetch fails
      - event is NOT mutually_exclusive (child markets are independent
        binaries, not a joint distribution — bailing is correct)
      - outcome coverage falls below MIN_OUTCOMES_COVERED

    Caller is responsible for normalizing to sum=1.
    """
    outcomes = event.get("outcomes") or []
    if len(outcomes) < 3:
        return None

    event_ticker = (
        event.get("event_ticker")
        or _derive_event_ticker(event.get("market_ticker"))
    )
    if not event_ticker:
        return None

    ev = get_event(event_ticker)
    if ev is None:
        return None

    # mutually_exclusive guarantees child markets sum (approximately) to 1.
    # When false, children are independent binaries and don't form a joint
    # distribution — bail.
    if not ev.get("mutually_exclusive", False):
        return None

    children = ev.get("markets") or []
    if not children:
        return None

    by_outcome: dict[str, float] = {}
    total_vol = 0.0
    for child in children:
        if not isinstance(child, dict):
            continue
        mapped = _map_kalshi_child_to_outcome(child, outcomes)
        if mapped is None or mapped in by_outcome:
            continue
        p = _kalshi_child_p_yes(child)
        if p is None:
            continue
        by_outcome[mapped] = max(0.0, min(1.0, p))
        total_vol += _read_volume(child)

    coverage = len(by_outcome) / len(outcomes)
    if coverage < MIN_OUTCOMES_COVERED:
        return None

    probs: list[dict[str, float]] = [
        {"market": o, "probability": by_outcome.get(o, 0.0)} for o in outcomes
    ]
    ev_title = (ev.get("title") or event_ticker)[:80]
    raw_sum = sum(by_outcome.values())
    rationale = (
        f"kalshi event '{ev_title}' "
        f"covered {len(by_outcome)}/{len(outcomes)} outcomes, "
        f"sum={raw_sum:.3f}, vol24h=${total_vol:.0f}"
    )
    return probs, total_vol, rationale
