"""Dev-only end-to-end validation against live Kalshi markets.

Why this exists: on this dev machine the local DNS resolver doesn't return
records for `api.elections.kalshi.com`. We resolve via Google/Cloudflare DNS
and inject the result into Python's socket layer. Production hosts (the
hackathon venue, Cloud Run) have working DNS and don't need this.

Run with:
    .venv/bin/python scripts/check_kalshi_live.py [N]

`N` is the number of markets to sample (default 10).
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Any

# ---- DNS workaround (dev machine only) ------------------------------------
#
# This dev machine's local resolver doesn't return records for Kalshi hosts.
# If `dnspython` is available, route those specific hostnames through 8.8.8.8.
# On a machine with working DNS this is a no-op.

_OVERRIDE_HOSTS = {"api.elections.kalshi.com", "external-api.kalshi.com"}

try:
    import dns.resolver  # type: ignore

    _resolver = dns.resolver.Resolver(configure=False)
    _resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
    _real_getaddrinfo = socket.getaddrinfo

    def _patched_getaddrinfo(host, port, *args, **kwargs):
        if host in _OVERRIDE_HOSTS:
            try:
                answers = _resolver.resolve(host, "A", lifetime=5.0)
                ip = answers[0].address
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
            except Exception as e:
                print(f"  ! dnspython resolve failed for {host}: {e}", file=sys.stderr)
        return _real_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = _patched_getaddrinfo
    _DNS_OVERRIDE = "active (dnspython via 8.8.8.8)"
except ImportError:
    _DNS_OVERRIDE = "not installed (relying on system resolver)"


# ---- Live test ------------------------------------------------------------

import requests  # noqa: E402

from agent.kalshi import kalshi_base_url  # noqa: E402
from agent.predict import predict  # noqa: E402

DEFAULT_CATEGORIES = [
    "Economics", "Politics", "Science and Technology", "Climate and Weather",
    "Sports", "Entertainment", "Financials", "World",
]


def _paginate(url: str, params: dict, key: str, page_limit: int = 5) -> list[dict[str, Any]]:
    """Walk paginated Kalshi endpoints (markets/events use ?cursor=)."""
    results: list[dict[str, Any]] = []
    for _ in range(page_limit):
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get(key, []))
        cursor = data.get("cursor")
        if not cursor:
            break
        params = {**params, "cursor": cursor}
    return results


def list_top_markets(deadline_hours: int = 72) -> list[dict[str, Any]]:
    """Pull open markets closing in the next [24h, deadline_hours] window.

    Mirrors upstream `select_events`: this is the windowing strategy the
    hackathon harness uses, so the result is representative of what we'll
    actually be asked to predict on.
    """
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    min_close_ts = int((now + timedelta(hours=24)).timestamp())
    max_close_ts = int((now + timedelta(hours=deadline_hours)).timestamp())

    url = f"{kalshi_base_url()}/trade-api/v2/markets"
    return _paginate(
        url,
        params={
            "limit": 200,
            "status": "open",
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
        },
        key="markets",
    )


_event_cat_cache: dict[str, str] = {}


def category_for_event(event_ticker: str) -> str:
    """Fetch one event's category. Cached per process."""
    if event_ticker in _event_cat_cache:
        return _event_cat_cache[event_ticker]
    url = f"{kalshi_base_url()}/trade-api/v2/events/{event_ticker}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        cat = resp.json().get("event", {}).get("category", "")
    except requests.RequestException:
        cat = ""
    _event_cat_cache[event_ticker] = cat
    return cat


def main(n: int) -> None:
    print(f"Kalshi base URL: {kalshi_base_url()}")
    print(f"DNS override: {_DNS_OVERRIDE}; hosts={sorted(_OVERRIDE_HOSTS)}")
    print("Fetching markets closing in next 24–72h …")

    markets = list_top_markets()
    print(f"  → fetched {len(markets)} markets")

    # Filter to markets with positive 24h volume, sort by volume desc.
    def _vol(m: dict) -> float:
        try:
            return float(m.get("volume_24h_fp", "0") or 0)
        except (ValueError, TypeError):
            return 0.0

    liquid = [m for m in markets if _vol(m) > 0 and m.get("ticker")]
    liquid.sort(key=_vol, reverse=True)
    print(f"  → {len(liquid)} with positive 24h volume")

    candidates: list[tuple[str, dict]] = []
    for m in liquid:
        cat = category_for_event(m.get("event_ticker", ""))
        if cat:
            candidates.append((cat, m))

    print(f"  → {len(candidates)} resolved to a category")
    print()

    print(f"{'ticker':<48}{'cat':<14}{'vol24h':>10}{'bid':>7}{'ask':>7}{'last':>7}{'p_yes':>8}  rationale")
    print("-" * 200)

    for cat, m in candidates[:n]:
        ticker = m["ticker"]
        event = {
            "event_ticker": m.get("event_ticker", ""),
            "market_ticker": ticker,
            "title": m.get("title", ""),
            "subtitle": m.get("subtitle"),
            "description": m.get("description"),
            "category": cat,
            "rules": m.get("rules_primary") or m.get("rules"),
            "close_time": m.get("close_time", ""),
        }
        out = predict(event)
        vol = float(m.get("volume_24h_fp", "0") or 0)
        bid = float(m.get("yes_bid_dollars", "0") or 0)
        ask = float(m.get("yes_ask_dollars", "0") or 0)
        last = float(m.get("last_price_dollars", "0") or 0)
        print(
            f"{ticker[:46]:<48}{cat[:12]:<14}{vol:>10.0f}{bid:>7.3f}{ask:>7.3f}{last:>7.3f}"
            f"{out['p_yes']:>8.3f}  {out['rationale']}"
        )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n)
