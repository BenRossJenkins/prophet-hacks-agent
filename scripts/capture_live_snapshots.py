"""Capture snapshots of currently-open Kalshi markets for clean backtest data.

Run periodically (e.g., daily) through resolution. Each snapshot saves the
market state at capture time alongside event metadata. A separate resolver
script polls captured markets for settlement and produces a clean fixture.

Output (append-mode): tests/fixtures/live_snapshots.jsonl

Each line:
{
  "captured_at": ISO ts,
  "event": { event_ticker, market_ticker, title, category, ..., close_time },
  "market_snapshot": { ...all useful price/volume fields },
  "result": ""  # filled in later by resolve_captures.py
}
"""

from __future__ import annotations

import json
import socket
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# DNS workaround (no-op when dnspython missing)
_OVERRIDE_HOSTS = {"api.elections.kalshi.com", "external-api.kalshi.com"}
try:
    import dns.resolver  # type: ignore

    _resolver = dns.resolver.Resolver(configure=False)
    _resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
    _real_getaddrinfo = socket.getaddrinfo

    def _patched_getaddrinfo(host, port, *args, **kwargs):
        if host in _OVERRIDE_HOSTS:
            try:
                ip = _resolver.resolve(host, "A", lifetime=5.0)[0].address
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
            except Exception:
                pass
        return _real_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = _patched_getaddrinfo
except ImportError:
    pass

import requests  # noqa: E402

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT_PATH = Path("tests/fixtures/live_snapshots.jsonl")

# Mirror hackathon eval-set shape: top-volume markets closing in next 24–168h.
CLOSE_WINDOW_HOURS = (24, 168)
TOP_PER_CATEGORY = 5

SNAPSHOT_KEYS = [
    "ticker",
    "event_ticker",
    "status",
    "open_time",
    "close_time",
    "expiration_time",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "no_bid_dollars",
    "no_ask_dollars",
    "yes_bid_size_fp",
    "yes_ask_size_fp",
    "last_price_dollars",
    "previous_yes_bid_dollars",
    "previous_yes_ask_dollars",
    "previous_price_dollars",
    "volume_fp",
    "volume_24h_fp",
    "open_interest_fp",
    "liquidity_dollars",
    "updated_time",
    "result",
]


def _f(d: dict, key: str) -> float:
    try:
        return float(d.get(key, "0") or 0)
    except (ValueError, TypeError):
        return 0.0


def fetch_market_window() -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    min_ts = int((now + timedelta(hours=CLOSE_WINDOW_HOURS[0])).timestamp())
    max_ts = int((now + timedelta(hours=CLOSE_WINDOW_HOURS[1])).timestamp())
    markets: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "limit": 200,
        "status": "open",
        "min_close_ts": min_ts,
        "max_close_ts": max_ts,
    }
    for _ in range(10):
        try:
            resp = requests.get(f"{BASE}/markets", params=params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException:
            break
        js = resp.json()
        markets.extend(js.get("markets", []))
        cursor = js.get("cursor")
        if not cursor:
            break
        params = {**params, "cursor": cursor}
    return markets


_event_meta_cache: dict[str, dict[str, str]] = {}


def event_meta(event_ticker: str) -> dict[str, str]:
    if event_ticker in _event_meta_cache:
        return _event_meta_cache[event_ticker]
    out = {"series_ticker": "", "category": ""}
    try:
        resp = requests.get(f"{BASE}/events/{event_ticker}", timeout=10)
        resp.raise_for_status()
        ev = resp.json().get("event", {})
        out["series_ticker"] = ev.get("series_ticker", "")
        out["category"] = ev.get("category", "")
    except requests.RequestException:
        pass
    _event_meta_cache[event_ticker] = out
    return out


def main() -> None:
    captured_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    print(f"Capturing live snapshots at {captured_at}")

    markets = fetch_market_window()
    print(f"  fetched {len(markets)} markets in {CLOSE_WINDOW_HOURS[0]}–{CLOSE_WINDOW_HOURS[1]}h close window")

    # Filter MVE + positive volume, group by category (via event lookup), take top N per category by volume.
    enriched: list[tuple[float, str, dict]] = []
    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker or ticker.startswith("KXMVE"):
            continue
        vol = _f(m, "volume_24h_fp")
        if vol <= 0:
            continue
        meta = event_meta(m.get("event_ticker", ""))
        cat = meta.get("category", "")
        if not cat:
            continue
        enriched.append((vol, cat, m))

    by_cat: dict[str, list[tuple[float, dict]]] = {}
    for vol, cat, m in enriched:
        by_cat.setdefault(cat, []).append((vol, m))
    selected: list[tuple[str, dict]] = []
    for cat, items in by_cat.items():
        items.sort(key=lambda t: t[0], reverse=True)
        for _, m in items[:TOP_PER_CATEGORY]:
            selected.append((cat, m))
    print(f"  selected {len(selected)} markets across {len(by_cat)} categories")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT_PATH.open("a") as f:
        for cat, m in selected:
            ticker = m["ticker"]
            entry = {
                "captured_at": captured_at,
                "event": {
                    "event_ticker": m.get("event_ticker", ""),
                    "market_ticker": ticker,
                    "title": m.get("title", ""),
                    "subtitle": m.get("subtitle"),
                    "description": m.get("description"),
                    "category": cat,
                    "rules": m.get("rules_primary") or m.get("rules"),
                    "close_time": m.get("close_time", ""),
                },
                "market_snapshot": {k: m.get(k) for k in SNAPSHOT_KEYS},
                "result": "",
            }
            f.write(json.dumps(entry) + "\n")
            written += 1
            time.sleep(0.02)
    print(f"  wrote {written} snapshots → {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
