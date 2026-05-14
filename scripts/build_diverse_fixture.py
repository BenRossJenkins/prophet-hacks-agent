"""Append non-weather settled markets to the backtest fixture.

The default builder paginates ``/markets?status=settled`` and that feed
is dominated by weather and MVE parlay markets — diverse categories
basically don't show up in the first thousand results.

This script targets specific known-active series across categories
(Politics, Sports, Crypto, Entertainment, Financials, Economics) and
synthesizes a candlestick snapshot for each settled market under those
series, appending to ``tests/fixtures/resolved_markets.jsonl``.

It's intentionally a separate script (not an option on
``build_backtest_fixture.py``) because the access pattern is different —
we're chasing series-by-series rather than paginating a global feed.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# DNS workaround (dev machines)
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

sys.path.insert(0, str(Path(__file__).parent))
from build_backtest_fixture import (  # noqa: E402
    FIXTURE_PATH,
    SNAPSHOT_LIFETIME_FRACTION,
    fetch_candlesticks,
    snapshot_from_candles,
)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
MIN_VOL_FOR_INCLUSION = 20.0


# Curated list of high-volume series across categories we know we want covered.
# Selected from earlier probes (2026-05-14) where 1261 unique non-MVE series
# appeared in /events?status=settled.
CATEGORIES_AND_SERIES: dict[str, list[str]] = {
    "Politics": [
        "KXTRUMPACT",
        "KXTRUMPENDORSEMENTS",
        "KXTRUMPPHOTO",
        "KXACAHOUSEVOTE",
        "KXADMINNASA",
        "KXABRAHAMSA",
    ],
    "Sports": [
        "KXNCAABBGAME",
        "KXATPCHALLENGERMATCH",
        "KXMLBHIT",
        "KXACBGAME",
    ],
    "Crypto": [
        "KXBTCD",
        "KXETHD",
    ],
    "Entertainment": [
        "KXPUREALBUMS",
        "KX1ALBUM",
        "KX1SONG",
        "KX10SONG",
    ],
    "Financials": [
        "KXACQUANNOUNCEPARAMOUNT",
    ],
    "Economics": [
        "KXAAAGASD",
    ],
}


def _f(d: dict, key: str) -> float:
    try:
        return float(d.get(key, "0") or 0)
    except (ValueError, TypeError):
        return 0.0


def fetch_settled_events(series_ticker: str, max_pages: int = 5) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    params: dict[str, Any] = {"series_ticker": series_ticker, "status": "settled", "limit": 200}
    for _ in range(max_pages):
        try:
            resp = requests.get(f"{BASE}/events", params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException:
            break
        js = resp.json()
        events.extend(js.get("events", []))
        cursor = js.get("cursor")
        if not cursor:
            break
        params = {**params, "cursor": cursor}
    return events


def fetch_event_markets(event_ticker: str) -> list[dict[str, Any]]:
    try:
        resp = requests.get(
            f"{BASE}/markets", params={"event_ticker": event_ticker, "limit": 200}, timeout=15
        )
        resp.raise_for_status()
        return resp.json().get("markets", [])
    except requests.RequestException:
        return []


def main(per_category_cap: int = 30) -> int:
    print(f"Diversifying fixture (cap {per_category_cap}/category) → {FIXTURE_PATH}")

    # Load existing entries to dedupe by (market_ticker).
    existing_tickers: set[str] = set()
    if FIXTURE_PATH.exists():
        with FIXTURE_PATH.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    existing_tickers.add(e["event"]["market_ticker"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"  existing fixture has {len(existing_tickers)} entries")

    added_per_cat: dict[str, int] = {}
    total_added = 0
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("a") as out:
        for category, series_list in CATEGORIES_AND_SERIES.items():
            added_per_cat[category] = 0
            for series in series_list:
                if added_per_cat[category] >= per_category_cap:
                    break
                events = fetch_settled_events(series)
                print(f"  [{category}] series={series}: {len(events)} settled events")
                for ev in events:
                    if added_per_cat[category] >= per_category_cap:
                        break
                    event_ticker = ev.get("event_ticker", "")
                    if not event_ticker:
                        continue
                    series_ticker = ev.get("series_ticker", "") or series

                    markets = fetch_event_markets(event_ticker)
                    for m in markets:
                        ticker = m.get("ticker", "")
                        if not ticker or ticker in existing_tickers:
                            continue
                        if m.get("result", "") not in ("yes", "no"):
                            continue
                        if _f(m, "volume_fp") < MIN_VOL_FOR_INCLUSION:
                            continue

                        try:
                            open_dt = datetime.fromisoformat(
                                str(m.get("open_time", "")).replace("Z", "+00:00")
                            )
                            end_str = m.get("settlement_ts") or m.get("expiration_time", "")
                            end_dt = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
                        except (ValueError, TypeError):
                            continue
                        lifetime = (end_dt - open_dt).total_seconds()
                        if lifetime < 60:
                            continue
                        snapshot_dt = (
                            open_dt + (end_dt - open_dt) * SNAPSHOT_LIFETIME_FRACTION
                        )
                        snapshot_ts = int(snapshot_dt.timestamp())
                        window_start = snapshot_ts - 86400

                        bars = fetch_candlesticks(
                            series_ticker, ticker, window_start, snapshot_ts
                        )
                        snapshot = snapshot_from_candles(bars, snapshot_ts)
                        if snapshot is None:
                            continue

                        entry = {
                            "event": {
                                "event_ticker": event_ticker,
                                "market_ticker": ticker,
                                "title": m.get("title", ""),
                                "subtitle": m.get("subtitle"),
                                "description": m.get("description"),
                                "category": category,
                                "rules": m.get("rules_primary") or m.get("rules"),
                                "close_time": m.get("close_time", ""),
                            },
                            "market_snapshot": snapshot,
                            "result": m["result"],
                            "snapshot_ts": snapshot_ts,
                            "snapshot_lifetime_frac": SNAPSHOT_LIFETIME_FRACTION,
                            "source": "diverse_by_series",
                        }
                        out.write(json.dumps(entry) + "\n")
                        existing_tickers.add(ticker)
                        added_per_cat[category] += 1
                        total_added += 1
                        if added_per_cat[category] >= per_category_cap:
                            break
                        time.sleep(0.03)

    print()
    print(f"Added {total_added} new entries:")
    for cat, n in added_per_cat.items():
        print(f"  {cat:<20} +{n}")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sys.exit(main(per_category_cap=n))
