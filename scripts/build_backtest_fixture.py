"""Build a backtest fixture from settled Kalshi markets using candlestick history.

Strategy:
  1. Paginate /markets?status=settled. Keep non-MVE markets with a result
     and a non-trivial volume_fp.
  2. For each market, query the event to get series_ticker + category.
  3. Pick a snapshot timestamp = halfway between open_time and expiration_time.
  4. Query 1-minute candlesticks for the [snapshot - 24h, snapshot] window.
  5. The latest bar before snapshot gives the actual bid/ask/last at that moment.
     Volume_24h_fp = sum(volume_fp across the 24h window).
  6. Save snapshot + event metadata + result to JSONL.

This avoids the leakage problem in the trade-replay approach because the
snapshot is picked from a time point we control, not from "all trades to
date" which would include resolution-correlated late trades.

Output: tests/fixtures/resolved_markets.jsonl
"""

from __future__ import annotations

import json
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# DNS workaround for dev machines (no-op when dnspython missing).
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
FIXTURE_PATH = Path("tests/fixtures/resolved_markets.jsonl")

MIN_VOL_FOR_INCLUSION = 20.0  # USD lifetime volume
SNAPSHOT_LIFETIME_FRACTION = 0.75  # snapshot at 75% through market lifetime
TARGET_FIXTURE_SIZE = 200


def _f(d: dict, key: str) -> float:
    try:
        return float(d.get(key, "0") or 0)
    except (ValueError, TypeError):
        return 0.0


def fetch_settled_markets(target: int) -> list[dict[str, Any]]:
    """Paginate /markets?status=settled, filter MVE, volume, has result."""
    keep: list[dict[str, Any]] = []
    params = {"limit": 200, "status": "settled"}
    pages = 0
    while True:
        try:
            resp = requests.get(f"{BASE}/markets", params=params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException:
            break
        js = resp.json()
        for m in js.get("markets", []):
            ticker = m.get("ticker", "")
            if not ticker or ticker.startswith("KXMVE"):
                continue
            if m.get("result", "") not in ("yes", "no"):
                continue
            if _f(m, "volume_fp") < MIN_VOL_FOR_INCLUSION:
                continue
            keep.append(m)
        cursor = js.get("cursor")
        pages += 1
        if pages % 5 == 0:
            print(f"  page {pages}: {len(keep)} candidate markets")
        if not cursor or len(keep) >= target or pages >= 50:
            break
        params = {**params, "cursor": cursor}
    return keep


_event_cache: dict[str, dict[str, str]] = {}


def event_metadata(event_ticker: str) -> dict[str, str]:
    """Returns {series_ticker, category, title} for an event. Cached."""
    if event_ticker in _event_cache:
        return _event_cache[event_ticker]
    out: dict[str, str] = {"series_ticker": "", "category": "", "title": ""}
    try:
        resp = requests.get(f"{BASE}/events/{event_ticker}", timeout=10)
        resp.raise_for_status()
        ev = resp.json().get("event", {})
        out["series_ticker"] = ev.get("series_ticker", "")
        out["category"] = ev.get("category", "")
        out["title"] = ev.get("title", "")
    except requests.RequestException:
        pass
    _event_cache[event_ticker] = out
    return out


def fetch_candlesticks(
    series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_min: int = 1
) -> list[dict[str, Any]]:
    """Pull 1-minute OHLC bars for a market between [start_ts, end_ts]."""
    url = f"{BASE}/series/{series_ticker}/markets/{ticker}/candlesticks"
    try:
        resp = requests.get(
            url,
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_min},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []
    return resp.json().get("candlesticks", [])


def snapshot_from_candles(
    bars: list[dict[str, Any]], snapshot_ts: int
) -> dict[str, Any] | None:
    """Synthesize a market-state dict from candles around snapshot_ts.

    Uses the latest bar at or before snapshot_ts for bid/ask/last; sums
    volume_fp across the 24h window before snapshot_ts.
    """
    if not bars:
        return None

    bars = sorted(bars, key=lambda b: b.get("end_period_ts", 0))
    pre = [b for b in bars if b.get("end_period_ts", 0) <= snapshot_ts]
    if not pre:
        return None
    latest = pre[-1]

    def _cd(bar: dict, side: str) -> float:
        block = bar.get(side) or {}
        try:
            return float(block.get("close_dollars", "0") or 0)
        except (ValueError, TypeError):
            return 0.0

    yes_bid = _cd(latest, "yes_bid")
    yes_ask = _cd(latest, "yes_ask")

    price_block = latest.get("price") or {}
    try:
        last_price = float(price_block.get("close_dollars", "0") or 0)
    except (ValueError, TypeError):
        last_price = 0.0

    if yes_bid > yes_ask:
        yes_bid, yes_ask = yes_ask, yes_bid

    # Sum volume in 24h before snapshot
    window_start = snapshot_ts - 86400
    vol_24h = 0.0
    for b in pre:
        ts = b.get("end_period_ts", 0)
        if ts < window_start:
            continue
        try:
            vol_24h += float(b.get("volume_fp", "0") or 0)
        except (ValueError, TypeError):
            continue

    snapshot_iso = (
        datetime.fromtimestamp(snapshot_ts, tz=UTC).isoformat().replace("+00:00", "Z")
    )

    return {
        "yes_bid_dollars": f"{yes_bid:.4f}",
        "yes_ask_dollars": f"{yes_ask:.4f}",
        "no_bid_dollars": f"{max(0.0, 1.0 - yes_ask):.4f}",
        "no_ask_dollars": f"{max(0.0, 1.0 - yes_bid):.4f}",
        "last_price_dollars": f"{last_price:.4f}",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "volume_24h_fp": f"{vol_24h:.2f}",
        "volume_fp": f"{vol_24h:.2f}",
        "open_interest_fp": "0.00",
        "liquidity_dollars": "0.0000",
        "updated_time": snapshot_iso,
        "status": "active",
        "result": "",
    }


def main(target: int = TARGET_FIXTURE_SIZE) -> None:
    print(f"Building backtest fixture (target {target}) …")
    print("Fetching settled markets …")
    candidates = fetch_settled_markets(target * 5)
    print(f"  {len(candidates)} candidates")

    out_path = FIXTURE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_candles = 0
    skipped_no_meta = 0

    with out_path.open("w") as f:
        for m in candidates:
            if written >= target:
                break
            ticker = m["ticker"]
            event_ticker = m.get("event_ticker", "")

            meta = event_metadata(event_ticker)
            if not meta["series_ticker"] or not meta["category"]:
                skipped_no_meta += 1
                continue

            try:
                open_dt = datetime.fromisoformat(
                    str(m.get("open_time", "")).replace("Z", "+00:00")
                )
                # settlement_ts is when the market actually resolved (trading ended);
                # expiration_time is the formal calendar deadline and can be days/weeks
                # after trading actually stopped, so its midpoint isn't useful.
                end_str = m.get("settlement_ts") or m.get("expiration_time", "")
                end_dt = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            lifetime = (end_dt - open_dt).total_seconds()
            if lifetime < 60:  # skip ultra-short markets
                continue
            snapshot_dt = open_dt + (end_dt - open_dt) * SNAPSHOT_LIFETIME_FRACTION
            snapshot_ts = int(snapshot_dt.timestamp())
            window_start = snapshot_ts - 86400

            bars = fetch_candlesticks(meta["series_ticker"], ticker, window_start, snapshot_ts)
            snapshot = snapshot_from_candles(bars, snapshot_ts)
            if snapshot is None:
                skipped_no_candles += 1
                continue

            entry = {
                "event": {
                    "event_ticker": event_ticker,
                    "market_ticker": ticker,
                    "title": m.get("title", ""),
                    "subtitle": m.get("subtitle"),
                    "description": m.get("description"),
                    "category": meta["category"],
                    "rules": m.get("rules_primary") or m.get("rules"),
                    "close_time": m.get("close_time", ""),
                },
                "market_snapshot": snapshot,
                "result": m["result"],
                "snapshot_ts": snapshot_ts,
                "snapshot_lifetime_frac": SNAPSHOT_LIFETIME_FRACTION,
                "n_candles_used": len(bars),
            }
            f.write(json.dumps(entry) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  wrote {written}/{target}")
            time.sleep(0.03)

    print(f"Done. {written} entries → {out_path}")
    print(f"  skipped: {skipped_no_meta} no-meta, {skipped_no_candles} no-candles")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_FIXTURE_SIZE
    main(n)
