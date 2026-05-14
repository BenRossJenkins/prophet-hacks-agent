"""Promote resolved live snapshots into the clean fixture.

Reads tests/fixtures/live_snapshots.jsonl (each line has a market we
captured earlier). For any snapshot whose market has since settled,
queries Kalshi for the result and appends a clean entry to
tests/fixtures/resolved_markets_live.jsonl.

Idempotent: tracks which (captured_at, market_ticker) pairs are already
resolved by reading the output file at start.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

# DNS workaround
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
SNAP_PATH = Path("tests/fixtures/live_snapshots.jsonl")
CLEAN_PATH = Path("tests/fixtures/resolved_markets_live.jsonl")


def main() -> None:
    if not SNAP_PATH.exists():
        print(f"No snapshots at {SNAP_PATH}; run capture_live_snapshots.py first.")
        return

    seen: set[tuple[str, str]] = set()
    if CLEAN_PATH.exists():
        with CLEAN_PATH.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    seen.add((e.get("captured_at", ""), e["event"]["market_ticker"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    with SNAP_PATH.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(entries)} snapshots, {len(seen)} already promoted")

    promoted = 0
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN_PATH.open("a") as out:
        for e in entries:
            key = (e.get("captured_at", ""), e["event"]["market_ticker"])
            if key in seen:
                continue
            ticker = e["event"]["market_ticker"]
            try:
                resp = requests.get(f"{BASE}/markets/{ticker}", timeout=10)
                resp.raise_for_status()
                m = resp.json().get("market", {})
            except requests.RequestException:
                continue
            result = m.get("result", "")
            if result not in ("yes", "no"):
                continue
            clean = {
                "event": e["event"],
                "market_snapshot": e["market_snapshot"],
                "result": result,
                "captured_at": e.get("captured_at", ""),
                "resolved_at": m.get("settlement_ts") or m.get("expiration_time", ""),
                "source": "live_capture",
            }
            out.write(json.dumps(clean) + "\n")
            promoted += 1
            seen.add(key)
            time.sleep(0.03)
    print(f"Promoted {promoted} newly-resolved snapshots → {CLEAN_PATH}")


if __name__ == "__main__":
    main()
