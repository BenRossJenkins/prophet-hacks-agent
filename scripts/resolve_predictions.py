"""Resolve logged predictions against Kalshi current state.

Reads ``data/predictions.jsonl`` (where the live agent writes every
forecast it emits), looks up each market's current state on Kalshi, and
if the market has settled (status=finalized, result in {yes, no}) writes
an enriched entry to ``data/resolved_predictions.jsonl``.

Idempotent: keeps track of already-resolved (ts, market_ticker) pairs by
scanning the output file at startup. Safe to re-run as often as you want.

Override paths with env vars:
  PREDICTION_LOG_PATH      input file (default data/predictions.jsonl)
  RESOLVED_LOG_PATH        output file (default data/resolved_predictions.jsonl)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

# Optional dev-machine DNS workaround
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
PRED_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "data/predictions.jsonl"))
RESOLVED_PATH = Path(os.environ.get("RESOLVED_LOG_PATH", "data/resolved_predictions.jsonl"))


def _key(entry: dict) -> tuple[str, str]:
    return (entry.get("ts", ""), entry.get("event", {}).get("market_ticker", ""))


def _fetch_market(ticker: str) -> dict | None:
    try:
        resp = requests.get(f"{BASE}/markets/{ticker}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("market") or resp.json()
    except requests.RequestException:
        return None


def main() -> int:
    if not PRED_PATH.exists():
        print(f"No predictions log at {PRED_PATH}", file=sys.stderr)
        return 1

    seen: set[tuple[str, str]] = set()
    if RESOLVED_PATH.exists():
        with RESOLVED_PATH.open() as f:
            for line in f:
                try:
                    seen.add(_key(json.loads(line)))
                except json.JSONDecodeError:
                    continue

    with PRED_PATH.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(entries)} predictions, {len(seen)} already resolved")

    promoted = 0
    skipped_unresolved = 0
    RESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESOLVED_PATH.open("a") as out:
        for entry in entries:
            if _key(entry) in seen:
                continue
            ticker = entry.get("event", {}).get("market_ticker", "")
            if not ticker:
                continue
            market = _fetch_market(ticker)
            if market is None:
                continue
            result = market.get("result", "")
            if result not in ("yes", "no"):
                skipped_unresolved += 1
                continue
            resolved = {
                **entry,
                "result": result,
                "settlement_ts": market.get("settlement_ts", ""),
                "final_yes_bid": market.get("yes_bid_dollars", ""),
                "final_yes_ask": market.get("yes_ask_dollars", ""),
            }
            out.write(json.dumps(resolved) + "\n")
            promoted += 1
            seen.add(_key(entry))
            time.sleep(0.03)

    print(f"Promoted {promoted} newly-resolved predictions → {RESOLVED_PATH}")
    print(f"Still unresolved (skipped): {skipped_unresolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
