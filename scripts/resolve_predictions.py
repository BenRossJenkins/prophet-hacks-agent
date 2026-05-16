"""Resolve logged predictions against actual outcomes.

Reads ``data/predictions.jsonl`` (where the live agent writes every
forecast it emits) and, for each not-yet-resolved entry, tries to find
its outcome from:

  1. Prophet Arena API (GET /forecast/events?status=closed). Preferred
     because eval events use dataset task_ids that aren't real Kalshi
     tickers, and the PA API is the authoritative source.
  2. Kalshi market state (fallback for actual Kalshi market_tickers,
     useful during pre-event testing).

For multi-outcome events the PA API returns `actual_outcome` as a list
of winner labels; we convert to binary by checking if outcomes[0] is in
that list (matching the server's scoring binarization).

Idempotent: keeps track of already-resolved (ts, market_ticker) pairs by
scanning the output file at startup. Safe to re-run.

Override paths with env vars:
  PREDICTION_LOG_PATH      input file (default data/predictions.jsonl)
  RESOLVED_LOG_PATH        output file (default data/resolved_predictions.jsonl)
  PA_SERVER_URL            base URL (default https://api.aiprophet.dev)
  PA_SERVER_API_KEY        API key for /forecast/events (required to use PA path)
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

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
PA_BASE = os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev").rstrip("/")
PA_KEY = os.environ.get("PA_SERVER_API_KEY", "")
PRED_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "data/predictions.jsonl"))
RESOLVED_PATH = Path(os.environ.get("RESOLVED_LOG_PATH", "data/resolved_predictions.jsonl"))


def _key(entry: dict) -> tuple[str, str]:
    return (entry.get("ts", ""), entry.get("event", {}).get("market_ticker", ""))


def _fetch_pa_closed_events() -> dict[str, dict]:
    """Pull all closed events from the Prophet Arena API, keyed by market_ticker.

    Returns empty dict on auth failure or network error.
    """
    if not PA_KEY:
        return {}
    try:
        resp = requests.get(
            f"{PA_BASE}/forecast/events",
            params={"status": "closed"},
            headers={"X-API-Key": PA_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  PA API fetch failed: {e}", file=sys.stderr)
        return {}
    if not isinstance(data, list):
        return {}
    return {str(ev.get("market_ticker", "")): ev for ev in data if ev.get("market_ticker")}


def _resolve_via_pa(entry: dict, pa_events: dict[str, dict]) -> dict | None:
    """If this prediction's market is in the PA closed-events index, build the
    resolved entry. None when not yet resolved server-side.
    """
    ticker = entry.get("event", {}).get("market_ticker", "")
    if not ticker or ticker not in pa_events:
        return None
    ev = pa_events[ticker]
    actual = ev.get("actual_outcome")
    if actual is None:
        return None
    # actual_outcome may be a single label or a list of winner labels.
    if isinstance(actual, list):
        winners = [str(x) for x in actual]
    else:
        winners = [str(actual)]
    outcomes = entry.get("event", {}).get("outcomes") or []
    if not outcomes:
        # No outcomes list — treat actual==Yes as positive.
        result = "yes" if any(w.lower() in ("yes", "1", "true") for w in winners) else "no"
    else:
        # Server scoring: positive iff outcomes[0] is among the winners.
        result = "yes" if outcomes[0] in winners else "no"
    return {
        **entry,
        "result": result,
        "settlement_ts": ev.get("resolved_at", ""),
        "actual_outcome": actual,
        "resolution_source": "prophet-arena",
    }


def _fetch_kalshi_market(ticker: str) -> dict | None:
    try:
        resp = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("market") or resp.json()
    except requests.RequestException:
        return None


def _resolve_via_kalshi(entry: dict) -> dict | None:
    ticker = entry.get("event", {}).get("market_ticker", "")
    if not ticker:
        return None
    market = _fetch_kalshi_market(ticker)
    if market is None:
        return None
    result = market.get("result", "")
    if result not in ("yes", "no"):
        return None
    return {
        **entry,
        "result": result,
        "settlement_ts": market.get("settlement_ts", ""),
        "final_yes_bid": market.get("yes_bid_dollars", ""),
        "final_yes_ask": market.get("yes_ask_dollars", ""),
        "resolution_source": "kalshi",
    }


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

    # Fetch the PA closed-events index once; ~14 days of eval has ≤200
    # events so a single request covers everything.
    pa_events = _fetch_pa_closed_events()
    print(f"  PA closed-events index: {len(pa_events)} entries")

    promoted_pa = 0
    promoted_kalshi = 0
    skipped_unresolved = 0
    RESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESOLVED_PATH.open("a") as out:
        for entry in entries:
            if _key(entry) in seen:
                continue
            # PA API path first (authoritative for eval), Kalshi fallback.
            resolved = _resolve_via_pa(entry, pa_events) if pa_events else None
            if resolved is not None:
                promoted_pa += 1
            else:
                resolved = _resolve_via_kalshi(entry)
                if resolved is not None:
                    promoted_kalshi += 1
                    time.sleep(0.03)  # rate-limit Kalshi
                else:
                    skipped_unresolved += 1
                    continue
            out.write(json.dumps(resolved) + "\n")
            seen.add(_key(entry))

    print(
        f"Promoted {promoted_pa} via PA API + {promoted_kalshi} via Kalshi "
        f"→ {RESOLVED_PATH}"
    )
    print(f"Still unresolved (skipped): {skipped_unresolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
