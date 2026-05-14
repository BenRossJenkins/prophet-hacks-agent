"""Backtest the agent against fixture(s) of (snapshot, result) pairs.

Usage:
    .venv/bin/python scripts/backtest.py [fixture_path]

Default fixture: tests/fixtures/resolved_markets.jsonl (candlestick-derived).

Mocks `agent.predict.get_market` to return the snapshot dict, then runs
`predict(event)` for each entry. Reports total Brier, by category, by
liquidity tier, and a p_yes calibration table.

By default, disables the LLM fallback by also mocking `llm_forecast` to
return None — backtests should be deterministic and cheap. Pass
--with-llm to exercise the LLM path against real Claude.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

DEFAULT_FIXTURE = Path("tests/fixtures/resolved_markets.jsonl")


def load_fixture(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def liquidity_tier(snapshot: dict) -> str:
    try:
        vol = float(snapshot.get("volume_24h_fp", "0") or 0)
    except (ValueError, TypeError):
        vol = 0.0
    if vol < 50:
        return "thin (<$50)"
    if vol < 500:
        return "light ($50–$500)"
    if vol < 5000:
        return "mid ($500–$5k)"
    return "heavy (>$5k)"


def brier(p: float, actual: float) -> float:
    return (p - actual) ** 2


def aggregate(entries: list[dict[str, Any]], with_llm: bool = False) -> dict:
    from agent.predict import predict

    results: list[dict] = []
    if with_llm:
        # Web search would leak resolution info on settled markets. Backtest
        # only exercises the model's own knowledge as of training cutoff.
        import functools
        from agent import llm as llm_mod

        no_search_llm = functools.partial(llm_mod.llm_forecast, with_web_search=False)
        ctx_mgrs = [
            patch("agent.predict.get_market", side_effect=_market_lookup_factory(entries)),
            patch("agent.predict.llm_forecast", side_effect=no_search_llm),
        ]
    else:
        ctx_mgrs = [
            patch("agent.predict.get_market", side_effect=_market_lookup_factory(entries)),
            patch("agent.predict.llm_forecast", return_value=None),
        ]
    for ctx in ctx_mgrs:
        ctx.__enter__()
    try:
        for entry in entries:
            event = entry["event"]
            actual = 1.0 if entry["result"] == "yes" else 0.0
            out = predict(event)
            p = float(out["p_yes"])
            results.append(
                {
                    "ticker": event["market_ticker"],
                    "category": event["category"],
                    "tier": liquidity_tier(entry["market_snapshot"]),
                    "p": p,
                    "actual": actual,
                    "brier": brier(p, actual),
                    "rationale": out["rationale"],
                }
            )
    finally:
        for ctx in reversed(ctx_mgrs):
            ctx.__exit__(None, None, None)
    return _summarize(results)


def _market_lookup_factory(entries: list[dict]):
    """Returns a side_effect function that returns the right snapshot per ticker call."""
    by_ticker = {e["event"]["market_ticker"]: e["market_snapshot"] for e in entries}

    def _lookup(ticker: str, **kwargs):
        return by_ticker.get(ticker)

    return _lookup


def _summarize(results: list[dict]) -> dict:
    total_brier = statistics.mean(r["brier"] for r in results) if results else 0.0
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_tier: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["brier"])
        by_tier[r["tier"]].append(r["brier"])

    # Calibration: bucket p_yes into deciles, compute actual-yes rate per bucket.
    buckets: list[list[dict]] = [[] for _ in range(10)]
    for r in results:
        idx = min(9, int(r["p"] * 10))
        buckets[idx].append(r)
    cal = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        lo, hi = i / 10, (i + 1) / 10
        mean_p = statistics.mean(b["p"] for b in bucket)
        actual_yes_rate = statistics.mean(b["actual"] for b in bucket)
        cal.append(
            {
                "bucket": f"[{lo:.1f}-{hi:.1f})",
                "n": len(bucket),
                "mean_p": round(mean_p, 3),
                "actual_yes_rate": round(actual_yes_rate, 3),
            }
        )

    return {
        "n": len(results),
        "brier": round(total_brier, 5),
        "by_category": {k: {"n": len(v), "brier": round(statistics.mean(v), 5)} for k, v in by_cat.items()},
        "by_tier": {k: {"n": len(v), "brier": round(statistics.mean(v), 5)} for k, v in by_tier.items()},
        "calibration": cal,
        "rows": results,
    }


def print_report(report: dict, show_rows: int = 0) -> None:
    print(f"N predictions: {report['n']}")
    print(f"Overall Brier: {report['brier']}")
    print(f"  baseline (always-0.5): 0.25")
    print()
    print("By category:")
    for cat, info in sorted(report["by_category"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {cat:<26} n={info['n']:<4} brier={info['brier']}")
    print()
    print("By liquidity tier:")
    for tier, info in sorted(report["by_tier"].items()):
        print(f"  {tier:<22} n={info['n']:<4} brier={info['brier']}")
    print()
    print("Calibration (p bucket → actual yes rate):")
    print(f"  {'bucket':<14}{'n':>5}{'mean_p':>10}{'actual':>10}")
    for c in report["calibration"]:
        print(f"  {c['bucket']:<14}{c['n']:>5}{c['mean_p']:>10}{c['actual_yes_rate']:>10}")
    if show_rows:
        print("\nDetail (worst-Brier rows):")
        worst = sorted(report["rows"], key=lambda r: -r["brier"])[:show_rows]
        for r in worst:
            print(
                f"  {r['ticker'][:48]:<50} p={r['p']:.3f} actual={r['actual']:.0f} "
                f"brier={r['brier']:.4f}  {r['rationale'][:80]}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fixture", nargs="?", default=str(DEFAULT_FIXTURE), help="path to a .jsonl fixture")
    p.add_argument("--with-llm", action="store_true", help="exercise the LLM path (slow + costs tokens)")
    p.add_argument("--worst", type=int, default=5, help="show worst-N rows by Brier")
    args = p.parse_args()

    entries = load_fixture(Path(args.fixture))
    if not entries:
        print(f"empty fixture: {args.fixture}", file=sys.stderr)
        return 1
    report = aggregate(entries, with_llm=args.with_llm)
    print_report(report, show_rows=args.worst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
