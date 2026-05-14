"""Simulated trading P&L on the resolved-markets fixture.

For each fixture entry:
  1. Mock `get_market` to return the snapshot.
  2. Call `agent.trading.decide()` to get a buy/sell/hold decision.
  3. If actionable, call `book.attempt_open(decision, category)`.
After all decisions, resolve every market against its known outcome and
print the realized P&L, win rate, and per-category breakdown.

Caveats:
- The candlestick fixture is biased: snapshot at 75% of market lifetime
  with the known outcome means we're paper-trading at a point closer to
  resolution than a real-time agent would be. P&L numbers should be
  read as "given a calibrated forecast, does the sizing/edge logic
  produce positive P&L?" — not as a forecast of live performance.
- LLM is disabled in backtest (same leakage concern as `backtest.py`).
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

import functools

DEFAULT_FIXTURE = Path("tests/fixtures/resolved_markets.jsonl")


def load_fixture(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _market_lookup_factory(entries: list[dict]):
    by_ticker = {e["event"]["market_ticker"]: e["market_snapshot"] for e in entries}

    def _lookup(ticker: str, **kwargs):
        return by_ticker.get(ticker)

    return _lookup


def run(entries: list[dict], *, with_llm: bool = False, starting_bankroll: float = 1000.0) -> dict:
    from agent import llm as llm_mod
    from agent.trading import PositionBook, decide

    book = PositionBook(starting_bankroll=starting_bankroll)
    market_lookup = _market_lookup_factory(entries)

    if with_llm:
        # Web search would leak resolution info on settled markets. Use the
        # ensemble in no-web-search mode so we measure training-cutoff
        # knowledge only.
        no_search_ensemble = functools.partial(
            llm_mod.llm_forecast_ensemble, with_web_search=False
        )
        llm_patch_independent = patch(
            "agent.independent.llm_forecast_ensemble", side_effect=no_search_ensemble
        )
    else:
        llm_patch_independent = patch(
            "agent.independent.llm_forecast_ensemble", return_value=None
        )

    decisions: list[dict] = []
    with patch("agent.trading.get_market", side_effect=market_lookup), patch(
        "agent.predict.get_market", side_effect=market_lookup
    ), llm_patch_independent:
        # First pass: collect decisions
        for entry in entries:
            event = entry["event"]
            d = decide(event)
            book.attempt_open(d, category=event.get("category", "?"))
            decisions.append(
                {
                    "ticker": event["market_ticker"],
                    "category": event.get("category", "?"),
                    "action": d.action,
                    "size": d.size_fraction,
                    "our_p": d.our_p,
                    "yes_ask": d.yes_ask,
                    "no_ask": d.no_ask,
                    "result": entry["result"],
                }
            )

    # Resolve everything
    realized_per_category: dict[str, float] = defaultdict(float)
    wins, losses = 0, 0
    for d in decisions:
        if d["action"] == "hold":
            continue
        pnl = book.resolve(d["ticker"], d["result"])
        realized_per_category[d["category"]] += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

    actionable = [d for d in decisions if d["action"] != "hold"]
    return {
        "starting_bankroll": starting_bankroll,
        "ending_cash": book.cash,
        "realized_pnl": book.realized_pnl,
        "n_decisions": len(decisions),
        "n_actionable": len(actionable),
        "n_holds": len(decisions) - len(actionable),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, wins + losses),
        "by_category": dict(realized_per_category),
        "decisions": decisions,
    }


def print_report(report: dict, show_rows: int = 10) -> None:
    print(f"Starting bankroll: ${report['starting_bankroll']:.2f}")
    print(f"Decisions:        {report['n_decisions']}")
    print(f"  actionable:     {report['n_actionable']}")
    print(f"  holds:          {report['n_holds']}")
    if report["n_actionable"] == 0:
        print("\n(No actionable trades — nothing to score.)")
        return

    print(f"Wins / Losses:    {report['wins']} / {report['losses']}")
    print(f"Win rate:         {report['win_rate']:.1%}")
    print(f"Realized P&L:     ${report['realized_pnl']:+.2f}")
    print(f"Return on bank:   {report['realized_pnl'] / report['starting_bankroll']:+.2%}")
    print()
    print("By category:")
    for cat, pnl in sorted(report["by_category"].items(), key=lambda kv: -kv[1]):
        n = sum(1 for d in report["decisions"] if d["category"] == cat and d["action"] != "hold")
        print(f"  {cat:<22} n={n:<4} pnl=${pnl:+.2f}")

    if show_rows:
        actionable_results = []
        for d in report["decisions"]:
            if d["action"] == "hold":
                continue
            price = d["yes_ask"] if d["action"] == "buy_yes" else d["no_ask"]
            won = (d["action"] == "buy_yes" and d["result"] == "yes") or (
                d["action"] == "buy_no" and d["result"] == "no"
            )
            payout = (1.0 / price) - 1.0 if won else -1.0  # return per $1 staked
            actionable_results.append({**d, "return_per_dollar": payout})

        actionable_results.sort(key=lambda r: -r["return_per_dollar"])
        print("\nTop winners (return per $1 staked):")
        for r in actionable_results[:show_rows]:
            if r["return_per_dollar"] <= 0:
                break
            print(
                f"  {r['ticker'][:46]:<48} {r['action']:<8} "
                f"p={r['our_p']:.2f} ret={r['return_per_dollar']:+.2f}"
            )
        print("\nWorst losers:")
        for r in actionable_results[-show_rows:]:
            if r["return_per_dollar"] >= 0:
                break
            print(
                f"  {r['ticker'][:46]:<48} {r['action']:<8} "
                f"p={r['our_p']:.2f} ret={r['return_per_dollar']:+.2f}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fixture", nargs="?", default=str(DEFAULT_FIXTURE))
    p.add_argument("--with-llm", action="store_true")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--worst", type=int, default=5)
    args = p.parse_args()

    entries = load_fixture(Path(args.fixture))
    if not entries:
        print(f"empty fixture: {args.fixture}", file=sys.stderr)
        return 1
    report = run(entries, with_llm=args.with_llm, starting_bankroll=args.bankroll)
    print_report(report, show_rows=args.worst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
