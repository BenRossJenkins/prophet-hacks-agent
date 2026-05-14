"""Inspect calibration and Brier on resolved predictions.

Reads ``data/resolved_predictions.jsonl`` (produced by
``scripts/resolve_predictions.py``) and prints overall Brier, breakdowns
by category and by predicted-probability bucket, and the worst-Brier
rows so you can eyeball what's hurting us.

This is a *manual* tuning loop — the script doesn't auto-adjust any
agent constants. Read the output, decide if a category or bucket is
miscalibrated, edit the constants in `agent/predict.py` (or `agent/
priors.py`) yourself and redeploy.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_resolved(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def brier(p: float, actual: float) -> float:
    return (p - actual) ** 2


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for e in entries:
        p = float(e.get("p_yes", 0.5))
        result = e.get("result", "")
        if result not in ("yes", "no"):
            continue
        actual = 1.0 if result == "yes" else 0.0
        rows.append(
            {
                "ts": e.get("ts", ""),
                "ticker": e.get("event", {}).get("market_ticker", ""),
                "category": e.get("event", {}).get("category", "?"),
                "p": p,
                "actual": actual,
                "brier": brier(p, actual),
                "rationale": e.get("rationale", ""),
            }
        )
    if not rows:
        return {"n": 0}

    total_brier = statistics.mean(r["brier"] for r in rows)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["brier"])

    buckets: list[list[dict]] = [[] for _ in range(10)]
    for r in rows:
        idx = min(9, int(r["p"] * 10))
        buckets[idx].append(r)
    cal = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        lo, hi = i / 10, (i + 1) / 10
        cal.append(
            {
                "bucket": f"[{lo:.1f}-{hi:.1f})",
                "n": len(bucket),
                "mean_p": round(statistics.mean(b["p"] for b in bucket), 3),
                "actual_yes_rate": round(statistics.mean(b["actual"] for b in bucket), 3),
                "brier": round(statistics.mean(b["brier"] for b in bucket), 5),
            }
        )

    return {
        "n": len(rows),
        "brier": round(total_brier, 5),
        "by_category": {
            k: {"n": len(v), "brier": round(statistics.mean(v), 5)} for k, v in by_cat.items()
        },
        "calibration": cal,
        "rows": rows,
    }


def print_report(report: dict, show_rows: int = 10) -> None:
    n = report["n"]
    if n == 0:
        print("No resolved predictions yet.")
        return
    print(f"N resolved predictions: {n}")
    print(f"Overall Brier:          {report['brier']}")
    print(f"  baseline (always-0.5): 0.25")
    print()
    print("By category:")
    for cat, info in sorted(report["by_category"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {cat:<26} n={info['n']:<4} brier={info['brier']}")
    print()
    print("Calibration (p bucket → actual yes rate):")
    print(f"  {'bucket':<14}{'n':>5}{'mean_p':>10}{'actual':>10}{'brier':>10}")
    for c in report["calibration"]:
        flag = ""
        gap = abs(c["mean_p"] - c["actual_yes_rate"])
        if c["n"] >= 5 and gap > 0.10:
            flag = "  ← MISCAL"
        print(
            f"  {c['bucket']:<14}{c['n']:>5}{c['mean_p']:>10}{c['actual_yes_rate']:>10}"
            f"{c['brier']:>10}{flag}"
        )
    if show_rows:
        worst = sorted(report["rows"], key=lambda r: -r["brier"])[:show_rows]
        print("\nWorst-Brier rows:")
        for r in worst:
            print(
                f"  {r['ticker'][:48]:<50} cat={r['category'][:10]:<12} "
                f"p={r['p']:.3f} actual={r['actual']:.0f} brier={r['brier']:.4f}"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "input",
        nargs="?",
        default=os.environ.get("RESOLVED_LOG_PATH", "data/resolved_predictions.jsonl"),
    )
    p.add_argument("--worst", type=int, default=10)
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"No resolved log at {path}", file=sys.stderr)
        return 1
    entries = load_resolved(path)
    report = aggregate(entries)
    print_report(report, show_rows=args.worst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
