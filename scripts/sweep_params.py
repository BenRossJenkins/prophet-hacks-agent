"""Brier-optimal parameter search for the agent's tunable constants.

Runs the backtest aggregator with different candidate values for each
constant in turn (others held at default) and reports the value that
minimizes Brier on `tests/fixtures/resolved_markets.jsonl`.

Caveats:
- Our fixture is weather-heavy (~95% Climate and Weather). Constants on
  the weather-prior path get good signal; constants on the market-anchor
  path get less (most fixture markets are thin and hit fallback).
- LLM is disabled for the sweep (same as `backtest --without-llm`). This
  measures the agent's deterministic paths only.
- Each sweep is univariate (one constant at a time). Multivariate
  interactions aren't explored.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable so we can reuse backtest.aggregate / load_fixture.
sys.path.insert(0, str(Path(__file__).parent))

import backtest as bt  # noqa: E402

import agent.predict as P  # noqa: E402
import agent.trading as T  # noqa: E402
import agent.weather as W  # noqa: E402


def sweep(label: str, module, attr: str, values: list, entries: list) -> list[tuple]:
    """Sweep one (module, attr) over candidate values; print Brier per value."""
    orig = getattr(module, attr)
    rows: list[tuple] = []
    for v in values:
        setattr(module, attr, v)
        try:
            report = bt.aggregate(entries, with_llm=False)
            rows.append((v, report["brier"]))
        except Exception as e:
            rows.append((v, f"err: {e}"))
    setattr(module, attr, orig)

    print(f"\n=== {label} ({module.__name__}.{attr}, default={orig}) ===")
    valid = [(v, b) for v, b in rows if isinstance(b, float)]
    if valid:
        best_brier = min(b for _, b in valid)
        for v, b in rows:
            mark = " ← best" if isinstance(b, float) and b == best_brier else ""
            print(f"  {v!r:<8} → brier={b}{mark}")
    return rows


def main() -> int:
    fixture = Path("tests/fixtures/resolved_markets.jsonl")
    if not fixture.exists():
        print(f"No fixture at {fixture}", file=sys.stderr)
        return 1
    entries = bt.load_fixture(fixture)
    print(f"Fixture: {len(entries)} entries")

    # Forecasting — shrinkage
    sweep("MAX_SHRINK_ALPHA", P, "MAX_SHRINK_ALPHA", [0.02, 0.05, 0.08, 0.10, 0.15, 0.20], entries)
    sweep("ALPHA_VOL_SCALE", P, "ALPHA_VOL_SCALE", [50.0, 100.0, 200.0, 400.0, 1000.0], entries)

    # Forecasting — liquidity gates
    sweep("MIN_VOL_24H", P, "MIN_VOL_24H", [10.0, 25.0, 50.0, 100.0, 200.0], entries)
    sweep("MAX_SPREAD", P, "MAX_SPREAD", [0.10, 0.15, 0.20, 0.30, 0.50], entries)

    # Forecasting — LLM shrinkage (only meaningful when LLM is exercised; with
    # LLM disabled in the sweep these are no-ops, but we list them so the
    # sweep file is the complete record of what's tunable).
    sweep("LLM_SHRINK_SPEC", P, "LLM_SHRINK_SPECULATIVE", [0.05, 0.10, 0.15, 0.20, 0.30], entries)
    sweep("LLM_SHRINK_GROUND", P, "LLM_SHRINK_GROUNDED", [0.02, 0.05, 0.10, 0.15], entries)

    # Weather — sigma (the sigmoid bandwidth in temperature units)
    sweep("TEMP_SIGMA_F", W, "TEMP_SIGMA_F", [1.5, 2.0, 3.0, 4.0, 5.0, 7.0], entries)

    # Trading — edge threshold (does not affect predict() Brier, but lets us
    # check that the value isn't on a cliff for some pathological reason)
    sweep("EDGE_THRESHOLD", T, "EDGE_THRESHOLD", [0.02, 0.05, 0.08, 0.10, 0.15], entries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
