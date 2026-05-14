"""Post-hoc calibration table — fit + apply.

Mirrors the upstream PR we shipped (feat/forecast-calibrate-command).
Bundled here so the live agent can apply a fitted calibration in
real time during the eval window, even before the upstream PR merges.

Workflow during the May 17-28 eval:
  1. Agent serves /predict; every prediction is logged via
     agent.prediction_log to data/predictions.jsonl.
  2. scripts/resolve_predictions.py runs daily, marking resolved
     predictions with their outcomes into data/resolved_predictions.jsonl.
  3. scripts/fit_calibration.py runs daily, fitting a calibration table
     from those resolutions into data/calibration.json.
  4. The live agent (this module) reads data/calibration.json on each
     predict() call and applies it before returning.

If the calibration file doesn't exist or is empty, predictions pass
through unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = 1
DEFAULT_PATH = "data/calibration.json"


def fit_calibration(
    rows: list[dict],  # entries from resolved_predictions.jsonl
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Fit a binned calibration table.

    `rows`: list of dicts with at least p_yes (float) and result ("yes"/"no").
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for row in rows:
        try:
            p = float(row.get("p_yes", 0.5))
        except (ValueError, TypeError):
            continue
        result = row.get("result", "")
        if result not in ("yes", "no"):
            continue
        actual = 1.0 if result == "yes" else 0.0
        p = max(0.0, min(0.9999, p))
        idx = min(n_bins - 1, int(p * n_bins))
        buckets[idx].append((p, actual))

    table: list[dict[str, Any]] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        n = len(bucket)
        mean_p = sum(b[0] for b in bucket) / n
        mean_actual = sum(b[1] for b in bucket) / n
        table.append(
            {
                "bucket_lo": round(i / n_bins, 4),
                "bucket_hi": round((i + 1) / n_bins, 4),
                "n": n,
                "mean_p": round(mean_p, 5),
                "mean_actual": round(mean_actual, 5),
            }
        )
    return table


def apply_calibration(p_yes: float, table: list[dict[str, Any]]) -> float:
    """Replace p_yes with the bucket's observed yes-rate."""
    if not table:
        return p_yes
    p = max(0.0, min(0.9999, float(p_yes)))
    for bucket in table:
        lo = float(bucket["bucket_lo"])
        hi = float(bucket["bucket_hi"])
        if hi >= 0.9999:
            if p >= lo:
                return max(0.01, min(0.99, float(bucket["mean_actual"])))
        elif lo <= p < hi:
            return max(0.01, min(0.99, float(bucket["mean_actual"])))
    return p_yes


def save_calibration(table: list[dict[str, Any]], path: str | Path) -> None:
    payload = {"version": CALIBRATION_VERSION, "buckets": table}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))


def load_calibration(path: str | Path) -> list[dict[str, Any]] | None:
    """Returns the table, or None if missing/unparseable. Never raises."""
    try:
        data = json.loads(Path(path).read_text())
        if data.get("version") != CALIBRATION_VERSION:
            return None
        return data.get("buckets") or None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def get_calibration_path() -> str:
    return os.environ.get("CALIBRATION_PATH", DEFAULT_PATH)


_cache: dict[str, tuple[float, list[dict] | None]] = {}
_CACHE_TTL = 60.0  # seconds


def get_calibration_table() -> list[dict[str, Any]] | None:
    """Read the calibration table from disk, with a 60s in-process cache.

    Cached so we don't disk-read on every /predict call but still pick up
    new tables produced by the daily fit script within a minute.
    """
    import time

    path = get_calibration_path()
    now = time.time()
    cached = _cache.get(path)
    if cached is not None and (now - cached[0]) < _CACHE_TTL:
        return cached[1]
    table = load_calibration(path)
    _cache[path] = (now, table)
    return table
