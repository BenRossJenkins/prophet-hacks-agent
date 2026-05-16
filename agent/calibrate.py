"""Post-hoc calibration table: fit + apply.

Path-stratified calibration. Each pipeline branch (tail-anchor,
kalshi-anchor, kalshi+poly-blend, llm-grounded, llm-speculative, etc.)
gets its own calibration map fit on resolved predictions that took that
branch. Falls back to a global table when a per-path bucket has too few
samples to trust.

The path-stratified approach beats per-category for our agent because
error distributions cluster by *how a prediction was produced*, not by
*what the prediction was about*. A Politics question resolved by a deep
Kalshi book has the same error shape as a Sports question resolved by a
deep Kalshi book; both are wildly different from an LLM-speculative
prediction with no market signal.

Workflow during the May 17-28 eval:
  1. Agent serves /predict; every prediction is logged via
     agent.prediction_log with `metadata.path` stamped on each entry.
  2. scripts/resolve_predictions.py runs daily, marking resolved
     predictions with their outcomes into
     data/resolved_predictions.jsonl.
  3. scripts/fit_calibration.py runs daily, fitting one table per path
     plus a global table, writing to data/calibration.json AND
     optionally uploading to a GCS bucket (`CALIBRATION_GCS_URI`).
  4. The live agent (this module) reads the table on each predict()
     call (60s in-process cache), dispatches by path label, falls back
     to global when path data is sparse.

Schema:

  v2 (current):
    {"version": 2,
     "global": [bucket, ...],
     "by_path": {"tail-anchor": [bucket, ...],
                 "llm-grounded": [bucket, ...], ...}}

  v1 (legacy, still loaded):
    {"version": 1, "buckets": [bucket, ...]}
    → coerced to {"global": <buckets>, "by_path": {}}

Storage backends, in order of precedence:

  - CALIBRATION_GCS_URI: preferred for deployed agents (Cloud Run picks
    up daily-pushed updates without redeploy).
  - CALIBRATION_PATH (filesystem): local dev + GCS fallback.

If neither produces a usable table, predictions pass through unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = 2
DEFAULT_PATH = "data/calibration.json"

# Minimum samples in a per-path bucket before we trust it over global.
# Per-path tables fall back to global aggressively at low N.
MIN_BUCKET_N_FOR_PATH = 5

# Maximum amount a single calibration correction can shift the raw
# prediction. Protects against a small bucket (e.g. 5-7 events) with an
# extreme mean_actual from pulling a confident prediction wildly off.
# The shift is bounded as |adjusted - raw| <= MAX_CALIBRATION_SHIFT.
# Set to None to disable.
MAX_CALIBRATION_SHIFT: float | None = 0.05


def fit_calibration(
    rows: list[dict],
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Fit a binned calibration table from one slice of resolved predictions.

    `rows`: list of dicts with at least `p_yes` (float) and `result`
    ("yes"/"no").
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


def _row_path(row: dict) -> str | None:
    """Pull the pipeline-branch label from a logged prediction row.

    Prefers `metadata.path` (set by the new prediction_log format);
    falls back to classifying the rationale for legacy entries.
    """
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        p = metadata.get("path")
        if isinstance(p, str) and p:
            return p
    rationale = row.get("rationale")
    if isinstance(rationale, str) and rationale:
        from agent.prediction_log import classify_path  # lazy import

        return classify_path(rationale)
    return None


def fit_calibration_by_path(
    rows: list[dict],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit both a global table and one per-path table from resolved rows.

    Returns the dict payload to save (without version wrapper).
    """
    by_path_rows: dict[str, list[dict]] = {}
    for row in rows:
        path = _row_path(row)
        if path is None:
            continue
        by_path_rows.setdefault(path, []).append(row)

    by_path: dict[str, list[dict[str, Any]]] = {}
    for path, path_rows in by_path_rows.items():
        table = fit_calibration(path_rows, n_bins=n_bins)
        if table:
            by_path[path] = table

    return {
        "global": fit_calibration(rows, n_bins=n_bins),
        "by_path": by_path,
    }


def _bucket_for(
    p: float, table: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find the bucket containing p, or None if no table or p out of range."""
    if not table:
        return None
    p = max(0.0, min(0.9999, float(p)))
    for bucket in table:
        lo = float(bucket["bucket_lo"])
        hi = float(bucket["bucket_hi"])
        if hi >= 0.9999:
            if p >= lo:
                return bucket
        elif lo <= p < hi:
            return bucket
    return None


def apply_calibration(p_yes: float, table: list[dict[str, Any]]) -> float:
    """Replace p_yes with the bucket's observed yes-rate. No-op if no bucket."""
    bucket = _bucket_for(p_yes, table)
    if bucket is None:
        return p_yes
    return max(0.01, min(0.99, float(bucket["mean_actual"])))


def _bound_shift(raw: float, adjusted: float) -> float:
    """Clip the magnitude of (adjusted - raw) to MAX_CALIBRATION_SHIFT.

    Prevents a noisy small-N bucket from yanking a prediction far from
    the model's actual signal. Returns the bounded value, clamped to
    the submission-contract range.
    """
    if MAX_CALIBRATION_SHIFT is None:
        return max(0.01, min(0.99, adjusted))
    delta = adjusted - raw
    if delta > MAX_CALIBRATION_SHIFT:
        adjusted = raw + MAX_CALIBRATION_SHIFT
    elif delta < -MAX_CALIBRATION_SHIFT:
        adjusted = raw - MAX_CALIBRATION_SHIFT
    return max(0.01, min(0.99, adjusted))


def apply_calibration_data(
    p_yes: float,
    data: dict[str, Any] | None,
    path: str | None = None,
    *,
    min_n: int = MIN_BUCKET_N_FOR_PATH,
) -> float:
    """Path-stratified calibration with global fallback.

    Lookup order:
      1. by_path[path] bucket containing p_yes, if its n >= min_n
      2. global bucket containing p_yes
      3. unchanged p_yes

    The final adjustment is bounded so |adjusted - raw| ≤ MAX_CALIBRATION_SHIFT.
    """
    if not data:
        return p_yes
    if path:
        path_table = (data.get("by_path") or {}).get(path) or []
        bucket = _bucket_for(p_yes, path_table)
        if bucket is not None and int(bucket.get("n", 0)) >= min_n:
            return _bound_shift(p_yes, float(bucket["mean_actual"]))
    adjusted = apply_calibration(p_yes, data.get("global") or [])
    return _bound_shift(p_yes, adjusted)


def save_calibration(
    payload: list[dict[str, Any]] | dict[str, Any], path: str | Path
) -> None:
    """Save either a flat bucket list (legacy) or a v2 dict to disk."""
    if isinstance(payload, list):
        wrapper: dict[str, Any] = {
            "version": CALIBRATION_VERSION,
            "global": payload,
            "by_path": {},
        }
    else:
        wrapper = {
            "version": CALIBRATION_VERSION,
            "global": payload.get("global") or [],
            "by_path": payload.get("by_path") or {},
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(wrapper, indent=2))


def _coerce_to_v2(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a loaded payload to the v2 shape. None on unrecognized version."""
    version = data.get("version")
    if version == 2:
        return {
            "global": data.get("global") or [],
            "by_path": data.get("by_path") or {},
        }
    if version == 1:
        # Legacy single-table format.
        buckets = data.get("buckets")
        if isinstance(buckets, list):
            return {"global": buckets, "by_path": {}}
    return None


def load_calibration(path: str | Path) -> list[dict[str, Any]] | None:
    """Load v1-style flat bucket list. Returns just the global table for
    backwards compatibility. New code should call `load_calibration_data`.
    """
    data = load_calibration_data(path)
    return (data or {}).get("global") if data else None


def load_calibration_data(path: str | Path) -> dict[str, Any] | None:
    """Load the full v2 payload (global + by_path). None if missing/invalid."""
    try:
        raw = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return _coerce_to_v2(raw)


def get_calibration_path() -> str:
    return os.environ.get("CALIBRATION_PATH", DEFAULT_PATH)


def get_calibration_gcs_uri() -> str | None:
    return os.environ.get("CALIBRATION_GCS_URI") or None


def _load_from_gcs(uri: str) -> dict[str, Any] | None:
    """Pull a calibration JSON from gs://bucket/object. Never raises.

    Returns the v2 payload dict; callers needing just `global` can pull
    that field.
    """
    if not uri.startswith("gs://"):
        return None
    try:
        from google.cloud import storage  # lazy import

        rest = uri[len("gs://"):]
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            return None
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        text = blob.download_as_text(timeout=10)
        raw = json.loads(text)
        if not isinstance(raw, dict):
            return None
        return _coerce_to_v2(raw)
    except Exception as e:
        logger.warning("GCS calibration fetch failed for %s: %s", uri, e)
        return None


_cache: dict[str, tuple[float, dict | None]] = {}
_CACHE_TTL = 60.0  # seconds


def get_calibration_data() -> dict[str, Any] | None:
    """Return the active v2 calibration payload, GCS-preferred, with 60s cache."""
    import time

    gcs_uri = get_calibration_gcs_uri()
    cache_key = gcs_uri or get_calibration_path()
    now = time.time()
    cached = _cache.get(cache_key)
    if cached is not None and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    data: dict[str, Any] | None = None
    if gcs_uri:
        data = _load_from_gcs(gcs_uri)
    if data is None:
        data = load_calibration_data(get_calibration_path())

    _cache[cache_key] = (now, data)
    return data


def get_calibration_table() -> list[dict[str, Any]] | None:
    """Backwards-compat: return just the global bucket list."""
    data = get_calibration_data()
    if data is None:
        return None
    return data.get("global") or None
