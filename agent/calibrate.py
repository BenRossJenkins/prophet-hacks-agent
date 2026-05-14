"""Post-hoc calibration table: fit + apply.

Mirrors the upstream PR we shipped (feat/forecast-calibrate-command).
Bundled here so the live agent can apply a fitted calibration in real
time during the eval window, even before the upstream PR merges.

Workflow during the May 17-28 eval:
  1. Agent serves /predict; every prediction is logged via
     agent.prediction_log to data/predictions.jsonl.
  2. scripts/resolve_predictions.py runs daily, marking resolved
     predictions with their outcomes into
     data/resolved_predictions.jsonl.
  3. scripts/fit_calibration.py runs daily, fitting a calibration
     table to data/calibration.json AND optionally uploading it to
     a GCS bucket (`CALIBRATION_GCS_URI`).
  4. The live agent (this module) reads the table on each predict()
     call (60s in-process cache).

Two storage backends, in order of precedence:

  - CALIBRATION_GCS_URI (gs://bucket/path/file.json): preferred for
    deployed agents. Lets the Cloud Run instance pick up daily fits
    pushed by the local daily-submit script with no redeploy.
  - CALIBRATION_PATH (filesystem path; defaults to data/calibration.json):
    used for local development and as the fallback when GCS is
    unavailable.

If neither produces a valid table, predictions pass through unchanged.
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


def get_calibration_gcs_uri() -> str | None:
    return os.environ.get("CALIBRATION_GCS_URI") or None


def _load_from_gcs(uri: str) -> list[dict[str, Any]] | None:
    """Pull a calibration JSON from gs://bucket/object. Never raises."""
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
        data = json.loads(text)
        if data.get("version") != CALIBRATION_VERSION:
            return None
        return data.get("buckets") or None
    except Exception as e:
        logger.warning("GCS calibration fetch failed for %s: %s", uri, e)
        return None


_cache: dict[str, tuple[float, list[dict] | None]] = {}
_CACHE_TTL = 60.0  # seconds


def get_calibration_table() -> list[dict[str, Any]] | None:
    """Return the active calibration table with a 60s in-process cache.

    Tries GCS first (CALIBRATION_GCS_URI), then falls back to the local
    path. Either backend returning None means "no calibration", and
    predictions pass through unchanged.
    """
    import time

    gcs_uri = get_calibration_gcs_uri()
    cache_key = gcs_uri or get_calibration_path()
    now = time.time()
    cached = _cache.get(cache_key)
    if cached is not None and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    table: list[dict[str, Any]] | None = None
    if gcs_uri:
        table = _load_from_gcs(gcs_uri)
    if table is None:
        table = load_calibration(get_calibration_path())

    _cache[cache_key] = (now, table)
    return table
