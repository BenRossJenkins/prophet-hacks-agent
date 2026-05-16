"""Append-only log of every prediction the agent emits.

Used during the live eval window so we can compute Brier and inspect
calibration as outcomes resolve. The resolver script
(`scripts/resolve_predictions.py`) later annotates each entry with the
actual outcome.

Log format (JSONL, one record per line):
{
  "ts":         ISO timestamp of prediction,
  "event":      event dict passed to predict(),
  "p_yes":      float (probability of outcomes[0]),
  "rationale":  str (verbose human-readable),
  "metadata":   {
    "path":            str — which pipeline branch produced this prediction
                       ("tail-anchor"/"safe-band-anchor"/"poly-only"/
                        "kalshi+poly-blend"/"kalshi-anchor"/"guardrail-anchored"/
                        "prior"/"llm-grounded"/"llm-speculative"/"uniform"/
                        "multi-outcome-poly"/"multi-outcome-llm"/"multi-outcome-uniform")
    "category":        str — event category for stratified Brier analysis
    "n_outcomes":      int — len(outcomes), useful for binary-vs-multi grouping
  }
}

The `metadata.path` field enables rolling backtest analysis during the
eval window: which paths win on Brier, which need parameter tuning, etc.
Classification is done by string-match on the rationale to avoid threading
structured signals all the way through the forecast pipeline.

Defensive: log_prediction never raises. A logging failure must not break
a live `/predict` response.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "data/predictions.jsonl"

# Optional: when set to a `gs://bucket/prefix` URI, every prediction is
# ALSO written as a per-event JSON object under that prefix. This makes
# predictions durable across Cloud Run container restarts and lets the
# daily calibration cron pull them back without needing FS access to the
# running container. Object key is "<prefix>/<ts>_<market_ticker>.json".
PREDICTION_LOG_GCS_PREFIX_ENV = "PREDICTION_LOG_GCS_PREFIX"


def get_log_path() -> Path:
    return Path(os.environ.get("PREDICTION_LOG_PATH", DEFAULT_LOG_PATH))


def _write_to_gcs(entry: dict[str, Any]) -> None:
    """Write a single prediction entry to GCS as `<prefix>/<ts>_<ticker>.json`.

    Never raises. Silently skips when PREDICTION_LOG_GCS_PREFIX isn't set,
    when google-cloud-storage isn't importable, or on any network error.
    """
    prefix = os.environ.get(PREDICTION_LOG_GCS_PREFIX_ENV)
    if not prefix or not prefix.startswith("gs://"):
        return
    try:
        from google.cloud import storage  # lazy import

        rest = prefix[len("gs://"):]
        bucket_name, _, key_prefix = rest.partition("/")
        if not bucket_name:
            return
        key_prefix = key_prefix.strip("/")
        ts = entry.get("ts", "").replace(":", "-")
        ticker = (entry.get("event") or {}).get("market_ticker", "unknown")
        safe_ticker = ticker.replace("/", "_")
        object_name = (
            f"{key_prefix}/{ts}_{safe_ticker}.json" if key_prefix
            else f"{ts}_{safe_ticker}.json"
        )
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(object_name)
        blob.upload_from_string(
            json.dumps(entry), content_type="application/json", timeout=10
        )
    except Exception as e:
        logger.warning("GCS prediction-log write failed: %s", e)


def classify_path(rationale: str) -> str:
    """Map a rationale string to a coarse pipeline-branch label.

    Order matters: more specific markers checked first. Each prediction
    came from exactly one branch, but rationales can compose (e.g.
    'tail-anchor ... guardrail anchored' would be very unusual).
    """
    r = rationale or ""
    if "multi-outcome" in r:
        if "kalshi event" in r and "poly event" in r:
            return "multi-outcome-blend"
        if "kalshi event" in r or "kalshi only" in r:
            return "multi-outcome-kalshi"
        if "poly event" in r or "poly only" in r:
            return "multi-outcome-poly"
        if "LLM unavailable" in r:
            return "multi-outcome-uniform"
        return "multi-outcome-llm"
    if "guardrail anchored" in r:
        return "guardrail-anchored"
    if "tail-anchor" in r:
        return "tail-anchor"
    if "polymarket-only" in r:
        return "poly-only"
    if "blend" in r and "kalshi" in r.lower():
        return "kalshi+poly-blend"
    if "LLM (decisive" in r:
        return "llm-decisive"
    if "LLM (grounded" in r:
        return "llm-grounded"
    if "LLM (speculative" in r:
        return "llm-speculative"
    if "LLM unavailable" in r or "uniform prior" in r:
        return "uniform"
    if "prior:" in r:
        return "prior"
    return "kalshi-anchor"


def log_prediction(
    event: dict, p_yes: float, rationale: str, metadata: dict[str, Any] | None = None
) -> None:
    """Append a single prediction to the log file. Never raises.

    `metadata` is auto-populated with the classified path if absent.
    Callers can pass extra fields (e.g. p_yes_pre_calibration) and they
    will be merged in.
    """
    try:
        path = get_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "path": classify_path(rationale),
            "category": (event.get("category") or "") if isinstance(event, dict) else "",
            "n_outcomes": (
                len(event.get("outcomes") or []) if isinstance(event, dict) else 0
            ),
        }
        if metadata:
            meta.update(metadata)
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "event": event,
            "p_yes": float(p_yes) if p_yes is not None else None,
            "rationale": rationale,
            "metadata": meta,
        }
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("prediction log write failed: %s", e)
        return
    # Best-effort GCS mirror so predictions survive Cloud Run restarts.
    # Never blocks or breaks /predict; failures are logged and dropped.
    try:
        _write_to_gcs(entry)
    except Exception as e:
        logger.warning("prediction log GCS mirror failed: %s", e)
