"""Append-only log of every prediction the agent emits.

Used during the live eval window so we can compute Brier and inspect
calibration as outcomes resolve. The resolver script
(`scripts/resolve_predictions.py`) later annotates each entry with the
actual outcome.

Log format (JSONL, one record per line):
{
  "ts":        ISO timestamp of prediction,
  "event":     event dict passed to predict(),
  "p_yes":     float,
  "rationale": str
}

Defensive: log_prediction never raises. A logging failure must not break
a live `/predict` response.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = "data/predictions.jsonl"


def get_log_path() -> Path:
    return Path(os.environ.get("PREDICTION_LOG_PATH", DEFAULT_LOG_PATH))


def log_prediction(event: dict, p_yes: float, rationale: str) -> None:
    """Append a single prediction to the log file. Never raises."""
    try:
        path = get_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "event": event,
            "p_yes": float(p_yes),
            "rationale": rationale,
        }
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("prediction log write failed: %s", e)
