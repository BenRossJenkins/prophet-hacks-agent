#!/usr/bin/env bash
# Daily calibration refit during the May 17-31 eval window.
#
# Flow:
#   1. Pull the prediction log from the running Cloud Run instance via GCS
#      (the agent persists predictions to GCS via prediction_log on every
#      call; this script reads them locally).
#   2. Resolve newly-closed predictions against the Prophet Arena API and
#      Kalshi.
#   3. Fit a path-stratified calibration table.
#   4. Push the updated table to GCS — Cloud Run picks it up automatically
#      on the next request (60s in-process cache).
#
# Idempotent. Safe to re-run any time. Designed to be triggered daily by
# launchd (macOS) or cron.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/benjenkins/Documents/AI FORECASTING HACKATHON}"
cd "$REPO_DIR"

# Load .env so API keys are present.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . .env
    set +a
fi

PYTHON=".venv/bin/python"
# Per-event prediction objects live under this prefix (one JSON per event).
PRED_GCS_PREFIX="${PREDICTION_LOG_GCS_PREFIX:-gs://prophet-hacks-2026-calibration/predictions}"
LOCAL_PRED_PATH="${PREDICTION_LOG_PATH:-data/predictions.jsonl}"

mkdir -p "$(dirname "$LOCAL_PRED_PATH")"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') daily calibration refit ==="

# 1. Pull all per-event prediction objects from GCS and assemble into a single
#    JSONL file. Idempotent: each rebuild reflects the current set of objects.
echo "[1/4] Pulling prediction objects from $PRED_GCS_PREFIX/"
TMPDIR_PREDS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_PREDS"' EXIT
if gcloud storage ls "$PRED_GCS_PREFIX/" >/dev/null 2>&1; then
    gcloud storage cp "$PRED_GCS_PREFIX/*.json" "$TMPDIR_PREDS/" --quiet 2>/dev/null || true
    # Concatenate every JSON object as one JSONL line, sorted by filename
    # (filenames begin with timestamp, so this preserves chronological order).
    : > "$LOCAL_PRED_PATH"
    for f in $(ls "$TMPDIR_PREDS"/*.json 2>/dev/null | sort); do
        cat "$f" >> "$LOCAL_PRED_PATH"
        echo "" >> "$LOCAL_PRED_PATH"
    done
    echo "      $(wc -l < "$LOCAL_PRED_PATH") prediction rows assembled"
else
    echo "      No remote predictions yet — using local file if present"
fi

# 2. Resolve newly-closed predictions.
echo "[2/4] Resolving predictions"
"$PYTHON" scripts/resolve_predictions.py

# 3. Fit path-stratified calibration.
echo "[3/4] Fitting calibration"
if "$PYTHON" scripts/fit_calibration.py --min-samples 20; then
    echo "      calibration table updated"
else
    rc=$?
    if [ "$rc" -eq 2 ]; then
        echo "      not enough samples yet — calibration unchanged"
        exit 0
    fi
    echo "      fit_calibration exited with code $rc" >&2
    exit "$rc"
fi

# 4. (fit_calibration.py already pushes to CALIBRATION_GCS_URI internally.)
echo "[4/4] Done — Cloud Run will pick up the new table within 60s"
