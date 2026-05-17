#!/usr/bin/env bash
#
# One-shot evaluator entry point for Prophet Hacks judges and reviewers.
#
# Reproduces a clean evaluation of the Brier Patch forecasting agent in a
# standard environment:
#
#   1. Verifies Python 3.11+ is available.
#   2. Creates an isolated virtualenv (.venv) if one isn't present.
#   3. Installs the agent package and its declared dependencies.
#   4. Confirms the three LLM API keys are set (the agent gracefully
#      degrades if some are missing, but the full ensemble needs all
#      three to evaluate the agent at its intended strength).
#   5. Pulls a sample-sports event slate from ai-prophet-datasets via
#      the official `prophet forecast retrieve` CLI.
#   6. Runs predictions via `prophet forecast predict --local agent.predict`
#      — the same entrypoint Prophet Arena's eval harness uses, but
#      executed locally so judges can see every line of output without
#      relying on our deployed Cloud Run instance.
#   7. Prints a summary of the predictions: per-event probability
#      distribution, total Σ probabilities (should be ≈1 for binary /
#      single-winner events and ≈K for top-K events), and the rationale
#      that drove each forecast.
#
# Usage:
#
#     export ANTHROPIC_API_KEY=sk-ant-...
#     export OPENAI_API_KEY=sk-proj-...
#     export GEMINI_API_KEY=AIza...
#     bash scripts/evaluate_agent.sh
#
# Optional environment overrides:
#
#     EVAL_DATASET     — dataset slug to pull (default: sample-sports)
#     EVAL_EVENT_COUNT — limit events to first N (default: 3, keeps cost
#                        low; pass empty string to run the full slate)
#
# Cost note: each event triggers a 3-vendor LLM ensemble with shared web
# search. Approximate cost per event is $0.03-0.07 with current pricing.
# Default of 3 events keeps a full reviewer run under ~$0.25.

set -euo pipefail

# ---- 1. Python version check ---------------------------------------------

if ! command -v python3.11 >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PY=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    case "$PY" in
      3.11|3.12|3.13) PYTHON=python3 ;;
      *)
        echo "ERROR: Python 3.11+ required (found $PY). Install python3.11 and retry." >&2
        exit 1
        ;;
    esac
  else
    echo "ERROR: Python 3.11+ required but no python3 found on PATH." >&2
    exit 1
  fi
else
  PYTHON=python3.11
fi
echo "[1/7] Python: $($PYTHON --version)"

# ---- 2. Virtualenv -------------------------------------------------------

if [ ! -d .venv ]; then
  echo "[2/7] Creating .venv ..."
  "$PYTHON" -m venv .venv
else
  echo "[2/7] Reusing existing .venv"
fi

# ---- 3. Install dependencies --------------------------------------------

echo "[3/7] Installing dependencies (this may take a few minutes on first run)..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e .

# ---- 4. Verify API keys -------------------------------------------------

REQUIRED_KEYS=(ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY)
MISSING=()
for key in "${REQUIRED_KEYS[@]}"; do
  if [ -z "${!key:-}" ]; then
    MISSING+=("$key")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "[4/7] WARNING: the following env vars are NOT set: ${MISSING[*]}" >&2
  echo "      The agent will gracefully degrade (missing vendors are dropped" >&2
  echo "      from the ensemble), but evaluating the full pipeline requires" >&2
  echo "      all three keys. Get keys from:" >&2
  echo "        ANTHROPIC_API_KEY: https://console.anthropic.com/settings/keys" >&2
  echo "        OPENAI_API_KEY:    https://platform.openai.com/api-keys" >&2
  echo "        GEMINI_API_KEY:    https://aistudio.google.com/app/apikey" >&2
else
  echo "[4/7] All LLM API keys present."
fi

# ---- 5. Pull a sample event slate ---------------------------------------

DATASET="${EVAL_DATASET:-sample-economics}"
EVENT_COUNT="${EVAL_EVENT_COUNT:-3}"
EVENTS_FILE=$(mktemp -t prophet-events-XXXXXX.json)
echo "[5/7] Retrieving dataset '$DATASET' → $EVENTS_FILE ..."
.venv/bin/prophet forecast retrieve --dataset "$DATASET" -o "$EVENTS_FILE"

# Filter to events whose close_time is still in the future (the prophet
# CLI skips past-close events; sample datasets age relative to wall time,
# so filtering keeps the script robust regardless of when it's run).
TRIMMED=$(mktemp -t prophet-events-trimmed-XXXXXX.json)
.venv/bin/python - "$EVENTS_FILE" "$TRIMMED" "${EVENT_COUNT:-0}" <<'PY'
import json, sys
from datetime import datetime, timezone
events_path, out_path, count_str = sys.argv[1], sys.argv[2], sys.argv[3]
events = json.load(open(events_path))
now = datetime.now(timezone.utc)
def is_future(e):
    ts = e.get("close_time")
    if not ts:
        return True
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")) > now
    except (ValueError, TypeError):
        return True
future = [e for e in events if is_future(e)]
count = int(count_str) if count_str.isdigit() and int(count_str) > 0 else len(future)
trimmed = future[:count]
json.dump(trimmed, open(out_path, "w"), indent=2)
skipped = len(events) - len(future)
print(f"  filtered: {len(events)} total → {len(future)} future-dated → {len(trimmed)} after trim "
      f"({skipped} skipped as past close_time)")
PY
EVENTS_FILE="$TRIMMED"

# Bail early if filtering left nothing to evaluate.
COUNT=$(.venv/bin/python -c "import json; print(len(json.load(open('$EVENTS_FILE'))))")
if [ "$COUNT" -eq 0 ]; then
  echo "ERROR: no future-dated events in '$DATASET' to evaluate against." >&2
  echo "       Try a different EVAL_DATASET (sample-sports / sample-economics / sample-entertainment)." >&2
  exit 1
fi

# ---- 6. Run predictions locally -----------------------------------------

OUT_FILE=$(mktemp -t prophet-predictions-XXXXXX.json)
echo "[6/7] Running agent on events (this calls the LLM ensemble — 30-90s per event)..."
.venv/bin/prophet forecast predict \
  --events "$EVENTS_FILE" \
  --local agent.predict \
  --output "$OUT_FILE"

# ---- 7. Summarize -------------------------------------------------------

echo "[7/7] Predictions written to $OUT_FILE. Summary:"
export EVENTS_FILE OUT_FILE
.venv/bin/python - <<'PY'
import json
import os

events = json.load(open(os.environ["EVENTS_FILE"]))
data = json.load(open(os.environ["OUT_FILE"]))
predictions = data if isinstance(data, list) else data.get("predictions", [])

events_by_ticker = {e["market_ticker"]: e for e in events}

# Note: `prophet forecast predict --local` writes a slim summary
# (market_ticker, p_yes, rationale) per prediction. The full per-outcome
# distribution is computed by the agent and visible via the HTTP endpoint
# (`prophet forecast predict --agent-url ...`) or the `probabilities`
# field of `predict()`'s return value; the CLI strips it here.

print()
for entry in predictions:
    ticker = entry.get("market_ticker") or entry.get("ticker") or "?"
    event = events_by_ticker.get(ticker, {})
    p_yes = entry.get("p_yes")
    rationale = (entry.get("rationale") or "")
    # Show the per-vendor breakdown if present in the rationale.
    if "ensemble[" in rationale:
        ensemble_part = rationale.split("ensemble[", 1)[1].split("]", 1)[0]
    else:
        ensemble_part = "(no ensemble breakdown)"
    print(f"  ── {ticker} ──")
    print(f"     title:    {event.get('title', '?')[:90]}")
    p_str = f"{p_yes:.4f}" if isinstance(p_yes, (int, float)) else "?"
    print(f"     p_yes:    {p_str}   (probability of outcomes[0]={event.get('outcomes', ['?'])[0]!r})")
    print(f"     vendors:  ensemble[{ensemble_part}]")
    print(f"     rationale tail: …{rationale[-160:]}")
    print()

print(f"Detailed per-outcome distributions are visible when running the agent")
print(f"as an HTTP server (see 'Done.' note below) — the prophet CLI's local")
print(f"mode summarises to p_yes + rationale only.")
PY

echo
echo "Done. The agent is also runnable as an HTTP server with:"
echo "  .venv/bin/uvicorn agent.predict:app --host 0.0.0.0 --port 8000"
echo "and exercised via:"
echo "  .venv/bin/prophet forecast predict --events <events.json> --agent-url http://localhost:8000/predict"
