#!/usr/bin/env bash
# Daily eval-window submit flow for Prophet Hacks 2026.
#
# Run once per day from May 17 through May 28. Sequences the full daily
# workflow: refresh resolutions, refit calibration, snapshot today's
# markets, fetch open events, generate predictions, submit. The server
# scores the latest prediction per market, so submitting once a day is
# fine; running this more often is also safe and only updates outputs.
#
# Cron example (run daily at 10:00 local):
#   0 10 * * * cd "$HOME/Documents/AI FORECASTING HACKATHON" && \
#       bash scripts/daily_submit.sh >> data/daily.log 2>&1
#
# launchd alternative (LaunchAgents/com.prophet-hacks.daily.plist):
#   ProgramArguments: bash -lc '"cd '"$PWD"' && bash scripts/daily_submit.sh"'
#   StartCalendarInterval: Hour=10, Minute=0
#
# Exit codes:
#   0  all critical steps succeeded
#   1  critical failure (events / predict / submit)
#   2  preflight env var missing

set -uo pipefail
cd "$(dirname "$0")/.."

# Source env so PA_SERVER_API_KEY and friends are present.
set -a
[[ -f .env ]] && . .env
set +a

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

log "=== Daily submit run start ==="

# Preflight: required env vars.
missing=()
for var in ANTHROPIC_API_KEY OPENAI_API_KEY PA_SERVER_API_KEY PA_TEAM_NAME; do
    if [[ -z "${!var:-}" ]]; then
        missing+=("$var")
    fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
    log "Missing required env vars: ${missing[*]}"
    log "  PA_SERVER_API_KEY and PA_TEAM_NAME are issued at hackathon kickoff"
    log "  See https://prophetarena.co/profile/api-keys"
    exit 2
fi

PYTHON=".venv/bin/python"
PROPHET=".venv/bin/prophet"
mkdir -p data

# Step 1: resolve newly-settled predictions into data/resolved_predictions.jsonl.
log "Step 1/7: resolve_predictions.py"
"$PYTHON" scripts/resolve_predictions.py || log "  (resolver had issues, continuing)"

# Step 2: refit calibration table from accumulated resolutions.
# Refuses to save if < 20 samples — agent will pass through unchanged.
log "Step 2/7: fit_calibration.py"
"$PYTHON" scripts/fit_calibration.py || log "  (calibration not yet trustworthy, continuing)"

# Step 3: snapshot today's top-volume open markets (grows the live fixture).
log "Step 3/7: capture_live_snapshots.py"
"$PYTHON" scripts/capture_live_snapshots.py || log "  (capture failed, continuing)"

# Step 4: fetch open events from the Prophet Arena server.
log "Step 4/7: prophet forecast events"
if ! "$PROPHET" forecast events -o events.json; then
    log "Critical: events fetch failed; aborting before submit"
    exit 1
fi
n_events=$(wc -l < events.json | tr -d ' ')
log "  events.json fetched (${n_events} lines)"

# Step 5: generate predictions. agent.predict auto-applies the calibration
# table at data/calibration.json if it exists.
log "Step 5/7: prophet forecast predict (--local agent.predict)"
if ! "$PROPHET" forecast predict \
        --events events.json \
        --local agent.predict \
        -o submission.json; then
    log "Critical: prediction generation failed; aborting before submit"
    exit 1
fi
n_preds=$(grep -c '"p_yes"' submission.json || echo 0)
log "  submission.json: ${n_preds} predictions"

# Step 6: submit to the server. Scorer takes the latest per market.
log "Step 6/7: prophet forecast submit"
if ! "$PROPHET" forecast submit --submission submission.json; then
    log "Critical: submit failed"
    exit 1
fi
log "  submission accepted"

# Step 7: post-run sanity analysis (best effort).
log "Step 7/7: analyze_predictions.py (post-run)"
"$PYTHON" scripts/analyze_predictions.py 2>&1 | tail -20 || log "  (analyze skipped)"

log "=== Daily submit run complete ==="
