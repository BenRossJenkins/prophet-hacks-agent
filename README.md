# Prophet Hacks 2026: Forecasting Agent

Team **Brier Patch** entry for the [Prophet Hacks](https://www.prophethacks.com)
AI forecasting hackathon. Forecasting Track. Build window May 16-17, 2026;
live evaluation through May 31 on [Prophet Arena](https://prophetarena.co).

## Team

- [@BenRossJenkins](https://github.com/BenRossJenkins)
- [@duckmoll](https://github.com/duckmoll)

## Live endpoint

The agent is deployed on Google Cloud Run and registered with the
Prophet Arena eval server:

- `POST https://prophet-hacks-agent-651046060481.us-central1.run.app/predict`
- `GET  https://prophet-hacks-agent-651046060481.us-central1.run.app/health`

Accepts one event per request, returns the probability distribution
across that event's outcomes. See "Agent contract" below.

## Stack

- Python 3.11+
- [`ai-prophet-core`](https://pypi.org/project/ai-prophet-core/) — SDK and API client
- [`ai-prophet`](https://pypi.org/project/ai-prophet/) — `prophet` CLI (retrieve / predict / evaluate)
- FastAPI + uvicorn for the live `/predict` endpoint
- Anthropic Claude + OpenAI + Google Gemini for the cross-vendor LLM ensemble
- Polymarket (gamma-api) and ESPN scoreboard for market-anchored priors
- NWS forecasts (weather) and yfinance (crypto) for typed external priors

## Pipeline

For binary events:

1. Fetch the Kalshi market by `market_ticker`.
2. **Tail-anchor triage**: confident liquid market (vol >= $500, price
   outside [0.05, 0.95]) returns directly with a 3% safety shrink.
3. **Cross-venue agreement gate** (politics / world / company / etc.):
   - In the safe band ([0.20, 0.80] with Kalshi vol >= $10k): fetch
     Polymarket and **only blend when |kalshi - poly| > 0.03**. When
     the venues agree, there's no signal in the cross-reference.
   - Outside the safe band: always volume-weighted-blend with Kalshi.
4. **Volume-weighted shrinkage** toward 0.5.
5. If no market signal, **category priors** (NWS for weather, yfinance
   for crypto, ESPN for sports, Manifold elsewhere).
6. Final fallback: **cross-vendor LLM ensemble** (Claude Opus
   extended-thinking + GPT-5-mini + Gemini 2.5 Flash) with shared web
   search (Anthropic anchors search; OpenAI + Gemini receive its
   findings as `search_context`). Median aggregation. **Three-tier
   tail-aware shrinkage:** `decisive` (alpha=0.02 when rationale
   describes a resolved outcome), `grounded` (alpha=0.05 when it cites
   current data), `speculative` (alpha=0.15 base-rate reasoning), with
   extra alpha at |p - 0.5| > 0.40 and a hard cap at alpha=0.50. If
   the whole ensemble fails, retry once with web search disabled
   before falling to uniform 0.5.
7. **Market sanity guardrail**: if final p deviates >0.30 from a deep
   liquid Kalshi mid, anchor 60/40 toward market.
8. **Path-stratified calibration** when a fitted table is present (GCS,
   refit daily during eval, ±0.05 shift cap). The pipeline branch is
   stamped at the producer (`tail-anchor`, `kalshi-anchor`,
   `kalshi+poly-blend`, `guardrail-anchored`, `prior`,
   `llm-{decisive,grounded,speculative}`, etc.) rather than re-derived
   from the rationale text, so composed rationales don't corrupt
   stratification. Per-bucket yes-rates are Beta-Bernoulli shrunk
   toward the bucket's `mean_p` with prior strength `N_0=10` so
   small-N buckets behave sensibly (`min_n=3`).

For multi-outcome events (3+ outcomes):

1. **Kalshi event lookup** via `/trade-api/v2/events/{event_ticker}
   ?with_nested_markets=true`. Requires `mutually_exclusive=true`,
   maps each child market to one of the event's outcomes, requires
   >=60% coverage.
2. **Polymarket event lookup** with the same coverage contract.
3. **Capped volume-weighted blend** of the two (KALSHI_POLY_MAX_WEIGHT
   = 0.75) when both are available; otherwise use whichever returned.
4. **LLM ensemble** with explicit `p_yes = P(outcomes[0])` framing and
   two worked top-K examples in the prompt; retry without web search
   on total failure.
5. **Uniform 1/N** as the final fallback.

All paths produce a `{probabilities: [...]}` response strictly summing
to 1, with `market` names matching the event's `outcomes` list exactly.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # fill in API keys (see Environment below)
```

## Run locally — full pipeline

```bash
# Pull a slate of sample events (no API key needed for dataset retrieval)
.venv/bin/prophet forecast retrieve --dataset sample-sports -o events.json

# Run our agent against those events via the prophet CLI
.venv/bin/prophet forecast predict \
    --events events.json \
    --local agent.predict \
    -o submission.json

cat submission.json
```

## Run as an HTTP server (matches the live deployment)

```bash
uvicorn agent.predict:app --host 0.0.0.0 --port 8000

# In another shell, exercise the endpoint via prophet CLI:
.venv/bin/prophet forecast predict \
    --events events.json \
    --agent-url http://localhost:8000/predict

# Or hit /predict directly with curl:
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @events.json
```

## For evaluators / organizers

One-shot script (recommended): runs a clean end-to-end evaluation against
a sample event slate.

```bash
git clone https://github.com/BenRossJenkins/prophet-hacks-agent.git
cd prophet-hacks-agent

export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
export GEMINI_API_KEY=AIza...

bash scripts/evaluate_agent.sh
```

`scripts/evaluate_agent.sh` handles Python version checks, virtualenv
setup, dependency installation, dataset retrieval, prediction, and a
human-readable summary of outputs. Approximate cost is `$0.25` for the
default 3-event smoke test; pass `EVAL_EVENT_COUNT=` (empty) to run the
full sample slate.

Manual setup, if you prefer step-by-step:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...

# Option A: run the agent as a local module via the official prophet CLI
prophet forecast retrieve --dataset sample-sports -o events.json
prophet forecast predict --events events.json --local agent.predict -o predictions.json

# Option B: run as an HTTP server (matches the deployed Cloud Run setup)
uvicorn agent.predict:app --host 0.0.0.0 --port 8000
prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
```

The container image is also available — `docker build -t agent .` then
`docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... agent` reproduces the
Cloud Run deployment locally.

## Reliability strategy

Completion rate is multiplicative on the final score, so a hung or crashed
request hurts as much as a wrong prediction. The agent is engineered so
no single failure can break a /predict response:

- **Ensemble hard deadline.** Each call to `llm_forecast_ensemble` is
  capped at 8 minutes (well below the 10-min per-event budget). Vendors
  still running at the deadline are abandoned and the median of whatever
  responses arrived is returned. A single hung LLM vendor cannot
  consume the whole budget.
- **Retry-without-search on total failure.** When the whole ensemble
  returns None (all 3 vendors failed simultaneously, typically a search
  tool rate-limit), we retry once with web search disabled. Base
  chat-completion APIs rate-limit independently of search tools, so
  this converts a 0.5-fallback into a real prediction roughly 50% of
  the time in stress tests. The eval server does NOT retry timed-out
  requests on our behalf, so this is on us.
- **Multi-tier market fallback chain.** When the LLM stack is
  unreachable, the agent falls through tail-anchor → Polymarket blend →
  category prior (NWS / yfinance / ESPN / Manifold) → uniform 0.5. Every
  external call is wrapped to return `None` on failure rather than
  raise, so any tier can fail without blocking the next.
- **Probabilities-only response contract is enforced at exactly one
  place.** Every code path produces its final distribution via
  `_wrap_binary` (binary events) or `_normalize_distribution`
  (multi-outcome), so the wire response is always well-formed JSON with
  probabilities summing to 1 and markets matching the event's outcomes.
- **Bounded calibration shift.** Any single calibration adjustment is
  capped at ±0.05 from the raw prediction so a noisy small-N bucket
  cannot yank a confident forecast off track. Per-bucket yes-rates
  are additionally Beta-Bernoulli shrunk toward the bucket's mean
  prediction so a 3-event "all yes" bucket doesn't output `1.0`.
- **Diff-sanity guard on calibration publishes.** The daily refit
  loads the previously-published table from GCS, compares each new
  small-N bucket against its predecessor, and refuses to publish
  (exit 3) when any small-N bucket moved by more than 0.20. Operators
  override with `--skip-diff-sanity` only after inspecting the data.
- **GCS-mirrored prediction log.** Every prediction is also written as a
  per-event JSON object to GCS so the daily calibration refit job has a
  durable read source — predictions survive Cloud Run container
  restarts and post-hoc audit is possible. The log records the
  producing pipeline branch and the agent `version` for each entry
  so post-eval analysis can attribute Brier deltas to specific
  versions.

The calibration table is refit nightly from resolved questions over the
eval window; this is a parameter update from observed data, not a
mid-eval code change. Architecture, prompts, and pipeline structure are
frozen at submission.

## Agent contract

The `/predict` endpoint matches the Prophet Arena spec
(<https://prophetarena.co/developer>):

**Input** (one event per request):

```json
{
  "event_ticker": "...",
  "market_ticker": "...",
  "title": "Who will win: A or B?",
  "category": "Sports",
  "rules": "Resolves to the winner.",
  "close_time": "2026-05-25T23:59:59Z",
  "outcomes": ["A", "B"]
}
```

**Output:**

```json
{
  "probabilities": [
    {"market": "A", "probability": 0.62},
    {"market": "B", "probability": 0.38}
  ]
}
```

- Probabilities sum to 1 across the event's outcomes
- Each `market` matches one of the event's `outcomes` exactly
- `p_yes` and `rationale` are included as extra fields for our own
  logging; the eval server ignores them

Brier scoring (lower is better) per the developer docs. Each outcome's
squared error contributes to the per-event Brier.

## Layout

```
agent/                 forecasting logic + FastAPI app
  predict.py             pipeline orchestration, /predict, /health
  kalshi.py              Kalshi market client
  polymarket.py          Polymarket binary + multi-outcome lookup
  sports.py              ESPN moneyline-derived prior
  weather.py             NWS-backed prior
  financials.py          yfinance-backed crypto prior
  manifold.py            Manifold fallback prior
  llm.py                 multi-vendor ensemble with shared web search
  calibrate.py           path-stratified calibration (GCS-loaded)
  prediction_log.py      defensive append-only log (GCS-mirrored)

scripts/               operational utilities
  daily_calibration.sh   daily cron wrapper (resolve + refit + push)
  resolve_predictions.py marks resolved predictions from PA + Kalshi
  fit_calibration.py     fits path-stratified calibration table
  backtest.py            local Brier evaluation against a fixture
  capture_live_snapshots.py / resolve_captures.py
                         daily capture flow for ongoing fixture growth
  build_backtest_fixture.py / build_diverse_fixture.py
                         reproducible candlestick-based fixtures

tests/                 unit tests; tests/fixtures/ for backtest data
```

320 tests under `pytest tests/`.

## Environment

See `.env.example`. Required for the full pipeline:

- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` — LLM vendors
- `PA_SERVER_URL=https://api.aiprophet.dev` — Prophet Arena base
- `PA_SERVER_API_KEY=prophet_...` — issued by organizers, X-API-Key header
- `PA_TEAM_NAME=Brier Patch` — registered team name

Optional:

- `PREDICTION_LOG_PATH` (default `data/predictions.jsonl`)
- `PREDICTION_LOG_GCS_PREFIX=gs://...` — mirror every prediction to GCS
- `CALIBRATION_GCS_URI=gs://.../calibration.json` — daily refit source
- `FORECAST_MODEL=claude-opus-4-7` — override the single-model path

## Architecture conventions and gotchas

See `CLAUDE.md` for the in-source architecture summary, defensive-
degradation rules, the path-stratified calibration design, and the
running gotchas list (Kalshi DNS, Polymarket DNS, the empty
LLM-denylist policy, etc.).
