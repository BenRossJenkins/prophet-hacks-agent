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
2. **Tail-anchor triage** — confident liquid market (vol >= $500, price
   outside [0.05, 0.95]) returns directly with a 3% safety shrink.
3. **Safe-band auto-anchor** — liquid book in [0.20, 0.80] skips the
   Polymarket blend (Kalshi is well-calibrated there).
4. Otherwise, **Polymarket cross-reference** (politics / world / etc.) →
   volume-weighted blend.
5. If no market signal, **category priors** (NWS for weather, yfinance
   for crypto, ESPN for sports, Manifold elsewhere).
6. Final fallback: **LLM ensemble** (Claude Opus + GPT-5-mini + Gemini
   2.5 Flash) with shared web search, median aggregation, tail-aware
   shrinkage.
7. **Sanity guardrail** — if final p deviates >0.30 from a deep liquid
   Kalshi mid, anchor 60/40 toward market.
8. **Path-stratified calibration** when a fitted table is present (GCS,
   refit daily during eval).

Multi-outcome events skip steps 1-5 and try: Polymarket multi-outcome
event → LLM ensemble with per-outcome distribution → uniform 1/N.

All paths produce a `{probabilities: [...]}` response strictly summing to
1, with `market` names matching the event's `outcomes` list exactly.

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

To run the agent in a standardized environment:

```bash
# 1. Clone + install
git clone https://github.com/BenRossJenkins/prophet-hacks-agent.git
cd prophet-hacks-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Provide LLM keys via environment (the agent reads from os.environ).
#    The full ensemble requires all three; missing keys gracefully degrade.
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...

# 3. Start the server
uvicorn agent.predict:app --host 0.0.0.0 --port 8000

# 4. POST one event at a time
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"event_ticker":"x","market_ticker":"x","title":"Will A beat B?","category":"Sports","close_time":"2026-12-31T23:59:59Z","outcomes":["A","B"]}'
```

The container image is also available — `docker build -t agent .` then
`docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... agent` reproduces the
Cloud Run deployment locally.

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

248 tests under `pytest tests/`.

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
