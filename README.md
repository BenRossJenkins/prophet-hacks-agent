# Prophet Hacks 2026: Forecasting Agent

Team entry for the [Prophet Hacks](https://www.prophethacks.com) AI forecasting hackathon.
Build window May 16-17, 2026 (9:00 AM CT kickoff to 5:00 PM CT deadline). Live evaluation on
[Prophet Arena](https://prophetarena.co) through ~May 31.

## Team

- [@BenRossJenkins](https://github.com/BenRossJenkins)
- [@duckmoll](https://github.com/duckmoll)

## Stack

- Python 3.11+
- [`ai-prophet-core`](https://pypi.org/project/ai-prophet-core/): SDK and API client
- [`ai-prophet`](https://pypi.org/project/ai-prophet/): provides the `prophet` CLI (retrieve / predict / submit)
- FastAPI for the live `/predict` endpoint
- Anthropic + OpenAI + Google Gemini for the LLM ensemble

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in keys
```

## Local pipeline

```bash
# 1. Pull markets for tomorrow
prophet forecast retrieve --deadline "2026-05-15T23:59:59Z" -o events.json

# 2. Run our agent locally over those events
prophet forecast predict --events events.json --local agent.predict -o submission.json

# 3. Inspect
cat submission.json

# 4. (optional) Score against actuals once they're known
prophet forecast evaluate --submission submission.json --actuals actuals.json
```

## Live HTTP server

```bash
uvicorn agent.predict:app --host 0.0.0.0 --port 8000
# then in another shell:
prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
```

## Layout

```
agent/            our forecasting logic: predict(event) and the FastAPI app
scripts/          local eval harness, backtest fixture builders, daily submit flow
tests/            unit tests + resolved-markets fixture
reference/        upstream ai-prophet clone (gitignored, for reading source)
```

## Contract and scoring

See `SUBMISSION_CONTRACT.md`. Scoring is Brier (lower is better), with `p_yes` enforced to [0.01, 0.99].

## Deployment

Live `/predict` endpoint on Cloud Run:
[`prophet-hacks-agent-651046060481.us-central1.run.app`](https://prophet-hacks-agent-651046060481.us-central1.run.app)

See `CLAUDE.md` for the architectural conventions and the in-source gotchas list.
