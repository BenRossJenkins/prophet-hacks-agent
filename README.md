# Prophet Hacks 2026 — Forecasting Agent

Solo entry for the [Prophet Hacks](https://www.prophethacks.com) AI forecasting hackathon
(May 16–17, 2026 build; May 17–28 live evaluation on [Prophet Arena](https://prophetarena.co)).

## Stack

- Python 3.11+
- [`ai-prophet-core`](https://pypi.org/project/ai-prophet-core/) — SDK & API client
- [`ai-prophet`](https://pypi.org/project/ai-prophet/) — provides the `prophet` CLI (retrieve / predict / submit)
- FastAPI for the live `/predict` endpoint

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
agent/            # our forecasting logic — `predict(event)` and FastAPI `app`
scripts/          # local eval harness, packaging helpers (TBD)
reference/        # upstream ai-prophet clone (gitignored, for reading source)
```

## Contract & scoring

See `SUBMISSION_CONTRACT.md`. Scoring is Brier (lower is better), with `p_yes` enforced to [0.01, 0.99].
