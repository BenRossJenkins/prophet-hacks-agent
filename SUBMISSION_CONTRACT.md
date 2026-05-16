# Submission contract

Authoritative sources:
- Prophet Arena developer docs (<https://prophetarena.co/developer>)
- Upstream `ai-prophet-core/forecast/schemas.py` Pydantic models

Updated 2026-05-16 to reflect the probabilities-only response shape
(replaced the legacy binary `p_yes` contract).

## Agent surfaces (pick either; both expose the same logic)

### Local module
```python
def predict(event: dict) -> dict:
    return {
        "probabilities": [
            {"market": "Pittsburgh", "probability": 0.68},
            {"market": "Atlanta",    "probability": 0.32},
        ]
    }
```
Called by `prophet forecast predict --events events.json --local <module>`.

### HTTP service
```
POST /predict
body: EventRequest JSON
200:  {"probabilities": [{"market": <outcome>, "probability": <float>}, ...]}
```
Called by `prophet forecast predict --events events.json --agent-url <url>`
during local development, and by the Prophet Arena eval server hitting our
registered `--endpoint-url` during live eval.

## Schemas

**EventRequest** (input):

| field            | type              | required |
|------------------|-------------------|----------|
| event_ticker     | str               | yes      |
| market_ticker    | str               | yes      |
| title            | str               | yes      |
| subtitle         | str \| None       | no       |
| description      | str \| None       | no       |
| category         | str               | yes      |
| rules            | str \| None       | no       |
| close_time       | str (ISO 8601)    | yes      |
| outcomes         | list[str] \| None | no       |
| resolved_outcome | dict \| None      | no       |

**PredictionResponse** (output):

| field         | type                    | constraint                                |
|---------------|-------------------------|-------------------------------------------|
| probabilities | list[MarketProbability] | required; sums to 1; markets ∈ outcomes   |
| p_yes         | float \| None           | optional; for our own logging/calibration |
| rationale     | str \| None             | optional; ignored by server               |

Where `MarketProbability = {market: str, probability: float in [0, 1]}`.

## Scoring

Per-event multi-class Brier:

```
event_brier = Σ_i (p_i − outcome_i)²
```

…across the event's outcomes, where `outcome_i = 1.0` if outcome `i` is
in `actual_outcome` (a list of winner labels) and `0.0` otherwise. The
final team score is the mean per-event Brier across the eval window,
weighted by completion rate. **Lower is better.**

Implication: confidently-wrong 0.99 on a binary event costs
`(0.99)² + (0.01)² ≈ 0.98`. Hedged-wrong 0.7 costs
`(0.7)² + (0.3)² = 0.58`. Calibration beats sharpness; don't push to
extremes without strong evidence.

## Live evaluation flow (May 17 → May 31)

1. `prophet forecast register --team-name <T> --endpoint-url <U>`
   registers our team + HTTP endpoint with the Prophet Arena server.
2. The eval server queries our `/predict` endpoint for each new event
   (one event per request, 10-minute response budget per event).
3. Brier scores accumulate as events resolve via `/forecast/events?status=closed`.
4. Leaderboard at `https://api.aiprophet.dev/forecast/scores` (or
   `prophet forecast leaderboard`).

## Implementation rules

- `/predict` must ALWAYS return a well-formed `probabilities`
  distribution. Pydantic 422 = silent miss = completion-rate hit.
- Clamp each per-outcome probability to `[0, 1]` and normalize the
  list to sum to 1 strictly.
- Every code path through `agent.predict._forecast` produces a
  `PredictionResponse` via `_wrap_binary` (binary events) or
  `_normalize_distribution` (multi-outcome) — single chokepoint that
  enforces the contract.
- 10-minute per-event budget gives generous latency room; the LLM
  ensemble has an 8-minute internal deadline so a hung vendor never
  consumes the whole budget.
