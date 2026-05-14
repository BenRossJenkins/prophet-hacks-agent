# Submission contract

Derived from reading `reference/ai-prophet-upstream/packages/cli/ai_prophet/forecast/main.py`
and `packages/core/ai_prophet_core/forecast/schemas.py` (upstream cloned 2026-05-14).

## Agent surfaces (pick either; both expose the same logic)

### Local module
```python
def predict(event: dict) -> dict:
    return {"p_yes": 0.5, "rationale": "..."}
```
Called by `prophet forecast predict --events events.json --local <module>`.

### HTTP service
```
POST /predict
body: EventRequest JSON
200:  {"p_yes": float, "rationale": str}
```
Called by `prophet forecast predict --events events.json --agent-url <url>`,
and (during the live window) by the Prophet Arena server hitting our
registered `--endpoint-url`.

## Schemas

**EventRequest** (input):
| field          | type            | required |
|----------------|-----------------|----------|
| event_ticker   | str             | yes      |
| market_ticker  | str             | yes      |
| title          | str             | yes      |
| subtitle       | str \| None     | no       |
| description    | str \| None     | no       |
| category       | str             | yes      |
| rules          | str \| None     | no       |
| close_time     | str (ISO 8601)  | yes      |

**PredictionResponse** (output):
| field      | type            | constraint                                          |
|------------|-----------------|-----------------------------------------------------|
| p_yes      | float           | `0.01 ≤ p_yes ≤ 0.99` (Pydantic rejects otherwise) |
| rationale  | str             | required in our code; optional in upstream schema  |

## Scoring

Brier score across markets that resolve during the evaluation window:

```
Brier = (1 / N) * Σ (p_yes − actual)²
```

`actual ∈ {0.0, 1.0}`. **Lower is better.** Implication: a confidently-wrong
0.99 vs. truth=0 costs `(0.99)² ≈ 0.98`. A hedged 0.7 wrong costs `0.49`.
Calibration matters more than sharpness — never push to extremes without
strong evidence.

## Live evaluation flow (May 17–28)

1. `prophet forecast register --team-name <T> --endpoint-url <U>` registers our team and HTTP endpoint.
2. The Prophet Arena server (presumably) calls our `/predict` endpoint daily for new markets.
3. Brier scores accumulate as markets resolve.
4. Leaderboard: `prophet forecast leaderboard`.

## Implications for the build

- `predict` must be fast and reliable — a 30s timeout per event is the default.
- Need a deployable host (Fly, Render, GCP Cloud Run, etc.) running before May 17.
- Clamp `p_yes` to `[0.01, 0.99]` defensively — Pydantic will 422 otherwise.
- Handle the full Kalshi category surface, not just the ones we expect.
