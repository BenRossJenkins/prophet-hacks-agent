"""Forecasting agent entrypoint.

Two interchangeable surfaces:
  - `predict(event: dict) -> dict` for `prophet forecast predict --local agent.predict`
  - FastAPI `app` with POST /predict for `--agent-url http://host:port/predict`

Return shape: {"p_yes": float in [0.01, 0.99], "rationale": str}.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field


class EventRequest(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str


class PredictionResponse(BaseModel):
    p_yes: float = Field(ge=0.01, le=0.99)
    rationale: str


def _forecast(event: EventRequest) -> PredictionResponse:
    # TODO(hackathon): replace stub with real forecasting logic.
    return PredictionResponse(p_yes=0.5, rationale="stub: uniform prior")


def predict(event: dict) -> dict:
    resp = _forecast(EventRequest(**event))
    return {"p_yes": resp.p_yes, "rationale": resp.rationale}


app = FastAPI(title="Prophet Hacks Forecast Agent")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    return _forecast(event)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
