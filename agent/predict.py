"""Forecasting agent entrypoint.

v1 strategy — market-anchor:
  1. Look up the Kalshi market by ticker.
  2. If liquid (volume + tight spread), use mid yes_bid/yes_ask as p_yes.
  3. Else fall back to last trade price, then to 0.5.
  4. Shrink toward 0.5 by `SHRINK_ALPHA` as calibration insurance.
  5. Clamp to [0.01, 0.99] per submission contract.

Two interchangeable surfaces:
  - `predict(event: dict) -> dict` for `prophet forecast predict --local agent.predict`
  - FastAPI `app` with POST /predict for `--agent-url http://host:port/predict`
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.kalshi import get_market


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


SHRINK_ALPHA = 0.05  # shrinkage toward 0.5; cheap insurance against overconfident markets
MIN_VOL_24H = 50.0   # USD — below this we don't trust the order book
MAX_SPREAD = 0.20    # dollars (= probability units) — wider than this and the book is uninformative


def _f(market: dict, key: str) -> float:
    try:
        return float(market.get(key, "0") or 0)
    except (ValueError, TypeError):
        return 0.0


def _market_implied_prob(market: dict) -> tuple[float | None, str]:
    """Derive a probability from a Kalshi market dict.

    Returns (p, rationale). p is None when the market is too illiquid to inform a forecast.
    """
    yes_bid = _f(market, "yes_bid_dollars")
    yes_ask = _f(market, "yes_ask_dollars")
    last = _f(market, "last_price_dollars")
    volume_24h = _f(market, "volume_24h_fp")

    if (
        yes_bid > 0
        and yes_ask > 0
        and (yes_ask - yes_bid) <= MAX_SPREAD
        and volume_24h >= MIN_VOL_24H
    ):
        p = (yes_bid + yes_ask) / 2
        return p, (
            f"midprice {p:.3f} (bid={yes_bid:.3f}/ask={yes_ask:.3f}, vol24h=${volume_24h:.0f})"
        )
    if last > 0:
        return last, (
            f"last trade {last:.3f} (illiquid book: bid={yes_bid:.3f}/ask={yes_ask:.3f}, "
            f"vol24h=${volume_24h:.0f})"
        )
    return None, (
        f"no price signal (bid={yes_bid:.3f}/ask={yes_ask:.3f}/last={last:.3f}, "
        f"vol24h=${volume_24h:.0f})"
    )


def _shrink_and_clamp(p: float, alpha: float = SHRINK_ALPHA) -> float:
    p = (1 - alpha) * p + alpha * 0.5
    return max(0.01, min(0.99, p))


def _forecast(event: EventRequest) -> PredictionResponse:
    market = get_market(event.market_ticker)
    if market is None:
        return PredictionResponse(
            p_yes=0.5,
            rationale=f"kalshi fetch failed for {event.market_ticker}; uniform prior",
        )

    raw_p, rationale = _market_implied_prob(market)
    if raw_p is None:
        return PredictionResponse(p_yes=0.5, rationale=f"{rationale}; uniform prior")

    p = _shrink_and_clamp(raw_p)
    return PredictionResponse(
        p_yes=p, rationale=f"{rationale}; shrunk α={SHRINK_ALPHA} → {p:.3f}"
    )


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
