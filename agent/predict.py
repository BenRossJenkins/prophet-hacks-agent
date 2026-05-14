"""Forecasting agent entrypoint.

v2.1 — market-anchored with calibration insurance:

  1. Look up the Kalshi market by ticker.
  2. Check no-arb: |yes_ask + no_ask − 1| must be small, else the book is broken.
  3. If liquid (vol + tight spread + no-arb OK), use depth-weighted midprice.
  4. Else fall back to last trade price; else 0.5.
  5. Volume-weighted shrinkage toward 0.5 (more volume → trust market more).
  6. Stale-book detection: Kalshi's `updated_time` tracks metadata changes,
     not book activity (active markets routinely show 4+ day old `updated_time`),
     so we don't penalize on it. Helper kept in place pending a better signal.
  7. Clamp to [0.01, 0.99] per submission contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.kalshi import get_market
from agent.llm import llm_forecast, llm_forecast_ensemble
from agent.priors import category_prior, llm_allowed_for


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


# Liquidity gates
MIN_VOL_24H = 50.0    # USD — below this the book is noise
MAX_SPREAD = 0.20     # dollars — wider and the midprice is uninformative
NO_ARB_TOL = 0.02     # |yes_ask + no_ask − 1| > this → book broken

# Shrinkage curve: alpha = scale / (vol + scale), clipped to [MIN, MAX]
MIN_SHRINK_ALPHA = 0.005
MAX_SHRINK_ALPHA = 0.10
ALPHA_VOL_SCALE = 200.0

# Staleness — currently dormant; see _staleness_hours docstring.
STALE_HOURS = 6.0
STALE_ALPHA_MULTIPLIER = 2.0
APPLY_STALENESS = False

# LLM-output shrinkage. Two tiers based on whether the rationale suggests
# the model did real research vs speculating from base rates.
LLM_SHRINK_GROUNDED = 0.05
LLM_SHRINK_SPECULATIVE = 0.15
_LLM_GROUNDED_MARKERS = (
    "search",
    "according to",
    "as of",
    "source",
    "report",
    "data shows",
    "polls",
    "polling",
)


def _f(market: dict, key: str) -> float:
    try:
        return float(market.get(key, "0") or 0)
    except (ValueError, TypeError):
        return 0.0


def _depth_weighted_mid(market: dict) -> float | None:
    """Midprice weighted by inverse depth.

    More demand on the bid (large bid_size) shifts the true price toward the
    ask, since liquidity is biased to one side. Returns None when sizes aren't
    informative.
    """
    bid = _f(market, "yes_bid_dollars")
    ask = _f(market, "yes_ask_dollars")
    bid_size = _f(market, "yes_bid_size_fp")
    ask_size = _f(market, "yes_ask_size_fp")
    total = bid_size + ask_size
    if total <= 0 or bid <= 0 or ask <= 0:
        return None
    return (ask_size * bid + bid_size * ask) / total


def _no_arb_violated(market: dict) -> bool:
    """True if the book offers free money — usually means it's stale/broken."""
    yes_ask = _f(market, "yes_ask_dollars")
    no_ask = _f(market, "no_ask_dollars")
    yes_bid = _f(market, "yes_bid_dollars")
    no_bid = _f(market, "no_bid_dollars")

    # Buying both sides for < $1 is a guaranteed profit, can't be real.
    if yes_ask > 0 and no_ask > 0 and yes_ask + no_ask < 1.0 - NO_ARB_TOL:
        return True
    # Selling both sides for > $1 is also arbitrage.
    if yes_bid > 0 and no_bid > 0 and yes_bid + no_bid > 1.0 + NO_ARB_TOL:
        return True
    return False


def _staleness_hours(market: dict) -> float | None:
    """Hours since the market's `updated_time` field.

    Note: empirically, Kalshi's `updated_time` reflects metadata changes,
    not book/order activity — an active market with $300k of 24h volume
    can still report `updated_time` 4 days ago. So this value is not a
    reliable book-staleness signal on its own. Helper is kept here for
    future use (e.g., combined with last_price/previous_price comparison)
    but is not currently used in the forecast path (see APPLY_STALENESS).

    Returns hours since `updated_time`, or None if missing/unparsable.
    """
    updated = market.get("updated_time", "")
    if not updated:
        return None
    try:
        ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (datetime.now(UTC) - ts).total_seconds() / 3600.0


def _shrink_alpha(vol_24h: float) -> float:
    """Shrinkage weight as a function of 24h volume. More vol → trust market more."""
    raw = ALPHA_VOL_SCALE / (vol_24h + ALPHA_VOL_SCALE)
    return max(MIN_SHRINK_ALPHA, min(MAX_SHRINK_ALPHA, raw))


def _shrink_and_clamp(p: float, alpha: float) -> float:
    p = (1 - alpha) * p + alpha * 0.5
    return max(0.01, min(0.99, p))


def _market_implied_prob(
    market: dict, *, arb_violated: bool
) -> tuple[float | None, str]:
    """Derive a probability from a Kalshi market dict.

    Returns (p, rationale). p is None when no usable signal exists.
    """
    yes_bid = _f(market, "yes_bid_dollars")
    yes_ask = _f(market, "yes_ask_dollars")
    last = _f(market, "last_price_dollars")
    vol = _f(market, "volume_24h_fp")

    book_usable = (
        not arb_violated
        and yes_bid > 0
        and yes_ask > 0
        and (yes_ask - yes_bid) <= MAX_SPREAD
        and vol >= MIN_VOL_24H
    )
    if book_usable:
        p = _depth_weighted_mid(market) or (yes_bid + yes_ask) / 2
        return p, (
            f"depth-mid {p:.3f} (bid={yes_bid:.3f}/ask={yes_ask:.3f}, vol24h=${vol:.0f})"
        )

    if last > 0:
        note = "no-arb violation" if arb_violated else "illiquid book"
        return last, (
            f"last trade {last:.3f} ({note}: bid={yes_bid:.3f}/ask={yes_ask:.3f}, "
            f"vol24h=${vol:.0f})"
        )
    return None, (
        f"no price signal (bid={yes_bid:.3f}/ask={yes_ask:.3f}/last={last:.3f}, "
        f"vol24h=${vol:.0f})"
    )


def _llm_shrink_alpha(rationale: str) -> float:
    """Pick LLM shrinkage strength based on whether it appears data-grounded."""
    rationale_lower = rationale.lower()
    if any(marker in rationale_lower for marker in _LLM_GROUNDED_MARKERS):
        return LLM_SHRINK_GROUNDED
    return LLM_SHRINK_SPECULATIVE


def _llm_fallback(event: EventRequest, *, reason: str) -> PredictionResponse:
    """Reach for an alternative when the market gives us no usable price.

    Order of preference:
      1. Category-specific external-data prior (Phase 4).
      2. LLM forecast, if the category isn't on the LLM denylist. The LLM
         output is shrunk toward 0.5 — more if the rationale doesn't show
         signs of having grounded the answer in current data.
      3. Uniform 0.5 prior.
    """
    event_d = event.model_dump()
    prior_out = category_prior(event_d)
    if prior_out is not None:
        p, prior_rationale = prior_out
        p = max(0.01, min(0.99, p))
        return PredictionResponse(p_yes=p, rationale=f"{reason}; prior: {prior_rationale}")

    if not llm_allowed_for(event.category):
        return PredictionResponse(
            p_yes=0.5,
            rationale=f"{reason}; LLM gated for category='{event.category}'; uniform prior",
        )

    # Ensemble over multiple Claude variants; gracefully degrades to a single
    # call if some members fail.
    out = llm_forecast_ensemble(event_d)
    if out is None:
        return PredictionResponse(p_yes=0.5, rationale=f"{reason}; LLM unavailable; uniform prior")
    p_raw, llm_rationale = out
    alpha = _llm_shrink_alpha(llm_rationale)
    p = _shrink_and_clamp(p_raw, alpha=alpha)
    tier = "grounded" if alpha == LLM_SHRINK_GROUNDED else "speculative"
    return PredictionResponse(
        p_yes=p,
        rationale=f"{reason}; LLM ({tier}, α={alpha}, raw={p_raw:.3f}): {llm_rationale}",
    )


def _forecast(event: EventRequest) -> PredictionResponse:
    market = get_market(event.market_ticker)
    if market is None:
        return _llm_fallback(event, reason=f"kalshi fetch failed for {event.market_ticker}")

    arb_violated = _no_arb_violated(market)
    raw_p, rationale = _market_implied_prob(market, arb_violated=arb_violated)
    if raw_p is None:
        return _llm_fallback(event, reason=rationale)

    vol_24h = _f(market, "volume_24h_fp")
    alpha = _shrink_alpha(vol_24h)

    if APPLY_STALENESS:
        age_h = _staleness_hours(market)
        if age_h is not None and age_h > STALE_HOURS:
            alpha = min(MAX_SHRINK_ALPHA, alpha * STALE_ALPHA_MULTIPLIER)
            rationale += f"; stale book {age_h:.1f}h (α ×{STALE_ALPHA_MULTIPLIER:g})"

    p = _shrink_and_clamp(raw_p, alpha=alpha)
    return PredictionResponse(
        p_yes=p,
        rationale=f"{rationale}; shrunk α={alpha:.3f} → {p:.3f}",
    )


def predict(event: dict) -> dict:
    resp = _forecast(EventRequest(**event))
    return {"p_yes": resp.p_yes, "rationale": resp.rationale}


app = FastAPI(title="Prophet Hacks Forecast Agent")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    return _forecast(event)


@app.post("/trade")
async def trade_endpoint(event: EventRequest) -> dict:
    from agent.trading import decide

    return decide(event.model_dump()).to_dict()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
