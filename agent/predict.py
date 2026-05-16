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

import logging
import re
from datetime import UTC, datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from agent.calibrate import apply_calibration, get_calibration_table
from agent.kalshi import get_market
from agent.llm import llm_forecast, llm_forecast_ensemble, llm_forecast_ensemble_full
from agent.polymarket import polymarket_quote
from agent.prediction_log import log_prediction
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
    outcomes: list[str] | None = None
    resolved_outcome: dict | None = None


class MarketProbability(BaseModel):
    market: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    p_yes: float = Field(ge=0.01, le=0.99)
    probabilities: list[MarketProbability] | None = None
    rationale: str


# Liquidity gates
MIN_VOL_24H = 10.0    # USD — below this the book is noise.
                      # Lowered from 50 → 10 after parameter sweep
                      # (scripts/sweep_params.py) showed -0.015 Brier on
                      # the candlestick fixture (2026-05-14): markets with
                      # $10-50 24h vol still carry useful price signal.
MAX_SPREAD = 0.50     # dollars — wider and the midprice is uninformative.
                      # Sweep on the diversified 267-entry fixture preferred
                      # 0.50 over 0.20 (~0.001 Brier). Wider tolerance lets
                      # more mid-tier markets use depth-mid; volume-weighted
                      # shrinkage already handles the inflated uncertainty.
NO_ARB_TOL = 0.02     # |yes_ask + no_ask − 1| > this → book broken

# Shrinkage curve: alpha = scale / (vol + scale), clipped to [MIN, MAX].
# MAX dropped 0.10 → 0.05 after sweep (marginal but consistent gain;
# protects high-confidence markets from over-pull toward 0.5).
MIN_SHRINK_ALPHA = 0.005
MAX_SHRINK_ALPHA = 0.05
ALPHA_VOL_SCALE = 200.0

# Staleness — currently dormant; see _staleness_hours docstring.
STALE_HOURS = 6.0
STALE_ALPHA_MULTIPLIER = 2.0
APPLY_STALENESS = False

# LLM-output shrinkage. Two tiers based on whether the rationale suggests
# the model did real research vs speculating from base rates.
LLM_SHRINK_GROUNDED = 0.05
LLM_SHRINK_SPECULATIVE = 0.15

# Polymarket cross-reference. Categories where a Polymarket sibling is most
# likely to exist + add signal. Weather/Crypto/Sports are excluded: Weather
# has no Poly equivalent, Crypto has its own quantitative prior, and Sports
# uses pre-game odds (added in the sports prior). Politics is the headline
# use-case.
POLYMARKET_CATEGORIES = frozenset(
    {
        "Politics",
        "Elections",
        "World",
        "Companies",
        "Financials",
        "Entertainment",
        "Economics",
    }
)
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


def _is_multi_outcome(event: EventRequest) -> bool:
    """True if the event has 3+ outcomes (e.g. Eurovision, award nominees)."""
    return event.outcomes is not None and len(event.outcomes) > 2


_TOP_K_PATTERN = re.compile(r"top\s+(\d+)|finish.+top\s+(\d+)", re.IGNORECASE)


def _estimate_winners_count(event: EventRequest) -> int:
    """How many positive outcomes are expected, given the question phrasing.

    Returns 1 by default (single-winner: "Who will win X?"). Returns K when
    the title clearly says "top K". For ordinal/bucket questions (e.g.
    "At least N million views") the resolution rule picks exactly one
    bucket, so K=1 there too.
    """
    title = event.title or ""
    rules = event.rules or ""
    m = _TOP_K_PATTERN.search(title) or _TOP_K_PATTERN.search(rules)
    if m:
        for group in m.groups():
            if group:
                try:
                    k = int(group)
                except ValueError:
                    continue
                if 1 <= k <= 20:
                    return k
    return 1


def _uniform_prior(event: EventRequest) -> float:
    """k/N uniform prior for outcomes[0] — assumes each outcome equally likely."""
    if not event.outcomes:
        return 0.5
    n = len(event.outcomes)
    if n <= 0:
        return 0.5
    k = _estimate_winners_count(event)
    p = k / n
    return max(0.01, min(0.99, p))


# Multi-outcome LLM shrinkage: we pull aggressively toward the uniform prior
# because (a) overconfidence costs more across many options, (b) LLMs are
# generally less reliable when answering "which of these 35 things" than
# binary yes/no. α=0.30 means a confident LLM 0.90 against a 0.143 uniform
# prior gets shrunk to 0.90*0.70 + 0.143*0.30 = 0.673. Still meaningful, but
# not catastrophic if wrong.
MULTI_LLM_SHRINK = 0.30


# Tail-market triage: when Kalshi is confidently at one tail AND backed by
# enough volume to trust, return the market price directly. Skip Polymarket
# blend, skip LLM, skip shrinkage.
#
# Rationale (per Prophet Arena paper + author write-up): tail markets have
# already aggregated the consensus information. LLM disagreement at the
# tails almost always hurts Brier — the squared-error penalty for being
# wrong at 0.05 vs the truth at 1.0 is brutal. Also a major cost savings.
TAIL_LOW = 0.05
TAIL_HIGH = 0.95
TAIL_MIN_VOL_24H = 500.0  # USD — high enough that the price reflects real consensus,
                          # not a $5 thin-book accident.


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


def _kalshi_volume_weight(market: dict | None) -> float:
    """Depth proxy for Kalshi side of the cross-market blend (24h volume in USD)."""
    if market is None:
        return 0.0
    return _f(market, "volume_24h_fp")


def _blend_with_polymarket(
    event: EventRequest,
    kalshi_market: dict | None,
    kalshi_p: float | None,
    kalshi_rationale: str,
) -> tuple[float | None, str]:
    """If a Polymarket sibling exists, fold it into the price.

    Three cases:
      - Both books have signal: volume-weighted average of the two prices.
      - Only Polymarket has signal (Kalshi illiquid/missing): use Poly price.
      - Neither: pass kalshi_p (which may itself be None) through unchanged.

    Returns (p_or_none, rationale). When category isn't on the Polymarket
    allowlist or no usable Poly match exists, this is a pass-through.
    """
    if event.category not in POLYMARKET_CATEGORIES:
        return kalshi_p, kalshi_rationale

    try:
        poly = polymarket_quote(event.model_dump())
    except Exception as e:  # poly is best-effort, never block on its failures
        logger.warning("polymarket lookup raised: %s", e)
        poly = None

    if poly is None:
        return kalshi_p, kalshi_rationale

    poly_p, poly_weight, poly_rationale = poly

    if kalshi_p is None:
        return poly_p, f"polymarket-only ({poly_rationale}); kalshi: {kalshi_rationale}"

    kalshi_weight = _kalshi_volume_weight(kalshi_market)
    total_weight = kalshi_weight + poly_weight
    if total_weight <= 0:
        # Both books exist but neither has measurable depth — straight average.
        blended = (kalshi_p + poly_p) / 2
        weight_note = "equal-weight"
    else:
        blended = (kalshi_p * kalshi_weight + poly_p * poly_weight) / total_weight
        weight_note = (
            f"vol-weighted (kalshi=${kalshi_weight:.0f} poly=${poly_weight:.0f})"
        )
    return blended, (
        f"blend {blended:.3f} {weight_note}; kalshi: {kalshi_rationale}; {poly_rationale}"
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


def _multi_outcome_forecast(event: EventRequest) -> PredictionResponse:
    """Forecast for events with 3+ outcomes.

    The market-anchor / sportsbook / manifold tiers are all designed for 2-
    outcome binary questions, so they're skipped entirely. Go straight to
    the LLM ensemble with explicit framing ('p_yes is P(outcomes[0] in
    resolved set)'), aggregate per-outcome probabilities across vendors,
    and shrink p_yes toward the k/N uniform prior.
    """
    event_d = event.model_dump()
    prior = _uniform_prior(event)
    n_out = len(event.outcomes or [])
    k = _estimate_winners_count(event)

    out = llm_forecast_ensemble_full(event_d)
    if out is None:
        return PredictionResponse(
            p_yes=max(0.01, min(0.99, prior)),
            rationale=(
                f"multi-outcome ({n_out} options, top-{k}); LLM unavailable; "
                f"uniform prior {prior:.3f}"
            ),
        )
    raw_p, probabilities, llm_rationale = out

    # Shrink toward the uniform prior, not toward 0.5 — for a 35-option
    # question, 0.5 is itself an absurd answer.
    shrunk = (1 - MULTI_LLM_SHRINK) * raw_p + MULTI_LLM_SHRINK * prior
    shrunk = max(0.01, min(0.99, shrunk))

    probs_payload: list[MarketProbability] | None = None
    if probabilities:
        # Clamp each per-outcome probability to [0, 1] but DON'T renormalize:
        # top-K questions have probabilities summing to ~K, not 1.
        probs_payload = [
            MarketProbability(
                market=p["market"], probability=max(0.0, min(1.0, p["probability"]))
            )
            for p in probabilities
        ]

    return PredictionResponse(
        p_yes=shrunk,
        probabilities=probs_payload,
        rationale=(
            f"multi-outcome ({n_out} options, top-{k}, uniform={prior:.3f}); "
            f"raw={raw_p:.3f} α={MULTI_LLM_SHRINK} → {shrunk:.3f}; {llm_rationale}"
        ),
    )


def _forecast(event: EventRequest) -> PredictionResponse:
    if _is_multi_outcome(event):
        return _multi_outcome_forecast(event)

    market = get_market(event.market_ticker)

    if market is None:
        # Kalshi fetch failed entirely. Try Polymarket as an alternative
        # market-anchor before reaching for priors / LLM.
        poly_p, poly_rationale = _blend_with_polymarket(
            event, None, None, f"kalshi fetch failed for {event.market_ticker}"
        )
        if poly_p is not None:
            p = _shrink_and_clamp(poly_p, alpha=MIN_SHRINK_ALPHA)
            return PredictionResponse(p_yes=p, rationale=poly_rationale)
        return _llm_fallback(event, reason=f"kalshi fetch failed for {event.market_ticker}")

    arb_violated = _no_arb_violated(market)
    raw_p, rationale = _market_implied_prob(market, arb_violated=arb_violated)
    vol_24h = _f(market, "volume_24h_fp")

    # Tail-market triage: a confident high-volume Kalshi price already
    # reflects market consensus. LLM disagreement here almost always hurts
    # Brier. Skip everything downstream and return the price directly.
    if (
        raw_p is not None
        and vol_24h >= TAIL_MIN_VOL_24H
        and (raw_p < TAIL_LOW or raw_p > TAIL_HIGH)
    ):
        p = max(0.01, min(0.99, raw_p))
        return PredictionResponse(
            p_yes=p,
            rationale=f"tail-anchor {p:.3f} (vol24h=${vol_24h:.0f}); {rationale}",
        )

    # Cross-reference Polymarket when category is on the allowlist.
    raw_p, rationale = _blend_with_polymarket(event, market, raw_p, rationale)

    if raw_p is None:
        return _llm_fallback(event, reason=rationale)

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


def _maybe_calibrate(resp: PredictionResponse) -> PredictionResponse:
    """If a calibration table is present on disk, apply it. Never raises."""
    try:
        table = get_calibration_table()
    except Exception:
        return resp
    if not table:
        return resp
    raw = resp.p_yes
    adjusted = apply_calibration(raw, table)
    if adjusted == raw:
        return resp
    return PredictionResponse(
        p_yes=max(0.01, min(0.99, adjusted)),
        rationale=f"{resp.rationale}; calibrated {raw:.3f}→{adjusted:.3f}",
    )


def predict(event: dict) -> dict:
    resp = _forecast(EventRequest(**event))
    resp = _maybe_calibrate(resp)
    log_prediction(event, resp.p_yes, resp.rationale)
    return {"p_yes": resp.p_yes, "rationale": resp.rationale}


app = FastAPI(title="Prophet Hacks Forecast Agent")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    resp = _forecast(event)
    resp = _maybe_calibrate(resp)
    log_prediction(event.model_dump(), resp.p_yes, resp.rationale)
    return resp


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
