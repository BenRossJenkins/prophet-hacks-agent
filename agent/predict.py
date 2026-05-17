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

from agent.calibrate import (
    apply_calibration,
    apply_calibration_data,
    get_calibration_data,
    get_calibration_table,
)
from agent.kalshi import get_market, kalshi_event_distribution
from agent.llm import llm_forecast, llm_forecast_ensemble, llm_forecast_ensemble_full
from agent.polymarket import polymarket_event_distribution, polymarket_quote
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
    """Server-facing response.

    Per the Prophet Arena developer docs (2026-05-16): every prediction must
    return `probabilities` as a list of {market, probability} entries that
    cover every event outcome and sum to 1. The `p_yes` and `rationale`
    fields are retained for internal use (calibration fitting, logging,
    backwards compat with older CLI versions) but are NOT the contract.
    """

    probabilities: list[MarketProbability]
    # Legacy convenience field — not required by the server, but the older
    # ai-prophet CLI falls back to it and our calibration / log machinery
    # still keys off a single binary probability for binary events.
    p_yes: float | None = Field(default=None, ge=0.01, le=0.99)
    rationale: str | None = None


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

# LLM-output shrinkage. Three tiers based on what the rationale signals.
#
#  - DECISIVE: web search uncovered concrete outcome evidence (the event
#    has already resolved, or a winner has been determined). Hedging the
#    LLM here is *worse* than trusting it; the squared-error penalty for
#    pulling a confidently-correct 0.02 up to 0.16 dominates any
#    protective benefit. Used very sparingly (requires explicit markers).
#  - GROUNDED: web-search-grounded rationale citing current data, no
#    decisive outcome signal yet.
#  - SPECULATIVE: rationale reads as base-rate speculation. Most
#    aggressive shrinkage applied.
LLM_SHRINK_DECISIVE = 0.02
LLM_SHRINK_GROUNDED = 0.05
LLM_SHRINK_SPECULATIVE = 0.15

# Non-linear tail boost. LLMs are reliably overconfident in the deep tails
# (p < 0.10 or p > 0.90). Beyond the fixed base α, every unit of distance
# from 0.5 past LLM_SHRINK_TAIL_THRESHOLD adds extra α. Tuned so that:
#   p=0.95 grounded   → α≈0.15 → final ≈ 0.88
#   p=0.99 grounded   → α≈0.23 → final ≈ 0.88
#   p=0.95 speculative → α≈0.25 → final ≈ 0.84
# Capped so we never push past 0.75/0.25 (which would flip the LLM's
# directional signal entirely).
LLM_SHRINK_TAIL_THRESHOLD = 0.40
LLM_SHRINK_TAIL_SLOPE = 2.0
LLM_SHRINK_MAX_ALPHA = 0.50

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

# Decisive-evidence markers. When the LLM rationale contains phrases that
# describe an OUTCOME — not just a forecast or a base rate — we're in the
# case where the event has effectively already resolved (game played,
# winner declared, team eliminated, etc.). At that point our usual
# protective shrinkage HURTS Brier: pulling 0.02 → 0.16 on a question
# that legitimately answers 0 costs us 0.04 per event vs ~0.001 if
# trusted. Requires fairly explicit phrasing to avoid false positives
# from speculative rationales that mention these words in other contexts.
_LLM_DECISIVE_MARKERS = (
    "already won",
    "already lost",
    "already happened",
    "already eliminated",
    "already clinched",
    "already advanced",
    "already finished",
    "already concluded",
    "already resolved",
    "did not win",
    "did not advance",
    "did not make",
    "was eliminated",
    "were eliminated",
    "has been eliminated",
    "have been eliminated",
    "is now confirmed",
    "has been confirmed",
    "winner is",
    "winner was",
    "outcome is",
    "outcome was",
    "has concluded",
    "game has been played",
    "match has been played",
    "series is over",
    "series has ended",
    "is impossible",
    "cannot win",
    "cannot happen",
    "is mathematically eliminated",
)


def _default_outcomes(event: EventRequest) -> list[str]:
    """Fall back to ['Yes', 'No'] when an event has no outcomes list.

    Per the developer docs, every Event should arrive with an `outcomes`
    field. This is a defensive fallback for malformed inputs / legacy
    callers; do not rely on it.
    """
    if event.outcomes and len(event.outcomes) >= 2:
        return list(event.outcomes)
    return ["Yes", "No"]


def _binary_distribution(p: float, outcomes: list[str]) -> list[MarketProbability]:
    """Convert P(outcomes[0]) → 2-outcome distribution summing to 1."""
    p = max(0.0, min(1.0, p))
    return [
        MarketProbability(market=outcomes[0], probability=p),
        MarketProbability(market=outcomes[1], probability=1.0 - p),
    ]


def _normalize_distribution(
    probs: list[dict] | list[MarketProbability], outcomes: list[str]
) -> list[MarketProbability]:
    """Align a probability list to `outcomes` order, fill missing, normalize to 1.0.

    Ensures every event outcome has an entry, missing ones get the
    uniform residual, then renormalize so probabilities sum to 1.0.
    Critical: server rejects (or rather mis-scores) outputs whose
    markets don't match event outcomes exactly.
    """
    by_market: dict[str, float] = {}
    for entry in probs:
        if isinstance(entry, MarketProbability):
            market, prob = entry.market, entry.probability
        else:
            market = str(entry.get("market", ""))
            try:
                prob = float(entry.get("probability", 0.0))
            except (ValueError, TypeError):
                continue
        if market in outcomes:
            by_market[market] = max(0.0, prob)

    missing = [o for o in outcomes if o not in by_market]
    if missing:
        covered_sum = sum(by_market.values())
        residual = max(0.0, 1.0 - covered_sum)
        per_missing = residual / len(missing) if missing else 0.0
        for m in missing:
            by_market[m] = per_missing

    total = sum(by_market.get(o, 0.0) for o in outcomes)
    if total <= 0:
        # All zero — uniform fallback.
        per = 1.0 / len(outcomes)
        return [MarketProbability(market=o, probability=per) for o in outcomes]
    return [
        MarketProbability(market=o, probability=by_market[o] / total)
        for o in outcomes
    ]


def _wrap_binary(
    p: float, rationale: str, event: EventRequest
) -> PredictionResponse:
    """Build the response from an internal P(outcomes[0])."""
    p_clamped = max(0.01, min(0.99, p))
    outs = _default_outcomes(event)
    return PredictionResponse(
        probabilities=_binary_distribution(p_clamped, outs),
        p_yes=p_clamped,
        rationale=rationale,
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

# Capped volume-weighted blend for multi-outcome events with both Kalshi and
# Polymarket distributions. Caps the share of either venue at this fraction
# so a single very-liquid venue (typically Kalshi for US-resolved questions,
# or Polymarket for politics) doesn't dominate while ignoring the other's
# information.
KALSHI_POLY_MAX_WEIGHT = 0.75


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
# Tiny shrinkage applied on the tail-anchor return path. Markets are usually
# slightly overconfident at the extremes; a 3% pull toward 0.5 is free Brier
# protection when the market is wrong, and costs almost nothing when it's
# right (0.97 vs 0.95 contribution to per-event squared error is negligible).
TAIL_ANCHOR_SHRINK = 0.03

# Cross-venue agreement gate. When Kalshi and Polymarket agree closely
# we skip the blend (the venue cross-reference adds no signal); when they
# disagree by more than CROSS_VENUE_DISAGREE_TOL we blend (disagreement
# carries information). This refines the older "safe-band skip" logic
# (which skipped Poly blindly in [0.20, 0.80] regardless of agreement)
# — disagreement in the mid-band is exactly where the cross-venue signal
# is most informative.
SAFE_BAND_LOW = 0.20
SAFE_BAND_HIGH = 0.80
SAFE_BAND_MIN_VOL_24H = 10_000  # USD — only consider skip when Kalshi is liquid
CROSS_VENUE_DISAGREE_TOL = 0.03   # |kalshi - poly|; below this → skip the blend

# Market-deviation sanity guardrail. When our final prediction differs
# materially from a deep liquid Kalshi book, one of us is wrong — and at
# this volume threshold the asymmetric Brier penalty (a confident
# disagree-and-be-wrong costs 4-9x staying near market) makes anchoring
# harder almost always the safer bet. This is a guardrail, not a clamp:
# we blend back toward market, we don't replace.
GUARDRAIL_DEVIATION = 0.30      # |our_p − kalshi_mid| above this triggers anchoring
GUARDRAIL_MIN_VOL_24H = 100_000  # USD — deep, calibrated, well-informed books only
GUARDRAIL_ANCHOR_WEIGHT = 0.60   # how much weight market gets in the blend-back


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
    *,
    skip_if_agree_within: float | None = None,
) -> tuple[float | None, str]:
    """If a Polymarket sibling exists, fold it into the price.

    Three cases:
      - Both books have signal: volume-weighted average of the two prices.
      - Only Polymarket has signal (Kalshi illiquid/missing): use Poly price.
      - Neither: pass kalshi_p (which may itself be None) through unchanged.

    When `skip_if_agree_within` is set, AND both books have a price, AND the
    two prices agree to within that tolerance, return the Kalshi price
    unchanged. This is the "safe-band agreement gate" — when the venues
    agree there's no signal to extract, skip the blend.

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

    # Cross-venue agreement check: when both venues quote the same answer,
    # there's no information in their agreement — skip the blend.
    if skip_if_agree_within is not None and abs(kalshi_p - poly_p) <= skip_if_agree_within:
        return kalshi_p, (
            f"{kalshi_rationale}; poly agrees ({poly_p:.3f}, |Δ|≤{skip_if_agree_within:.2f}) → skip blend"
        )

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


def _market_sanity_check(
    resp: PredictionResponse, market: dict | None, event: EventRequest
) -> PredictionResponse:
    """Anchor `resp.p_yes` back toward Kalshi mid when deviation is large.

    Only triggers on:
      - Binary events with a known outcomes list and a non-None resp.p_yes
      - A Kalshi market fetched successfully with vol_24h >= GUARDRAIL_MIN_VOL_24H
      - A usable bid+ask midpoint
      - |resp.p_yes − market_mid| > GUARDRAIL_DEVIATION

    Doesn't fire on multi-outcome paths or low-volume markets.
    """
    if market is None or resp.p_yes is None or _is_multi_outcome(event):
        return resp

    vol_24h = _f(market, "volume_24h_fp")
    if vol_24h < GUARDRAIL_MIN_VOL_24H:
        return resp

    yes_bid = _f(market, "yes_bid_dollars")
    yes_ask = _f(market, "yes_ask_dollars")
    if yes_bid <= 0 or yes_ask <= 0:
        return resp
    market_mid = (yes_bid + yes_ask) / 2

    deviation = abs(resp.p_yes - market_mid)
    if deviation <= GUARDRAIL_DEVIATION:
        return resp

    anchored = (
        GUARDRAIL_ANCHOR_WEIGHT * market_mid
        + (1 - GUARDRAIL_ANCHOR_WEIGHT) * resp.p_yes
    )
    anchored = max(0.01, min(0.99, anchored))
    note = (
        f"; guardrail anchored {resp.p_yes:.3f}→{anchored:.3f} "
        f"(kalshi mid={market_mid:.3f}, vol24h=${vol_24h:.0f}, dev={deviation:.2f})"
    )
    logger.warning(
        "guardrail: %s deviated %.2f from Kalshi mid (%.3f→%.3f, vol=%s)",
        event.market_ticker,
        deviation,
        resp.p_yes,
        anchored,
        vol_24h,
    )
    return _wrap_binary(anchored, (resp.rationale or "") + note, event)


def _llm_shrink_alpha(rationale: str) -> float:
    """Pick LLM shrinkage strength based on rationale content.

    Three tiers, checked in order of decreasing specificity:
      DECISIVE  → rationale describes an outcome that has already
                  occurred or been determined (game played, winner
                  declared, contender eliminated, etc.). Trust the LLM.
      GROUNDED  → rationale cites current data ("according to", "as of",
                  "polls show", etc.) but no decisive outcome.
      SPECULATIVE → rationale reads as base-rate speculation.
    """
    rationale_lower = rationale.lower()
    if any(marker in rationale_lower for marker in _LLM_DECISIVE_MARKERS):
        return LLM_SHRINK_DECISIVE
    if any(marker in rationale_lower for marker in _LLM_GROUNDED_MARKERS):
        return LLM_SHRINK_GROUNDED
    return LLM_SHRINK_SPECULATIVE


def _llm_shrink_with_tail(p: float, alpha_base: float) -> float:
    """Shrink LLM output toward 0.5 with extra penalty in the deep tails.

    For p in the central range (distance from 0.5 ≤ THRESHOLD), uses
    alpha_base linearly. Beyond the threshold, adds extra alpha
    proportional to how far past the threshold the prediction sits.
    Caps at LLM_SHRINK_MAX_ALPHA to preserve the LLM's directional
    signal even at the extreme tails.
    """
    p = max(0.0, min(1.0, p))
    distance = abs(p - 0.5)
    extra = 0.0
    if distance > LLM_SHRINK_TAIL_THRESHOLD:
        extra = (distance - LLM_SHRINK_TAIL_THRESHOLD) * LLM_SHRINK_TAIL_SLOPE
    alpha = min(LLM_SHRINK_MAX_ALPHA, alpha_base + extra)
    shrunk = (1 - alpha) * p + alpha * 0.5
    return max(0.01, min(0.99, shrunk))


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
        return _wrap_binary(p, f"{reason}; prior: {prior_rationale}", event)

    if not llm_allowed_for(event.category):
        return _wrap_binary(
            0.5,
            f"{reason}; LLM gated for category='{event.category}'; uniform prior",
            event,
        )

    # Ensemble over multiple Claude variants; gracefully degrades to a single
    # call if some members fail. If the full ensemble returns None (all
    # vendors failed simultaneously), retry once without web search —
    # search tools rate-limit independently of base chat completions, and
    # the eval server doesn't retry timed-out requests on our behalf.
    out = llm_forecast_ensemble(event_d)
    retry_note = ""
    if out is None:
        logger.warning("ensemble returned None; retrying without web search")
        out = llm_forecast_ensemble(event_d, with_web_search=False)
        retry_note = " (retry, no-search)"
    if out is None:
        return _wrap_binary(0.5, f"{reason}; LLM unavailable; uniform prior", event)
    p_raw, llm_rationale = out
    llm_rationale = llm_rationale + retry_note
    alpha_base = _llm_shrink_alpha(llm_rationale)
    p = _llm_shrink_with_tail(p_raw, alpha_base=alpha_base)
    tier = (
        "decisive" if alpha_base == LLM_SHRINK_DECISIVE
        else "grounded" if alpha_base == LLM_SHRINK_GROUNDED
        else "speculative"
    )
    return _wrap_binary(
        p,
        f"{reason}; LLM ({tier}, α_base={alpha_base}, raw={p_raw:.3f}→{p:.3f}): {llm_rationale}",
        event,
    )


def _blend_multi_outcome_distributions(
    kalshi_out: tuple[list[dict[str, float]], float, str],
    poly_out: tuple[list[dict[str, float]], float, str],
    outcomes: list[str],
) -> tuple[list[dict[str, float]], float, str]:
    """Capped volume-weighted blend of two multi-outcome distributions.

    Per-outcome blend; missing values on either venue pass through (the
    caller's _normalize_distribution handles the sum-to-1 contract). Cap
    enforces neither venue can exceed KALSHI_POLY_MAX_WEIGHT, so even
    when one venue's volume dwarfs the other we still incorporate the
    minority signal.
    """
    k_probs, k_vol, k_rat = kalshi_out
    p_probs, p_vol, p_rat = poly_out

    k_by = {p["market"]: p["probability"] for p in k_probs}
    p_by = {p["market"]: p["probability"] for p in p_probs}

    total_vol = max(k_vol + p_vol, 1.0)
    raw_k_weight = k_vol / total_vol
    k_weight = min(
        max(raw_k_weight, 1.0 - KALSHI_POLY_MAX_WEIGHT),
        KALSHI_POLY_MAX_WEIGHT,
    )
    p_weight = 1.0 - k_weight

    blended: list[dict[str, float]] = []
    for o in outcomes:
        k_p = k_by.get(o)
        p_p = p_by.get(o)
        if k_p is not None and p_p is not None:
            blended_p = k_weight * k_p + p_weight * p_p
        elif k_p is not None:
            blended_p = k_p
        elif p_p is not None:
            blended_p = p_p
        else:
            blended_p = 0.0
        blended.append({"market": o, "probability": blended_p})

    rationale = (
        f"blend k_w={k_weight:.2f} p_w={p_weight:.2f} "
        f"(k_vol=${k_vol:.0f} p_vol=${p_vol:.0f}); "
        f"kalshi: {k_rat}; poly: {p_rat}"
    )
    return blended, k_vol + p_vol, rationale


def _multi_outcome_forecast(event: EventRequest) -> PredictionResponse:
    """Forecast for events with 3+ outcomes.

    Order of preference:
      1. Kalshi event with mutually_exclusive=True. Canonical resolution
         venue and clean per-outcome enumeration via child markets.
      2. Polymarket multi-outcome event with sufficient coverage.
      3. Blend (1) and (2) when both available — capped volume-weighted.
      4. LLM ensemble with explicit "p_yes is P(outcomes[0])" framing
         and per-outcome distribution output.
      5. Uniform 1/N when all fail.

    Final distribution always normalized to sum=1 (server contract).
    """
    event_d = event.model_dump()
    outcomes = event.outcomes or []
    n_out = len(outcomes)
    k = _estimate_winners_count(event)
    prior = _uniform_prior(event)

    # Step 1: try Kalshi multi-outcome event.
    try:
        kalshi_out = kalshi_event_distribution(event_d)
    except Exception as e:
        logger.warning("kalshi event lookup raised: %s", e)
        kalshi_out = None

    # Step 2: try Polymarket multi-outcome event.
    try:
        poly_out = polymarket_event_distribution(event_d)
    except Exception as e:
        logger.warning("polymarket event lookup raised: %s", e)
        poly_out = None

    # Step 3: combine market signals.
    market_dist: list[dict[str, float]] | None = None
    market_rationale: str = ""
    if kalshi_out is not None and poly_out is not None:
        blended, _vol, rat = _blend_multi_outcome_distributions(
            kalshi_out, poly_out, outcomes
        )
        market_dist = blended
        market_rationale = rat
    elif kalshi_out is not None:
        market_dist = kalshi_out[0]
        market_rationale = f"kalshi only; {kalshi_out[2]}"
    elif poly_out is not None:
        market_dist = poly_out[0]
        market_rationale = f"poly only; {poly_out[2]}"

    if market_dist is not None:
        dist = _normalize_distribution(market_dist, outcomes)
        p_yes_value = next(
            (p.probability for p in dist if p.market == outcomes[0]), prior
        )
        p_yes_value = max(0.01, min(0.99, p_yes_value))
        return PredictionResponse(
            probabilities=dist,
            p_yes=p_yes_value,
            rationale=(
                f"multi-outcome ({n_out} options, top-{k}); {market_rationale}"
            ),
        )

    # Step 4: LLM ensemble. Same retry-without-search dance as the binary
    # path — if all vendors fail simultaneously (rare), search tools may
    # be the saturated dimension; base chat completions are often still
    # available.
    out = llm_forecast_ensemble_full(event_d)
    if out is None:
        logger.warning(
            "multi-outcome ensemble returned None; retrying without web search"
        )
        out = llm_forecast_ensemble_full(event_d, with_web_search=False)
    if out is None:
        # Uniform fallback. For top-K questions, every outcome is equally
        # likely, so the distribution is just 1/N across outcomes (which
        # would naively sum to 1 anyway — perfect).
        uniform = [
            MarketProbability(market=o, probability=1.0 / n_out)
            for o in outcomes
        ] if n_out > 0 else []
        return PredictionResponse(
            probabilities=uniform,
            p_yes=max(0.01, min(0.99, prior)),
            rationale=(
                f"multi-outcome ({n_out} options, top-{k}); LLM unavailable; "
                f"uniform 1/N across outcomes"
            ),
        )
    raw_p, probabilities, llm_rationale = out

    # Build the per-outcome distribution. If the LLM gave us one, normalize
    # it to sum to 1 (the LLM was instructed to sum to K for top-K, but the
    # server contract is strict sum-to-1). If no distribution was provided,
    # synthesize one from p_yes for outcomes[0] + uniform across the rest.
    if probabilities:
        dist = _normalize_distribution(probabilities, outcomes)
    else:
        # Shrink raw_p toward uniform prior k/N first (variance protection),
        # then distribute the remaining mass uniformly across other outcomes.
        shrunk = (1 - MULTI_LLM_SHRINK) * raw_p + MULTI_LLM_SHRINK * prior
        shrunk = max(0.01, min(0.99, shrunk))
        synthesized = [{"market": outcomes[0], "probability": shrunk}]
        if n_out > 1:
            per_other = (1.0 - shrunk) / (n_out - 1)
            for o in outcomes[1:]:
                synthesized.append({"market": o, "probability": per_other})
        dist = _normalize_distribution(synthesized, outcomes)

    # Surface the p_yes (probability of outcomes[0]) for logging / calibration.
    p_yes_value = next(
        (p.probability for p in dist if p.market == outcomes[0]),
        prior,
    )
    p_yes_value = max(0.01, min(0.99, p_yes_value))

    return PredictionResponse(
        probabilities=dist,
        p_yes=p_yes_value,
        rationale=(
            f"multi-outcome ({n_out} options, top-{k}, uniform={prior:.3f}); "
            f"raw={raw_p:.3f}; {llm_rationale}"
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
            return _wrap_binary(p, poly_rationale, event)
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
        # Free Brier protection: pull market a tiny bit toward 0.5.
        shrunk = (1 - TAIL_ANCHOR_SHRINK) * raw_p + TAIL_ANCHOR_SHRINK * 0.5
        p = max(0.01, min(0.99, shrunk))
        return _wrap_binary(
            p,
            f"tail-anchor {raw_p:.3f}→{p:.3f} (α={TAIL_ANCHOR_SHRINK}, vol24h=${vol_24h:.0f}); {rationale}",
            event,
        )

    # Cross-venue agreement gate: in the central band with a liquid book
    # we only blend Polymarket when it actually disagrees with Kalshi
    # (≥ CROSS_VENUE_DISAGREE_TOL). When the venues agree, the blend adds
    # noise without signal. Outside the safe band we always blend (a thin
    # Kalshi tail-price is exactly where Poly disagreement can help).
    in_safe_band = (
        raw_p is not None
        and vol_24h >= SAFE_BAND_MIN_VOL_24H
        and SAFE_BAND_LOW <= raw_p <= SAFE_BAND_HIGH
    )
    if in_safe_band:
        raw_p, rationale = _blend_with_polymarket(
            event, market, raw_p, rationale,
            skip_if_agree_within=CROSS_VENUE_DISAGREE_TOL,
        )
    else:
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
    resp = _wrap_binary(p, f"{rationale}; shrunk α={alpha:.3f} → {p:.3f}", event)
    # Sanity guardrail: if Polymarket blend or anything else pulled us far
    # from a deep liquid Kalshi book, anchor back. No-op otherwise.
    return _market_sanity_check(resp, market, event)


def _maybe_calibrate(
    resp: PredictionResponse, event: EventRequest
) -> PredictionResponse:
    """If a calibration table is present, apply path-stratified lookup.

    Path is classified from the rationale via prediction_log.classify_path.
    Per-path bucket used when its `n >= MIN_BUCKET_N_FOR_PATH`; otherwise
    falls back to the global table. Multi-outcome events skip calibration
    entirely (the binary p_yes map doesn't apply to a 35-way distribution).
    Never raises.
    """
    if resp.p_yes is None or _is_multi_outcome(event):
        return resp
    try:
        data = get_calibration_data()
    except Exception:
        return resp
    if not data:
        return resp
    from agent.prediction_log import classify_path  # lazy import

    path = classify_path(resp.rationale or "")
    raw = resp.p_yes
    adjusted = apply_calibration_data(raw, data, path=path)
    if adjusted == raw:
        return resp
    adjusted = max(0.01, min(0.99, adjusted))
    return _wrap_binary(
        adjusted,
        f"{resp.rationale}; calibrated[{path}] {raw:.3f}→{adjusted:.3f}",
        event,
    )


def predict(event: dict) -> dict:
    event_obj = EventRequest(**event)
    resp = _forecast(event_obj)
    resp = _maybe_calibrate(resp, event_obj)
    log_prediction(event, resp.p_yes, resp.rationale)
    return {
        "probabilities": [
            {"market": p.market, "probability": p.probability}
            for p in resp.probabilities
        ],
        "p_yes": resp.p_yes,
        "rationale": resp.rationale,
    }


app = FastAPI(title="Prophet Hacks Forecast Agent")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    resp = _forecast(event)
    resp = _maybe_calibrate(resp, event)
    log_prediction(event.model_dump(), resp.p_yes, resp.rationale)
    return resp


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
