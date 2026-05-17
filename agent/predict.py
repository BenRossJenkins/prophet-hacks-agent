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
    # Internal field stamped at the producing pipeline branch. Excluded
    # from the wire response — used by log_prediction → calibration
    # path-stratification so the table doesn't have to re-derive the
    # branch from free-text rationale. Re-deriving (classify_path) is
    # order-dependent and corrupts the table when rationales compose.
    path: str | None = Field(default=None, exclude=True)


# v3.17 — defensive exception handling on the LLM ensemble's sequential
# paths. Previously the search anchor and the single-model short-circuit
# both let exceptions propagate out, meaning a single Anthropic SDK
# exception (network, parse, rate-limit-not-caught-internally) would kill
# the whole ensemble — including preventing the other two vendors from
# running. Now any exception from the anchor or single-model call is
# caught and treated identically to a None return, so /predict survives
# any single-vendor failure. Builds on v3.16 (Kalshi settled-market
# coverage), v3.15 (sum-to-K), v3.14 (path-stamping).
AGENT_VERSION = "v3.17"


# Liquidity gates
MIN_VOL_24H = 10.0    # USD — below this the book is noise.
                      # Lowered from 50 → 10 after parameter sweep
                      # (scripts/sweep_params.py) showed -0.015 Brier on
                      # the candlestick fixture (2026-05-14): markets with
                      # $10-50 24h vol still carry useful price signal.
MAX_SPREAD = 0.50     # dollars — wider and the midprice is uninformative.
TIGHT_SPREAD_FOR_LOW_VOL = 0.03  # spread below this trusts the book at any volume.
                                 # Captures settled-direction markets where bid/ask
                                 # is pinned (e.g. 0.99/1.00 on a resolved-but-not-
                                 # yet-settled question) but 24h volume is zero
                                 # because nobody's bothered to trade. 3-cent
                                 # spread is "the market clearly believes this"
                                 # territory; wider needs volume to confirm.
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
    """Use the event's `outcomes` when present; else fall back to ['Yes', 'No'].

    1-outcome events (e.g. "Will X happen by date D?" with outcomes=['By D']):
    preserve the single outcome rather than appending a synthetic 'No' that
    would silently break the response contract (server requires every
    response market to match an event outcome exactly).
    """
    if event.outcomes and len(event.outcomes) >= 1:
        return list(event.outcomes)
    return ["Yes", "No"]


def _binary_distribution(p: float, outcomes: list[str]) -> list[MarketProbability]:
    """Convert P(outcomes[0]) → response distribution.

    For 2-outcome events: standard {outcomes[0]: p, outcomes[1]: 1-p}.
    For 1-outcome events: return just that single outcome with p; the
    server's score is (p - actual)² for that one label. Don't fabricate
    a complement outcome — its label wouldn't match anything in the
    event's outcomes list.
    """
    p = max(0.0, min(1.0, p))
    if len(outcomes) == 1:
        return [MarketProbability(market=outcomes[0], probability=p)]
    return [
        MarketProbability(market=outcomes[0], probability=p),
        MarketProbability(market=outcomes[1], probability=1.0 - p),
    ]


def _normalize_distribution(
    probs: list[dict] | list[MarketProbability],
    outcomes: list[str],
    *,
    target_sum: float = 1.0,
) -> list[MarketProbability]:
    """Align a probability list to `outcomes` order, fill missing, scale to target_sum.

    target_sum controls what the per-outcome probabilities sum to:
      - 1.0 (default): single-winner. Each outcome's probability is P(this
        is THE resolved outcome); they sum to 1 by definition.
      - K > 1: top-K event. Each outcome's probability is P(this outcome
        is among the K resolved outcomes); they sum to K by linearity of
        expectation. Per-outcome values remain in [0, 1].

    The wire-shape rule "each market must match one of the event's outcomes"
    is enforced here — entries with unknown markets are dropped, missing
    outcomes are backfilled with the uniform residual against target_sum.
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
        residual = max(0.0, target_sum - covered_sum)
        per_missing = residual / len(missing) if missing else 0.0
        for m in missing:
            by_market[m] = per_missing

    total = sum(by_market.get(o, 0.0) for o in outcomes)
    if total <= 0:
        # All zero — uniform fallback at target_sum/N (each outcome equally
        # likely, sum to target_sum). For K=1 this is 1/N; for K=k this is k/N.
        per = target_sum / len(outcomes) if outcomes else 0.0
        per = max(0.0, min(1.0, per))
        return [MarketProbability(market=o, probability=per) for o in outcomes]
    # Scale so the sum equals target_sum; clamp each per-outcome value to
    # [0, 1] (a probability cannot exceed 1 even when the sum is K).
    scale = target_sum / total
    return [
        MarketProbability(market=o, probability=max(0.0, min(1.0, by_market[o] * scale)))
        for o in outcomes
    ]


def _wrap_binary(
    p: float, rationale: str, event: EventRequest, *, path: str
) -> PredictionResponse:
    """Build the response from an internal P(outcomes[0]).

    `path` is the producing pipeline branch (tail-anchor, kalshi-anchor,
    kalshi+poly-blend, guardrail-anchored, poly-only, prior, llm-decisive,
    llm-grounded, llm-speculative, uniform). Required: keeping it
    mandatory catches missing call sites at test time.
    """
    p_clamped = max(0.01, min(0.99, p))
    outs = _default_outcomes(event)
    return PredictionResponse(
        probabilities=_binary_distribution(p_clamped, outs),
        p_yes=p_clamped,
        rationale=rationale,
        path=path,
    )


def _is_multi_outcome(event: EventRequest) -> bool:
    """True if the event has 3+ outcomes (e.g. Eurovision, award nominees)."""
    return event.outcomes is not None and len(event.outcomes) > 2


# Top-K detection patterns. Each captures the number from an explicit
# top-K phrasing. Ordered most-specific to least-specific. Required to
# have an actual integer right after the K-cue word to fire — qualitative
# phrasings ("multiple winners", "several teams") never trigger sum-to-K.
_TOP_K_PATTERNS = (
    re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bfinish(?:es)?\s+(?:in\s+)?(?:the\s+)?top\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bfirst\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bleading\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s+(?:winners?|qualif(?:y|iers?)|advance(?:rs?)?)\b", re.IGNORECASE),
)


def _detect_top_k(event: EventRequest) -> int:
    """How many outcomes resolve YES in this event. Returns 1 for single-winner.

    Conservative by design: only returns K > 1 when ALL of the following hold:
      - An explicit integer follows a top-K cue word in the title or rules
      - 2 ≤ K < len(outcomes) — single-winner if K is degenerate
      - len(outcomes) >= 3 — binary events are always K=1

    False positives are catastrophic (sum-to-K distribution submitted for a
    single-winner event scores wrong on every outcome). False negatives are
    status quo (today's known leak). So we bias hard toward K=1 on any
    ambiguity. The Prophet Arena spec confirmed (organizer answer 2026-05-17)
    that probabilities are scored as-is without normalization, so top-K
    events should be returned summing to ~K, not ~1.
    """
    outcomes = event.outcomes or []
    n_out = len(outcomes)
    if n_out < 3:
        return 1
    title = event.title or ""
    rules = event.rules or ""
    haystack = f"{title}\n{rules}"
    for pattern in _TOP_K_PATTERNS:
        m = pattern.search(haystack)
        if m is None:
            continue
        try:
            k = int(m.group(1))
        except (ValueError, IndexError):
            continue
        # Range checks. K must be plausible: at least 2, strictly less than
        # the number of outcomes (otherwise the question is degenerate), and
        # not absurdly large.
        if 2 <= k < n_out and k <= 20:
            return k
    return 1


# Kept for backwards compatibility — older code and tests reference the
# old name. Returns the same value as _detect_top_k for clarity.
_estimate_winners_count = _detect_top_k


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

# Minimum Polymarket 24h volume (USD) to include in the cross-venue blend
# WHEN Kalshi also has a price. Below this floor, a Polymarket quote is
# probably a thin / stale secondary listing whose price contains more
# noise than signal — blending it in gives an outlier 25% weight (per
# KALSHI_POLY_MAX_WEIGHT-style cap) to bad data. When only Polymarket
# has a quote (no Kalshi signal) we keep using it regardless; some
# signal beats no signal.
MIN_POLYMARKET_VOLUME_FOR_BLEND = 5_000.0

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

    # Tight spread (e.g. 0.99/1.00 on a settled-direction market) carries
    # the market's signal even when volume is zero. Trust those without
    # requiring vol >= MIN_VOL_24H. Wider spreads still require volume to
    # prevent us from anchoring to a single stale order on a noisy book.
    spread = yes_ask - yes_bid if (yes_bid > 0 and yes_ask > 0) else 1.0
    tight_spread = spread <= TIGHT_SPREAD_FOR_LOW_VOL
    book_usable = (
        not arb_violated
        and yes_bid > 0
        and yes_ask > 0
        and spread <= MAX_SPREAD
        and (tight_spread or vol >= MIN_VOL_24H)
    )
    if book_usable:
        p = _depth_weighted_mid(market) or (yes_bid + yes_ask) / 2
        note = "tight-spread" if (tight_spread and vol < MIN_VOL_24H) else "depth-mid"
        return p, (
            f"{note} {p:.3f} (bid={yes_bid:.3f}/ask={yes_ask:.3f}, "
            f"spread={spread:.3f}, vol24h=${vol:.0f})"
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
) -> tuple[float | None, str, str]:
    """If a Polymarket sibling exists, fold it into the price.

    Returns (p_or_none, rationale, blend_kind). `blend_kind` is one of:
      - "no-poly"     — category gated, no quote, or pass-through; kalshi unchanged
      - "poly-only"   — kalshi had no signal; poly carried the price
      - "skip-floor"  — poly volume below MIN_POLYMARKET_VOLUME_FOR_BLEND
      - "skip-agree"  — both venues agreed within tolerance
      - "blend"       — volume-weighted blend of both venues
      - "blend-equal" — equal-weight blend (both volumes zero)

    Callers use blend_kind to stamp the correct path label without
    re-deriving it from the free-text rationale.
    """
    if event.category not in POLYMARKET_CATEGORIES:
        return kalshi_p, kalshi_rationale, "no-poly"

    try:
        poly = polymarket_quote(event.model_dump())
    except Exception as e:  # poly is best-effort, never block on its failures
        logger.warning("polymarket lookup raised: %s", e)
        poly = None

    if poly is None:
        return kalshi_p, kalshi_rationale, "no-poly"

    poly_p, poly_weight, poly_rationale = poly

    if kalshi_p is None:
        return (
            poly_p,
            f"polymarket-only ({poly_rationale}); kalshi: {kalshi_rationale}",
            "poly-only",
        )

    # Minimum-volume floor: when Kalshi has a real price, a thin
    # Polymarket quote (volume < MIN_POLYMARKET_VOLUME_FOR_BLEND) is
    # likely a secondary listing with stale or wide-spread prices.
    # Including it in the blend gives bad data ~25% weight (cap is the
    # ratio cap, not an absolute volume floor). Skip the blend in that
    # case — use Kalshi alone.
    if poly_weight < MIN_POLYMARKET_VOLUME_FOR_BLEND:
        return (
            kalshi_p,
            (
                f"{kalshi_rationale}; poly vol ${poly_weight:.0f} < "
                f"${MIN_POLYMARKET_VOLUME_FOR_BLEND:.0f} floor → skip blend"
            ),
            "skip-floor",
        )

    # Cross-venue agreement check: when both venues quote the same answer,
    # there's no information in their agreement — skip the blend.
    if skip_if_agree_within is not None and abs(kalshi_p - poly_p) <= skip_if_agree_within:
        return (
            kalshi_p,
            (
                f"{kalshi_rationale}; poly agrees ({poly_p:.3f}, "
                f"|Δ|≤{skip_if_agree_within:.2f}) → skip blend"
            ),
            "skip-agree",
        )

    kalshi_weight = _kalshi_volume_weight(kalshi_market)
    total_weight = kalshi_weight + poly_weight
    if total_weight <= 0:
        # Both books exist but neither has measurable depth — straight average.
        blended = (kalshi_p + poly_p) / 2
        weight_note = "equal-weight"
        kind = "blend-equal"
    else:
        blended = (kalshi_p * kalshi_weight + poly_p * poly_weight) / total_weight
        weight_note = (
            f"vol-weighted (kalshi=${kalshi_weight:.0f} poly=${poly_weight:.0f})"
        )
        kind = "blend"
    return (
        blended,
        f"blend {blended:.3f} {weight_note}; kalshi: {kalshi_rationale}; {poly_rationale}",
        kind,
    )


# Map _blend_with_polymarket blend_kind → path label for binary events.
# The "kalshi-anchor" rollup covers the three cases where Kalshi's price
# was the canonical signal (no poly, poly skipped at floor, poly agreed
# so blend skipped). "kalshi+poly-blend" only when poly actually changed
# the price. "poly-only" when Kalshi had no signal.
_BLEND_KIND_TO_PATH = {
    "no-poly": "kalshi-anchor",
    "skip-floor": "kalshi-anchor",
    "skip-agree": "kalshi-anchor",
    "blend": "kalshi+poly-blend",
    "blend-equal": "kalshi+poly-blend",
    "poly-only": "poly-only",
}


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
    return _wrap_binary(
        anchored, (resp.rationale or "") + note, event, path="guardrail-anchored"
    )


def _llm_shrink_alpha(rationale: str, p: float | None = None) -> float:
    """Pick LLM shrinkage strength based on rationale content.

    Three tiers, checked in order of decreasing specificity:
      DECISIVE  → rationale describes an outcome that has already
                  occurred or been determined (game played, winner
                  declared, contender eliminated, etc.). Trust the LLM.
      GROUNDED  → rationale cites current data ("according to", "as of",
                  "polls show", etc.) but no decisive outcome.
      SPECULATIVE → rationale reads as base-rate speculation.

    Decisive false-positive guard: when `p` is provided AND the
    decisive markers fire BUT the prediction is mid-band [0.20, 0.80],
    downgrade to grounded. A decisive marker (e.g. "did not win") can
    misfire on counterfactual rationale ("Team A did not win in 2024
    but might in 2026"); the structural sanity check is that genuine
    decisive-evidence forecasts should land near the tails, not the
    middle. False-negatives here are cheap (we just apply slightly
    more shrinkage); false-positives at α=0.02 in the mid-band cost
    real Brier.
    """
    rationale_lower = rationale.lower()
    if any(marker in rationale_lower for marker in _LLM_DECISIVE_MARKERS):
        if p is not None and 0.20 <= p <= 0.80:
            return LLM_SHRINK_GROUNDED
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
        return _wrap_binary(
            p, f"{reason}; prior: {prior_rationale}", event, path="prior"
        )

    if not llm_allowed_for(event.category):
        return _wrap_binary(
            0.5,
            f"{reason}; LLM gated for category='{event.category}'; uniform prior",
            event,
            path="uniform",
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
        return _wrap_binary(
            0.5, f"{reason}; LLM unavailable; uniform prior", event, path="uniform"
        )
    p_raw, llm_rationale = out
    llm_rationale = llm_rationale + retry_note
    # Force speculative tier on the no-search retry: without web search,
    # the LLM is operating on training-cutoff knowledge alone and any
    # rationale-text classification (decisive / grounded) is unreliable.
    # Belt-and-suspenders insurance — no real cost on the happy path
    # since retry_note only sets when the first attempt failed entirely.
    if retry_note:
        alpha_base = LLM_SHRINK_SPECULATIVE
    else:
        alpha_base = _llm_shrink_alpha(llm_rationale, p=p_raw)
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
        path=f"llm-{tier}",
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
    prior = _uniform_prior(event)

    # Step 1: try Kalshi multi-outcome event. Returns target_sum which
    # is the authoritative K signal — 1.0 for mutex=True single-winner
    # events, K for mutex=False top-K events (where children naturally
    # sum to ~K). Kalshi's mutex flag is more reliable than text regex.
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

    # Determine target_sum (K signal hierarchy):
    #
    # The QUESTION TEXT is authoritative when it gives an explicit K
    # ("top 4 Bundesliga finishers" → K=4 by definition; Kalshi's
    # Σ children = 4.83 rounding to 5 is market noise, not the truth).
    # Kalshi's mutex flag is canonical for the SINGLE-WINNER vs TOP-K
    # split, but K itself comes from the title when stated.
    #
    # Resolution order:
    #   1. Kalshi mutex=True → target_sum=1 (definitive single-winner)
    #   2. Explicit text K (≥2) → target_sum=K
    #   3. Kalshi mutex=False with no text K → target_sum from Σ children
    #   4. No Kalshi data → text K (regex only)
    text_k = _detect_top_k(event)
    if kalshi_out is not None:
        kalshi_target_sum = float(kalshi_out[3])
        if kalshi_target_sum == 1.0:
            # mutex=True from Kalshi: definitively single-winner.
            target_sum = 1.0
        elif text_k >= 2:
            # Kalshi says top-K AND text gives explicit K. Trust text.
            target_sum = float(text_k)
        else:
            # Kalshi says top-K but text is silent. Trust Kalshi's K.
            target_sum = kalshi_target_sum
    else:
        target_sum = float(text_k)
    k = int(round(target_sum))

    # Step 3: combine market signals.
    market_dist: list[dict[str, float]] | None = None
    market_rationale: str = ""
    market_path: str = ""
    if kalshi_out is not None and poly_out is not None:
        # _blend_multi_outcome_distributions takes 3-tuples; pull off the
        # target_sum field before blending. We keep the kalshi target_sum
        # since Polymarket doesn't yet surface a comparable signal.
        kalshi_3 = (kalshi_out[0], kalshi_out[1], kalshi_out[2])
        blended, _vol, rat = _blend_multi_outcome_distributions(
            kalshi_3, poly_out, outcomes
        )
        market_dist = blended
        market_rationale = rat
        market_path = "multi-outcome-blend"
    elif kalshi_out is not None:
        market_dist = kalshi_out[0]
        market_rationale = f"kalshi only; {kalshi_out[2]}"
        market_path = "multi-outcome-kalshi"
    elif poly_out is not None:
        market_dist = poly_out[0]
        market_rationale = f"poly only; {poly_out[2]}"
        market_path = "multi-outcome-poly"

    if market_dist is not None:
        dist = _normalize_distribution(market_dist, outcomes, target_sum=target_sum)
        # Align p_yes with dist[0] so calibration / logging see the same
        # outcomes[0] probability the server scores against.
        p_yes_value = max(0.01, min(0.99, dist[0].probability)) if dist else prior
        return PredictionResponse(
            probabilities=dist,
            p_yes=p_yes_value,
            rationale=(
                f"multi-outcome ({n_out} options, top-{k}, Σ={target_sum:g}); "
                f"{market_rationale}"
            ),
            path=market_path,
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
        # Uniform fallback. For top-K (K>=2) events the uniform marginal is
        # K/N per outcome (sums to K). For single-winner it's 1/N (sums to 1).
        per_outcome = (target_sum / n_out) if n_out > 0 else 0.0
        per_outcome = max(0.0, min(1.0, per_outcome))
        uniform = [
            MarketProbability(market=o, probability=per_outcome) for o in outcomes
        ] if n_out > 0 else []
        return PredictionResponse(
            probabilities=uniform,
            p_yes=max(0.01, min(0.99, per_outcome)),
            rationale=(
                f"multi-outcome ({n_out} options, top-{k}, Σ={target_sum:g}); "
                f"LLM unavailable; uniform K/N across outcomes"
            ),
            path="multi-outcome-uniform",
        )
    raw_p, probabilities, llm_rationale = out

    # Build the per-outcome distribution scaled to target_sum (=K). Safety
    # clamp: if any per-outcome value after scaling exceeds 0.99, the LLM
    # gave us a malformed distribution for this K (e.g., 0.5 on a K=5 event
    # would scale to 2.5). Fall back to uniform K/N to ship a defensible
    # sum-to-K distribution rather than a clamped-and-distorted one.
    safety_triggered = False
    if probabilities:
        dist = _normalize_distribution(probabilities, outcomes, target_sum=target_sum)
    else:
        # Shrink raw_p toward uniform prior k/N first (variance protection),
        # then distribute the remaining mass uniformly across other outcomes.
        # The synthesized distribution is built to sum to 1; _normalize_
        # distribution will scale it to target_sum.
        shrunk = (1 - MULTI_LLM_SHRINK) * raw_p + MULTI_LLM_SHRINK * prior
        shrunk = max(0.01, min(0.99, shrunk))
        synthesized = [{"market": outcomes[0], "probability": shrunk}]
        if n_out > 1:
            per_other = (1.0 - shrunk) / (n_out - 1)
            for o in outcomes[1:]:
                synthesized.append({"market": o, "probability": per_other})
        dist = _normalize_distribution(synthesized, outcomes, target_sum=target_sum)

    if target_sum > 1.0 and any(p.probability > 0.99 for p in dist):
        # LLM distribution doesn't fit K cleanly. Ship uniform K/N as
        # safety. Bounded error: |dist - uniform_K_over_N| is at most K-1
        # per-outcome in absolute Brier.
        safety_triggered = True
        per_outcome = max(0.0, min(1.0, target_sum / n_out)) if n_out > 0 else 0.0
        dist = [
            MarketProbability(market=o, probability=per_outcome)
            for o in outcomes
        ]

    # Align p_yes with dist[0] post-normalization — calibration / logging
    # see the same outcomes[0] probability the server scores against.
    p_yes_value = max(0.01, min(0.99, dist[0].probability)) if dist else prior

    rationale_suffix = (
        " [safety: scaled prob > 0.99 → uniform K/N]" if safety_triggered else ""
    )
    return PredictionResponse(
        probabilities=dist,
        p_yes=p_yes_value,
        rationale=(
            f"multi-outcome ({n_out} options, top-{k}, Σ={target_sum:g}, "
            f"uniform={prior:.3f}); raw={raw_p:.3f}; {llm_rationale}"
            f"{rationale_suffix}"
        ),
        path="multi-outcome-llm",
    )


def _forecast(event: EventRequest) -> PredictionResponse:
    if _is_multi_outcome(event):
        return _multi_outcome_forecast(event)

    market = get_market(event.market_ticker)

    if market is None:
        # Kalshi fetch failed entirely. Try Polymarket as an alternative
        # market-anchor before reaching for priors / LLM.
        poly_p, poly_rationale, blend_kind = _blend_with_polymarket(
            event, None, None, f"kalshi fetch failed for {event.market_ticker}"
        )
        if poly_p is not None:
            p = _shrink_and_clamp(poly_p, alpha=MIN_SHRINK_ALPHA)
            return _wrap_binary(
                p, poly_rationale, event,
                path=_BLEND_KIND_TO_PATH.get(blend_kind, "poly-only"),
            )
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
            path="tail-anchor",
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
        raw_p, rationale, blend_kind = _blend_with_polymarket(
            event, market, raw_p, rationale,
            skip_if_agree_within=CROSS_VENUE_DISAGREE_TOL,
        )
    else:
        raw_p, rationale, blend_kind = _blend_with_polymarket(
            event, market, raw_p, rationale
        )

    if raw_p is None:
        return _llm_fallback(event, reason=rationale)

    alpha = _shrink_alpha(vol_24h)

    if APPLY_STALENESS:
        age_h = _staleness_hours(market)
        if age_h is not None and age_h > STALE_HOURS:
            alpha = min(MAX_SHRINK_ALPHA, alpha * STALE_ALPHA_MULTIPLIER)
            rationale += f"; stale book {age_h:.1f}h (α ×{STALE_ALPHA_MULTIPLIER:g})"

    p = _shrink_and_clamp(raw_p, alpha=alpha)
    resp = _wrap_binary(
        p, f"{rationale}; shrunk α={alpha:.3f} → {p:.3f}", event,
        path=_BLEND_KIND_TO_PATH.get(blend_kind, "kalshi-anchor"),
    )
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

    # Prefer the path stamped at the producer; fall back to classifying
    # the rationale for legacy responses that didn't stamp.
    path = resp.path or classify_path(resp.rationale or "")
    raw = resp.p_yes
    adjusted = apply_calibration_data(raw, data, path=path)
    if adjusted == raw:
        return resp
    adjusted = max(0.01, min(0.99, adjusted))
    return _wrap_binary(
        adjusted,
        f"{resp.rationale}; calibrated[{path}] {raw:.3f}→{adjusted:.3f}",
        event,
        path=path,
    )


def _log_metadata(resp: PredictionResponse) -> dict:
    """Build the log_prediction metadata dict from a response.

    Stamps the producer's path (preferred over rationale-regex classification)
    and the agent version so post-eval analysis can attribute predictions
    to the version that produced them.
    """
    meta: dict = {"version": AGENT_VERSION}
    if resp.path:
        meta["path"] = resp.path
    return meta


def predict(event: dict) -> dict:
    event_obj = EventRequest(**event)
    resp = _forecast(event_obj)
    resp = _maybe_calibrate(resp, event_obj)
    log_prediction(event, resp.p_yes, resp.rationale, metadata=_log_metadata(resp))
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
    log_prediction(
        event.model_dump(), resp.p_yes, resp.rationale, metadata=_log_metadata(resp)
    )
    return resp


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
