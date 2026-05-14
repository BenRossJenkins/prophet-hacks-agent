"""Trading decisions and position management for the Prophet Hacks track.

Three pieces:

1. `decide(event)` — single-market decision. Combines our forecaster
   (`agent.predict.predict`) with the live Kalshi book to produce a
   buy-YES / buy-NO / hold decision sized via fractional Kelly.

2. `Position` / `PositionBook` — track open positions across the
   eval window, mark-to-market against current Kalshi prices, and
   apply resolutions when markets settle.

3. Portfolio-level guardrails — per-category exposure caps and
   per-market exposure caps so a single decision can't blow up the
   bankroll.

The Prophet Arena trading API isn't published yet (server still 404s);
this module produces structured decisions and tracks state that we can
plug into the platform's submission format on hackathon weekend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from agent.independent import independent_forecast
from agent.kalshi import get_market

# ---- Sizing constants ----------------------------------------------------

EDGE_THRESHOLD = 0.08         # don't trade unless edge >= 8pp (wider than the
                              # forecaster's 5pp; trading is leveraged so a
                              # bigger edge floor is appropriate)
KELLY_FRACTION = 0.25         # fractional Kelly — 1/4 Kelly is conservative-standard
MAX_PER_MARKET = 0.05         # cap exposure per market at 5% of bankroll
MAX_PER_CATEGORY = 0.25       # cap exposure per category at 25% of bankroll

# Confidence-based Kelly scaling. Multiplied with KELLY_FRACTION above.
# Tradeable confidences: typed prior beats grounded LLM beats speculative LLM.
#
# 2026-05-14: bumped "high" from 1.0 → 2.0 (i.e. 0.5x full Kelly instead of
# 0.25x) to be more aggressive when our typed external-data priors fire.
# "medium" bumped 0.5 → 1.0 to preserve the 0.5x relative scaling. The
# starting-bankroll-fraction cap (MAX_PER_MARKET=5%) still limits per-bet
# exposure so we can't blow up on a single decision.
KELLY_BY_CONFIDENCE = {
    "high": 2.0,        # typed prior — 0.5x full Kelly
    "medium": 1.0,      # grounded LLM — 0.25x full Kelly
    "low": 0.0,         # speculative LLM — don't trade
    "none": 0.0,        # no signal — never
}

# ---- Decision dataclass --------------------------------------------------


@dataclass(frozen=True)
class TradeDecision:
    market_ticker: str
    action: str  # "buy_yes" | "buy_no" | "hold"
    our_p: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    edge: float
    size_fraction: float          # of bankroll
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Kelly sizing -------------------------------------------------------


def kelly_fraction(our_p: float, price: float) -> float:
    """Full-Kelly fraction for buying at `price` with our forecast `our_p`.

    f* = (our_p * (1 - price) - (1 - our_p) * price) / (1 - price)
       = (our_p - price) / (1 - price)

    Returns 0 when the edge isn't positive (don't bet) or when price >= 1.
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    edge = our_p - price
    if edge <= 0:
        return 0.0
    return edge / (1.0 - price)


def sized_fraction(our_p: float, price: float, *, confidence: str = "high") -> float:
    """Position size as a bankroll fraction, scaled by confidence."""
    full_kelly = kelly_fraction(our_p, price)
    confidence_mult = KELLY_BY_CONFIDENCE.get(confidence, 0.0)
    return min(KELLY_FRACTION * confidence_mult * full_kelly, MAX_PER_MARKET)


# ---- Decision -----------------------------------------------------------


def _f(d: dict, key: str) -> float:
    try:
        return float(d.get(key, "0") or 0)
    except (ValueError, TypeError):
        return 0.0


def _build_hold(
    market_ticker: str,
    our_p: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    edge: float,
    rationale: str,
) -> TradeDecision:
    return TradeDecision(
        market_ticker=market_ticker,
        action="hold",
        our_p=our_p,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        edge=edge,
        size_fraction=0.0,
        rationale=rationale,
    )


def decide(event: dict) -> TradeDecision:
    """Make a single-market trading decision from an event dict.

    Uses the independent forecast (priors / LLM ensemble — never market-
    anchored) as our second opinion. We only trade when (1) the forecast
    is confident enough to bet AND (2) it disagrees with the market by at
    least EDGE_THRESHOLD.
    """
    market_ticker = event.get("market_ticker", "")
    our_p, forecast_rationale, confidence = independent_forecast(event)

    # No tradeable signal → hold. (Speculative LLM and total fallback both
    # land here; we don't risk bankroll on guesses.)
    if confidence not in ("high", "medium"):
        return _build_hold(
            market_ticker, our_p, 0.0, 0.0, 0.0, 0.0, 0.0,
            rationale=f"hold: confidence={confidence}; {forecast_rationale}",
        )

    market = get_market(market_ticker)
    if market is None:
        return _build_hold(
            market_ticker,
            our_p,
            0.0, 0.0, 0.0, 0.0, 0.0,
            rationale=f"hold: kalshi unavailable for {market_ticker}; forecast: {forecast_rationale}",
        )

    yes_bid = _f(market, "yes_bid_dollars")
    yes_ask = _f(market, "yes_ask_dollars")
    no_bid = _f(market, "no_bid_dollars")
    no_ask = _f(market, "no_ask_dollars")

    edge_buy_yes = our_p - yes_ask if yes_ask > 0 else float("-inf")
    edge_buy_no = (1.0 - our_p) - no_ask if no_ask > 0 else float("-inf")

    if edge_buy_yes >= EDGE_THRESHOLD and edge_buy_yes >= edge_buy_no:
        size = sized_fraction(our_p, yes_ask, confidence=confidence)
        if size <= 0:
            return _build_hold(
                market_ticker, our_p, yes_bid, yes_ask, no_bid, no_ask, edge_buy_yes,
                rationale=f"hold: zero size at confidence={confidence}; {forecast_rationale}",
            )
        return TradeDecision(
            market_ticker=market_ticker,
            action="buy_yes",
            our_p=our_p,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            edge=edge_buy_yes,
            size_fraction=size,
            rationale=(
                f"buy YES @ {yes_ask:.3f}: our_p={our_p:.3f}, edge={edge_buy_yes:.3f}, "
                f"size={size:.3%}, confidence={confidence}; {forecast_rationale}"
            ),
        )
    if edge_buy_no >= EDGE_THRESHOLD:
        size = sized_fraction(1.0 - our_p, no_ask, confidence=confidence)
        if size <= 0:
            return _build_hold(
                market_ticker, our_p, yes_bid, yes_ask, no_bid, no_ask, edge_buy_no,
                rationale=f"hold: zero size at confidence={confidence}; {forecast_rationale}",
            )
        return TradeDecision(
            market_ticker=market_ticker,
            action="buy_no",
            our_p=our_p,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            edge=edge_buy_no,
            size_fraction=size,
            rationale=(
                f"buy NO @ {no_ask:.3f}: 1-our_p={1 - our_p:.3f}, edge={edge_buy_no:.3f}, "
                f"size={size:.3%}, confidence={confidence}; {forecast_rationale}"
            ),
        )

    best_edge = max(edge_buy_yes, edge_buy_no)
    return _build_hold(
        market_ticker, our_p, yes_bid, yes_ask, no_bid, no_ask, best_edge,
        rationale=(
            f"hold: best edge {best_edge:.3f} below threshold "
            f"{EDGE_THRESHOLD:.2f}; {forecast_rationale}"
        ),
    )


# ---- Position management ------------------------------------------------


@dataclass
class Position:
    market_ticker: str
    category: str
    side: str           # "yes" or "no"
    qty: float          # number of contracts (each pays $1 if right side wins)
    cost_basis: float   # total dollars paid

    @property
    def avg_price(self) -> float:
        return self.cost_basis / self.qty if self.qty > 0 else 0.0


@dataclass
class PositionBook:
    """Tracks open positions and bankroll over the trading session.

    Designed for the eval-window flow: each call to /predict produces a
    decision; the trader applies the decision by calling `attempt_open`,
    which respects bankroll, per-market, and per-category caps.
    """

    starting_bankroll: float = 1000.0
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.cash = self.starting_bankroll

    # ----- exposure tracking -----

    def exposure_in_market(self, market_ticker: str) -> float:
        pos = self.positions.get(market_ticker)
        return pos.cost_basis if pos else 0.0

    def exposure_in_category(self, category: str) -> float:
        return sum(p.cost_basis for p in self.positions.values() if p.category == category)

    def total_exposure(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def bankroll(self) -> float:
        """Cash + cost-basis of open positions (post-realization)."""
        return self.cash + self.total_exposure() + self.realized_pnl

    # ----- opening / applying decisions -----

    def attempt_open(
        self, decision: TradeDecision, category: str, *, bankroll_override: float | None = None
    ) -> Position | None:
        """Try to open the position implied by `decision`. Returns the Position
        opened (which may be smaller than requested if caps bite) or None if
        the decision was a hold / no edge / no remaining capacity.
        """
        if decision.action not in ("buy_yes", "buy_no"):
            return None
        side = "yes" if decision.action == "buy_yes" else "no"
        price = decision.yes_ask if side == "yes" else decision.no_ask
        if price <= 0.0 or price >= 1.0:
            return None

        # Compounding bankroll basis: starting capital plus realized P&L.
        # Open positions are still tracked separately (their cost basis sits
        # in cash already) so this represents capital ALLOCATABLE to new
        # bets. Caller can override for fixed-bankroll backtests or stress
        # tests.
        if bankroll_override is not None:
            basis = bankroll_override
        else:
            basis = max(0.0, self.starting_bankroll + self.realized_pnl)
        desired_cost = basis * decision.size_fraction

        # Apply per-market cap
        existing_in_market = self.exposure_in_market(decision.market_ticker)
        market_cap = basis * MAX_PER_MARKET
        allowed_by_market = max(0.0, market_cap - existing_in_market)

        # Apply per-category cap
        existing_in_cat = self.exposure_in_category(category)
        cat_cap = basis * MAX_PER_CATEGORY
        allowed_by_cat = max(0.0, cat_cap - existing_in_cat)

        # Apply remaining cash
        allowed_by_cash = max(0.0, self.cash)

        actual_cost = min(desired_cost, allowed_by_market, allowed_by_cat, allowed_by_cash)
        if actual_cost <= 0:
            return None

        qty = actual_cost / price
        existing = self.positions.get(decision.market_ticker)
        if existing is None or existing.side != side:
            # Either no existing position or one on the opposite side; we treat
            # opposite-side as ignored here (would require netting logic we
            # don't need yet).
            if existing is not None and existing.side != side:
                return None
            self.positions[decision.market_ticker] = Position(
                market_ticker=decision.market_ticker,
                category=category,
                side=side,
                qty=qty,
                cost_basis=actual_cost,
            )
        else:
            existing.qty += qty
            existing.cost_basis += actual_cost

        self.cash -= actual_cost
        return self.positions[decision.market_ticker]

    # ----- resolution -----

    def resolve(self, market_ticker: str, result: str) -> float:
        """Apply a market resolution. `result` ∈ {"yes", "no"}. Returns realized P&L."""
        pos = self.positions.pop(market_ticker, None)
        if pos is None or result not in ("yes", "no"):
            return 0.0
        payout = pos.qty if pos.side == result else 0.0
        pnl = payout - pos.cost_basis
        self.realized_pnl += pnl
        self.cash += payout
        return pnl

    def resolve_many(self, resolutions: Iterable[tuple[str, str]]) -> float:
        return sum(self.resolve(t, r) for t, r in resolutions)

    # ----- introspection -----

    def summary(self) -> dict:
        return {
            "cash": round(self.cash, 4),
            "open_positions": len(self.positions),
            "total_cost_basis": round(self.total_exposure(), 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "by_category": {
                cat: round(sum(p.cost_basis for p in self.positions.values() if p.category == cat), 4)
                for cat in {p.category for p in self.positions.values()}
            },
        }
