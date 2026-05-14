"""Trading decisions for the Prophet Hacks trading track.

Combines our forecaster (`agent.predict.predict`) with the current Kalshi book.
For each market we compute the edge from buying YES (our_p - yes_ask) and
the edge from buying NO ((1 - our_p) - no_ask), and trade the better side if
its edge exceeds a threshold. Sizing is fixed-fraction of bankroll.

The Prophet Arena trading API isn't published yet (server still 404s
2026-05-14); this module is shape-only and returns structured TradeDecision
objects that can be wired to whatever submission format the platform exposes
on hackathon weekend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agent.kalshi import get_market
from agent.predict import predict

# Tunables — exposed as module-level so they can be overridden in tests.
EDGE_THRESHOLD = 0.05       # don't trade unless edge >= 5 percentage points
BANKROLL_FRACTION = 0.02    # 2% of bankroll per trade
MAX_PER_MARKET = 0.05       # cap exposure per market at 5%


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
    size_fraction: float
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


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

    The event must have at least `market_ticker`; predict(event) fills in the
    forecast and we look up the current Kalshi book directly.
    """
    market_ticker = event.get("market_ticker", "")
    forecast = predict(event)
    our_p = float(forecast["p_yes"])
    forecast_rationale = forecast["rationale"]

    market = get_market(market_ticker)
    if market is None:
        return _build_hold(
            market_ticker,
            our_p,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            rationale=f"hold: kalshi unavailable for {market_ticker}; forecast: {forecast_rationale}",
        )

    yes_bid = _f(market, "yes_bid_dollars")
    yes_ask = _f(market, "yes_ask_dollars")
    no_bid = _f(market, "no_bid_dollars")
    no_ask = _f(market, "no_ask_dollars")

    edge_buy_yes = our_p - yes_ask if yes_ask > 0 else float("-inf")
    edge_buy_no = (1.0 - our_p) - no_ask if no_ask > 0 else float("-inf")

    if edge_buy_yes >= EDGE_THRESHOLD and edge_buy_yes >= edge_buy_no:
        size = min(BANKROLL_FRACTION, MAX_PER_MARKET)
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
                f"buy YES: our_p={our_p:.3f} > yes_ask={yes_ask:.3f} "
                f"(edge={edge_buy_yes:.3f}); {forecast_rationale}"
            ),
        )
    if edge_buy_no >= EDGE_THRESHOLD:
        size = min(BANKROLL_FRACTION, MAX_PER_MARKET)
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
                f"buy NO: 1-our_p={1 - our_p:.3f} > no_ask={no_ask:.3f} "
                f"(edge={edge_buy_no:.3f}); {forecast_rationale}"
            ),
        )

    best_edge = max(edge_buy_yes, edge_buy_no)
    return _build_hold(
        market_ticker,
        our_p,
        yes_bid,
        yes_ask,
        no_bid,
        no_ask,
        best_edge,
        rationale=(
            f"hold: best edge {best_edge:.3f} below threshold "
            f"{EDGE_THRESHOLD:.2f}; {forecast_rationale}"
        ),
    )
