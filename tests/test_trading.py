from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.trading import (
    EDGE_THRESHOLD,
    KELLY_FRACTION,
    MAX_PER_CATEGORY,
    MAX_PER_MARKET,
    Position,
    PositionBook,
    TradeDecision,
    decide,
    kelly_fraction,
    sized_fraction,
)


def _event(market_ticker: str = "TEST-MKT") -> dict:
    return {
        "event_ticker": "TEST-EVT",
        "market_ticker": market_ticker,
        "title": "Test market",
        "category": "Sports",
        "close_time": "2026-12-31T23:59:59Z",
    }


def _market(yes_bid: float, yes_ask: float, *, no_bid: float | None = None, no_ask: float | None = None) -> dict:
    if no_bid is None:
        no_bid = 1.0 - yes_ask
    if no_ask is None:
        no_ask = 1.0 - yes_bid
    return {
        "yes_bid_dollars": f"{yes_bid:.4f}",
        "yes_ask_dollars": f"{yes_ask:.4f}",
        "no_bid_dollars": f"{no_bid:.4f}",
        "no_ask_dollars": f"{no_ask:.4f}",
    }


def _mocks(our_p: float, market: dict | None):
    return (
        patch("agent.trading.predict", return_value={"p_yes": our_p, "rationale": "test forecast"}),
        patch("agent.trading.get_market", return_value=market),
    )


# ---- Kelly math ----------------------------------------------------------


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.40, 0.42) == 0
    assert kelly_fraction(0.50, 0.50) == 0


def test_kelly_full_formula():
    # p=0.70, price=0.50 → f* = (0.70-0.50)/(1-0.50) = 0.40
    assert kelly_fraction(0.70, 0.50) == pytest.approx(0.40)
    # p=0.60, price=0.40 → f* = 0.20/0.60 = 0.333
    assert kelly_fraction(0.60, 0.40) == pytest.approx(1 / 3)


def test_kelly_grows_as_edge_grows():
    assert kelly_fraction(0.55, 0.50) < kelly_fraction(0.70, 0.50)


def test_sized_fraction_applies_fractional_kelly_and_market_cap():
    # full Kelly 0.40 → fractional 0.25 * 0.40 = 0.10, capped at MAX_PER_MARKET=0.05
    s = sized_fraction(0.70, 0.50)
    assert s == pytest.approx(MAX_PER_MARKET)


def test_sized_fraction_below_cap():
    # Small edge: p=0.52, price=0.50 → full Kelly 0.04 → fractional 0.01 (below cap)
    s = sized_fraction(0.52, 0.50)
    assert s == pytest.approx(KELLY_FRACTION * 0.04)


# ---- decide() ------------------------------------------------------------


def test_buy_yes_when_our_p_well_above_ask():
    market = _market(yes_bid=0.40, yes_ask=0.42)
    p1, p2 = _mocks(our_p=0.70, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "buy_yes"
    assert d.size_fraction > 0
    assert d.size_fraction <= MAX_PER_MARKET


def test_buy_no_when_our_p_well_below_bid():
    market = _market(yes_bid=0.60, yes_ask=0.62)
    p1, p2 = _mocks(our_p=0.20, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "buy_no"
    assert d.size_fraction > 0


def test_hold_when_edge_below_threshold():
    market = _market(yes_bid=0.49, yes_ask=0.51)
    p1, p2 = _mocks(our_p=0.52, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "hold"
    assert d.size_fraction == 0.0


def test_hold_when_kalshi_unavailable():
    p1, p2 = _mocks(our_p=0.70, market=None)
    with p1, p2:
        d = decide(_event())
    assert d.action == "hold"
    assert "kalshi unavailable" in d.rationale


def test_hold_when_forecast_is_no_signal_fallback():
    """If the agent fell back to 0.5 because it had no info, the trader must
    refuse to trade — the market price is a better estimator and any apparent
    edge is illusory.
    """
    market = _market(yes_bid=0.40, yes_ask=0.42)
    no_signal_rationale = (
        "no price signal (bid=0.000/ask=0.000/last=0.000); LLM unavailable; uniform prior"
    )
    with patch(
        "agent.trading.predict",
        return_value={"p_yes": 0.5, "rationale": no_signal_rationale},
    ), patch("agent.trading.get_market", return_value=market):
        d = decide(_event())
    assert d.action == "hold"
    assert "no-signal" in d.rationale


def test_hold_when_category_gated_uniform_prior():
    market = _market(yes_bid=0.40, yes_ask=0.42)
    gated_rationale = "no price signal; LLM gated for category='Climate and Weather'; uniform prior"
    with patch(
        "agent.trading.predict",
        return_value={"p_yes": 0.5, "rationale": gated_rationale},
    ), patch("agent.trading.get_market", return_value=market):
        d = decide(_event())
    assert d.action == "hold"


def test_just_above_threshold_triggers_buy():
    market = _market(yes_bid=0.40, yes_ask=0.45)
    p1, p2 = _mocks(our_p=0.51, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "buy_yes"


def test_decision_is_serializable():
    market = _market(yes_bid=0.40, yes_ask=0.42)
    p1, p2 = _mocks(our_p=0.70, market=market)
    with p1, p2:
        d = decide(_event())
    obj = d.to_dict()
    assert obj["market_ticker"] == "TEST-MKT"
    assert obj["action"] == "buy_yes"


# ---- PositionBook --------------------------------------------------------


def _decision_buy_yes(ticker: str = "M1", price: float = 0.50, size: float = 0.02) -> TradeDecision:
    return TradeDecision(
        market_ticker=ticker,
        action="buy_yes",
        our_p=0.70,
        yes_bid=price - 0.01,
        yes_ask=price,
        no_bid=1.0 - price - 0.01,
        no_ask=1.0 - price,
        edge=0.20,
        size_fraction=size,
        rationale="test",
    )


def test_position_book_opens_position():
    book = PositionBook(starting_bankroll=1000.0)
    d = _decision_buy_yes(price=0.50, size=0.02)
    pos = book.attempt_open(d, category="Sports")

    assert pos is not None
    assert pos.qty == pytest.approx(20.0 / 0.50)  # $20 / $0.50 = 40 contracts
    assert pos.cost_basis == pytest.approx(20.0)
    assert book.cash == pytest.approx(980.0)


def test_position_book_enforces_per_market_cap():
    book = PositionBook(starting_bankroll=1000.0)
    d = _decision_buy_yes(price=0.50, size=0.20)  # asks for 20% but cap is 5%
    pos = book.attempt_open(d, category="Sports")
    assert pos is not None
    assert pos.cost_basis == pytest.approx(50.0)  # 5% of $1000


def test_position_book_enforces_per_category_cap():
    book = PositionBook(starting_bankroll=1000.0)
    # Open 5 separate markets, each at 5% per market in same category
    for i in range(5):
        d = _decision_buy_yes(ticker=f"M{i}", price=0.50, size=0.05)
        book.attempt_open(d, category="Sports")
    # 5 * 5% = 25% = category cap. A 6th market should hit zero capacity.
    d6 = _decision_buy_yes(ticker="M6", price=0.50, size=0.05)
    pos = book.attempt_open(d6, category="Sports")
    assert pos is None


def test_position_book_resolves_winning_yes():
    book = PositionBook(starting_bankroll=1000.0)
    d = _decision_buy_yes(price=0.50, size=0.02)
    book.attempt_open(d, category="Sports")
    pnl = book.resolve("M1", "yes")
    # 40 contracts paid $1 each = $40 payout. Cost basis was $20. PnL = +$20.
    assert pnl == pytest.approx(20.0)
    assert book.realized_pnl == pytest.approx(20.0)
    assert "M1" not in book.positions


def test_position_book_resolves_losing_yes():
    book = PositionBook(starting_bankroll=1000.0)
    d = _decision_buy_yes(price=0.50, size=0.02)
    book.attempt_open(d, category="Sports")
    pnl = book.resolve("M1", "no")
    # Payout 0, cost basis 20 → PnL = -$20.
    assert pnl == pytest.approx(-20.0)
    assert book.realized_pnl == pytest.approx(-20.0)


def test_position_book_holds_decision_returns_none():
    book = PositionBook(starting_bankroll=1000.0)
    hold = TradeDecision(
        market_ticker="M1",
        action="hold",
        our_p=0.5,
        yes_bid=0.49, yes_ask=0.51,
        no_bid=0.49, no_ask=0.51,
        edge=0.0,
        size_fraction=0.0,
        rationale="no edge",
    )
    pos = book.attempt_open(hold, category="Sports")
    assert pos is None
    assert book.cash == 1000.0


def test_position_book_summary_shape():
    book = PositionBook(starting_bankroll=1000.0)
    book.attempt_open(_decision_buy_yes("A", 0.5, 0.02), category="Sports")
    book.attempt_open(_decision_buy_yes("B", 0.3, 0.02), category="Politics")
    s = book.summary()
    assert s["open_positions"] == 2
    assert "Sports" in s["by_category"]
    assert "Politics" in s["by_category"]
