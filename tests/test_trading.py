from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.trading import (
    BANKROLL_FRACTION,
    EDGE_THRESHOLD,
    MAX_PER_MARKET,
    TradeDecision,
    decide,
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
    """Convenience: returns context-manager patches for predict + get_market."""
    return (
        patch(
            "agent.trading.predict",
            return_value={"p_yes": our_p, "rationale": "test forecast"},
        ),
        patch("agent.trading.get_market", return_value=market),
    )


def test_buy_yes_when_our_p_well_above_ask():
    market = _market(yes_bid=0.40, yes_ask=0.42)
    p1, p2 = _mocks(our_p=0.70, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "buy_yes"
    # edge = 0.70 - 0.42 = 0.28
    assert d.edge == pytest.approx(0.28)
    assert d.size_fraction == pytest.approx(min(BANKROLL_FRACTION, MAX_PER_MARKET))


def test_buy_no_when_our_p_well_below_bid():
    market = _market(yes_bid=0.60, yes_ask=0.62)
    p1, p2 = _mocks(our_p=0.20, market=market)
    with p1, p2:
        d = decide(_event())
    # 1 - 0.20 = 0.80; no_ask = 1 - 0.60 = 0.40; edge = 0.80 - 0.40 = 0.40
    assert d.action == "buy_no"
    assert d.edge == pytest.approx(0.40)


def test_hold_when_edge_below_threshold():
    market = _market(yes_bid=0.49, yes_ask=0.51)
    # our_p = 0.52 → edge_buy_yes = 0.01, edge_buy_no = -0.01 → hold
    p1, p2 = _mocks(our_p=0.52, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "hold"
    assert d.size_fraction == 0.0
    assert "below threshold" in d.rationale


def test_hold_when_kalshi_unavailable():
    p1, p2 = _mocks(our_p=0.70, market=None)
    with p1, p2:
        d = decide(_event())
    assert d.action == "hold"
    assert "kalshi unavailable" in d.rationale


def test_chooses_higher_edge_side():
    # YES edge 0.06 (just above threshold). NO edge 0.20 (clearly higher).
    # our_p = 0.30, yes_ask = 0.24 → edge_buy_yes = 0.06
    # no_ask = 1 - yes_bid. If yes_bid = 0.50 then no_ask = 0.50.
    # 1 - 0.30 = 0.70. edge_buy_no = 0.70 - 0.50 = 0.20.
    market = _market(yes_bid=0.50, yes_ask=0.24)  # wide intentional
    p1, p2 = _mocks(our_p=0.30, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "buy_no"


def test_decision_is_serializable():
    market = _market(yes_bid=0.40, yes_ask=0.42)
    p1, p2 = _mocks(our_p=0.70, market=market)
    with p1, p2:
        d = decide(_event())
    obj = d.to_dict()
    assert obj["market_ticker"] == "TEST-MKT"
    assert obj["action"] == "buy_yes"
    assert obj["size_fraction"] > 0


def test_just_above_threshold_triggers_buy():
    market = _market(yes_bid=0.40, yes_ask=0.45)
    # edge_buy_yes = 0.51 - 0.45 = 0.06 (clearly > threshold 0.05, no float noise)
    p1, p2 = _mocks(our_p=0.51, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "buy_yes"


def test_hold_when_book_has_no_prices():
    market = {
        "yes_bid_dollars": "0",
        "yes_ask_dollars": "0",
        "no_bid_dollars": "0",
        "no_ask_dollars": "0",
    }
    p1, p2 = _mocks(our_p=0.80, market=market)
    with p1, p2:
        d = decide(_event())
    assert d.action == "hold"


def test_decision_carries_forecast_rationale():
    market = _market(yes_bid=0.40, yes_ask=0.42)
    p1, p2 = _mocks(our_p=0.70, market=market)
    with p1, p2:
        d = decide(_event())
    assert "test forecast" in d.rationale
