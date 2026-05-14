from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.predict import (
    MAX_SPREAD,
    MIN_VOL_24H,
    SHRINK_ALPHA,
    _market_implied_prob,
    _shrink_and_clamp,
    predict,
)


def _event(market_ticker: str = "TEST-MKT") -> dict:
    return {
        "event_ticker": "TEST-EVT",
        "market_ticker": market_ticker,
        "title": "Test market",
        "category": "Test",
        "close_time": "2026-12-31T23:59:59Z",
    }


def test_shrink_pulls_extremes_toward_half():
    assert _shrink_and_clamp(0.5) == pytest.approx(0.5)
    assert _shrink_and_clamp(0.9) == pytest.approx(0.9 * (1 - SHRINK_ALPHA) + 0.5 * SHRINK_ALPHA)
    # Shrinkage alone keeps inputs in [0,1] within the contract range.
    assert _shrink_and_clamp(1.0) == pytest.approx(1.0 * (1 - SHRINK_ALPHA) + 0.5 * SHRINK_ALPHA)
    assert _shrink_and_clamp(0.0) == pytest.approx(0.0 * (1 - SHRINK_ALPHA) + 0.5 * SHRINK_ALPHA)


def test_clamp_engages_only_on_out_of_band_input():
    # Inputs outside [0,1] (shouldn't happen normally) get clamped to [0.01, 0.99].
    assert _shrink_and_clamp(1.5) == pytest.approx(0.99)
    assert _shrink_and_clamp(-0.5) == pytest.approx(0.01)


def test_market_implied_uses_midprice_when_liquid():
    market = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.42",
        "last_price_dollars": "0.41",
        "volume_24h_fp": str(MIN_VOL_24H * 10),
    }
    p, rationale = _market_implied_prob(market)
    assert p == pytest.approx(0.41)
    assert "midprice" in rationale


def test_market_implied_falls_back_to_last_when_spread_too_wide():
    market = {
        "yes_bid_dollars": "0.10",
        "yes_ask_dollars": "0.10" if MAX_SPREAD == 0 else f"{0.10 + MAX_SPREAD + 0.01}",
        "last_price_dollars": "0.30",
        "volume_24h_fp": str(MIN_VOL_24H * 10),
    }
    p, rationale = _market_implied_prob(market)
    assert p == pytest.approx(0.30)
    assert "last trade" in rationale


def test_market_implied_falls_back_to_last_when_no_volume():
    market = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.42",
        "last_price_dollars": "0.30",
        "volume_24h_fp": "0",
    }
    p, _ = _market_implied_prob(market)
    assert p == pytest.approx(0.30)


def test_market_implied_returns_none_when_no_signal():
    market = {
        "yes_bid_dollars": "0",
        "yes_ask_dollars": "0",
        "last_price_dollars": "0",
        "volume_24h_fp": "0",
    }
    p, rationale = _market_implied_prob(market)
    assert p is None
    assert "no price signal" in rationale


def test_market_implied_handles_garbage_strings():
    market = {
        "yes_bid_dollars": None,
        "yes_ask_dollars": "",
        "last_price_dollars": "not-a-number",
        "volume_24h_fp": "0",
    }
    p, _ = _market_implied_prob(market)
    assert p is None


def test_predict_falls_back_when_kalshi_fails():
    with patch("agent.predict.get_market", return_value=None):
        out = predict(_event())
    assert out["p_yes"] == 0.5
    assert "kalshi fetch failed" in out["rationale"]


def test_predict_uses_shrunk_midprice_on_liquid_market():
    market = {
        "yes_bid_dollars": "0.65",
        "yes_ask_dollars": "0.67",
        "last_price_dollars": "0.66",
        "volume_24h_fp": "500",
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    expected = _shrink_and_clamp(0.66)
    assert out["p_yes"] == pytest.approx(expected)


def test_predict_output_always_in_contract_range():
    # Extreme market (yes ~1.0 — should be clamped after shrink)
    market = {
        "yes_bid_dollars": "0.99",
        "yes_ask_dollars": "1.00",
        "last_price_dollars": "0.995",
        "volume_24h_fp": "10000",
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    assert 0.01 <= out["p_yes"] <= 0.99
