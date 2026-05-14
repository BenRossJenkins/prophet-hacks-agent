from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from agent.financials import (
    _asset_for,
    crypto_prior,
    lognormal_p,
    parse_market,
)


# ---- asset mapping -------------------------------------------------------


def test_asset_for_btc():
    assert _asset_for("KXBTCD-26MAY1413") == "BTC-USD"
    assert _asset_for("KXBTC-26MAY1413") == "BTC-USD"
    assert _asset_for("KXBTCMAXY-26DEC31") == "BTC-USD"


def test_asset_for_eth():
    assert _asset_for("KXETHD-26MAY1413") == "ETH-USD"


def test_asset_for_unknown():
    assert _asset_for("KXFOO-26MAY") is None
    assert _asset_for("") is None


# ---- parser --------------------------------------------------------------


def _event(
    ticker_prefix: str = "KXBTCD",
    market_suffix: str = "-T90299.99",
    subtitle: str = "$90,300 or above",
    close_in_hours: float = 1.0,
) -> dict:
    deadline = datetime.now(UTC) + timedelta(hours=close_in_hours)
    return {
        "event_ticker": f"{ticker_prefix}-26MAY1413",
        "market_ticker": f"{ticker_prefix}-26MAY1413{market_suffix}",
        "title": "Bitcoin price on May 14, 2026?",
        "subtitle": subtitle,
        "description": None,
        "category": "Crypto",
        "rules": None,
        "close_time": deadline.isoformat().replace("+00:00", "Z"),
    }


def test_parse_subtitle_above():
    p = parse_market(_event(subtitle="$90,300 or above"))
    assert p is not None
    assert p["asset"] == "BTC-USD"
    assert p["threshold"] == pytest.approx(90300)
    assert p["comparison"] == "above"


def test_parse_subtitle_below():
    p = parse_market(_event(subtitle="$80,000 or below"))
    assert p is not None
    assert p["comparison"] == "below"
    assert p["threshold"] == pytest.approx(80000)


def test_parse_falls_back_to_ticker_threshold():
    p = parse_market(_event(subtitle="", market_suffix="-T90299.99"))
    assert p is not None
    assert p["threshold"] == pytest.approx(90299.99)
    # Default comparison without subtitle is "above"
    assert p["comparison"] == "above"


def test_parse_returns_none_for_unknown_asset():
    assert parse_market(_event(ticker_prefix="KXFOO")) is None


def test_parse_returns_none_without_threshold():
    e = _event(subtitle="", market_suffix="-XYZ")
    assert parse_market(e) is None


# ---- probability model ---------------------------------------------------


def test_lognormal_at_strike_is_half():
    # When spot == strike, sigma > 0 and t > 0, P(S_T > K) → 0.5.
    assert lognormal_p(100.0, 100.0, 0.5, 0.01) == pytest.approx(0.5)


def test_lognormal_far_above_strike_is_near_one():
    # Spot $100k, strike $80k, low time → almost certainly above.
    p = lognormal_p(100_000, 80_000, 0.3, 1 / 365)
    assert p > 0.99


def test_lognormal_far_below_strike_is_near_zero():
    p = lognormal_p(80_000, 100_000, 0.3, 1 / 365)
    assert p < 0.01


def test_lognormal_handles_zero_inputs():
    # Defensive: returns 0.5 rather than ZeroDivisionError.
    assert lognormal_p(0, 100, 0.5, 0.01) == 0.5
    assert lognormal_p(100, 100, 0, 0.01) == 0.5
    assert lognormal_p(100, 100, 0.5, 0) == 0.5


# ---- end-to-end with mocked yfinance ------------------------------------


def test_crypto_prior_above_strike_high_probability():
    with patch("agent.financials._spot_and_vol", return_value=(95000.0, 0.30)):
        out = crypto_prior(_event(subtitle="$90,300 or above", close_in_hours=2))
    assert out is not None
    p, rationale = out
    # Spot ~95k vs threshold 90.3k in 2h is highly likely above.
    assert p > 0.95
    assert "yfinance" in rationale


def test_crypto_prior_below_strike_high_no():
    with patch("agent.financials._spot_and_vol", return_value=(95000.0, 0.30)):
        out = crypto_prior(_event(subtitle="$120,000 or above", close_in_hours=2))
    assert out is not None
    p, _ = out
    assert p < 0.05  # very unlikely to jump 25k in 2h


def test_crypto_prior_below_compare_flips():
    with patch("agent.financials._spot_and_vol", return_value=(95000.0, 0.30)):
        above = crypto_prior(_event(subtitle="$120,000 or above", close_in_hours=2))
        below = crypto_prior(_event(subtitle="$120,000 or below", close_in_hours=2))
    assert above is not None and below is not None
    assert above[0] + below[0] == pytest.approx(1.0, abs=1e-6)


def test_crypto_prior_returns_none_when_yfinance_fails():
    with patch("agent.financials._spot_and_vol", return_value=None):
        assert crypto_prior(_event()) is None


def test_crypto_prior_returns_none_when_past_deadline():
    with patch("agent.financials._spot_and_vol", return_value=(95000.0, 0.30)):
        assert crypto_prior(_event(close_in_hours=-1)) is None
