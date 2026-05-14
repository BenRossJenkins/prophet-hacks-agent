from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.priors import LLM_DENIED_CATEGORIES, category_prior, llm_allowed_for


def test_weather_is_denied():
    assert llm_allowed_for("Climate and Weather") is False


def test_crypto_is_denied():
    assert llm_allowed_for("Crypto") is False


def test_financials_is_allowed():
    # Financials is dominated by IPO/CEO speculation — knowledge questions
    # where LLM-with-web-search performs reasonably. Not denied.
    assert llm_allowed_for("Financials") is True


def test_other_categories_allowed():
    for cat in ("Politics", "Sports", "Entertainment", "Economics", "World"):
        assert llm_allowed_for(cat) is True


def test_denied_set_is_explicit():
    assert LLM_DENIED_CATEGORIES == frozenset({"Climate and Weather", "Crypto"})


def test_category_prior_returns_none_by_default():
    # Phase 4 will populate handlers; for now everything returns None.
    assert category_prior({"category": "Climate and Weather", "title": "x"}) is None
    assert category_prior({"category": "Politics", "title": "x"}) is None


# ---- predict() integration -----------------------------------------------


def _event(category: str, market_ticker: str = "TEST") -> dict:
    return {
        "event_ticker": "TEST-EVT",
        "market_ticker": market_ticker,
        "title": "test",
        "category": category,
        "close_time": "2026-12-31T23:59:59Z",
    }


def test_predict_skips_llm_for_denied_category():
    from agent.predict import predict

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=None
    ), patch("agent.predict.llm_forecast") as llm_mock:
        out = predict(_event("Climate and Weather"))
    assert out["p_yes"] == 0.5
    assert "LLM gated" in out["rationale"]
    llm_mock.assert_not_called()


def test_predict_uses_llm_for_allowed_category():
    from agent.predict import predict

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast", return_value=(0.72, "base rate")
    ):
        out = predict(_event("Politics"))
    # Output shrunk by speculative α=0.15 (rationale lacks "grounded" markers).
    assert out["p_yes"] == pytest.approx(0.72 * 0.85 + 0.5 * 0.15)
    assert "LLM" in out["rationale"]


def test_predict_prefers_category_prior_over_llm():
    from agent.predict import predict

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=(0.85, "NWS forecast")
    ), patch("agent.predict.llm_forecast") as llm_mock:
        out = predict(_event("Climate and Weather"))
    assert out["p_yes"] == pytest.approx(0.85)
    assert "prior" in out["rationale"]
    assert "NWS forecast" in out["rationale"]
    llm_mock.assert_not_called()


def test_predict_prior_can_override_llm_gate():
    # Even on a denied category, a prior should be used (it's the whole point).
    from agent.predict import predict

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=(0.3, "from forecast")
    ):
        out = predict(_event("Climate and Weather"))
    assert out["p_yes"] == pytest.approx(0.3)
    assert "prior" in out["rationale"]
