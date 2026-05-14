from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.independent import independent_forecast


def _event(category: str = "Sports") -> dict:
    return {
        "event_ticker": "TEST-EVT",
        "market_ticker": "TEST-MKT",
        "title": "Test market",
        "category": category,
        "close_time": "2026-12-31T23:59:59Z",
    }


def test_uses_prior_when_available():
    with patch(
        "agent.independent.category_prior", return_value=(0.78, "NWS forecast says 78%")
    ), patch("agent.independent.llm_forecast_ensemble") as llm_mock:
        p, rationale, conf = independent_forecast(_event("Climate and Weather"))
    assert p == pytest.approx(0.78)
    assert conf == "high"
    assert "prior" in rationale
    llm_mock.assert_not_called()


def test_falls_back_to_llm_when_no_prior():
    with patch("agent.independent.category_prior", return_value=None), patch(
        "agent.independent.llm_forecast_ensemble",
        return_value=(0.62, "web search found Reuters article"),
    ):
        p, rationale, conf = independent_forecast(_event("Politics"))
    assert p == pytest.approx(0.62)
    assert conf == "medium"
    assert "search" in rationale


def test_llm_without_grounding_is_low_confidence():
    with patch("agent.independent.category_prior", return_value=None), patch(
        "agent.independent.llm_forecast_ensemble",
        return_value=(0.55, "general knowledge"),
    ):
        p, _, conf = independent_forecast(_event("Politics"))
    assert p == pytest.approx(0.55)
    assert conf == "low"


def test_denied_category_with_no_prior_returns_none_confidence():
    with patch("agent.independent.category_prior", return_value=None), patch(
        "agent.independent.llm_forecast_ensemble"
    ) as llm_mock:
        p, _, conf = independent_forecast(_event("Climate and Weather"))
    assert p == 0.5
    assert conf == "none"
    llm_mock.assert_not_called()


def test_all_failures_returns_none_confidence():
    with patch("agent.independent.category_prior", return_value=None), patch(
        "agent.independent.llm_forecast_ensemble", return_value=None
    ):
        p, _, conf = independent_forecast(_event("Politics"))
    assert p == 0.5
    assert conf == "none"


def test_output_is_clamped_to_contract_range():
    with patch(
        "agent.independent.category_prior", return_value=(1.5, "buggy prior returns >1")
    ):
        p, _, conf = independent_forecast(_event("Politics"))
    assert p == 0.99
    assert conf == "high"

    with patch(
        "agent.independent.category_prior", return_value=(-0.2, "buggy prior returns <0")
    ):
        p, _, _ = independent_forecast(_event("Politics"))
    assert p == 0.01
