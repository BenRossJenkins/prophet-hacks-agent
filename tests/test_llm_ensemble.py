from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.llm import llm_forecast_ensemble


def test_ensemble_returns_none_when_all_members_fail():
    with patch("agent.llm.llm_forecast", return_value=None):
        out = llm_forecast_ensemble({"title": "x"}, models=("a", "b", "c"))
    assert out is None


def test_ensemble_returns_median_of_three():
    # Three members return 0.20, 0.50, 0.80 → median 0.50.
    rationales = iter([(0.20, "low"), (0.50, "mid"), (0.80, "high")])
    with patch("agent.llm.llm_forecast", side_effect=lambda *a, **kw: next(rationales)):
        out = llm_forecast_ensemble({"title": "x"}, models=("a", "b", "c"))
    assert out is not None
    p, rationale = out
    assert p == pytest.approx(0.50)
    assert "median=0.500" in rationale


def test_ensemble_handles_partial_failure():
    # One model fails (returns None), two succeed at 0.4 and 0.6 → median 0.5
    rationales = iter([None, (0.40, "model A"), (0.60, "model B")])
    with patch("agent.llm.llm_forecast", side_effect=lambda *a, **kw: next(rationales)):
        out = llm_forecast_ensemble({"title": "x"}, models=("a", "b", "c"))
    assert out is not None
    p, _ = out
    assert p == pytest.approx(0.50)


def test_ensemble_single_member_short_circuits():
    # When only one model is configured, ensemble should just call llm_forecast once.
    with patch("agent.llm.llm_forecast", return_value=(0.42, "single")) as m:
        out = llm_forecast_ensemble({"title": "x"}, models=("a",))
    assert out == (0.42, "single")
    assert m.call_count == 1


def test_ensemble_empty_model_list_returns_none():
    out = llm_forecast_ensemble({"title": "x"}, models=())
    assert out is None
