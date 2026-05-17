from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.llm import llm_forecast_ensemble


def test_ensemble_returns_none_when_all_members_fail():
    with patch("agent.llm.llm_forecast_full", return_value=None):
        out = llm_forecast_ensemble({"title": "x"}, models=("a", "b", "c"))
    assert out is None


def test_ensemble_returns_median_of_three():
    # Three members return 0.20, 0.50, 0.80 → median 0.50.
    rationales = iter([(0.20, None, "low"), (0.50, None, "mid"), (0.80, None, "high")])
    with patch("agent.llm.llm_forecast_full", side_effect=lambda *a, **kw: next(rationales)):
        out = llm_forecast_ensemble({"title": "x"}, models=("a", "b", "c"))
    assert out is not None
    p, rationale = out
    assert p == pytest.approx(0.50)
    assert "median=0.500" in rationale


def test_ensemble_handles_partial_failure():
    # One model fails (returns None), two succeed at 0.4 and 0.6 → median 0.5
    rationales = iter([None, (0.40, None, "model A"), (0.60, None, "model B")])
    with patch("agent.llm.llm_forecast_full", side_effect=lambda *a, **kw: next(rationales)):
        out = llm_forecast_ensemble({"title": "x"}, models=("a", "b", "c"))
    assert out is not None
    p, _ = out
    assert p == pytest.approx(0.50)


def test_ensemble_single_member_short_circuits():
    # When only one model is configured, ensemble should just call llm_forecast_full once.
    with patch("agent.llm.llm_forecast_full", return_value=(0.42, None, "single")) as m:
        out = llm_forecast_ensemble({"title": "x"}, models=("a",))
    assert out == (0.42, "single")
    assert m.call_count == 1


def test_ensemble_empty_model_list_returns_none():
    out = llm_forecast_ensemble({"title": "x"}, models=())
    assert out is None


def test_shared_search_anchor_runs_first_and_injects_context():
    """Anthropic anchor runs sequentially with search; OpenAI/Gemini get its findings."""
    seen_events: list[dict] = []

    def fake(event, *, model, with_web_search):
        seen_events.append({"model": model, "search": with_web_search, "event": dict(event)})
        return 0.5, None, f"answer-from-{model}"

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        out = llm_forecast_ensemble(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini", "gemini-2.5-flash"),
            with_web_search=True,
        )
    assert out is not None
    # Anthropic model should have been called with search=True.
    anthropic_calls = [c for c in seen_events if c["model"].startswith("claude")]
    assert anthropic_calls and all(c["search"] is True for c in anthropic_calls)
    # Non-anthropic models should NOT have web_search and SHOULD see search_context.
    others = [c for c in seen_events if not c["model"].startswith("claude")]
    assert others
    for c in others:
        assert c["search"] is False
        assert "search_context" in c["event"]
        assert "answer-from-claude" in c["event"]["search_context"]


def test_shared_search_falls_back_to_parallel_when_anchor_fails():
    """If the Anthropic anchor returns None, others still run (with search ON)."""
    seen: list[tuple[str, bool]] = []

    def fake(event, *, model, with_web_search):
        seen.append((model, with_web_search))
        if model.startswith("claude"):
            return None
        return 0.4, None, f"answer-{model}"

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        out = llm_forecast_ensemble(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini", "gemini-2.5-flash"),
            with_web_search=True,
        )
    assert out is not None
    # When anchor failed, the other models should run WITH search on
    # (we never got context to share).
    non_anchor = [s for s in seen if not s[0].startswith("claude")]
    assert non_anchor
    assert all(search is True for _, search in non_anchor)


def test_ensemble_deadline_returns_partial_results():
    """If a vendor hangs past the ensemble deadline, return what arrived."""
    import time as _time

    from agent.llm import llm_forecast_ensemble_full

    def slow(event, *, model, with_web_search):
        if model == "claude-opus-4-7":
            # Anchor returns quickly.
            return 0.5, None, "anchor"
        if model == "gpt-5-mini":
            return 0.6, None, "fast"
        if model == "hangs-forever":
            _time.sleep(30)  # exceeds the patched deadline
            return 0.7, None, "should never reach this"
        return None

    with patch("agent.llm.llm_forecast_full", side_effect=slow), patch(
        "agent.llm.ENSEMBLE_HARD_DEADLINE_SECONDS", 0.5
    ):
        out = llm_forecast_ensemble_full(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini", "hangs-forever"),
            with_web_search=True,
        )
    assert out is not None
    p, _, rationale = out
    # The hung vendor was abandoned; only anchor + fast vendor contributed.
    assert "claude-opus-4-7" in rationale
    assert "gpt-5-mini" in rationale
    # Median of {0.5, 0.6} = 0.55.
    assert p == pytest.approx(0.55)


# ---- exception-degradation tests ----------------------------------------
#
# A returned `None` from a vendor (handled-failure) and a raised exception
# from a vendor (uncaught-failure) are distinct paths through the ensemble
# code. The existing tests cover the None path; these cover the exception
# path. The contract: one or two vendors raising must not bring down the
# whole ensemble — the other vendors' answers still flow through and
# produce a valid median.


def test_ensemble_parallel_vendor_raising_does_not_kill_ensemble():
    """If a parallel vendor RAISES (not returns None), the other vendors'
    results still drive the median."""

    def fake(event, *, model, with_web_search):
        if model == "claude-opus-4-7":
            # Anchor succeeds.
            return 0.5, None, "anchor ok"
        if model == "gpt-5-mini":
            raise RuntimeError("simulated OpenAI rate limit / 500")
        if model == "gemini-2.5-flash":
            return 0.7, None, "gemini ok"
        return None

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        out = llm_forecast_ensemble(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini", "gemini-2.5-flash"),
            with_web_search=True,
        )
    # Anchor 0.5 + gemini 0.7 → median 0.6.
    assert out is not None
    p, _ = out
    assert p == pytest.approx(0.60)


def test_ensemble_anchor_raising_falls_back_to_all_parallel():
    """If the search anchor RAISES (not returns None), the rest of the
    ensemble must still produce a result. Without a defensive try/except
    around the anchor call, an Anthropic library exception would propagate
    out of llm_forecast_ensemble and crash /predict."""

    def fake(event, *, model, with_web_search):
        if model == "claude-opus-4-7":
            raise RuntimeError("simulated Anthropic 503")
        return 0.55, None, f"answer-{model}"

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        out = llm_forecast_ensemble(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini", "gemini-2.5-flash"),
            with_web_search=True,
        )
    # Anchor crashed; the other two ran (with search on, since no context).
    # Both returned 0.55 → median 0.55.
    assert out is not None
    p, _ = out
    assert p == pytest.approx(0.55)


def test_ensemble_all_vendors_raising_returns_none():
    """When every vendor raises, return None so the caller can retry
    without web search (a separate code path)."""

    def fake(event, *, model, with_web_search):
        raise RuntimeError(f"simulated total failure on {model}")

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        out = llm_forecast_ensemble(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini", "gemini-2.5-flash"),
            with_web_search=True,
        )
    assert out is None


def test_ensemble_single_vendor_raising_returns_none():
    """For a 1-vendor ensemble, the short-circuit path must also handle
    a raised exception."""

    def fake(event, *, model, with_web_search):
        raise RuntimeError("single-vendor crash")

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        out = llm_forecast_ensemble(
            {"title": "x"}, models=("claude-opus-4-7",), with_web_search=True,
        )
    assert out is None


def test_shared_search_disabled_runs_all_parallel():
    seen: list[tuple[str, bool]] = []

    def fake(event, *, model, with_web_search):
        seen.append((model, with_web_search))
        return 0.5, None, "x"

    with patch("agent.llm.llm_forecast_full", side_effect=fake):
        llm_forecast_ensemble(
            {"title": "x"},
            models=("claude-opus-4-7", "gpt-5-mini"),
            with_web_search=False,
        )
    # With search OFF entirely, no sequencing — all calls go with search=False.
    assert seen
    assert all(search is False for _, search in seen)
