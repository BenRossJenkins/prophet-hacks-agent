from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.llm import _aggregate_probabilities, parse_response_full
from agent.predict import (
    MULTI_LLM_SHRINK,
    EventRequest,
    _estimate_winners_count,
    _is_multi_outcome,
    _uniform_prior,
    predict,
)


def _multi_event(outcomes: list[str], **overrides) -> dict:
    base = {
        "event_ticker": "EUR-2026",
        "market_ticker": "EUR-2026",
        "title": "Which acts will finish in the top 5 at Eurovision 2026?",
        "category": "Entertainment",
        "close_time": "2026-05-30T23:00:00Z",
        "outcomes": outcomes,
    }
    base.update(overrides)
    return base


def _binary_event(**overrides) -> dict:
    base = {
        "event_ticker": "TEST-EVT",
        "market_ticker": "TEST-MKT",
        "title": "Will Cleveland beat Detroit on May 15?",
        "category": "Sports",
        "close_time": "2026-05-15T20:00:00Z",
        "outcomes": ["Cleveland", "Detroit"],
    }
    base.update(overrides)
    return base


# ---- detection ----


def test_is_multi_outcome_three_plus():
    e = EventRequest(**_multi_event(["A", "B", "C"]))
    assert _is_multi_outcome(e) is True


def test_is_multi_outcome_binary_returns_false():
    e = EventRequest(**_binary_event())
    assert _is_multi_outcome(e) is False


def test_is_multi_outcome_no_outcomes_returns_false():
    # Legacy event missing the outcomes field entirely.
    raw = _binary_event()
    raw.pop("outcomes")
    e = EventRequest(**raw)
    assert _is_multi_outcome(e) is False


# ---- winners-count estimator ----


def test_top_k_parsed_from_title():
    e = EventRequest(**_multi_event(["A"] * 35, title="Which acts finish in the top 5?"))
    assert _estimate_winners_count(e) == 5


def test_top_k_parsed_from_rules():
    e = EventRequest(
        **_multi_event(
            ["A"] * 10,
            title="Eurovision podium",
            rules="The top 3 finishers resolve YES.",
        )
    )
    assert _estimate_winners_count(e) == 3


def test_default_single_winner():
    e = EventRequest(**_multi_event(["A"] * 6, title="Who will win Female Artist?"))
    assert _estimate_winners_count(e) == 1


# ---- uniform prior ----


def test_uniform_prior_single_winner():
    e = EventRequest(**_multi_event(["A"] * 10, title="Who will win?"))
    assert _uniform_prior(e) == pytest.approx(0.1)


def test_uniform_prior_top_k():
    e = EventRequest(**_multi_event(["A"] * 35, title="top 5 finishers"))
    assert _uniform_prior(e) == pytest.approx(5 / 35)


# ---- parse_response_full ----


def test_parse_full_extracts_probabilities():
    raw = """
    {
      "p_yes": 0.18,
      "probabilities": [
        {"market": "Albania", "probability": 0.18},
        {"market": "Armenia", "probability": 0.05}
      ],
      "rationale": "based on betting markets"
    }
    """
    out = parse_response_full(raw)
    assert out is not None
    p, probs, rationale = out
    assert p == pytest.approx(0.18)
    assert probs is not None
    assert len(probs) == 2
    assert probs[0] == {"market": "Albania", "probability": 0.18}
    assert "betting markets" in rationale


def test_parse_full_no_probabilities_returns_none_for_field():
    raw = '{"p_yes": 0.5, "rationale": "binary"}'
    out = parse_response_full(raw)
    assert out is not None
    p, probs, rationale = out
    assert p == 0.5
    assert probs is None


def test_parse_full_ignores_malformed_probabilities():
    raw = """
    {
      "p_yes": 0.2,
      "probabilities": [
        {"market": "A", "probability": 1.5},
        {"market": "B"},
        {"probability": 0.3}
      ],
      "rationale": "x"
    }
    """
    out = parse_response_full(raw)
    assert out is not None
    _, probs, _ = out
    # All three entries are invalid; result should be None (not empty list).
    assert probs is None


# ---- ensemble aggregation ----


def test_aggregate_probabilities_takes_median_per_outcome():
    per_model = [
        ("m1", [{"market": "A", "probability": 0.2}, {"market": "B", "probability": 0.5}]),
        ("m2", [{"market": "A", "probability": 0.3}, {"market": "B", "probability": 0.4}]),
        ("m3", [{"market": "A", "probability": 0.4}, {"market": "B", "probability": 0.6}]),
    ]
    out = _aggregate_probabilities(per_model, outcomes=["A", "B"])
    assert out is not None
    by_market = {e["market"]: e["probability"] for e in out}
    assert by_market["A"] == pytest.approx(0.3)
    assert by_market["B"] == pytest.approx(0.5)


def test_aggregate_returns_none_when_too_few_contributors():
    # 1 of 4 models with a distribution → not enough.
    per_model = [
        ("m1", [{"market": "A", "probability": 0.1}]),
        ("m2", None),
        ("m3", None),
        ("m4", None),
    ]
    out = _aggregate_probabilities(per_model, outcomes=["A", "B"])
    assert out is None


def test_aggregate_skips_outcomes_no_vendor_covered():
    per_model = [
        ("m1", [{"market": "A", "probability": 0.5}]),
        ("m2", [{"market": "A", "probability": 0.6}]),
    ]
    out = _aggregate_probabilities(per_model, outcomes=["A", "B"])
    assert out is not None
    assert len(out) == 1
    assert out[0]["market"] == "A"


# ---- predict() end-to-end for multi-outcome ----


def test_predict_multi_outcome_skips_market_anchor():
    # Even if get_market would return data, multi-outcome should never touch it.
    eurovision_outcomes = [f"Country{i}" for i in range(35)]
    event = _multi_event(eurovision_outcomes, title="Top 5 acts at Eurovision 2026?")
    with patch("agent.predict.get_market") as market_mock, patch(
        "agent.predict.llm_forecast_ensemble_full",
        return_value=(0.20, None, "raw rationale"),
    ):
        out = predict(event)
    market_mock.assert_not_called()
    # raw=0.20, uniform=5/35≈0.143, shrink=0.30
    # shrunk = 0.20*0.70 + 0.143*0.30 ≈ 0.183
    assert out["p_yes"] == pytest.approx(0.20 * 0.70 + (5 / 35) * 0.30)
    assert "multi-outcome" in out["rationale"]
    assert "top-5" in out["rationale"]


def test_predict_multi_outcome_passes_probabilities_through():
    outcomes = ["A", "B", "C", "D"]
    event = _multi_event(outcomes, title="Who will win?")
    with patch("agent.predict.get_market") as market_mock, patch(
        "agent.predict.llm_forecast_ensemble_full",
        return_value=(
            0.40,
            [
                {"market": "A", "probability": 0.40},
                {"market": "B", "probability": 0.30},
                {"market": "C", "probability": 0.20},
                {"market": "D", "probability": 0.10},
            ],
            "with distribution",
        ),
    ):
        # Call the FastAPI route via async to verify probabilities make it to
        # the response. We can simulate by going through PredictionResponse.
        from agent.predict import _forecast
        from agent.predict import EventRequest as ER

        resp = _forecast(ER(**event))
    market_mock.assert_not_called()
    assert resp.probabilities is not None
    assert len(resp.probabilities) == 4
    by_market = {p.market: p.probability for p in resp.probabilities}
    assert by_market["A"] == pytest.approx(0.40)
    assert by_market["D"] == pytest.approx(0.10)


def test_predict_multi_outcome_fallback_to_uniform_when_llm_dies():
    outcomes = ["A"] * 10
    event = _multi_event(outcomes, title="Who will win?")
    with patch("agent.predict.llm_forecast_ensemble_full", return_value=None):
        out = predict(event)
    # Single-winner uniform with N=10 → 0.1
    assert out["p_yes"] == pytest.approx(0.1)
    assert "LLM unavailable" in out["rationale"]


def test_binary_event_still_uses_market_anchor_path():
    # Binary events with outcomes=[X, Y] should NOT enter the multi-outcome path.
    market = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.42",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.58",
        "no_ask_dollars": "0.60",
        "last_price_dollars": "0.41",
        "volume_24h_fp": "1000",
    }
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.llm_forecast_ensemble_full"
    ) as multi_mock:
        out = predict(_binary_event())
    multi_mock.assert_not_called()
    # Should produce a market-anchored result.
    assert 0.35 < out["p_yes"] < 0.45
