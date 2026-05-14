from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from agent.predict import (
    ALPHA_VOL_SCALE,
    MAX_SHRINK_ALPHA,
    MIN_SHRINK_ALPHA,
    NO_ARB_TOL,
    STALE_HOURS,
    _depth_weighted_mid,
    _market_implied_prob,
    _no_arb_violated,
    _shrink_alpha,
    _shrink_and_clamp,
    _staleness_hours,
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


def _fresh_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# ---- _shrink_and_clamp ---------------------------------------------------


def test_shrink_at_half_is_noop():
    assert _shrink_and_clamp(0.5, alpha=0.05) == pytest.approx(0.5)


def test_shrink_pulls_toward_half():
    assert _shrink_and_clamp(0.9, alpha=0.10) == pytest.approx(0.86)


def test_clamp_engages_only_on_out_of_band_input():
    assert _shrink_and_clamp(1.5, alpha=0.0) == pytest.approx(0.99)
    assert _shrink_and_clamp(-0.5, alpha=0.0) == pytest.approx(0.01)


# ---- _shrink_alpha (volume-weighted) -------------------------------------


def test_shrink_alpha_decreases_with_volume():
    a_low = _shrink_alpha(100.0)
    a_med = _shrink_alpha(2_000.0)
    a_high = _shrink_alpha(100_000.0)
    assert a_low > a_med > a_high


def test_shrink_alpha_respects_ceiling_and_floor():
    assert _shrink_alpha(0.0) == pytest.approx(MAX_SHRINK_ALPHA)
    assert _shrink_alpha(10_000_000.0) == pytest.approx(MIN_SHRINK_ALPHA)


def test_shrink_alpha_curve_shape():
    # At scale point, raw alpha is 0.5 before clipping → caps at MAX.
    assert _shrink_alpha(ALPHA_VOL_SCALE) == pytest.approx(MAX_SHRINK_ALPHA)


# ---- _depth_weighted_mid -------------------------------------------------


def test_depth_mid_equal_sizes_matches_plain_mid():
    m = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.60",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
    }
    assert _depth_weighted_mid(m) == pytest.approx(0.50)


def test_depth_mid_bid_heavy_moves_toward_ask():
    # Lots of bid demand → true price closer to ask (0.60), not the plain mid (0.50).
    m = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.60",
        "yes_bid_size_fp": "900",
        "yes_ask_size_fp": "100",
    }
    p = _depth_weighted_mid(m)
    assert p == pytest.approx(0.58)
    assert p > 0.50


def test_depth_mid_ask_heavy_moves_toward_bid():
    m = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.60",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "900",
    }
    p = _depth_weighted_mid(m)
    assert p == pytest.approx(0.42)
    assert p < 0.50


def test_depth_mid_returns_none_when_sizes_zero():
    m = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.60",
        "yes_bid_size_fp": "0",
        "yes_ask_size_fp": "0",
    }
    assert _depth_weighted_mid(m) is None


def test_depth_mid_returns_none_when_bid_zero():
    m = {
        "yes_bid_dollars": "0",
        "yes_ask_dollars": "0.10",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
    }
    assert _depth_weighted_mid(m) is None


# ---- _no_arb_violated ----------------------------------------------------


def test_no_arb_clean_book_passes():
    m = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.42",
        "no_bid_dollars": "0.58",
        "no_ask_dollars": "0.60",
    }
    assert _no_arb_violated(m) is False


def test_no_arb_triggers_when_yes_ask_plus_no_ask_too_low():
    # Buying both sides for less than $1 — guaranteed profit, can't be real.
    m = {
        "yes_ask_dollars": "0.30",
        "no_ask_dollars": "0.30",
        "yes_bid_dollars": "0.20",
        "no_bid_dollars": "0.20",
    }
    assert _no_arb_violated(m) is True


def test_no_arb_triggers_when_yes_bid_plus_no_bid_too_high():
    m = {
        "yes_ask_dollars": "0.90",
        "no_ask_dollars": "0.90",
        "yes_bid_dollars": "0.55",
        "no_bid_dollars": "0.55",
    }
    assert _no_arb_violated(m) is True


def test_no_arb_tolerates_small_deviation():
    m = {
        "yes_ask_dollars": "0.50",
        "no_ask_dollars": "0.50",  # sum = 1.00, OK
    }
    assert _no_arb_violated(m) is False
    m_edge = {
        "yes_ask_dollars": "0.50",
        "no_ask_dollars": str(0.50 - NO_ARB_TOL / 2),  # within tolerance
    }
    assert _no_arb_violated(m_edge) is False


# ---- _staleness_hours ----------------------------------------------------


def test_staleness_returns_none_for_missing():
    assert _staleness_hours({}) is None


def test_staleness_returns_none_for_garbage():
    assert _staleness_hours({"updated_time": "not-a-date"}) is None


def test_staleness_parses_z_suffix():
    one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace(
        "+00:00", "Z"
    )
    age = _staleness_hours({"updated_time": one_hour_ago})
    assert age is not None
    assert 0.9 < age < 1.1


# ---- _market_implied_prob ------------------------------------------------


def test_implied_prob_uses_depth_mid_when_liquid():
    m = {
        "yes_bid_dollars": "0.40",
        "yes_ask_dollars": "0.42",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "last_price_dollars": "0.41",
        "volume_24h_fp": "1000",
    }
    p, rationale = _market_implied_prob(m, arb_violated=False)
    assert p == pytest.approx(0.41)
    assert "depth-mid" in rationale


def test_implied_prob_falls_back_to_last_on_arb_violation():
    m = {
        "yes_ask_dollars": "0.30",
        "no_ask_dollars": "0.30",
        "yes_bid_dollars": "0.20",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "last_price_dollars": "0.50",
        "volume_24h_fp": "1000",
    }
    p, rationale = _market_implied_prob(m, arb_violated=True)
    assert p == pytest.approx(0.50)
    assert "no-arb violation" in rationale


def test_implied_prob_none_when_no_signal():
    m = {
        "yes_bid_dollars": "0",
        "yes_ask_dollars": "0",
        "last_price_dollars": "0",
        "volume_24h_fp": "0",
    }
    p, rationale = _market_implied_prob(m, arb_violated=False)
    assert p is None
    assert "no price signal" in rationale


# ---- predict() end-to-end ------------------------------------------------


def test_predict_falls_back_to_uniform_when_both_market_and_llm_fail():
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast", return_value=None
    ):
        out = predict(_event())
    assert out["p_yes"] == 0.5
    assert "kalshi fetch failed" in out["rationale"]
    assert "LLM unavailable" in out["rationale"]


def test_predict_uses_llm_when_no_market_data():
    # No "grounded" marker in rationale → speculative shrink (α=0.15).
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast", return_value=(0.72, "base rate ~70%")
    ):
        out = predict(_event())
    expected = 0.72 * 0.85 + 0.5 * 0.15
    assert out["p_yes"] == pytest.approx(expected)
    assert "LLM" in out["rationale"]
    assert "speculative" in out["rationale"]
    assert "base rate" in out["rationale"]


def test_predict_grounded_llm_gets_less_shrinkage():
    # Rationale mentions "search" → grounded shrink (α=0.05).
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast",
        return_value=(0.95, "web search found Reuters article confirming"),
    ):
        out = predict(_event())
    expected = 0.95 * 0.95 + 0.5 * 0.05
    assert out["p_yes"] == pytest.approx(expected)
    assert "grounded" in out["rationale"]


def test_predict_speculative_llm_gets_more_shrinkage():
    grounded_p = None
    speculative_p = None
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast",
        return_value=(0.95, "based on general knowledge of similar events"),
    ):
        speculative_p = predict(_event())["p_yes"]
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast",
        return_value=(0.95, "as of today's news search"),
    ):
        grounded_p = predict(_event())["p_yes"]
    # Both shrink the 0.95 toward 0.5, but speculative pulls harder.
    assert grounded_p > speculative_p
    assert grounded_p < 0.95


def test_predict_uses_llm_when_no_price_signal():
    # Book has zero everywhere → _market_implied_prob returns None.
    market = {
        "yes_bid_dollars": "0",
        "yes_ask_dollars": "0",
        "last_price_dollars": "0",
        "volume_24h_fp": "0",
    }
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.llm_forecast", return_value=(0.18, "rare event")
    ):
        out = predict(_event())
    # 0.18 shrunk toward 0.5 with speculative α=0.15 → 0.18*0.85 + 0.5*0.15 = 0.228
    assert out["p_yes"] == pytest.approx(0.18 * 0.85 + 0.5 * 0.15)
    assert "no price signal" in out["rationale"]
    assert "LLM" in out["rationale"]


def test_predict_does_not_call_llm_when_market_is_usable():
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
        "agent.predict.llm_forecast"
    ) as llm_mock:
        predict(_event())
    llm_mock.assert_not_called()


def test_predict_liquid_market_with_fresh_book():
    market = {
        "yes_bid_dollars": "0.65",
        "yes_ask_dollars": "0.67",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.33",
        "no_ask_dollars": "0.35",
        "last_price_dollars": "0.66",
        "volume_24h_fp": "5000",
        "updated_time": _fresh_now_iso(),
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    # depth-mid = 0.66 (equal sizes), shrunk with α ≈ 200/5200 ≈ 0.038
    alpha = _shrink_alpha(5000)
    expected = _shrink_and_clamp(0.66, alpha=alpha)
    assert out["p_yes"] == pytest.approx(expected)


def test_predict_applies_extra_shrinkage_on_stale_book(monkeypatch):
    # When the staleness check is enabled, a 6h+ old book should pull
    # harder toward 0.5 than a fresh one.
    monkeypatch.setattr("agent.predict.APPLY_STALENESS", True)
    old = (
        (datetime.now(UTC) - timedelta(hours=STALE_HOURS + 2))
        .isoformat()
        .replace("+00:00", "Z")
    )
    market = {
        "yes_bid_dollars": "0.65",
        "yes_ask_dollars": "0.67",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.33",
        "no_ask_dollars": "0.35",
        "last_price_dollars": "0.66",
        "volume_24h_fp": "5000",
        "updated_time": old,
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    assert "stale book" in out["rationale"]
    fresh_alpha = _shrink_alpha(5000)
    fresh_expected = _shrink_and_clamp(0.66, alpha=fresh_alpha)
    assert out["p_yes"] < fresh_expected  # closer to 0.5 from above


def test_predict_does_not_apply_staleness_by_default():
    # With APPLY_STALENESS=False (the default), an old updated_time is ignored.
    old = (datetime.now(UTC) - timedelta(hours=240)).isoformat().replace("+00:00", "Z")
    market = {
        "yes_bid_dollars": "0.65",
        "yes_ask_dollars": "0.67",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.33",
        "no_ask_dollars": "0.35",
        "last_price_dollars": "0.66",
        "volume_24h_fp": "5000",
        "updated_time": old,
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    assert "stale book" not in out["rationale"]


def test_predict_output_always_in_contract_range():
    market = {
        "yes_bid_dollars": "0.99",
        "yes_ask_dollars": "1.00",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.00",
        "no_ask_dollars": "0.01",
        "last_price_dollars": "0.995",
        "volume_24h_fp": "100000",
        "updated_time": _fresh_now_iso(),
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    assert 0.01 <= out["p_yes"] <= 0.99
