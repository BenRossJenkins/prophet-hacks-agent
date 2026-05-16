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
    # Pick volumes that land in the monotonic region (between MAX clamp and MIN clamp).
    a_low = _shrink_alpha(5_000.0)
    a_med = _shrink_alpha(20_000.0)
    a_high = _shrink_alpha(200_000.0)
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
        "agent.predict.llm_forecast_ensemble", return_value=None
    ):
        out = predict(_event())
    assert out["p_yes"] == 0.5
    assert "kalshi fetch failed" in out["rationale"]
    assert "LLM unavailable" in out["rationale"]


def test_predict_uses_llm_when_no_market_data():
    # No "grounded" marker in rationale → speculative shrink (α=0.15).
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble", return_value=(0.72, "base rate ~70%")
    ):
        out = predict(_event())
    expected = 0.72 * 0.85 + 0.5 * 0.15
    assert out["p_yes"] == pytest.approx(expected)
    assert "LLM" in out["rationale"]
    assert "speculative" in out["rationale"]
    assert "base rate" in out["rationale"]


def test_predict_grounded_llm_gets_less_shrinkage():
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble",
        return_value=(0.95, "web search found Reuters article confirming"),
    ):
        out = predict(_event())
    expected = 0.95 * 0.95 + 0.5 * 0.05
    assert out["p_yes"] == pytest.approx(expected)
    assert "grounded" in out["rationale"]


def test_predict_speculative_llm_gets_more_shrinkage():
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble",
        return_value=(0.95, "based on general knowledge of similar events"),
    ):
        speculative_p = predict(_event())["p_yes"]
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble",
        return_value=(0.95, "as of today's news search"),
    ):
        grounded_p = predict(_event())["p_yes"]
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
        "agent.predict.llm_forecast_ensemble", return_value=(0.18, "rare event")
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
        "agent.predict.llm_forecast_ensemble"
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


# ---- Tail-market triage -----------------------------------------------


def _tail_high_market(vol: float = 1000.0) -> dict:
    return {
        "yes_bid_dollars": "0.96",
        "yes_ask_dollars": "0.98",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.02",
        "no_ask_dollars": "0.04",
        "last_price_dollars": "0.97",
        "volume_24h_fp": str(vol),
    }


def _tail_low_market(vol: float = 1000.0) -> dict:
    return {
        "yes_bid_dollars": "0.02",
        "yes_ask_dollars": "0.04",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.96",
        "no_ask_dollars": "0.98",
        "last_price_dollars": "0.03",
        "volume_24h_fp": str(vol),
    }


def test_tail_anchor_high_skips_llm_and_polymarket():
    e = _event()
    e["category"] = "Politics"  # on the Polymarket allowlist
    with patch("agent.predict.get_market", return_value=_tail_high_market(vol=1000)), patch(
        "agent.predict.polymarket_quote"
    ) as poly_mock, patch("agent.predict.llm_forecast_ensemble") as llm_mock:
        out = predict(e)
    poly_mock.assert_not_called()
    llm_mock.assert_not_called()
    # depth-mid = 0.97 (equal sizes), returned WITHOUT shrinkage.
    assert out["p_yes"] == pytest.approx(0.97)
    assert "tail-anchor" in out["rationale"]


def test_tail_anchor_low_skips_llm():
    with patch("agent.predict.get_market", return_value=_tail_low_market(vol=1000)), patch(
        "agent.predict.llm_forecast_ensemble"
    ) as llm_mock:
        out = predict(_event())
    llm_mock.assert_not_called()
    assert out["p_yes"] == pytest.approx(0.03)
    assert "tail-anchor" in out["rationale"]


def test_tail_anchor_requires_minimum_volume():
    # Same prices but volume below TAIL_MIN_VOL_24H → triage skipped, normal
    # path shrinks toward 0.5.
    from agent.predict import TAIL_MIN_VOL_24H

    market = _tail_high_market(vol=TAIL_MIN_VOL_24H - 100)
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    # Should NOT be the raw market price — should be shrunk.
    assert "tail-anchor" not in out["rationale"]
    assert out["p_yes"] < 0.97  # shrunk away from raw


def test_tail_anchor_does_not_engage_in_mid_range():
    # Market at 0.50 with high vol: NOT a tail, normal pipeline runs.
    market = {
        "yes_bid_dollars": "0.49",
        "yes_ask_dollars": "0.51",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.49",
        "no_ask_dollars": "0.51",
        "last_price_dollars": "0.50",
        "volume_24h_fp": "5000",
    }
    with patch("agent.predict.get_market", return_value=market):
        out = predict(_event())
    assert "tail-anchor" not in out["rationale"]


# ---- Polymarket blend --------------------------------------------------


def _politics_event() -> dict:
    e = _event()
    e["category"] = "Politics"
    return e


def test_predict_blends_kalshi_and_polymarket_volume_weighted():
    market = {
        "yes_bid_dollars": "0.30",
        "yes_ask_dollars": "0.32",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.68",
        "no_ask_dollars": "0.70",
        "last_price_dollars": "0.31",
        "volume_24h_fp": "1000",  # kalshi $1000
    }
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.50, 9000.0, "poly p=0.500 vol24h=$9000 (match=0.80)"),
    ):
        out = predict(_politics_event())
    # Kalshi p≈0.31 weight $1000, poly p=0.50 weight $9000.
    # Blended ≈ (0.31 * 1000 + 0.50 * 9000) / 10000 = 0.481
    # Then shrunk by α from $1000 vol.
    assert 0.40 < out["p_yes"] < 0.50
    assert "blend" in out["rationale"]
    assert "poly" in out["rationale"].lower()


def test_predict_uses_polymarket_when_kalshi_illiquid():
    # Kalshi book with no signal at all → polymarket should rescue.
    market = {
        "yes_bid_dollars": "0",
        "yes_ask_dollars": "0",
        "last_price_dollars": "0",
        "volume_24h_fp": "0",
    }
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.72, 5000.0, "poly p=0.720 vol24h=$5000 (match=0.90)"),
    ):
        out = predict(_politics_event())
    assert 0.70 < out["p_yes"] < 0.75
    assert "polymarket-only" in out["rationale"].lower()


def test_predict_uses_polymarket_when_kalshi_fetch_fails():
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.65, 3000.0, "poly p=0.650 vol24h=$3000 (match=0.75)"),
    ):
        out = predict(_politics_event())
    assert 0.63 < out["p_yes"] < 0.67
    assert "polymarket-only" in out["rationale"].lower()


def test_predict_skips_polymarket_for_off_allowlist_categories():
    # Test category isn't on POLYMARKET_CATEGORIES, so poly should never be called.
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
        "agent.predict.polymarket_quote"
    ) as poly_mock:
        predict(_event())  # category="Test", not on allowlist
    poly_mock.assert_not_called()


def test_predict_falls_through_when_kalshi_fails_and_poly_no_match():
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.polymarket_quote", return_value=None
    ), patch("agent.predict.category_prior", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble", return_value=(0.4, "base rate")
    ):
        out = predict(_politics_event())
    # Speculative LLM shrink: 0.4 * 0.85 + 0.5 * 0.15 = 0.415
    assert out["p_yes"] == pytest.approx(0.4 * 0.85 + 0.5 * 0.15)
    assert "LLM" in out["rationale"]


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
