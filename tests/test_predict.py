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


def test_implied_prob_trusts_tight_spread_at_zero_volume():
    """Regression: a settled-direction market with bid/ask pinned at 0.99/1.00
    and zero 24h volume still carries the market's signal. We trust the
    depth-mid (~0.995) rather than rejecting as illiquid."""
    m = {
        "yes_bid_dollars": "0.99",
        "yes_ask_dollars": "1.00",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "last_price_dollars": "0.0",
        "volume_24h_fp": "0",
    }
    p, rationale = _market_implied_prob(m, arb_violated=False)
    assert p is not None
    assert p == pytest.approx(0.995, abs=0.005)
    assert "tight-spread" in rationale


def test_implied_prob_still_requires_volume_for_wide_spread():
    """Regression guard: wide spread (e.g. 0.10/0.40) at zero volume is
    still rejected because the midprice is uninformative."""
    m = {
        "yes_bid_dollars": "0.10",
        "yes_ask_dollars": "0.40",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "last_price_dollars": "0",
        "volume_24h_fp": "0",
    }
    p, rationale = _market_implied_prob(m, arb_violated=False)
    # Spread 0.30 > TIGHT_SPREAD_FOR_LOW_VOL → still requires volume
    assert p is None
    assert "no price signal" in rationale


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


def test_predict_retries_ensemble_without_search_on_total_failure():
    """When the first ensemble call returns None (all vendors failed),
    retry once with with_web_search=False before falling to 0.5."""
    # First call (with search): None. Second call (no search): returns a value.
    call_args: list[dict] = []

    def fake_ensemble(event_d, **kwargs):
        call_args.append(kwargs)
        if kwargs.get("with_web_search", True) is True:
            return None
        return (0.65, "fallback no-search response")

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble", side_effect=fake_ensemble
    ), patch("agent.predict.category_prior", return_value=None):
        out = predict(_event())
    # Verify the retry path fired
    assert len(call_args) == 2
    assert call_args[0].get("with_web_search", True) is True
    assert call_args[1].get("with_web_search") is False
    # Verify the retry's response landed in the final prediction
    assert "retry, no-search" in out["rationale"]
    # Verify the actual probability came from the retry call (raw 0.65 → shrunk)
    assert out["p_yes"] != 0.5


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
    # p=0.95 grounded: tail-aware shrink. distance=0.45, extra=0.10, α=0.15.
    # shrunk = 0.85*0.95 + 0.15*0.5 = 0.8825
    assert out["p_yes"] == pytest.approx(0.8825)
    assert "grounded" in out["rationale"]


def test_predict_decisive_marker_minimises_shrinkage():
    """When the rationale describes a confirmed outcome, trust the LLM nearly fully."""
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble",
        return_value=(0.02, "Detroit defeated Cleveland 115-94 in Game 6; Cleveland was eliminated"),
    ):
        out = predict(_event())
    # Decisive tier α_base=0.02. p=0.02 distance=0.48, tail_extra=2.0*0.08=0.16,
    # α=0.18; shrunk = 0.82*0.02 + 0.18*0.5 = 0.0164 + 0.09 = 0.1064.
    # Without decisive detection it'd be speculative α_base=0.15 → ~0.20.
    assert "decisive" in out["rationale"]
    # Final probability stays close to the LLM's raw signal.
    assert out["p_yes"] < 0.15


def test_decisive_beats_grounded_at_extreme_tail():
    """Decisive marker present + grounded marker present → still decisive (lower shrink)."""
    decisive_rationale = "according to multiple sources, Cleveland is already eliminated"
    grounded_only_rationale = "according to multiple sources, Cleveland is the underdog"
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble", return_value=(0.05, decisive_rationale)
    ):
        decisive_p = predict(_event())["p_yes"]
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble", return_value=(0.05, grounded_only_rationale)
    ):
        grounded_p = predict(_event())["p_yes"]
    # Decisive shrinkage is smaller → final stays closer to the raw 0.05.
    assert decisive_p < grounded_p


def test_decisive_marker_in_mid_band_downgrades_to_grounded():
    """Regression for the false-positive case: a rationale with a decisive
    marker (e.g. "did not win") but a mid-band probability should NOT use
    the decisive tier — that combination almost certainly means the marker
    matched a counterfactual phrase ("did not win in 2024 but might in 2026")
    rather than a confirmed outcome.
    """
    # Mid-band p=0.55 with a decisive-marker phrase that's clearly counterfactual.
    counterfactual_rationale = (
        "Team A has not yet won the title this season but is well-positioned"
    )
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.llm_forecast_ensemble", return_value=(0.55, counterfactual_rationale)
    ):
        out = predict(_event())
    # Should report 'grounded' tier (downgraded from decisive), NOT 'decisive'.
    assert "grounded" in out["rationale"] or "speculative" in out["rationale"]
    assert "decisive" not in out["rationale"]


def test_no_search_retry_forces_speculative_tier():
    """When the initial ensemble fails and we retry without search, the
    resulting forecast should be classified as speculative regardless of
    rationale content. Web-search-less LLM operating on training cutoff
    knowledge alone is by definition speculation; any "grounded" markers
    in the rationale are fabricated citations.
    """
    # First call returns None (ensemble failure), second call (no-search) returns
    # a rationale that WOULD normally trigger 'grounded' or 'decisive'.
    call_count = [0]

    def fake_ensemble(event_d, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return None  # First attempt: fail
        # Retry: return a response whose rationale would mis-classify as grounded
        # if not for the force-speculative override.
        return (0.85, "according to multiple sources, this is very likely")

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=None
    ), patch("agent.predict.llm_forecast_ensemble", side_effect=fake_ensemble):
        out = predict(_event())
    # The no-search retry's rationale should force speculative tier.
    assert "speculative" in out["rationale"]
    assert "(retry, no-search)" in out["rationale"]
    # Speculative shrinkage at p=0.85: distance=0.35 < 0.40 threshold, no extra
    # tail alpha; shrunk = 0.85*0.85 + 0.15*0.5 = 0.7225 + 0.075 = 0.7975.
    assert out["p_yes"] == pytest.approx(0.7975)


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


# ---- Tail-aware LLM shrinkage ---------------------------------------------


def test_tail_shrink_noop_in_central_range():
    """For p near 0.5, behaves like normal linear shrinkage."""
    from agent.predict import _llm_shrink_with_tail

    # At p=0.6, distance=0.10 < 0.40 threshold, no extra alpha.
    out = _llm_shrink_with_tail(0.6, alpha_base=0.05)
    expected = 0.95 * 0.6 + 0.05 * 0.5  # 0.595
    assert out == pytest.approx(expected)


def test_tail_shrink_kicks_in_at_high_tail():
    """At p=0.95, extra shrinkage pulls more aggressively toward 0.5."""
    from agent.predict import _llm_shrink_with_tail

    out = _llm_shrink_with_tail(0.95, alpha_base=0.05)
    # distance=0.45, extra=2.0*0.05=0.10, alpha=0.15
    # shrunk = 0.85 * 0.95 + 0.15 * 0.5 = 0.8825
    assert out == pytest.approx(0.8825)


def test_tail_shrink_kicks_in_at_low_tail():
    """Symmetric: at p=0.05, extra shrinkage pulls toward 0.5 from below."""
    from agent.predict import _llm_shrink_with_tail

    out = _llm_shrink_with_tail(0.05, alpha_base=0.05)
    # distance=0.45, alpha=0.15; shrunk = 0.85 * 0.05 + 0.15 * 0.5 = 0.1175
    assert out == pytest.approx(0.1175)


def test_tail_shrink_caps_alpha():
    """The cap fires only when alpha_base is high enough that the tail boost
    would push alpha past 0.50. With base=0.40 and p=0.99, alpha would be
    0.40 + 0.99·2 - 0.80 = 0.58 → capped at 0.50."""
    from agent.predict import _llm_shrink_with_tail

    out = _llm_shrink_with_tail(0.99, alpha_base=0.40)
    # alpha=0.50, shrunk = 0.50*0.99 + 0.50*0.5 = 0.745
    assert out == pytest.approx(0.745)


def test_tail_shrink_preserves_directional_signal():
    """Even at extreme tail, output is still on the correct side of 0.5."""
    from agent.predict import _llm_shrink_with_tail

    high = _llm_shrink_with_tail(0.99, alpha_base=0.15)
    low = _llm_shrink_with_tail(0.01, alpha_base=0.15)
    assert high > 0.5
    assert low < 0.5


# ---- Safe-band auto-anchor (Polymarket skip) -----------------------------------


def test_safe_band_skips_blend_when_polymarket_agrees():
    """Liquid Kalshi in [0.20, 0.80] + Polymarket agrees within tol → no blend."""
    e = _event()
    e["category"] = "Politics"  # Polymarket-eligible
    e["outcomes"] = ["TeamA", "TeamB"]
    market = {
        "yes_bid_dollars": "0.49",
        "yes_ask_dollars": "0.51",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.49",
        "no_ask_dollars": "0.51",
        "last_price_dollars": "0.50",
        "volume_24h_fp": "20000",  # above SAFE_BAND_MIN_VOL_24H
    }
    # Polymarket at 0.51 → |0.50 - 0.51| = 0.01 < CROSS_VENUE_DISAGREE_TOL (0.03)
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.51, 5000.0, "poly p=0.51"),
    ) as poly_mock:
        out = predict(e)
    # Poly IS called (we check agreement), but result is "skip blend".
    poly_mock.assert_called_once()
    assert "skip blend" in out["rationale"] or "poly agrees" in out["rationale"]
    # Final p stays close to Kalshi mid 0.50.
    assert 0.49 < out["p_yes"] < 0.51


def test_blend_skipped_when_polymarket_volume_below_floor():
    """Regression for the thin-Polymarket concern: when Kalshi has signal
    AND Polymarket vol is below MIN_POLYMARKET_VOLUME_FOR_BLEND, skip the
    blend entirely (use Kalshi alone). Prevents a stale secondary listing
    with $500 of volume from getting ~25% weight against a $1M Kalshi book.
    """
    from agent.predict import MIN_POLYMARKET_VOLUME_FOR_BLEND

    e = _event()
    e["category"] = "Politics"
    e["outcomes"] = ["TeamA", "TeamB"]
    # Kalshi outside safe band (so blend would normally fire) with deep volume
    market = {
        "yes_bid_dollars": "0.84",
        "yes_ask_dollars": "0.86",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.14",
        "no_ask_dollars": "0.16",
        "last_price_dollars": "0.85",
        "volume_24h_fp": "200000",
    }
    # Polymarket with stale low-volume quote
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.30, MIN_POLYMARKET_VOLUME_FOR_BLEND - 100, "poly p=0.30 thin"),
    ):
        out = predict(e)
    # Blend should be skipped; final p ≈ kalshi mid 0.85 (after small shrinkage).
    # Sanity guardrail would NOT fire because we're using kalshi mid directly.
    assert "skip blend" in out["rationale"]
    assert out["p_yes"] > 0.70  # closer to Kalshi 0.85 than to Poly 0.30


def test_safe_band_blends_when_polymarket_disagrees():
    """Liquid Kalshi in safe band BUT Polymarket disagrees > tol → blend fires."""
    e = _event()
    e["category"] = "Politics"
    e["outcomes"] = ["TeamA", "TeamB"]
    market = {
        "yes_bid_dollars": "0.49",
        "yes_ask_dollars": "0.51",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.49",
        "no_ask_dollars": "0.51",
        "last_price_dollars": "0.50",
        "volume_24h_fp": "20000",
    }
    # Polymarket at 0.65 → |0.50 - 0.65| = 0.15 > CROSS_VENUE_DISAGREE_TOL
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.65, 5000.0, "poly p=0.65"),
    ) as poly_mock:
        out = predict(e)
    poly_mock.assert_called_once()
    # Blend should fire — final probability between Kalshi and Polymarket.
    assert "blend" in out["rationale"]
    assert 0.50 < out["p_yes"] < 0.65


def test_outside_safe_band_still_uses_polymarket():
    """At p=0.85 we're outside safe band → blend still runs."""
    e = _event()
    e["category"] = "Politics"
    e["outcomes"] = ["TeamA", "TeamB"]
    market = {
        "yes_bid_dollars": "0.84",
        "yes_ask_dollars": "0.86",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.14",
        "no_ask_dollars": "0.16",
        "last_price_dollars": "0.85",
        "volume_24h_fp": "20000",
    }
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.80, 5000.0, "poly p=0.80"),
    ) as poly_mock:
        predict(e)
    poly_mock.assert_called_once()


def test_safe_band_below_min_vol_still_blends():
    """Low-volume Kalshi in safe band: still gets Polymarket cross-reference."""
    e = _event()
    e["category"] = "Politics"
    e["outcomes"] = ["TeamA", "TeamB"]
    from agent.predict import SAFE_BAND_MIN_VOL_24H

    market = {
        "yes_bid_dollars": "0.49",
        "yes_ask_dollars": "0.51",
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": "0.49",
        "no_ask_dollars": "0.51",
        "last_price_dollars": "0.50",
        "volume_24h_fp": str(SAFE_BAND_MIN_VOL_24H - 100),
    }
    with patch("agent.predict.get_market", return_value=market), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.55, 2000.0, "poly p=0.55"),
    ) as poly_mock:
        predict(e)
    poly_mock.assert_called_once()


# ---- Market sanity guardrail -----------------------------------------------


def _deep_market(yes_bid: str, yes_ask: str, vol: float = 200_000) -> dict:
    return {
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "yes_bid_size_fp": "100",
        "yes_ask_size_fp": "100",
        "no_bid_dollars": str(round(1 - float(yes_ask), 2)),
        "no_ask_dollars": str(round(1 - float(yes_bid), 2)),
        "last_price_dollars": str((float(yes_bid) + float(yes_ask)) / 2),
        "volume_24h_fp": str(vol),
    }


def test_guardrail_anchors_when_polymarket_pulls_far_from_kalshi():
    """Kalshi at 0.85 (outside safe band), Polymarket at 0.30 → blend pulls far → anchor back."""
    e = _event()
    e["category"] = "Politics"
    # Kalshi at 0.85 (outside [0.20, 0.80] safe band) so Polymarket blend fires.
    kalshi = _deep_market("0.84", "0.86", vol=200_000)
    with patch("agent.predict.get_market", return_value=kalshi), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.30, 500_000.0, "poly p=0.30 vol24h=$500000 (match=0.85)"),
    ):
        out = predict(e)
    # Without guardrail: vol-weighted blend = (0.85*200k + 0.30*500k) / 700k ≈ 0.457 — far from Kalshi 0.85.
    # Guardrail blends 0.6*0.85 + 0.4*final back toward market.
    assert "guardrail" in out["rationale"]
    assert abs(out["p_yes"] - 0.85) < abs(out["p_yes"] - 0.30)


def test_guardrail_no_op_for_low_volume_kalshi():
    """Guardrail requires >= GUARDRAIL_MIN_VOL_24H; small books don't trigger anchor."""
    from agent.predict import GUARDRAIL_MIN_VOL_24H

    e = _event()
    e["category"] = "Politics"
    kalshi = _deep_market("0.49", "0.51", vol=GUARDRAIL_MIN_VOL_24H - 1)
    with patch("agent.predict.get_market", return_value=kalshi), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.95, 50_000.0, "poly"),
    ):
        out = predict(e)
    assert "guardrail" not in out["rationale"]


def test_guardrail_no_op_when_deviation_small():
    """When our final p is close to Kalshi mid, no anchoring happens."""
    e = _event()
    e["category"] = "Politics"
    kalshi = _deep_market("0.49", "0.51", vol=200_000)
    # Polymarket close to Kalshi → blended price stays near 0.50.
    with patch("agent.predict.get_market", return_value=kalshi), patch(
        "agent.predict.polymarket_quote",
        return_value=(0.52, 100_000.0, "poly p=0.52"),
    ):
        out = predict(e)
    assert "guardrail" not in out["rationale"]


def test_guardrail_does_not_apply_to_multi_outcome():
    """Multi-outcome events shouldn't trigger the binary guardrail."""
    e = _event()
    e["category"] = "Entertainment"
    e["outcomes"] = ["A", "B", "C", "D", "E"]
    # llm_forecast_ensemble_full would be called; mock it to return a distribution.
    with patch("agent.predict.get_market") as market_mock, patch(
        "agent.predict.llm_forecast_ensemble_full",
        return_value=(0.95, None, "confident"),
    ):
        out = predict(e)
    market_mock.assert_not_called()
    assert "guardrail" not in out["rationale"]


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
    # depth-mid 0.97 with TAIL_ANCHOR_SHRINK=0.03 → 0.97*0.97 + 0.03*0.5 = 0.9559
    assert out["p_yes"] == pytest.approx(0.97 * 0.97 + 0.03 * 0.5)
    assert "tail-anchor" in out["rationale"]


def test_tail_anchor_low_skips_llm():
    with patch("agent.predict.get_market", return_value=_tail_low_market(vol=1000)), patch(
        "agent.predict.llm_forecast_ensemble"
    ) as llm_mock:
        out = predict(_event())
    llm_mock.assert_not_called()
    # 0.03 * 0.97 + 0.5 * 0.03 = 0.0441
    assert out["p_yes"] == pytest.approx(0.03 * 0.97 + 0.5 * 0.03)
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


# ---- Probabilities-only contract ---------------------------------------


def test_predict_returns_probabilities_for_binary_event():
    """Every response must include probabilities matching outcomes, summing to 1."""
    e = _event()
    e["outcomes"] = ["Pittsburgh", "Atlanta"]
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=None
    ), patch("agent.predict.llm_forecast_ensemble", return_value=(0.62, "research")):
        out = predict(e)
    assert "probabilities" in out
    probs = out["probabilities"]
    assert len(probs) == 2
    markets = {p["market"] for p in probs}
    assert markets == {"Pittsburgh", "Atlanta"}
    total = sum(p["probability"] for p in probs)
    assert total == pytest.approx(1.0)


def test_predict_probabilities_match_outcomes_order():
    """outcomes[0] should be the first entry in probabilities (or at least present)."""
    e = _event()
    e["outcomes"] = ["TeamA", "TeamB"]
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=None
    ), patch("agent.predict.llm_forecast_ensemble", return_value=(0.7, "rationale")):
        out = predict(e)
    by_market = {p["market"]: p["probability"] for p in out["probabilities"]}
    # p_yes is for outcomes[0]; after shrink α=0.15 (speculative): 0.7*0.85 + 0.5*0.15 = 0.67
    expected = 0.7 * 0.85 + 0.5 * 0.15
    assert by_market["TeamA"] == pytest.approx(expected)
    assert by_market["TeamB"] == pytest.approx(1.0 - expected)


def test_predict_falls_back_to_yes_no_when_outcomes_missing():
    """If event has no outcomes, default to ['Yes', 'No'] distribution."""
    e = _event()
    e.pop("outcomes", None)
    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=None
    ), patch("agent.predict.llm_forecast_ensemble", return_value=(0.3, "x")):
        out = predict(e)
    markets = {p["market"] for p in out["probabilities"]}
    assert markets == {"Yes", "No"}


def test_predict_multi_outcome_distribution_sums_to_one():
    """For multi-outcome events, the full distribution must sum to 1 strictly."""
    e = _event()
    e["outcomes"] = ["A", "B", "C", "D"]
    e["title"] = "Who will win?"  # single-winner
    with patch("agent.predict.get_market") as market_mock, patch(
        "agent.predict.llm_forecast_ensemble_full",
        return_value=(
            0.4,
            [
                {"market": "A", "probability": 0.40},
                {"market": "B", "probability": 0.30},
                {"market": "C", "probability": 0.20},
                {"market": "D", "probability": 0.10},
            ],
            "rat",
        ),
    ):
        out = predict(e)
    market_mock.assert_not_called()
    total = sum(p["probability"] for p in out["probabilities"])
    assert total == pytest.approx(1.0)


def test_predict_multi_outcome_normalizes_summing_to_more_than_one():
    """Top-K LLM outputs sum to K. We must normalize to 1."""
    e = _event()
    e["outcomes"] = [f"C{i}" for i in range(35)]
    e["title"] = "top 5 finishers"
    # LLM gives a top-5-style distribution that sums to ~5
    raw_probs = [
        {"market": f"C{i}", "probability": 0.20 if i < 25 else 0.0}
        for i in range(35)
    ]
    with patch("agent.predict.llm_forecast_ensemble_full",
               return_value=(0.20, raw_probs, "r")):
        out = predict(e)
    total = sum(p["probability"] for p in out["probabilities"])
    assert total == pytest.approx(1.0)


def test_predict_multi_outcome_uniform_when_llm_fails():
    """Uniform 1/N distribution when the LLM can't be reached."""
    e = _event()
    e["outcomes"] = ["A", "B", "C", "D", "E"]
    with patch("agent.predict.llm_forecast_ensemble_full", return_value=None):
        out = predict(e)
    probs = out["probabilities"]
    assert len(probs) == 5
    for p in probs:
        assert p["probability"] == pytest.approx(0.2)


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
