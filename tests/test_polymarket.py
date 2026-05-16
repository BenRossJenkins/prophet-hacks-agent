from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.polymarket import (
    MATCH_THRESHOLD,
    MIN_VOLUME_24H,
    _is_usable,
    _market_p_yes,
    _overlap,
    _parse_outcome_prices,
    _tokens,
    find_match,
    polymarket_quote,
)


def _market(**overrides) -> dict:
    base = {
        "question": "Will Donald Trump win the 2028 election?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.42", "0.58"]',
        "bestBid": 0.41,
        "bestAsk": 0.43,
        "lastTradePrice": 0.42,
        "volume24hr": 5000.0,
        "closed": False,
        "archived": False,
        "active": True,
    }
    base.update(overrides)
    return base


# ---- token + overlap utilities ----


def test_tokens_strips_stopwords_and_short_tokens():
    out = _tokens("Will Donald Trump be elected in 2028?")
    assert "donald" in out
    assert "trump" in out
    assert "2028" in out
    assert "will" not in out
    assert "be" not in out
    assert "in" not in out


def test_overlap_perfect_match_is_one():
    a = {"trump", "2028", "election"}
    b = {"trump", "2028", "election"}
    assert _overlap(a, b) == 1.0


def test_overlap_uses_smaller_denominator():
    # Polymarket question with extra context shouldn't penalize match score.
    kalshi = {"trump", "2028"}
    poly = {"trump", "2028", "presidential", "election", "winner", "republican"}
    # All Kalshi tokens are in Poly → score = 2/2 = 1.0 under min-denominator.
    assert _overlap(kalshi, poly) == 1.0


def test_overlap_zero_when_disjoint():
    assert _overlap({"a", "b"}, {"c", "d"}) == 0.0


def test_overlap_zero_when_empty():
    assert _overlap(set(), {"a"}) == 0.0


# ---- price extraction ----


def test_parse_outcome_prices_stringified_json():
    assert _parse_outcome_prices({"outcomePrices": '["0.7", "0.3"]'}) == (0.7, 0.3)


def test_parse_outcome_prices_list_passthrough():
    assert _parse_outcome_prices({"outcomePrices": ["0.6", "0.4"]}) == (0.6, 0.4)


def test_parse_outcome_prices_bad_json_returns_none():
    assert _parse_outcome_prices({"outcomePrices": "not-json"}) is None


def test_parse_outcome_prices_out_of_range_returns_none():
    assert _parse_outcome_prices({"outcomePrices": '["1.5", "-0.5"]'}) is None


def test_market_p_yes_prefers_bid_ask_mid():
    p = _market_p_yes(_market(bestBid=0.30, bestAsk=0.40, lastTradePrice=0.99))
    assert p == 0.35


def test_market_p_yes_falls_back_to_outcome_prices():
    m = _market(bestBid=0.0, bestAsk=0.0, outcomePrices='["0.65", "0.35"]', lastTradePrice=0.0)
    assert _market_p_yes(m) == 0.65


def test_market_p_yes_falls_back_to_last_trade():
    m = _market(bestBid=0.0, bestAsk=0.0, outcomePrices='["0.0", "0.0"]', lastTradePrice=0.55)
    assert _market_p_yes(m) == 0.55


def test_market_p_yes_none_when_no_signal():
    m = _market(bestBid=0.0, bestAsk=0.0, outcomePrices='["0.0", "0.0"]', lastTradePrice=0.0)
    assert _market_p_yes(m) is None


# ---- usability gate ----


def test_is_usable_passes_clean_market():
    assert _is_usable(_market()) is True


def test_is_usable_rejects_closed():
    assert _is_usable(_market(closed=True)) is False


def test_is_usable_rejects_archived():
    assert _is_usable(_market(archived=True)) is False


def test_is_usable_rejects_inactive():
    assert _is_usable(_market(active=False)) is False


def test_is_usable_rejects_low_volume():
    assert _is_usable(_market(volume24hr=MIN_VOLUME_24H - 1)) is False


def test_is_usable_rejects_multi_outcome():
    assert _is_usable(_market(outcomes='["A", "B", "C"]')) is False


# ---- find_match / polymarket_quote ----


def test_find_match_returns_best_overlap():
    candidates = [
        _market(question="Will Donald Trump win the 2028 election?"),
        _market(question="Will Bitcoin reach $200k in 2027?"),
    ]
    with patch("agent.polymarket._search", return_value=candidates):
        match = find_match("Will Trump win the 2028 presidential election")
    assert match is not None
    market, score = match
    assert "Trump" in market["question"]
    assert score >= MATCH_THRESHOLD


def test_find_match_returns_none_when_no_candidate_clears_threshold():
    candidates = [_market(question="Will it rain in Seattle tomorrow?")]
    with patch("agent.polymarket._search", return_value=candidates):
        match = find_match("Will Donald Trump pardon Hunter Biden?")
    assert match is None


def test_find_match_returns_none_on_empty_search():
    with patch("agent.polymarket._search", return_value=[]):
        assert find_match("anything") is None


def test_polymarket_quote_returns_price_and_weight():
    with patch(
        "agent.polymarket._search",
        return_value=[_market(question="Will Donald Trump win the 2028 election?")],
    ):
        out = polymarket_quote(
            {"title": "Will Trump win 2028 election", "category": "Politics"}
        )
    assert out is not None
    p, weight, rationale = out
    assert 0.01 <= p <= 0.99
    assert weight == 5000.0
    assert "poly" in rationale.lower()


def test_polymarket_quote_returns_none_when_no_match():
    with patch("agent.polymarket._search", return_value=[]):
        assert polymarket_quote({"title": "obscure event", "category": "Politics"}) is None


def test_polymarket_quote_handles_network_failure():
    import requests

    def _raise(*a, **kw):
        raise requests.RequestException("boom")

    with patch("agent.polymarket.requests.get", side_effect=_raise):
        assert polymarket_quote({"title": "anything", "category": "Politics"}) is None


# ---- multi-outcome event lookup ----

from agent.polymarket import (  # noqa: E402
    _find_event_match,
    _map_child_to_outcome,
    polymarket_event_distribution,
)


def _event_payload(title: str, child_markets: list[dict]) -> dict:
    return {
        "title": title,
        "ticker": "TEST-EVENT",
        "closed": False,
        "archived": False,
        "markets": child_markets,
    }


def _child(question: str, yes_price: float, vol: float = 1000.0) -> dict:
    return {
        "question": question,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{yes_price}", "{1.0 - yes_price}"]',
        "bestBid": max(0.0, yes_price - 0.01),
        "bestAsk": min(1.0, yes_price + 0.01),
        "lastTradePrice": yes_price,
        "volume24hr": vol,
        "closed": False,
        "archived": False,
        "active": True,
    }


def test_map_child_outcome_matches_subset_tokens():
    outcomes = ["Albania", "France", "Sweden", "Italy"]
    assert _map_child_to_outcome("Will Albania win Eurovision 2026?", outcomes) == "Albania"
    assert _map_child_to_outcome("Will France finish top 5?", outcomes) == "France"
    # No outcome mentioned → None
    assert _map_child_to_outcome("Will it rain in Geneva tomorrow?", outcomes) is None


def test_event_match_requires_three_plus_children():
    sparse = _event_payload("Eurovision 2026", [_child("Will Albania win?", 0.05)])
    with patch("agent.polymarket._search_events", return_value=[sparse]):
        assert _find_event_match("Eurovision 2026 winner") is None


def test_event_distribution_covers_outcomes():
    event_payload = _event_payload(
        "Eurovision 2026 Winner",
        [
            _child("Will Albania win Eurovision 2026?", 0.05),
            _child("Will France win Eurovision 2026?", 0.30),
            _child("Will Sweden win Eurovision 2026?", 0.20),
            _child("Will Italy win Eurovision 2026?", 0.15),
        ],
    )
    with patch("agent.polymarket._search_events", return_value=[event_payload]):
        out = polymarket_event_distribution(
            {
                "title": "Who will win Eurovision 2026?",
                "outcomes": ["Albania", "France", "Sweden", "Italy"],
            }
        )
    assert out is not None
    probs, vol, rationale = out
    by_market = {p["market"]: p["probability"] for p in probs}
    assert by_market["France"] == pytest.approx(0.30)
    assert by_market["Albania"] == pytest.approx(0.05)
    assert "covered 4/4" in rationale


def test_event_distribution_returns_none_when_coverage_too_sparse():
    """Need MIN_OUTCOMES_COVERED of event outcomes mapped, else None."""
    event_payload = _event_payload(
        "Eurovision 2026 Winner",
        [
            _child("Will Albania win Eurovision 2026?", 0.05),
            # Other children don't match our outcomes list
            _child("Will Belarus win Eurovision 2026?", 0.02),
            _child("Will Russia win Eurovision 2026?", 0.01),
        ],
    )
    with patch("agent.polymarket._search_events", return_value=[event_payload]):
        out = polymarket_event_distribution(
            {
                "title": "Who will win Eurovision 2026?",
                "outcomes": ["Albania", "France", "Sweden", "Italy", "Germany"],
            }
        )
    # 1 of 5 = 20% < MIN_OUTCOMES_COVERED → None.
    assert out is None


def test_event_distribution_returns_none_for_binary_events():
    """Binary events skip the multi-outcome event lookup entirely."""
    out = polymarket_event_distribution(
        {"title": "Will A win?", "outcomes": ["A", "B"]}
    )
    assert out is None


def test_event_distribution_handles_network_failure():
    import requests

    def _raise(*a, **kw):
        raise requests.RequestException("boom")

    with patch("agent.polymarket.requests.get", side_effect=_raise):
        out = polymarket_event_distribution(
            {"title": "anything", "outcomes": ["A", "B", "C", "D"]}
        )
    assert out is None
