from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.kalshi import (
    _derive_event_ticker,
    _kalshi_child_p_yes,
    _map_kalshi_child_to_outcome,
    _read_price,
    _read_volume,
    get_event,
    get_market,
    kalshi_event_distribution,
)


# ---- price reading helpers ----


def test_read_price_prefers_cents_when_integer():
    assert _read_price({"yes_bid": 28}, "yes_bid", dollars_keys=("yes_bid_dollars",)) == 0.28


def test_read_price_falls_back_to_dollars():
    assert _read_price(
        {"yes_bid_dollars": 0.42}, "yes_bid", dollars_keys=("yes_bid_dollars",)
    ) == 0.42


def test_read_price_handles_cents_as_float():
    # Some Kalshi responses give 28.5 instead of 28 — still cents.
    assert _read_price({"yes_bid": 28.5}, "yes_bid", dollars_keys=()) == 0.285


def test_read_price_treats_subdollar_floats_as_dollars():
    # If the cents key has a value < 1, it's actually already a probability.
    assert _read_price({"yes_bid": 0.42}, "yes_bid", dollars_keys=()) == 0.42


def test_read_price_none_for_missing_or_garbage():
    assert _read_price({}, "yes_bid", dollars_keys=("yes_bid_dollars",)) is None
    assert _read_price({"yes_bid": "garbage"}, "yes_bid", dollars_keys=()) is None
    assert _read_price({"yes_bid": -5}, "yes_bid", dollars_keys=()) is None


def test_read_volume_tries_multiple_field_names():
    assert _read_volume({"volume_24h": 1500}) == 1500.0
    assert _read_volume({"volume_24h_fp": 800}) == 800.0
    assert _read_volume({"volume24hr": 200}) == 200.0
    assert _read_volume({}) == 0.0


# ---- _kalshi_child_p_yes ----


def test_child_p_yes_uses_mid_when_tight_spread():
    child = {"yes_bid": 28, "yes_ask": 32, "volume_24h": 200, "status": "active"}
    assert _kalshi_child_p_yes(child) == pytest.approx(0.30)


def test_child_p_yes_uses_last_on_wide_thin_book():
    # Wide spread (0.20), low vol → fall back to last trade.
    child = {"yes_bid": 10, "yes_ask": 30, "last_price": 18, "volume_24h": 5, "status": "active"}
    assert _kalshi_child_p_yes(child) == pytest.approx(0.18)


def test_child_p_yes_blends_when_wide_but_liquid():
    # Wide spread, but volume is high → blend 0.4*mid + 0.6*last.
    child = {"yes_bid": 10, "yes_ask": 30, "last_price": 22, "volume_24h": 5000, "status": "active"}
    expected = 0.4 * 0.20 + 0.6 * 0.22
    assert _kalshi_child_p_yes(child) == pytest.approx(expected)


def test_child_p_yes_accepts_settled_status():
    """v3.15: settled / finalized / closed are now accepted; the post-
    resolution price still carries meaningful signal. Only truly unusable
    statuses (cancelled, unknown) return None."""
    # Settled market with price-only signal (no `result` set) falls through
    # to the existing price logic.
    child = {"yes_bid": 28, "yes_ask": 30, "status": "settled"}
    assert _kalshi_child_p_yes(child) == pytest.approx(0.29)
    # Truly bad status → None.
    assert _kalshi_child_p_yes({"yes_bid": 28, "yes_ask": 30, "status": "cancelled"}) is None


def test_child_p_yes_uses_result_field_for_finalized_markets():
    """Kalshi pattern: status='finalized' + result='yes'/'no' is the
    market's definitive answer. Return 1.0/0.0 (the [0, 1] endpoints —
    Brier-0 on correct settled outcomes)."""
    child_no = {
        "status": "finalized", "result": "no",
        "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0100",
        "last_price_dollars": "0.0100",
    }
    assert _kalshi_child_p_yes(child_no) == pytest.approx(0.0)
    child_yes = {
        "status": "finalized", "result": "yes",
        "yes_bid_dollars": "0.9900", "yes_ask_dollars": "1.0000",
        "last_price_dollars": "0.9900",
    }
    assert _kalshi_child_p_yes(child_yes) == pytest.approx(1.0)


def test_child_p_yes_ignores_unexpected_result_values():
    """Result values other than 'yes'/'no' (e.g., 'void', empty string)
    fall through to the existing price extraction path."""
    child = {
        "status": "finalized", "result": "void",
        "yes_bid_dollars": "0.50", "yes_ask_dollars": "0.52",
        "volume_24h_fp": "100",
    }
    # spread = 0.02 ≤ 0.10 → mid = 0.51
    assert _kalshi_child_p_yes(child) == pytest.approx(0.51)


def test_child_p_yes_closed_market_falls_through_to_last_price():
    """Bundesliga pattern: status='closed', result='' (empty), bid/ask
    collapsed to 0/1, but last_price still carries the market's belief."""
    child = {
        "status": "closed", "result": "",
        "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000",
        "last_price_dollars": "0.9900",
        "volume_24h_fp": "0.00",
    }
    # bid=0, ask=1, spread=1 > 0.10, vol=0 < min_liquid → falls through
    # to last_price branch which returns 0.99.
    assert _kalshi_child_p_yes(child) == pytest.approx(0.99)


def test_child_p_yes_handles_dollars_variant():
    """Cents and dollars conventions both work for the same logic."""
    child = {
        "yes_bid_dollars": 0.28,
        "yes_ask_dollars": 0.30,
        "volume_24h_fp": 200,
        "status": "active",
    }
    assert _kalshi_child_p_yes(child) == pytest.approx(0.29)


# ---- _map_kalshi_child_to_outcome ----


def test_map_exact_match_case_insensitive():
    outcomes = ["Boston Celtics", "Denver Nuggets"]
    child = {"subtitle": "Boston Celtics"}
    assert _map_kalshi_child_to_outcome(child, outcomes) == "Boston Celtics"
    child = {"subtitle": "DENVER NUGGETS"}
    assert _map_kalshi_child_to_outcome(child, outcomes) == "Denver Nuggets"


def test_map_token_subset_fallback():
    outcomes = ["Boston Celtics", "Denver Nuggets"]
    child = {"subtitle": "Celtics"}
    # "Celtics" is a token subset of "Boston Celtics" → matches.
    assert _map_kalshi_child_to_outcome(child, outcomes) == "Boston Celtics"


def test_map_returns_none_when_no_match():
    outcomes = ["Boston Celtics", "Denver Nuggets"]
    child = {"subtitle": "Phoenix Suns"}
    assert _map_kalshi_child_to_outcome(child, outcomes) is None


def test_map_uses_yes_sub_title_as_fallback():
    outcomes = ["Yes", "No"]
    child = {"yes_sub_title": "Yes"}
    assert _map_kalshi_child_to_outcome(child, outcomes) == "Yes"


def test_map_returns_none_when_subtitle_empty():
    assert _map_kalshi_child_to_outcome({}, ["A", "B"]) is None
    assert _map_kalshi_child_to_outcome({"subtitle": "  "}, ["A", "B"]) is None


# ---- _derive_event_ticker ----


def test_derive_event_ticker_strips_child_suffix():
    assert _derive_event_ticker("KXNBACHAMP-26-BOS") == "KXNBACHAMP-26"
    assert _derive_event_ticker("KXNBASERIES-26MINSAS-MIN") == "KXNBASERIES-26MINSAS"


def test_derive_event_ticker_none_when_no_dash():
    assert _derive_event_ticker("FLATTICKER") is None
    assert _derive_event_ticker("") is None
    assert _derive_event_ticker(None) is None


# ---- get_event (mocked) ----


def test_get_event_returns_event_dict():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"event": {"event_ticker": "X", "markets": []}}
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        ev = get_event("X")
    assert ev is not None
    assert ev["event_ticker"] == "X"


def test_get_event_handles_unwrapped_response():
    """Some Kalshi endpoints return the event dict directly, not under "event"."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"event_ticker": "X", "markets": []}
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        ev = get_event("X")
    assert ev is not None
    assert ev["event_ticker"] == "X"


def test_get_event_returns_none_on_network_failure():
    import requests as _r

    with patch("agent.kalshi.requests.get", side_effect=_r.RequestException("boom")):
        assert get_event("X") is None


def test_get_market_still_works():
    """Regression: the existing get_market path shouldn't break."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"market": {"ticker": "X-1", "yes_bid_dollars": 0.5}}
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        m = get_market("X-1")
    assert m is not None
    assert m["ticker"] == "X-1"


# ---- kalshi_event_distribution end-to-end ----


def _nba_event_response() -> dict:
    return {
        "event": {
            "event_ticker": "KXNBACHAMP-26",
            "title": "Who wins the 2026 NBA Championship?",
            "mutually_exclusive": True,
            "markets": [
                {
                    "ticker": "KXNBACHAMP-26-BOS", "subtitle": "Boston Celtics",
                    "yes_bid": 28, "yes_ask": 30, "last_price": 29,
                    "volume_24h": 12000, "status": "active",
                },
                {
                    "ticker": "KXNBACHAMP-26-DEN", "subtitle": "Denver Nuggets",
                    "yes_bid": 18, "yes_ask": 20, "last_price": 19,
                    "volume_24h": 8000, "status": "active",
                },
                {
                    "ticker": "KXNBACHAMP-26-OKC", "subtitle": "Oklahoma City Thunder",
                    "yes_bid": 12, "yes_ask": 14, "last_price": 13,
                    "volume_24h": 5000, "status": "active",
                },
            ],
        }
    }


def test_kalshi_event_distribution_mutually_exclusive():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nba_event_response()
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        result = kalshi_event_distribution({
            "event_ticker": "KXNBACHAMP-26",
            "outcomes": ["Boston Celtics", "Denver Nuggets", "Oklahoma City Thunder"],
        })
    assert result is not None
    probs, vol, _, target_sum = result
    by_outcome = {p["market"]: p["probability"] for p in probs}
    assert by_outcome["Boston Celtics"] == pytest.approx(0.29)
    assert by_outcome["Denver Nuggets"] == pytest.approx(0.19)
    assert by_outcome["Oklahoma City Thunder"] == pytest.approx(0.13)
    assert vol == 25000
    # Single-winner event → target_sum = 1.0
    assert target_sum == pytest.approx(1.0)


def test_kalshi_event_distribution_not_mutually_exclusive_returns_topk():
    """v3.15: mutex=False is no longer auto-rejected. We now return the
    children with their natural probabilities and target_sum=K (round of
    Σ children clamped to [2, n_out-1]), provided the sum is close enough
    to an integer to be unambiguous (within AMBIGUITY_TOL=0.30)."""
    mock_resp = MagicMock()
    # 5 children with mids summing to 2.95 (≈ K=3, well within tolerance)
    mock_resp.json.return_value = {
        "event": {
            "event_ticker": "X", "mutually_exclusive": False, "title": "Top 3 event",
            "markets": [
                {"ticker": "X-A", "subtitle": "A", "yes_bid_dollars": 0.95, "yes_ask_dollars": 0.97, "volume_24h_fp": 1000, "status": "active"},
                {"ticker": "X-B", "subtitle": "B", "yes_bid_dollars": 0.85, "yes_ask_dollars": 0.87, "volume_24h_fp": 1000, "status": "active"},
                {"ticker": "X-C", "subtitle": "C", "yes_bid_dollars": 0.75, "yes_ask_dollars": 0.77, "volume_24h_fp": 1000, "status": "active"},
                {"ticker": "X-D", "subtitle": "D", "yes_bid_dollars": 0.25, "yes_ask_dollars": 0.27, "volume_24h_fp": 1000, "status": "active"},
                {"ticker": "X-E", "subtitle": "E", "yes_bid_dollars": 0.04, "yes_ask_dollars": 0.06, "volume_24h_fp": 1000, "status": "active"},
            ],
        }
    }
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        result = kalshi_event_distribution({
            "event_ticker": "X", "outcomes": ["A", "B", "C", "D", "E"],
        })
    assert result is not None
    probs, _vol, _rat, target_sum = result
    # Σ mids = 0.96 + 0.86 + 0.76 + 0.26 + 0.05 = 2.89 → round → K=3
    # |2.89 - 3| = 0.11 < 0.30 tol → target_sum = 3.0
    assert target_sum == pytest.approx(3.0)
    # Per-outcome probabilities are the raw mids (passed through)
    by_outcome = {p["market"]: p["probability"] for p in probs}
    assert by_outcome["A"] == pytest.approx(0.96)


def test_kalshi_event_distribution_ambiguous_sum_falls_back_to_single_winner():
    """When mutex=False but the children's sum is ambiguous (e.g., 4.5,
    equidistant between K=4 and K=5), default to single-winner so we
    don't ship a wrong-shape distribution."""
    mock_resp = MagicMock()
    # Sum = 4.50 — ambiguous. Should fall back to target_sum=1.0.
    mock_resp.json.return_value = {
        "event": {
            "event_ticker": "X", "mutually_exclusive": False, "title": "Ambiguous K",
            "markets": [
                {"ticker": f"X-{i}", "subtitle": chr(ord("A") + i),
                 "yes_bid_dollars": 0.49, "yes_ask_dollars": 0.51,
                 "volume_24h_fp": 1000, "status": "active"}
                for i in range(9)
            ],
        }
    }
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        result = kalshi_event_distribution({
            "event_ticker": "X",
            "outcomes": [chr(ord("A") + i) for i in range(9)],
        })
    assert result is not None
    _, _, _, target_sum = result
    # Σ mids = 9 * 0.5 = 4.5; |4.5 - 4| = 0.5 > 0.3, |4.5 - 5| = 0.5 > 0.3 → ambiguous → 1.0
    assert target_sum == pytest.approx(1.0)


def test_kalshi_event_distribution_insufficient_coverage():
    """Only 1 of 3 outcomes matched a child → below 60% threshold."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "event": {
            "event_ticker": "X", "mutually_exclusive": True,
            "markets": [
                {"ticker": "X-1", "subtitle": "Outcome A",
                 "yes_bid": 50, "yes_ask": 52, "volume_24h": 1000, "status": "active"},
            ],
        }
    }
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp):
        result = kalshi_event_distribution({
            "event_ticker": "X",
            "outcomes": ["Outcome A", "Outcome B", "Outcome C"],
        })
    assert result is None


def test_kalshi_event_distribution_returns_none_for_binary():
    """Only events with 3+ outcomes should hit this path."""
    assert kalshi_event_distribution({
        "event_ticker": "X", "outcomes": ["A", "B"],
    }) is None


def test_kalshi_event_distribution_handles_network_failure():
    import requests as _r

    with patch("agent.kalshi.requests.get", side_effect=_r.RequestException("boom")):
        result = kalshi_event_distribution({
            "event_ticker": "X", "outcomes": ["A", "B", "C", "D"],
        })
    assert result is None


def test_kalshi_event_distribution_derives_event_ticker_from_market_ticker():
    """If event_ticker isn't in the input, derive from market_ticker."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nba_event_response()
    mock_resp.raise_for_status = lambda: None
    with patch("agent.kalshi.requests.get", return_value=mock_resp) as mock_get:
        kalshi_event_distribution({
            "market_ticker": "KXNBACHAMP-26-BOS",  # event ticker derived as KXNBACHAMP-26
            "outcomes": ["Boston Celtics", "Denver Nuggets", "Oklahoma City Thunder"],
        })
    # Verify we asked Kalshi for the parent event, not the child market.
    called_url = mock_get.call_args[0][0]
    assert called_url.endswith("/events/KXNBACHAMP-26")
