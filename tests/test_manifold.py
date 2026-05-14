from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.manifold import _clean_query, _is_usable, manifold_prior


def _event(title: str, category: str = "Politics") -> dict:
    return {
        "event_ticker": "TEST",
        "market_ticker": "TEST-MKT",
        "title": title,
        "category": category,
        "close_time": "2026-12-31T23:59:59Z",
    }


# ---- _clean_query --------------------------------------------------------


def test_clean_query_strips_will():
    assert _clean_query("Will Trump be impeached?") == "Trump be impeached"


def test_clean_query_strips_who_will():
    assert _clean_query("Who will win the 2028 election?") == "win the 2028 election"


def test_clean_query_strips_how_much():
    # "How many" → strip "many" form
    assert _clean_query("How many goals will be scored?") == "goals will be scored"


def test_clean_query_passes_unrelated_titles_through():
    # Doesn't strip when no Kalshi pattern matches
    assert _clean_query("Bitcoin price on May 14") == "Bitcoin price on May 14"


def test_clean_query_truncates_huge_titles():
    long_title = "Will " + "x " * 500 + "?"
    out = _clean_query(long_title)
    assert len(out) <= 200


# ---- _is_usable ----------------------------------------------------------


def test_is_usable_normal_binary_market():
    m = {
        "outcomeType": "BINARY",
        "isResolved": False,
        "volume": 1000,
        "probability": 0.42,
    }
    assert _is_usable(m) is True


def test_is_usable_rejects_resolved():
    m = {
        "outcomeType": "BINARY",
        "isResolved": True,
        "volume": 1000,
        "probability": 0.99,
    }
    assert _is_usable(m) is False


def test_is_usable_rejects_multiple_choice():
    m = {
        "outcomeType": "MULTIPLE_CHOICE",
        "isResolved": False,
        "volume": 1000,
        "probability": 0.42,
    }
    assert _is_usable(m) is False


def test_is_usable_rejects_low_volume():
    m = {
        "outcomeType": "BINARY",
        "isResolved": False,
        "volume": 5,
        "probability": 0.5,
    }
    assert _is_usable(m) is False


def test_is_usable_rejects_missing_or_bad_probability():
    base = {"outcomeType": "BINARY", "isResolved": False, "volume": 1000}
    assert _is_usable({**base}) is False  # no probability
    assert _is_usable({**base, "probability": 1.5}) is False
    assert _is_usable({**base, "probability": "high"}) is False


# ---- manifold_prior end-to-end ------------------------------------------


def test_returns_first_usable_match():
    fake_results = [
        {  # Best match — usable
            "outcomeType": "BINARY",
            "isResolved": False,
            "volume": 5000,
            "probability": 0.74,
            "question": "Will Trump pardon X by end of term?",
        },
    ]
    with patch("agent.manifold._search", return_value=fake_results):
        out = manifold_prior(_event("Will Trump pardon X?"))
    assert out is not None
    p, rationale = out
    assert p == pytest.approx(0.74)
    assert "Manifold market" in rationale


def test_skips_resolved_markets_in_results():
    fake_results = [
        {"outcomeType": "BINARY", "isResolved": True, "volume": 5000, "probability": 1.0, "question": "stale resolved one"},
        {"outcomeType": "BINARY", "isResolved": False, "volume": 1000, "probability": 0.30, "question": "live one"},
    ]
    with patch("agent.manifold._search", return_value=fake_results):
        out = manifold_prior(_event("anything"))
    assert out is not None
    p, _ = out
    assert p == pytest.approx(0.30)


def test_returns_none_when_no_results():
    with patch("agent.manifold._search", return_value=[]):
        assert manifold_prior(_event("nothing matches")) is None


def test_returns_none_when_no_usable_results():
    fake_results = [
        {"outcomeType": "MULTIPLE_CHOICE", "isResolved": False, "volume": 10000, "question": "x"},
        {"outcomeType": "BINARY", "isResolved": False, "volume": 5, "probability": 0.5, "question": "low vol"},
    ]
    with patch("agent.manifold._search", return_value=fake_results):
        assert manifold_prior(_event("anything")) is None


def test_clamps_to_contract_range():
    fake_results = [
        {"outcomeType": "BINARY", "isResolved": False, "volume": 5000, "probability": 0.999, "question": "x"},
    ]
    with patch("agent.manifold._search", return_value=fake_results):
        out = manifold_prior(_event("x"))
    assert out is not None
    assert out[0] == 0.99

    fake_results[0]["probability"] = 0.001
    with patch("agent.manifold._search", return_value=fake_results):
        out = manifold_prior(_event("x"))
    assert out is not None
    assert out[0] == 0.01
