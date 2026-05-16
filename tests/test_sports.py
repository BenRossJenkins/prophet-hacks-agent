from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from agent.sports import (
    _detect_league,
    _devig,
    _extract_moneyline_probs,
    _find_team_in_title,
    _yyyymmdd_for_event,
    american_to_prob,
    sports_prior,
)


def _future_iso(hours: float = 6.0) -> str:
    return (
        (datetime.now(UTC) + timedelta(hours=hours))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _event(**overrides) -> dict:
    # market_ticker keys off event_ticker so overrides stay consistent.
    base_evt = overrides.get("event_ticker", "KXNBA-26MAY16-LAL")
    base = {
        "event_ticker": base_evt,
        "market_ticker": f"{base_evt}-T1",
        "title": "Will the Lakers beat the Celtics on May 16?",
        "subtitle": "",
        "category": "Sports",
        "close_time": _future_iso(),
    }
    base.update(overrides)
    return base


def _espn_event(
    home_name: str = "Los Angeles Lakers",
    away_name: str = "Boston Celtics",
    home_ml: float = -150,
    away_ml: float = 130,
) -> dict:
    return {
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {
                            "displayName": home_name,
                            "shortDisplayName": home_name.split()[-1],
                            "name": home_name.split()[-1],
                            "abbreviation": "HOM",
                        },
                    },
                    {
                        "homeAway": "away",
                        "team": {
                            "displayName": away_name,
                            "shortDisplayName": away_name.split()[-1],
                            "name": away_name.split()[-1],
                            "abbreviation": "AWY",
                        },
                    },
                ],
                "odds": [
                    {
                        "homeTeamOdds": {"moneyLine": home_ml, "favorite": home_ml < 0},
                        "awayTeamOdds": {"moneyLine": away_ml, "favorite": away_ml < 0},
                        "provider": {"name": "ESPN BET"},
                    }
                ],
            }
        ]
    }


# ---- league detection ----


def test_detect_league_from_ticker_prefix():
    assert _detect_league(_event(event_ticker="KXNBA-...")) == ("basketball", "nba")
    assert _detect_league(_event(event_ticker="KXNFL-...")) == ("football", "nfl")
    assert _detect_league(_event(event_ticker="KXMLB-...")) == ("baseball", "mlb")
    assert _detect_league(_event(event_ticker="KXNHL-...")) == ("hockey", "nhl")


def test_detect_league_from_title_keyword():
    e = _event(event_ticker="OTHER", market_ticker="OTHER-T1", title="Will the Yankees win this MLB game?")
    assert _detect_league(e) == ("baseball", "mlb")


def test_detect_league_returns_none_for_unknown_sport():
    e = _event(
        event_ticker="KXTENNIS",
        market_ticker="KXTENNIS-T1",
        title="Will Djokovic win?",
        category="Tennis",
    )
    assert _detect_league(e) is None


# ---- date formatting ----


def test_yyyymmdd_from_close_time():
    e = _event(close_time="2026-05-17T23:30:00Z")
    assert _yyyymmdd_for_event(e) == "20260517"


def test_yyyymmdd_returns_none_on_bad_date():
    assert _yyyymmdd_for_event({"close_time": ""}) is None
    assert _yyyymmdd_for_event({"close_time": "not-a-date"}) is None


# ---- moneyline conversion ----


def test_american_to_prob_favorite():
    # -150 → 150 / 250 = 0.60
    assert american_to_prob(-150) == pytest.approx(0.60)


def test_american_to_prob_underdog():
    # +130 → 100 / 230 ≈ 0.4348
    assert american_to_prob(130) == pytest.approx(100 / 230)


def test_american_to_prob_invalid():
    assert american_to_prob(0) is None
    assert american_to_prob("foo") is None
    assert american_to_prob(None) is None


def test_devig_sums_to_one():
    p_home, p_away = _devig(0.60, 0.4348)
    assert p_home + p_away == pytest.approx(1.0)


# ---- team matching ----


def test_find_team_picks_longest_match():
    competitors = [
        {"team": {"displayName": "Los Angeles Lakers", "shortDisplayName": "Lakers"}},
        {"team": {"displayName": "Boston Celtics", "shortDisplayName": "Celtics"}},
    ]
    chosen = _find_team_in_title("Will the Lakers win?", competitors)
    assert chosen is not None
    assert "Lakers" in chosen["team"]["displayName"]


def test_find_team_returns_none_when_no_match():
    competitors = [
        {"team": {"displayName": "Knicks"}},
        {"team": {"displayName": "Heat"}},
    ]
    assert _find_team_in_title("Will the Lakers win?", competitors) is None


# ---- odds extraction ----


def test_extract_moneyline_probs_devigs():
    comp = {
        "odds": [
            {
                "homeTeamOdds": {"moneyLine": -150},
                "awayTeamOdds": {"moneyLine": 130},
            }
        ]
    }
    probs = _extract_moneyline_probs(comp)
    assert probs is not None
    p_home, p_away = probs
    assert p_home + p_away == pytest.approx(1.0)
    assert p_home > p_away  # favorite has higher prob


def test_extract_moneyline_probs_missing_returns_none():
    assert _extract_moneyline_probs({"odds": []}) is None
    assert _extract_moneyline_probs({}) is None
    assert _extract_moneyline_probs({"odds": [{"homeTeamOdds": {}}]}) is None


# ---- sports_prior end-to-end ----


def test_sports_prior_returns_favorite_prob():
    with patch("agent.sports._fetch_scoreboard", return_value=[_espn_event()]):
        out = sports_prior(_event())
    assert out is not None
    p, rationale = out
    # Lakers favored at -150 → de-vigged p ≈ 0.580
    assert 0.55 < p < 0.62
    assert "Lakers" in rationale


def test_sports_prior_returns_underdog_prob():
    # Same scoreboard, but title asks about underdog Celtics.
    e = _event(title="Will the Celtics beat the Lakers?")
    with patch("agent.sports._fetch_scoreboard", return_value=[_espn_event()]):
        out = sports_prior(e)
    assert out is not None
    p, _ = out
    assert 0.38 < p < 0.45


def test_sports_prior_returns_none_for_unknown_sport():
    e = _event(event_ticker="KXFOO", title="Will some other team win tennis?")
    assert sports_prior(e) is None


def test_sports_prior_returns_none_when_team_not_found():
    e = _event(title="Will the Heat win?")
    with patch("agent.sports._fetch_scoreboard", return_value=[_espn_event()]):
        assert sports_prior(e) is None


def test_sports_prior_returns_none_when_scoreboard_empty():
    with patch("agent.sports._fetch_scoreboard", return_value=[]):
        assert sports_prior(_event()) is None


def test_sports_prior_skips_far_future_markets():
    # Markets more than 7 days out don't have pre-game odds.
    e = _event(close_time=_future_iso(hours=24 * 30))
    with patch("agent.sports._fetch_scoreboard", return_value=[_espn_event()]):
        assert sports_prior(e) is None


def test_sports_prior_negation_inverts_probability():
    e = _event(title="Will the Lakers lose to the Celtics?")
    with patch("agent.sports._fetch_scoreboard", return_value=[_espn_event()]):
        out = sports_prior(e)
    assert out is not None
    p, _ = out
    # Original favorite prob ≈ 0.58 → negated ≈ 0.42
    assert 0.38 < p < 0.45
