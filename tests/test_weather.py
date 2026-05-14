from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from agent.weather import (
    TEMP_SIGMA_F,
    _find_period_for,
    _prob_above,
    city_to_coords,
    comparison_to_prob,
    parse_title,
    weather_prior,
)


# ---- title parser --------------------------------------------------------


def test_parse_above_with_edt():
    out = parse_title(
        "Will the temp in New York City be above 60.99° on May 14, 2026 at 11am EDT?"
    )
    assert out is not None
    assert out["city"] == "New York City"
    assert out["comparison"] == "above"
    assert out["threshold_f"] == pytest.approx(60.99)
    # 11am EDT = 15:00 UTC
    assert out["target_utc"] == datetime(2026, 5, 14, 15, 0)


def test_parse_below():
    out = parse_title(
        "Will the temp in Chicago be below 32° on December 15, 2026 at 6pm CST?"
    )
    assert out is not None
    assert out["city"] == "Chicago"
    assert out["comparison"] == "below"
    assert out["threshold_f"] == pytest.approx(32.0)
    # 6pm CST = 0:00 UTC next day
    assert out["target_utc"] == datetime(2026, 12, 16, 0, 0)


def test_parse_negative_threshold():
    out = parse_title("Will the temp in Denver be above -10.50° on January 5, 2026 at 3am MST?")
    assert out is not None
    assert out["threshold_f"] == pytest.approx(-10.50)


def test_parse_does_not_match_unrelated_title():
    assert parse_title("Will it rain in NYC tomorrow?") is None
    assert parse_title("Will Trump win the election?") is None


def test_parse_missing_timezone_defaults_to_edt():
    out = parse_title("Will the temp in New York City be above 70° on June 1, 2026 at 12pm?")
    assert out is not None
    assert out["tz"] == "EDT"


# ---- coords lookup -------------------------------------------------------


def test_coords_lookup_case_insensitive():
    assert city_to_coords("New York City") is not None
    assert city_to_coords("NEW YORK CITY") is not None


def test_coords_lookup_unknown_returns_none():
    assert city_to_coords("Atlantis") is None


# ---- probability conversion ---------------------------------------------


def test_prob_above_at_threshold_is_half():
    assert _prob_above(60.0, 60.0) == pytest.approx(0.5)


def test_prob_above_warmer_forecast_increases_prob():
    p = _prob_above(65.0, 60.0)
    assert p > 0.5
    assert p == pytest.approx(1 / (1 + pow(2.718281828, -5 / TEMP_SIGMA_F)), abs=1e-4)


def test_prob_above_colder_forecast_decreases_prob():
    assert _prob_above(55.0, 60.0) < 0.5


def test_comparison_below_inverts():
    p_above = _prob_above(65.0, 60.0)
    p_below = comparison_to_prob("below", 65.0, 60.0)
    assert p_below == pytest.approx(1 - p_above)


def test_comparison_at_most_treated_as_below():
    p1 = comparison_to_prob("at most", 65.0, 60.0)
    p2 = comparison_to_prob("below", 65.0, 60.0)
    assert p1 == pytest.approx(p2)


# ---- period selection ---------------------------------------------------


def test_find_period_for_picks_closest():
    periods = [
        {"startTime": "2026-05-14T13:00:00+00:00", "temperature": 60},
        {"startTime": "2026-05-14T14:00:00+00:00", "temperature": 63},
        {"startTime": "2026-05-14T15:00:00+00:00", "temperature": 66},
        {"startTime": "2026-05-14T16:00:00+00:00", "temperature": 68},
    ]
    target = datetime(2026, 5, 14, 15, 10)
    p = _find_period_for(periods, target)
    assert p is not None
    assert p["temperature"] == 66


def test_find_period_for_returns_none_when_off_by_hours():
    periods = [{"startTime": "2026-05-14T13:00:00+00:00", "temperature": 60}]
    target = datetime(2026, 5, 15, 12, 0)  # 23 hours later
    assert _find_period_for(periods, target) is None


# ---- end-to-end ---------------------------------------------------------


def test_weather_prior_full_path_with_mocks():
    event = {
        "title": "Will the temp in New York City be above 60.99° on May 14, 2026 at 11am EDT?",
        "category": "Climate and Weather",
    }
    fake_periods = [
        {"startTime": "2026-05-14T15:00:00+00:00", "temperature": 68},
    ]
    with patch("agent.weather._points_lookup", return_value=("OKX", 30, 40)), patch(
        "agent.weather._hourly_forecast", return_value=fake_periods
    ):
        out = weather_prior(event)
    assert out is not None
    p, rationale = out
    # 68°F forecast, 60.99° threshold (above), sigma 3 → p ≈ 0.91
    assert 0.88 < p < 0.94
    assert "NWS hourly forecast" in rationale
    assert "68" in rationale


def test_weather_prior_returns_none_when_title_unparseable():
    assert weather_prior({"title": "Will it rain?", "category": "Climate and Weather"}) is None


def test_weather_prior_returns_none_when_no_nws_grid():
    event = {
        "title": "Will the temp in New York City be above 60° on May 14, 2026 at 11am EDT?",
        "category": "Climate and Weather",
    }
    with patch("agent.weather._points_lookup", return_value=None):
        assert weather_prior(event) is None


def test_weather_prior_returns_none_when_forecast_period_missing():
    event = {
        "title": "Will the temp in New York City be above 60° on May 14, 2026 at 11am EDT?",
        "category": "Climate and Weather",
    }
    # No matching hour in the forecast (far in the past).
    fake_periods = [{"startTime": "2099-01-01T00:00:00+00:00", "temperature": 50}]
    with patch("agent.weather._points_lookup", return_value=("OKX", 30, 40)), patch(
        "agent.weather._hourly_forecast", return_value=fake_periods
    ):
        assert weather_prior(event) is None
