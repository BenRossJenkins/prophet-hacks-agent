"""Weather-category prior for Kalshi temperature markets.

Parses market titles of the form
    "Will the temp in <CITY> be above <THRESHOLD>° on <DATE> at <TIME>?"
and queries the National Weather Service hourly-forecast API to compute
a calibrated probability.

Implementation notes:
- NWS is free and unauthenticated, but rate-limits casual clients. We
  cache grid lookups per process.
- NWS forecasts are point estimates; we apply a sigmoid around the
  threshold to express uncertainty (sigma defaults to 3°F).
- If anything fails (parser miss, NWS down, missing forecast row), the
  handler returns None and the agent falls through to 0.5.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
USER_AGENT = "(prophet-hacks-agent, benrossjenkins@gmail.com)"
TEMP_SIGMA_F = 3.0  # forecast uncertainty in °F around the threshold
NWS_TIMEOUT = 8.0


# Map common Kalshi city labels (and likely full names) to (lat, lon).
# Central-Park-style station for NYC because that's what KXTEMPNYC* resolves on.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york city": (40.7794, -73.9692),
    "los angeles": (33.9416, -118.4085),
    "chicago": (41.9742, -87.9073),
    "denver": (39.8617, -104.6731),
    "miami": (25.7959, -80.2870),
    "philadelphia": (39.8722, -75.2407),
    "phoenix": (33.4373, -112.0078),
    "boston": (42.3601, -71.0589),
    "austin": (30.1975, -97.6664),
    "seattle": (47.4502, -122.3088),
    "washington": (38.8951, -77.0364),
    "atlanta": (33.6407, -84.4277),
}


# ---- Parsing -------------------------------------------------------------

TITLE_RE = re.compile(
    r"^Will the temp in (?P<city>.+?) be "
    r"(?P<cmp>above|below|at most|at least) "
    r"(?P<threshold>-?\d+(?:\.\d+)?)° "
    r"on (?P<date>[A-Za-z]+ \d{1,2}, \d{4}) "
    r"at (?P<time>\d{1,2}(?::\d{2})?\s?(?:am|pm))\s*(?P<tz>[A-Z]{2,4})?",
    re.IGNORECASE,
)

_TZ_OFFSETS = {
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "UTC": 0,
}


def parse_title(title: str) -> dict[str, Any] | None:
    """Return a parsed dict, or None if the title doesn't match the pattern."""
    m = TITLE_RE.match(title.strip())
    if not m:
        return None
    try:
        threshold_f = float(m["threshold"])
    except (ValueError, TypeError):
        return None

    date_str = m["date"]
    time_str = m["time"].lower().replace(" ", "")
    tz = (m["tz"] or "EDT").upper()
    if tz not in _TZ_OFFSETS:
        return None

    try:
        target_local = datetime.strptime(f"{date_str} {time_str}", "%B %d, %Y %I%p")
    except ValueError:
        try:
            target_local = datetime.strptime(f"{date_str} {time_str}", "%B %d, %Y %I:%M%p")
        except ValueError:
            return None

    # Convert local time to UTC.
    target_utc = target_local - timedelta(hours=_TZ_OFFSETS[tz])

    return {
        "city": m["city"].strip(),
        "comparison": m["cmp"].lower(),
        "threshold_f": threshold_f,
        "target_utc": target_utc,
        "tz": tz,
    }


def city_to_coords(city: str) -> tuple[float, float] | None:
    return CITY_COORDS.get(city.lower())


# ---- NWS API -------------------------------------------------------------

_session: requests.Session | None = None
_grid_cache: dict[tuple[float, float], tuple[str, int, int]] = {}


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    return _session


def _points_lookup(lat: float, lon: float) -> tuple[str, int, int] | None:
    """Look up the NWS gridpoint for (lat, lon). Cached."""
    key = (round(lat, 4), round(lon, 4))
    if key in _grid_cache:
        return _grid_cache[key]
    try:
        r = _get_session().get(f"{NWS_BASE}/points/{key[0]},{key[1]}", timeout=NWS_TIMEOUT)
        r.raise_for_status()
        props = r.json().get("properties", {})
        office = props.get("gridId")
        x = props.get("gridX")
        y = props.get("gridY")
        if not office or x is None or y is None:
            return None
        _grid_cache[key] = (office, x, y)
        return _grid_cache[key]
    except requests.RequestException as e:
        logger.warning("NWS /points failed for %s: %s", key, e)
        return None


def _hourly_forecast(office: str, x: int, y: int) -> list[dict] | None:
    """Fetch the hourly forecast for an NWS grid cell."""
    try:
        r = _get_session().get(
            f"{NWS_BASE}/gridpoints/{office}/{x},{y}/forecast/hourly",
            timeout=NWS_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("properties", {}).get("periods", [])
    except requests.RequestException as e:
        logger.warning("NWS /gridpoints hourly failed for %s/%s,%s: %s", office, x, y, e)
        return None


def _find_period_for(periods: list[dict], target_utc: datetime) -> dict | None:
    """Pick the hourly period whose start time is closest to target_utc."""
    best: dict | None = None
    best_delta: float = float("inf")
    for p in periods:
        start = p.get("startTime", "")
        try:
            start_dt = datetime.fromisoformat(start)
        except (ValueError, TypeError):
            continue
        # Compare UTC. start_dt may be tz-aware (it usually is).
        if start_dt.tzinfo is not None:
            start_naive_utc = start_dt.astimezone(tz=None).replace(tzinfo=None)
            # actually convert to UTC properly
            start_naive_utc = (
                start_dt - start_dt.utcoffset()
            ).replace(tzinfo=None) if start_dt.utcoffset() else start_dt.replace(tzinfo=None)
        else:
            start_naive_utc = start_dt
        delta = abs((start_naive_utc - target_utc).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = p
    # If the closest period is > 90 minutes off, no forecast for that hour.
    if best is None or best_delta > 90 * 60:
        return None
    return best


# ---- Probability conversion ---------------------------------------------


def _prob_above(forecast_f: float, threshold_f: float, sigma: float = TEMP_SIGMA_F) -> float:
    """Sigmoid probability that actual temp > threshold given forecast=mean."""
    return 1.0 / (1.0 + math.exp(-(forecast_f - threshold_f) / sigma))


def comparison_to_prob(
    comparison: str, forecast_f: float, threshold_f: float, sigma: float = TEMP_SIGMA_F
) -> float:
    """Convert a parsed comparison into a YES probability."""
    p_above = _prob_above(forecast_f, threshold_f, sigma)
    if comparison == "above" or comparison == "at least":
        return p_above
    if comparison == "below" or comparison == "at most":
        return 1.0 - p_above
    return 0.5


# ---- Public handler ------------------------------------------------------


def weather_prior(event: dict) -> tuple[float, str] | None:
    """Return (p_yes, rationale) from NWS forecast, or None on any failure."""
    title = event.get("title", "")
    parsed = parse_title(title)
    if parsed is None:
        return None

    coords = city_to_coords(parsed["city"])
    if coords is None:
        return None

    grid = _points_lookup(*coords)
    if grid is None:
        return None

    periods = _hourly_forecast(*grid)
    if not periods:
        return None

    period = _find_period_for(periods, parsed["target_utc"])
    if period is None:
        return None

    try:
        forecast_temp = float(period.get("temperature", "nan"))
    except (ValueError, TypeError):
        return None

    p = comparison_to_prob(parsed["comparison"], forecast_temp, parsed["threshold_f"])
    rationale = (
        f"NWS hourly forecast {forecast_temp:.0f}°F at {parsed['target_utc']:%Y-%m-%d %H:%M}UTC "
        f"vs threshold {parsed['threshold_f']:.2f}°F ({parsed['comparison']})"
    )
    return p, rationale
