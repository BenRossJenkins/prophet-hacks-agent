"""Sportsbook-implied-probability prior for Kalshi 'Sports' category.

Kalshi sports markets ask "Will TEAM beat TEAM?" or "Will TEAM win?"
on a specific date. Pre-game moneyline odds from sportsbooks are the
gold-standard answer: they're the consensus of bettors with skin in
the game, and closing-line research shows them calibrated within
1-2 percentage points of true outcome frequency.

LLMs in contrast are mediocre at sports — they don't have live injury
reports, lineup data, or weather-at-stadium. So a sportsbook-derived
prior should beat the LLM ensemble on every sports market with thin
Kalshi liquidity.

Data source: ESPN site.api scoreboard endpoint
(site.api.espn.com/apis/site/v2/sports/<sport>/<league>/scoreboard).
Public, no auth, no documented rate limits. Per-day in-process cache.

Supported leagues: NBA, NFL, MLB, NHL, NCAA football, NCAA men's
basketball, MLS, EPL. Sports outside this list return None and the
agent falls through to Manifold / LLM.

Strategy:
  1. Detect league from Kalshi event_ticker prefix or title keywords.
  2. Pull scoreboard for the resolution date.
  3. Match team mention(s) in Kalshi title to ESPN competitor name.
  4. Convert American moneyline → de-vigged implied probability.

Defensive throughout: any unclear case returns None so the agent falls
through to the next tier.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
TIMEOUT = 8.0


# ESPN sport/league path segments. Order matters for keyword detection:
# more specific (NCAAF before NFL) should come first so 'Alabama' doesn't
# get matched against the NFL scoreboard.
LEAGUE_PATHS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("basketball", "mens-college-basketball", ("ncaab", "march madness", "ncaa basketball", "kxncaab")),
    ("football", "college-football", ("ncaaf", "college football", "kxncaaf")),
    ("basketball", "nba", ("nba", "kxnba")),
    ("football", "nfl", ("nfl", "kxnfl")),
    ("baseball", "mlb", ("mlb", "kxmlb")),
    ("hockey", "nhl", ("nhl", "kxnhl")),
    ("soccer", "usa.1", ("mls", "kxmls")),
    ("soccer", "eng.1", ("premier league", "epl", "kxepl")),
)


# Per-day scoreboard cache: {(sport, league, yyyymmdd): events}
_scoreboard_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}


def _detect_league(event: dict) -> tuple[str, str] | None:
    """Return (sport, league) path tuple for an event, or None if unknown."""
    haystack = " ".join(
        str(event.get(k, "") or "").lower()
        for k in ("event_ticker", "market_ticker", "title", "subtitle", "category")
    )
    for sport, league, keywords in LEAGUE_PATHS:
        for kw in keywords:
            if kw in haystack:
                return sport, league
    return None


def _yyyymmdd_for_event(event: dict) -> str | None:
    """Resolution date as YYYYMMDD (ESPN scoreboard's `dates` param format)."""
    close = event.get("close_time")
    if not close:
        return None
    try:
        dt = datetime.fromisoformat(str(close).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt.strftime("%Y%m%d")


def _fetch_scoreboard(sport: str, league: str, yyyymmdd: str) -> list[dict[str, Any]]:
    """Return events list for the given league + date. Empty on failure."""
    key = (sport, league, yyyymmdd)
    if key in _scoreboard_cache:
        return _scoreboard_cache[key]
    url = f"{ESPN_BASE}/{sport}/{league}/scoreboard"
    try:
        r = requests.get(url, params={"dates": yyyymmdd}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning("ESPN scoreboard fetch failed (%s/%s, %s): %s", sport, league, yyyymmdd, e)
        return []
    except ValueError:
        return []
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        events = []
    _scoreboard_cache[key] = events
    return events


_NAME_FIELDS = ("displayName", "shortDisplayName", "name", "nickname", "abbreviation", "location")


def _team_name_variants(team: dict[str, Any]) -> set[str]:
    """Every name a Kalshi title might use for this team, lowercased."""
    out: set[str] = set()
    for field in _NAME_FIELDS:
        v = team.get(field)
        if isinstance(v, str) and v:
            out.add(v.lower())
            # Add last word too — e.g. "Los Angeles Lakers" → "lakers"
            tail = v.lower().rsplit(" ", 1)[-1]
            if tail and tail != v.lower():
                out.add(tail)
    return out


def _find_team_in_title(title: str, competitors: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the competitor that is the *subject* of `title`.

    Kalshi titles phrase the question around one team ("Will the Lakers
    beat the Celtics?"). The subject is whichever team appears earliest;
    ties broken by longer name match. If only one team appears at all, it
    wins regardless of position.
    """
    if not title or not competitors:
        return None
    title_lower = title.lower()
    # (earliest_position, -match_length, competitor) — sort minimum wins.
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for comp in competitors:
        team = comp.get("team")
        if not isinstance(team, dict):
            continue
        best_pos = -1
        best_len = 0
        for variant in _team_name_variants(team):
            if len(variant) < 3:
                continue
            pos = title_lower.find(variant)
            if pos == -1:
                continue
            if best_pos == -1 or pos < best_pos or (pos == best_pos and len(variant) > best_len):
                best_pos = pos
                best_len = len(variant)
        if best_pos != -1:
            scored.append((best_pos, -best_len, comp))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]


def american_to_prob(moneyline: float) -> float | None:
    """American moneyline → raw (vig-inclusive) probability."""
    try:
        m = float(moneyline)
    except (ValueError, TypeError):
        return None
    if m == 0:
        return None
    if m < 0:
        return abs(m) / (abs(m) + 100.0)
    return 100.0 / (m + 100.0)


def _devig(p_home: float, p_away: float) -> tuple[float, float] | None:
    total = p_home + p_away
    if total <= 0:
        return None
    return p_home / total, p_away / total


def _extract_moneyline_probs(
    competition: dict[str, Any],
) -> tuple[float, float] | None:
    """Pull (home_prob, away_prob) from a competition's odds. None if missing."""
    odds_list = competition.get("odds")
    if not isinstance(odds_list, list) or not odds_list:
        return None
    # Use the first provider that gives us both sides.
    for odds in odds_list:
        home_ml = ((odds or {}).get("homeTeamOdds") or {}).get("moneyLine")
        away_ml = ((odds or {}).get("awayTeamOdds") or {}).get("moneyLine")
        p_home = american_to_prob(home_ml) if home_ml is not None else None
        p_away = american_to_prob(away_ml) if away_ml is not None else None
        if p_home is None or p_away is None:
            continue
        devigged = _devig(p_home, p_away)
        if devigged is None:
            continue
        return devigged
    return None


def _detect_negation(title: str) -> bool:
    """True if the question asks the negative (TEAM *not* win / lose).

    Kalshi's typical phrasing is positive ("Will the Lakers win?"). Negations
    are rare but possible. Conservative default: assume positive.
    """
    if not title:
        return False
    t = title.lower()
    return bool(re.search(r"\b(not win|lose|loses|fail to win|miss the playoffs)\b", t))


def sports_prior(event: dict) -> tuple[float, str] | None:
    """Pre-game moneyline-derived probability prior. None when unclear.

    Matches the contract of other priors: returns (p_yes, rationale) on
    success, or None to let the agent fall through to the next tier.
    """
    league_path = _detect_league(event)
    if league_path is None:
        return None
    sport, league = league_path

    yyyymmdd = _yyyymmdd_for_event(event)
    if yyyymmdd is None:
        return None

    # Markets resolving more than a few days out won't have pre-game odds yet.
    try:
        deadline = datetime.fromisoformat(str(event.get("close_time", "")).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    days_until = (deadline - datetime.now(UTC)).total_seconds() / 86400.0
    if days_until < -1 or days_until > 7:
        return None

    events = _fetch_scoreboard(sport, league, yyyymmdd)
    if not events:
        return None

    title = event.get("title", "") or ""
    for espn_event in events:
        comps = espn_event.get("competitions")
        if not isinstance(comps, list) or not comps:
            continue
        competition = comps[0]
        competitors = competition.get("competitors") or []
        chosen = _find_team_in_title(title, competitors)
        if chosen is None:
            continue
        probs = _extract_moneyline_probs(competition)
        if probs is None:
            continue
        p_home, p_away = probs
        # competitor['homeAway'] tells us which side `chosen` is on.
        side = (chosen.get("homeAway") or "").lower()
        if side == "home":
            p_chosen = p_home
        elif side == "away":
            p_chosen = p_away
        else:
            continue
        if _detect_negation(title):
            p_chosen = 1.0 - p_chosen
        p_chosen = max(0.01, min(0.99, p_chosen))
        chosen_name = ((chosen.get("team") or {}).get("displayName")) or "?"
        return p_chosen, (
            f"ESPN moneyline ({sport}/{league}, {yyyymmdd}): "
            f"{chosen_name} → p={p_chosen:.3f} (de-vigged)"
        )
    return None
