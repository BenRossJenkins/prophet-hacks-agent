"""Category-specific priors and the LLM-fallback gate.

Two responsibilities:

1. **LLM gate.** A safety mechanism for categories where running a naked
   LLM (no domain prior) historically hurt Brier. Currently empty — see
   the docstring on LLM_DENIED_CATEGORIES below for why.

2. **Category priors.** Plugs in external-data handlers here (e.g., NWS
   forecasts for weather markets, yfinance quotes for crypto markets,
   ESPN moneylines for sports). When a handler returns a probability,
   it takes precedence over the LLM fallback.
"""

from __future__ import annotations

# History: an early backtest (95% temperature questions) showed LLM-enabled
# Brier 0.282 vs LLM-disabled 0.245, so "Climate and Weather" and "Crypto"
# were gated to 0.5 when the typed prior couldn't handle them. But that
# fixture was extremely narrow:
#
#   - Temp questions are handled directly by weather_prior (NWS sigmoid).
#     The denylist only fires when weather_prior returns None — i.e., for
#     hurricane / named-storm / snowfall questions the prior CAN'T handle.
#     Hard-capping those at p=0.5 gives Brier ≤ 0.25 even when the LLM
#     ensemble (with web search) would know the answer.
#
#   - Same for Crypto: crypto_prior handles spot-vs-threshold via yfinance
#     + lognormal. The denylist fires on IPO / CEO / regulatory questions
#     where the LLM with web search performs reasonably well.
#
# So the gate is now empty: the typed priors handle what they're good at,
# and the LLM fallback (with grounded/speculative shrinkage) handles the
# rest. If a specific subcategory regresses, add it back.
LLM_DENIED_CATEGORIES: frozenset[str] = frozenset()


def llm_allowed_for(category: str) -> bool:
    """Is the LLM fallback considered safe for events in this category?"""
    return category not in LLM_DENIED_CATEGORIES


def category_prior(event: dict) -> tuple[float, str] | None:
    """Look up an external-data prior for this event's category.

    Returns (p_yes, rationale) on success, or None to delegate to the LLM
    fallback / uniform prior.
    """
    category = event.get("category", "")
    if category == "Climate and Weather":
        from agent.weather import weather_prior

        return weather_prior(event)
    if category == "Crypto":
        from agent.financials import crypto_prior

        return crypto_prior(event)
    if category == "Sports":
        # Sportsbook moneyline (ESPN) is the canonical answer for individual
        # game markets. Falls through to Manifold for season-long / tournament
        # questions ESPN's scoreboard doesn't cover.
        from agent.sports import sports_prior

        result = sports_prior(event)
        if result is not None:
            return result
        from agent.manifold import manifold_prior

        return manifold_prior(event)
    if category in {"Politics", "Elections", "World", "Companies"}:
        # Manifold has wide coverage of political/world/company events.
        # If no match is found, we return None and fall through to the LLM.
        from agent.manifold import manifold_prior

        return manifold_prior(event)
    return None
