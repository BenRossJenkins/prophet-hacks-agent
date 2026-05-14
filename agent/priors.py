"""Category-specific priors and the LLM-fallback gate.

Two responsibilities:

1. **LLM gate.** Some categories (Climate and Weather, Financials) have
   shown systematic LLM error in backtests because the LLM lacks current
   external data (forecasts, market prices). For those categories the
   uniform-prior fallback is *less wrong* than a confident LLM guess.

2. **Category priors.** Phase 4 plugs in category-specific external-data
   handlers here (e.g., NWS forecasts for weather markets, yfinance
   quotes for financial markets). When a handler returns a probability,
   it takes precedence over the LLM fallback.
"""

from __future__ import annotations

# Categories where the LLM fallback is suppressed.
#
# 2026-05-14 backtest (n=117, 95% Climate and Weather): LLM-enabled Brier 0.282
# vs LLM-disabled 0.245. The miss was concentrated in p ∈ [0.2, 0.5]: LLM
# under-predicted YES by 30-50 percentage points because it reasons from base
# rates without current weather/market data. Same dynamic expected for live
# crypto price markets — these need a live spot quote, not LLM speculation.
#
# Note: "Financials" is NOT on the denylist — that category is dominated by
# IPO/CEO-succession markets, which are knowledge/news questions where the
# LLM with web search performs reasonably.
LLM_DENIED_CATEGORIES = frozenset(
    {
        "Climate and Weather",
        "Crypto",
    }
)


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
    return None
