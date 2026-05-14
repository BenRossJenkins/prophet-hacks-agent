"""Independent forecast — same evidence sources as `predict()` minus the
market anchor.

`predict()` is built around a market-anchored design: when the Kalshi book
has signal, we trust it (with mild shrinkage). That's the right call for
the forecasting track (Brier rewards calibrated probabilities and the
market IS a calibrated probability for liquid contracts).

For the *trading* track, that design is structurally broken: our forecast
≈ market price, so we never see edge against it. We can't disagree with
the market we're pricing off of.

`independent_forecast()` runs the same evidence stack but skips the
market-anchor step entirely. The trader uses it as a true second opinion.

Returns (p_yes, rationale, confidence) where confidence ∈ {high, medium, low, none}:
  high     - typed external-data prior (NWS, yfinance) fired
  medium   - LLM ensemble fired with at least one grounding marker
             ("search", "according to", "as of", "report", ...)
  low      - LLM ensemble fired but rationale looks speculative
  none     - everything failed, returns 0.5
"""

from __future__ import annotations

from agent.llm import llm_forecast_ensemble
from agent.predict import _LLM_GROUNDED_MARKERS
from agent.priors import category_prior, llm_allowed_for


def _looks_grounded(rationale: str) -> bool:
    rationale_lower = (rationale or "").lower()
    return any(marker in rationale_lower for marker in _LLM_GROUNDED_MARKERS)


def independent_forecast(event: dict) -> tuple[float, str, str]:
    """Return (p_yes, rationale, confidence). Never falls back to market price.

    Confidence guides the trader: only "high" or "medium" should be traded
    on; "low" and "none" produce holds.
    """
    # First, try the typed external-data prior — most reliable when present.
    prior_out = category_prior(event)
    if prior_out is not None:
        p, prior_rationale = prior_out
        p = max(0.01, min(0.99, p))
        return p, f"prior: {prior_rationale}", "high"

    # If the LLM is gated for this category and we have no prior, no trade.
    category = event.get("category", "")
    if not llm_allowed_for(category):
        return 0.5, f"LLM gated for category='{category}' and no prior available", "none"

    out = llm_forecast_ensemble(event)
    if out is None:
        return 0.5, "LLM ensemble unavailable", "none"
    p_raw, llm_rationale = out
    p = max(0.01, min(0.99, p_raw))
    confidence = "medium" if _looks_grounded(llm_rationale) else "low"
    return p, f"LLM ({confidence}): {llm_rationale}", confidence
