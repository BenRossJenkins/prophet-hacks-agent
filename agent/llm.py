"""Minimal LLM client for forecasting illiquid markets.

Used as the fallback when Kalshi gives us no usable price signal at all.
Defensive: returns None on any failure (missing key, parse error, timeout,
out-of-range output) so the caller can fall back to a uniform 0.5.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
MAX_TOKENS = 400
TIMEOUT_SECONDS = 10.0

SYSTEM_PROMPT = """\
You are an expert forecaster trained for calibrated probability estimation.

Your task: estimate the probability that the given binary event resolves YES.

CALIBRATION RULES:
- Consider base rates for similar events before adjusting on details.
- Weight evidence by reliability AND recency.
- Account for uncertainty - don't be overconfident.
- Extremes (p < 0.05 or p > 0.95) require very strong, specific evidence.
- If you don't have enough information, return p between 0.30 and 0.70.

Output ONLY a single-line valid JSON object:
{"p_yes": <float in [0.01, 0.99]>, "rationale": "<one short sentence>"}
No other text."""


def _build_user_prompt(event: dict) -> str:
    parts = [f"Event title: {event.get('title', '?')}"]
    if subtitle := event.get("subtitle"):
        parts.append(f"Subtitle: {subtitle}")
    if description := event.get("description"):
        parts.append(f"Description: {description}")
    if rules := event.get("rules"):
        parts.append(f"Resolution rules: {rules}")
    if category := event.get("category"):
        parts.append(f"Category: {category}")
    if close_time := event.get("close_time"):
        parts.append(f"Resolution deadline: {close_time}")
    parts.append("\nProvide your probability estimate as JSON.")
    return "\n".join(parts)


def parse_response(text: str) -> tuple[float, str] | None:
    """Pull a JSON forecast out of the model's response. None on any failure."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"p_yes"[^{}]*\}', s, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    try:
        p = float(data["p_yes"])
    except (KeyError, ValueError, TypeError):
        return None
    if not (0.0 <= p <= 1.0):
        return None
    p = max(0.01, min(0.99, p))
    rationale = str(data.get("rationale", "")).strip()
    return p, rationale or "LLM forecast"


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        _client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("Anthropic client init failed: %s", e)
        return None
    return _client


def llm_forecast(event: dict, *, model: str | None = None) -> tuple[float, str] | None:
    """Call Claude to forecast the event. Returns (p_yes, rationale) or None on failure."""
    client = _get_client()
    if client is None:
        return None
    model = model or os.environ.get("FORECAST_MODEL", DEFAULT_MODEL)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(event)}],
        )
        text = resp.content[0].text
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None
    return parse_response(text)
