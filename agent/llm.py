"""Multi-vendor LLM client for forecasting illiquid markets.

Supports Anthropic Claude, OpenAI GPT-5 family, and Google Gemini. Each
vendor gets a native web-search tool wired in so the model can ground its
answer in current data. All vendors share the same calibration-focused
system prompt and JSON-output parser; the median across an ensemble is
what the agent ultimately uses.

Defensive throughout: any single vendor failure (missing key, network,
parse error, out-of-range output) returns None and the caller falls back.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import statistics

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
MAX_TOKENS = 2000
TIMEOUT_SECONDS = 45.0
MAX_WEB_SEARCHES = 3

SYSTEM_PROMPT = """\
You are an expert forecaster producing calibrated probability estimates.

Your task: estimate the probability that the given binary event resolves YES.

You have a web_search tool. Use it whenever the event depends on
current/recent information you can't be confident about from training
data alone — election state, current weather, sports outcomes today,
markets that have already moved, news within the last few weeks.

CALIBRATION RULES:
- Consider base rates for similar events before adjusting on details.
- Weight evidence by reliability AND recency.
- Don't be overconfident. Extremes (p < 0.05 or p > 0.95) require
  very strong, specific evidence.
- If you don't have enough information even after searching, return
  p between 0.30 and 0.70.

When you have your final answer, output ONLY a single JSON object on
its own line:
{"p_yes": <float in [0.01, 0.99]>, "rationale": "<one short sentence>"}
No other text in the final message."""


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
    s = (text or "").strip()
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


def _vendor_for(model: str) -> str:
    """Detect vendor from model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    return "anthropic"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL_ANTHROPIC = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": MAX_WEB_SEARCHES,
}

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        _anthropic_client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("Anthropic client init failed: %s", e)
        return None
    return _anthropic_client


def _anthropic_extract_text(content_blocks) -> str:
    parts: list[str] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text" or (hasattr(block, "text") and block_type is None):
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _anthropic_forecast(event: dict, model: str, with_web_search: bool) -> tuple[float, str] | None:
    client = _get_anthropic_client()
    if client is None:
        return None
    kwargs: dict = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_user_prompt(event)}],
    }
    if with_web_search:
        kwargs["tools"] = [WEB_SEARCH_TOOL_ANTHROPIC]
    try:
        resp = client.messages.create(**kwargs)
        text = _anthropic_extract_text(resp.content)
    except Exception as e:
        logger.warning("Anthropic call failed (%s): %s", model, e)
        return None
    return parse_response(text)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=api_key, timeout=TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("OpenAI client init failed: %s", e)
        return None
    return _openai_client


def _openai_forecast(event: dict, model: str, with_web_search: bool) -> tuple[float, str] | None:
    client = _get_openai_client()
    if client is None:
        return None
    tools = [{"type": "web_search"}] if with_web_search else None
    try:
        kwargs: dict = {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": _build_user_prompt(event),
            "max_output_tokens": MAX_TOKENS,
        }
        if tools:
            kwargs["tools"] = tools
        resp = client.responses.create(**kwargs)
        text = getattr(resp, "output_text", None) or ""
    except Exception as e:
        logger.warning("OpenAI call failed (%s): %s", model, e)
        return None
    return parse_response(text)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.warning("Gemini client init failed: %s", e)
        return None
    return _gemini_client


def _gemini_forecast(event: dict, model: str, with_web_search: bool) -> tuple[float, str] | None:
    client = _get_gemini_client()
    if client is None:
        return None
    try:
        from google.genai import types

        config_kwargs: dict = {"system_instruction": SYSTEM_PROMPT}
        if with_web_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        config = types.GenerateContentConfig(**config_kwargs)
        resp = client.models.generate_content(
            model=model,
            contents=_build_user_prompt(event),
            config=config,
        )
        text = getattr(resp, "text", None) or ""
    except Exception as e:
        logger.warning("Gemini call failed (%s): %s", model, e)
        return None
    return parse_response(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_forecast(
    event: dict, *, model: str | None = None, with_web_search: bool = True
) -> tuple[float, str] | None:
    """Call a single LLM (vendor auto-detected). Returns (p_yes, rationale) or None."""
    model = model or os.environ.get("FORECAST_MODEL", DEFAULT_MODEL)
    vendor = _vendor_for(model)
    if vendor == "anthropic":
        return _anthropic_forecast(event, model, with_web_search)
    if vendor == "openai":
        return _openai_forecast(event, model, with_web_search)
    if vendor == "google":
        return _gemini_forecast(event, model, with_web_search)
    logger.warning("Unknown vendor for model %s", model)
    return None


# Four-model cross-vendor ensemble: 2 Anthropic + 1 OpenAI + 1 Google.
ENSEMBLE_MODELS = (
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "gpt-5-mini",
    "gemini-2.5-flash",
)


def llm_forecast_ensemble(
    event: dict, *, models: tuple[str, ...] = ENSEMBLE_MODELS, with_web_search: bool = True
) -> tuple[float, str] | None:
    """Run several models in parallel, return median p_yes + concatenated rationales."""
    if not models:
        return None
    if len(models) == 1:
        return llm_forecast(event, model=models[0], with_web_search=with_web_search)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as ex:
        futures = {
            ex.submit(llm_forecast, event, model=m, with_web_search=with_web_search): m
            for m in models
        }
        results: list[tuple[str, tuple[float, str]]] = []
        for fut in concurrent.futures.as_completed(futures):
            model = futures[fut]
            try:
                out = fut.result()
            except Exception as e:
                logger.warning("ensemble member %s raised: %s", model, e)
                continue
            if out is not None:
                results.append((model, out))

    if not results:
        return None

    p_values = [out[0] for _, out in results]
    p_median = statistics.median(p_values)

    def _short(model: str) -> str:
        parts = model.split("-")
        return parts[1] if len(parts) > 1 else model

    parts = [f"{_short(m)}={out[0]:.3f}" for m, out in results]
    rationale = f"ensemble[{','.join(parts)}] → median={p_median:.3f}; " + "; ".join(
        f"{m}: {out[1][:120]}" for m, out in results
    )
    return p_median, rationale


# Backwards-compat: the old internal helper name. Tests use _extract_text.
_extract_text = _anthropic_extract_text
