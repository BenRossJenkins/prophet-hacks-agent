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
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
MAX_TOKENS = 3000  # bumped from 2000 for the scenarios-then-critique procedure
TIMEOUT_SECONDS = 45.0
MAX_WEB_SEARCHES = 3

# Hard wall-clock deadline for the whole ensemble (anchor + parallel
# fanout). Sized comfortably below the Prophet Arena 10-minute per-event
# budget (600s) so we ALWAYS have time to assemble a response. If we hit
# this, we return whatever vendors did respond; a single hanging vendor
# can't tank our completion rate.
ENSEMBLE_HARD_DEADLINE_SECONDS = 480.0  # 8 minutes

SYSTEM_PROMPT = """\
You are an expert forecaster producing calibrated probability estimates.

Your task: estimate the probability that the given binary event resolves YES.

You have a web_search tool. Use it whenever the event depends on
current/recent information — election state, current weather, sports
outcomes today, markets that have already moved, news within the last
few weeks.

PROCEDURE — work through these steps in order:

Step 1 — SCENARIOS. Identify 2-3 plausible scenarios for how this event
resolves. For each, state the rough base rate / historical frequency
that would apply if you knew nothing else.

Step 2 — INITIAL FORECAST. Combine the scenarios into an initial
probability estimate. Show the weighted reasoning.

Step 3 — COUNTER-ARGUMENT. State the strongest single argument against
your initial forecast. What evidence or scenario would push the
probability in the opposite direction? Be specific.

Step 4 — FINAL FORECAST. Incorporate the counter-argument and produce
your final probability. If the counter-argument materially weakens
your initial position, your final should reflect that. If it doesn't,
explain briefly why.

CALIBRATION RULES:
- Don't be overconfident. Extremes (p < 0.05 or p > 0.95) require very
  strong, specific evidence.
- If still uncertain after searching, return p between 0.30 and 0.70.

OUTPUT FORMAT — your reasoning may appear first, but the FINAL line of
your response MUST be a single valid JSON object:
{"p_yes": <float in [0.01, 0.99]>, "rationale": "<one short sentence summarizing the key evidence + the strongest counter-argument considered>"}
Nothing after the JSON."""


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

    # Injected by the shared-search ensemble: another model has already run
    # web search; their reasoning becomes additional context here. Treat as
    # ONE source — corroborate with your own knowledge, don't blindly defer.
    if search_context := event.get("search_context"):
        parts.append(
            "\n=== Research from a sibling model (treat as one source — "
            "corroborate, don't defer blindly) ===\n"
            f"{search_context}\n"
            "=== End sibling research ==="
        )

    outcomes = event.get("outcomes") or []
    if isinstance(outcomes, list) and len(outcomes) > 2:
        # Multi-outcome: anchor the question to outcomes[0] explicitly so the
        # model knows what "YES" means. Without this it will guess at the
        # question's framing and may answer the wrong thing entirely.
        first = outcomes[0]
        n = len(outcomes)
        parts.append(
            f"\nThis is a {n}-option question with outcomes:\n"
            f"  {outcomes}\n"
            f"YOUR p_yes IS THE PROBABILITY THAT '{first}' IS AMONG THE RESOLVED "
            f"POSITIVE OUTCOMES — not the probability of any other option. If the "
            f"question asks for a top-K winner set, p_yes is P('{first}' makes "
            f"that set). Uniform prior across {n} equally-likely options would "
            f"be ~{1/n:.3f}; a top-K question with K winners has uniform "
            f"prior ~K/{n}."
        )
        # Two worked examples to ground the model in the right reasoning.
        # Without these, LLMs tend to answer "probability this team wins
        # the championship" even when the question is "is this team in the
        # top 4", which is a much higher marginal probability.
        parts.append(
            "\nWorked examples of the right reasoning:\n"
            "Example 1 (single-winner, 30 options, FAVORITE): 'Who wins the "
            "2026 NBA championship? outcomes[0] = Boston Celtics, 30 teams "
            "total.' Boston is a strong favorite at ~22% on prediction "
            "markets → p_yes = 0.22. Uniform prior K/N = 1/30 ≈ 0.033; we "
            "EXCEED it because Boston is a market favorite.\n"
            "Example 2 (top-K, 35 options, K=5, FAVORITE): 'Which acts "
            "finish top 5 at Eurovision? outcomes[0] = France, 35 acts "
            "total.' France is a perennial top finisher; ~40% chance to "
            "make top 5 → p_yes = 0.40. Uniform K/N = 5/35 ≈ 0.143; we "
            "EXCEED it because France is more likely than the uniform "
            "baseline.\n"
            "Example 3 (single-winner, 30 options, LONGSHOT): 'Who wins "
            "the 2026 NBA championship? outcomes[0] = Sacramento Kings, "
            "30 teams total.' Sacramento is a clear non-contender; "
            "prediction markets give them ~0.5%. p_yes = 0.005. Uniform "
            "K/N = 0.033; we go BELOW uniform because Sacramento is a "
            "longshot. This direction is equally important — outcomes[0] "
            "is not always a favorite, and reflexively returning the "
            "uniform prior on unfamiliar candidates loses real Brier.\n"
            "Note: in ALL three examples p_yes is the MARGINAL probability "
            f"for '{first}' alone, not the joint probability of any "
            "specific winner set."
        )
        parts.append(
            "\nAdditionally: include in the JSON a `probabilities` array "
            f"of {{market, probability}} entries — one per outcome — that "
            "sums to the expected number of positive outcomes (1.0 for "
            "single-winner, K for top-K). This is optional but improves "
            "scoring when the server supports multi-class Brier."
        )
    parts.append("\nProvide your probability estimate as JSON.")
    return "\n".join(parts)


def parse_response(text: str) -> tuple[float, str] | None:
    """Pull a JSON forecast out of the model's response. None on any failure.

    For backwards compatibility with existing callers, this returns only
    (p_yes, rationale). Callers that need the multi-outcome distribution
    should use `parse_response_full` instead.
    """
    out = parse_response_full(text)
    if out is None:
        return None
    return out[0], out[2]


def parse_response_full(text: str) -> tuple[float, list[dict] | None, str] | None:
    """Parse the model's JSON output into (p_yes, probabilities, rationale).

    `probabilities` is the optional per-outcome distribution: a list of
    {"market": str, "probability": float} dicts. None if the model didn't
    provide one (binary questions, or model declined).
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # Fallback: scan for a {...} block containing "p_yes". Use a more
        # forgiving regex that tolerates nested {} (for probabilities lists).
        match = re.search(r'\{.*"p_yes".*\}', s, re.DOTALL)
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

    probabilities = None
    raw_probs = data.get("probabilities")
    if isinstance(raw_probs, list) and raw_probs:
        parsed: list[dict] = []
        for entry in raw_probs:
            if not isinstance(entry, dict):
                continue
            market = entry.get("market") or entry.get("outcome")
            prob = entry.get("probability") or entry.get("p")
            if market is None or prob is None:
                continue
            try:
                prob_f = float(prob)
            except (ValueError, TypeError):
                continue
            if not (0.0 <= prob_f <= 1.0):
                continue
            parsed.append({"market": str(market), "probability": prob_f})
        if parsed:
            probabilities = parsed

    rationale = str(data.get("rationale", "")).strip()
    return p, probabilities, rationale or "LLM forecast"


def _vendor_for(model: str) -> str:
    """Detect vendor from model name. Strips a trailing '-thinking' suffix."""
    base = model[: -len("-thinking")] if model.endswith("-thinking") else model
    if base.startswith("claude"):
        return "anthropic"
    if base.startswith("gpt") or base.startswith("o3") or base.startswith("o4"):
        return "openai"
    if base.startswith("gemini"):
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


THINKING_BUDGET_TOKENS = 1500   # extended-thinking allotment


def _anthropic_forecast_full(
    event: dict, model: str, with_web_search: bool
) -> tuple[float, list[dict] | None, str] | None:
    client = _get_anthropic_client()
    if client is None:
        return None

    # Models with the synthetic "-thinking" suffix turn on Claude's
    # extended-thinking mode; the underlying model name is the prefix.
    thinking_enabled = model.endswith("-thinking")
    real_model = model[: -len("-thinking")] if thinking_enabled else model

    kwargs: dict = {
        "model": real_model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": _build_user_prompt(event)}],
    }
    if thinking_enabled:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": "high"}
    if with_web_search:
        kwargs["tools"] = [WEB_SEARCH_TOOL_ANTHROPIC]
    try:
        resp = client.messages.create(**kwargs)
        text = _anthropic_extract_text(resp.content)
    except Exception as e:
        logger.warning("Anthropic call failed (%s): %s", model, e)
        return None
    return parse_response_full(text)


def _anthropic_forecast(event: dict, model: str, with_web_search: bool) -> tuple[float, str] | None:
    out = _anthropic_forecast_full(event, model, with_web_search)
    return None if out is None else (out[0], out[2])


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


def _openai_forecast_full(
    event: dict, model: str, with_web_search: bool
) -> tuple[float, list[dict] | None, str] | None:
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
    return parse_response_full(text)


def _openai_forecast(event: dict, model: str, with_web_search: bool) -> tuple[float, str] | None:
    out = _openai_forecast_full(event, model, with_web_search)
    return None if out is None else (out[0], out[2])


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


def _gemini_forecast_full(
    event: dict, model: str, with_web_search: bool
) -> tuple[float, list[dict] | None, str] | None:
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
    return parse_response_full(text)


def _gemini_forecast(event: dict, model: str, with_web_search: bool) -> tuple[float, str] | None:
    out = _gemini_forecast_full(event, model, with_web_search)
    return None if out is None else (out[0], out[2])


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


def llm_forecast_full(
    event: dict, *, model: str | None = None, with_web_search: bool = True
) -> tuple[float, list[dict] | None, str] | None:
    """Call a single LLM and return (p_yes, probabilities, rationale) or None."""
    model = model or os.environ.get("FORECAST_MODEL", DEFAULT_MODEL)
    vendor = _vendor_for(model)
    if vendor == "anthropic":
        return _anthropic_forecast_full(event, model, with_web_search)
    if vendor == "openai":
        return _openai_forecast_full(event, model, with_web_search)
    if vendor == "google":
        return _gemini_forecast_full(event, model, with_web_search)
    logger.warning("Unknown vendor for model %s", model)
    return None


# Production ensemble: 1 Anthropic (extended-thinking) + 1 OpenAI + 1 Google.
#
# Cross-vendor: Anthropic decorrelated from OpenAI decorrelated from Google.
# At N=3 the median is a real tiebreaker (not just an average), which is
# what makes ensembles meaningfully better than single calls.
#
# Currently pinned to gemini-2.5-flash because the current Gemini API key
# is on a free-tier-only project. To upgrade to gemini-3-pro-preview
# (latest flagship): enable paid billing on the API key's GCP project at
# https://aistudio.google.com/app/apikey, OR generate a new key under the
# prophet-hacks-2026 GCP project (which already has billing), then swap
# the model string below.
ENSEMBLE_MODELS = (
    "claude-opus-4-7-thinking",
    "gpt-5-mini",
    "gemini-2.5-flash",
)


def llm_forecast_ensemble(
    event: dict, *, models: tuple[str, ...] = ENSEMBLE_MODELS, with_web_search: bool = True
) -> tuple[float, str] | None:
    """Run several models, return median p_yes + concatenated rationales.

    Backwards-compatible 2-tuple wrapper around llm_forecast_ensemble_full.
    Uses the shared-search behavior internally — one search-grounded
    Anthropic call feeds its findings to OpenAI + Gemini.
    """
    out = llm_forecast_ensemble_full(event, models=models, with_web_search=with_web_search)
    if out is None:
        return None
    return out[0], out[2]


def _aggregate_probabilities(
    per_model: list[tuple[str, list[dict] | None]],
    outcomes: list[str],
) -> list[dict] | None:
    """Mixture-mean aggregation of per-vendor distributions.

    Critical that this preserves the sum-to-1 invariant: server contract
    requires it, and the previous per-outcome MEDIAN didn't (vendors that
    concentrate mass on different outcomes produced medians summing to
    much less than 1, which downstream normalization then inflated by
    multiplying every probability — including outcomes the vendors had
    explicitly said were ~0).

    Approach:
      1. For each vendor: fill in 0 for outcomes they didn't mention.
      2. Renormalize that vendor's distribution to sum=1.
      3. Per-outcome MEAN across the renormalized vendor distributions.
    The mean of N probability distributions is itself a probability
    distribution, so the result sums to 1 by construction.

    Returns None if fewer than half the vendors supplied a usable
    distribution (so consumer falls back to a synthesized one).
    """
    contributing = [probs for _, probs in per_model if probs]
    if not contributing or len(contributing) < max(1, len(per_model) // 2):
        return None

    normalized: list[dict[str, float]] = []
    for probs in contributing:
        vendor: dict[str, float] = {o: 0.0 for o in outcomes}
        for entry in probs:
            market = entry.get("market")
            if market in vendor:
                try:
                    vendor[market] = max(0.0, float(entry["probability"]))
                except (KeyError, ValueError, TypeError):
                    continue
        total = sum(vendor.values())
        if total <= 0:
            continue
        for o in outcomes:
            vendor[o] /= total
        normalized.append(vendor)

    if not normalized:
        return None

    return [
        {
            "market": o,
            "probability": sum(d[o] for d in normalized) / len(normalized),
        }
        for o in outcomes
    ]


def _split_anchor(models: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    """Pick the Anthropic model as the search-grounding anchor.

    Returns (anchor_model, other_models). Falls back to None when no
    Anthropic model is in the ensemble — caller should use parallel-all path.
    """
    anchor = next((m for m in models if _vendor_for(m) == "anthropic"), None)
    others = tuple(m for m in models if m != anchor) if anchor else models
    return anchor, others


def llm_forecast_ensemble_full(
    event: dict,
    *,
    models: tuple[str, ...] = ENSEMBLE_MODELS,
    with_web_search: bool = True,
) -> tuple[float, list[dict] | None, str] | None:
    """Multi-outcome aware ensemble. Returns (p_yes, probabilities, rationale).

    When `with_web_search=True` and the ensemble contains an Anthropic model,
    we run Anthropic FIRST with web_search enabled, then run the other
    vendors in parallel with web_search OFF but the Anthropic rationale
    injected as `search_context`. This shares one set of search findings
    across N models — meaningful cost + latency win.
    """
    if not models:
        return None
    outcomes = event.get("outcomes") or []
    is_multi = isinstance(outcomes, list) and len(outcomes) > 2

    if len(models) == 1:
        # Defensive: an unexpected exception from the single vendor (e.g.,
        # SDK bug, malformed response) must not propagate out of the
        # ensemble. Treat it like a None return so the caller can retry
        # without web search or fall to uniform.
        try:
            out = llm_forecast_full(
                event, model=models[0], with_web_search=with_web_search
            )
        except Exception as e:
            logger.warning("single-vendor ensemble (%s) raised: %s", models[0], e)
            return None
        if out is None:
            return None
        return out

    results: list[tuple[str, tuple[float, list[dict] | None, str]]] = []
    augmented_event = event
    started_at = time.monotonic()

    def _remaining_budget() -> float:
        return max(1.0, ENSEMBLE_HARD_DEADLINE_SECONDS - (time.monotonic() - started_at))

    # Step 1: search anchor (sequential). Only fires when search is requested.
    if with_web_search:
        anchor_model, other_models = _split_anchor(models)
        if anchor_model is not None:
            # Defensive: any exception from the anchor (SDK bug, network
            # error not caught internally, parse failure) must not bring
            # down the whole ensemble — the other vendors should still
            # run. Treat a raised exception identically to a None return.
            try:
                anchor_out = llm_forecast_full(
                    event, model=anchor_model, with_web_search=True
                )
            except Exception as e:
                logger.warning(
                    "ensemble anchor (%s) raised: %s", anchor_model, e
                )
                anchor_out = None
            if anchor_out is not None:
                results.append((anchor_model, anchor_out))
                # Share findings with the parallel callers. We strip the
                # JSON tail from rationale so the prompt stays clean.
                augmented_event = dict(event)
                augmented_event["search_context"] = anchor_out[2]
            else:
                logger.warning(
                    "shared-search anchor (%s) failed; falling back to all-parallel",
                    anchor_model,
                )
                other_models = models  # nothing was contributed, redo all
        else:
            other_models = models
    else:
        other_models = models

    # Step 2: parallel calls for the remaining models. The whole ensemble
    # is bounded by ENSEMBLE_HARD_DEADLINE_SECONDS — any vendor still
    # outstanding when the deadline hits is abandoned. This prevents one
    # hanging vendor from eating our entire per-event budget.
    parallel_search = with_web_search and not results
    if other_models:
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=len(other_models))
        try:
            futures = {
                ex.submit(
                    llm_forecast_full,
                    augmented_event,
                    model=m,
                    with_web_search=parallel_search,
                ): m
                for m in other_models
            }
            try:
                for fut in concurrent.futures.as_completed(
                    futures, timeout=_remaining_budget()
                ):
                    model = futures[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        logger.warning("ensemble member %s raised: %s", model, e)
                        continue
                    if out is not None:
                        results.append((model, out))
            except concurrent.futures.TimeoutError:
                still_running = [m for f, m in futures.items() if not f.done()]
                logger.warning(
                    "ensemble deadline hit (%.1fs); abandoning %s",
                    ENSEMBLE_HARD_DEADLINE_SECONDS,
                    still_running,
                )
        finally:
            # Don't block the calling thread waiting for hung vendors. Any
            # outstanding HTTP connections will be torn down by their own
            # per-vendor timeouts.
            ex.shutdown(wait=False, cancel_futures=True)

    if not results:
        return None

    p_values = [out[0] for _, out in results]
    p_median = statistics.median(p_values)

    aggregated_probs = None
    if is_multi:
        aggregated_probs = _aggregate_probabilities(
            [(m, out[1]) for m, out in results], outcomes
        )

    def _short(model: str) -> str:
        parts = model.split("-")
        return parts[1] if len(parts) > 1 else model

    parts = [f"{_short(m)}={out[0]:.3f}" for m, out in results]
    rationale = f"ensemble[{','.join(parts)}] → median={p_median:.3f}; " + "; ".join(
        f"{m}: {out[2][:120]}" for m, out in results
    )
    if aggregated_probs is not None:
        rationale += f"; distribution over {len(aggregated_probs)} outcomes"
    return p_median, aggregated_probs, rationale


# Backwards-compat: the old internal helper name. Tests use _extract_text.
_extract_text = _anthropic_extract_text
