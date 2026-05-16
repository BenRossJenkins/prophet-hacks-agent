# Architecture decisions

A running record of the non-obvious design choices in the Brier Patch
forecasting agent. Captured during the build window for the ICML
workshop talk if we place, and so the rationale survives outside git
history.

## Core framing

**Markets are informative priors, LLMs are evidence sources.**
Most LLM-based forecasting agents in the field treat the LLM as the
forecaster: take a question, ask Claude / GPT for a probability, return
it. That ignores the most calibrated source of forecasts available —
the prediction market price itself, which already aggregates the
beliefs of human bettors with skin in the game. Prophet Arena's own
paper documents that LLMs aggregate information slower than markets
near resolution.

Our pipeline inverts the usual stack: the Kalshi (and Polymarket) price
is the prior. The LLM ensemble is consulted only when the market lacks
signal (illiquid book) or when a structural reason exists for the model
to add value (multi-outcome questions with no Kalshi sibling, edge-band
questions where the market consensus is genuinely uncertain).

This connects to AGM-Bench-style bounded belief revision: every
deviation from the market prior is a belief update with associated
confidence, and we cap the magnitude of those updates so a single
noisy signal can't override a well-calibrated market.

## Decisions

### Empty LLM_DENIED_CATEGORIES

Earlier in the build we maintained a denylist of categories where the
LLM ensemble had measurably underperformed (`Climate and Weather`,
`Crypto`). The denylist gated those categories to a uniform 0.5
fallback when the typed prior (NWS, yfinance) couldn't handle the
question.

**Why removed:** the denylist was only firing on subcategories the
typed prior couldn't handle — hurricane tracks, IPO timing, crypto
exchange events. Hard-capping those at p=0.5 gives Brier ≤ 0.25 even
when the LLM ensemble with web search would correctly answer 0.05 or
0.95. Removing the denylist lets the LLM handle these (with
tail-aware shrinkage as protection); typed priors still run first on
the cases they're good at.

**Risk accepted:** if the LLM regresses badly on some specific
subcategory we'll see it in the path-stratified calibration log
post-eval; can add back surgically if needed.

### Safe-band auto-anchor + cross-venue agreement gate

Kalshi prices in the central band [0.20, 0.80] with vol_24h ≥ $10k
have historically been well-calibrated. Naive blending with another
venue (Polymarket) on every question would add variance without
information when both venues agree.

**Decision:** in the safe band, fetch Polymarket but skip the blend
when `|kalshi - poly| ≤ 0.03`. Disagreement above that threshold
triggers the blend (disagreement carries information). Outside the
safe band, blend unconditionally — a thin Kalshi tail-price is
exactly where cross-venue disagreement is most informative.

**Why "fetch but skip" instead of "skip entirely":** the cost of the
Polymarket API call is negligible (~50ms) compared to the latency
budget. Always fetching gives us the disagreement signal when it
matters.

### Path-stratified calibration

Original plan was per-category calibration buckets. Re-analysis
showed that at our N (~200 events / 14 days / 6-8 categories =
25-35 per category), per-bucket sample sizes are too small for
category-level binning to be reliable.

**Decision:** stratify by **pipeline branch** instead of category.
Each prediction is labeled with the path that produced it
(tail-anchor / kalshi-anchor / kalshi+poly-blend / llm-grounded /
llm-speculative / etc.). Path-stratified tables capture the
fundamentally different error distributions of each branch — a
Politics question resolved by a deep Kalshi book has the same error
shape as a Sports question resolved the same way; both differ
wildly from an LLM-speculative call with no market signal.

**Bounded update:** the final calibration shift is capped at ±0.05
from raw. A noisy small-N bucket can correct a prediction by no more
than 5 percentage points. This is the AGM-style bounded-update rule
applied to calibration: each correction has bounded influence.

### Shared web search across the ensemble

Three vendors each running their own `web_search` tool means three
independent searches that converge on the same news anyway, at 3x
cost and 3x latency.

**Decision:** Anthropic anchors the search (one call with
web_search=True). Its rationale gets injected as `search_context`
into the OpenAI and Gemini calls, which run in parallel with
web_search=False. The three models still produce independent
forecasts but reason on a shared evidence base.

**Trade-off:** reduces ensemble independence on the evidence side
(all models see the same Anthropic-mediated context). Gains:
~30% latency reduction, ~50% search-cost reduction, and the LLM
prompts can be a bit denser because we don't worry about each
model finding the same facts independently. If the anchor fails,
we fall back to the original all-parallel-with-search path.

### Tail-aware non-linear LLM shrinkage

Linear shrinkage at α=0.05 only pulls p=0.95 to 0.928 — not much
protection against the LLM-confidently-wrong case that dominates
Brier under squared loss.

**Decision:** linear baseline α for the central range, plus extra
α proportional to (distance from 0.5 - 0.40) at the tails. At p=0.95
grounded this gives final α≈0.15, shrinking to 0.88. Capped at α=0.50
so we never push past 0.75 / 0.25 (which would flip the LLM's
directional signal entirely).

**Why:** asymmetric Brier means a confident-wrong at 0.95 costs us
0.90; hedged-wrong at 0.85 costs 0.72; hedged-wrong at 0.75 costs
0.56. Pulling the LLM down to 0.88 when it claims 0.95 saves us
real Brier on the miss case and costs us very little on the hit
case (0.88 vs 0.95 contribution to squared error is small).

### Async-with-deadline ensemble

The eval server gives us 10 minutes per event. A naive ensemble that
waits for all vendors can stall on a single hung Anthropic / OpenAI /
Gemini call.

**Decision:** hard ensemble deadline at 480s (8 min, well under the
10-min budget). Any vendor still outstanding at the deadline is
abandoned and we return the median of whatever did respond. Falls
back to market price if fewer than 2 vendors arrived.

**Why:** completion rate is multiplicative on the final score, so a
hung request hurts as much as a wrong prediction. Trading vendor
abandonment for guaranteed-completion is the right Brier trade.

### Tail-anchor return-path shrinkage (3%)

When Kalshi is at p>0.95 or p<0.05 with vol_24h ≥ $500, we return
the market price directly (the tail-anchor triage). But markets
are slightly overconfident at the extremes — a 0.97 market that
resolves NO costs (0.97)² ≈ 0.94 per event.

**Decision:** apply a tiny 3% pull toward 0.5 on the tail-anchor
return path. p_final = 0.97 × p_market + 0.03 × 0.5. Costs almost
nothing on correct markets; recovers a slice of Brier on the
occasional surprise.

### GCS-mirrored prediction log

Cloud Run containers have ephemeral local filesystems. A revision
rollout or cold start would wipe `data/predictions.jsonl`.

**Decision:** every prediction is also written as a per-event JSON
object to `PREDICTION_LOG_GCS_PREFIX` (one file per event, key =
`<ts>_<market_ticker>.json`). The daily calibration cron reads from
GCS, not local FS. Defensive: GCS write failures are silent — they
never block /predict.

### What we explicitly did NOT do

- **Self-consistency on the LLM ensemble** (run each model twice with
  different temperatures): Opus extended-thinking doesn't expose a
  temperature knob, and averaging two same-model calls just averages
  noise without adding real signal.
- **Adversarial second-pass** ("explain why this forecast is wrong"
  then re-blend): the system prompt already enforces a
  counter-argument step in the SCENARIOS → INITIAL → COUNTER → FINAL
  procedure. Adding another LLM call mostly duplicates this.
- **Per-category calibration**: at our N, per-bucket sample sizes are
  too small. Path-stratified subsumes it: most categorical error
  patterns map back to pipeline-branch error patterns.
- **Inter-model agreement as confidence signal**: the cross-vendor
  ensemble exists specifically to capture *disagreement* — using
  agreement as confidence would undermine the ensemble's purpose.

## Workshop framing

If we place at ICML, the architectural story is:

1. **Treat markets as informative priors, not competition.** The agent
   anchors on the prediction market price and uses the LLM ensemble
   only when the market lacks signal (illiquid, multi-outcome) or
   when a structural reason exists for the model to add value.
2. **Bounded belief revision.** Every deviation from the market prior
   is bounded: per-LLM tail shrinkage prevents directional flips,
   the market sanity guardrail anchors back when blends drift too
   far, calibration corrections are capped at ±0.05 magnitude.
3. **Pipeline-branch calibration.** Standard calibration treats the
   forecasting agent as a single emitter. Stratifying by which
   internal branch produced the forecast captures the
   heterogeneity of error sources — a market-anchored prediction
   has fundamentally different miscalibration than an LLM-speculative
   one, and the calibration map adapts accordingly.

This is distinct from "Claude forecasts everything" or "ensemble of
LLMs" framings dominant in the literature, and the ±0.05 bounded
update connects naturally to AGM-Bench belief revision.
