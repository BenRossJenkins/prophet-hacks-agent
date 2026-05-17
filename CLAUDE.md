# CLAUDE.md

Project notes for AI assistants working on this codebase. Briefer than
the README, denser than the source comments. Read this before suggesting
changes to the agent pipeline.

## What this is

Two-person entry (Ben + @duckmoll) for [Prophet Hacks](https://www.prophethacks.com),
a 30-hour AI hackathon. Build window May 16-17, 2026; live evaluation
May 17-31 on [Prophet Arena](https://prophetarena.co).
**Forecasting track only** - organizers required teams to pick a single
track (Forecasting or Trading). All trading-track code was removed.
Scoring is Brier (lower is better). See `README.md` for setup and
`SUBMISSION_CONTRACT.md` for the agent interface.

## Architecture

```
predict(event):
  Multi-outcome (3+ outcomes):
    1. Kalshi event lookup: GET /trade-api/v2/events/{event_ticker}
       ?with_nested_markets=true. Require mutually_exclusive=true,
       map child subtitles to outcomes (exact match + token-subset),
       extract per-child YES probability, require ≥60% coverage.
    2. Polymarket event lookup via /events with the same coverage
       contract.
    3. Capped volume-weighted blend (KALSHI_POLY_MAX_WEIGHT=0.75) when
       both present; otherwise use whichever returned.
    4. If neither: LLM ensemble with explicit p_yes=P(outcomes[0])
       framing + top-K worked examples in the prompt. Retry once
       with web_search=False on total failure.
    5. If LLM also fails: uniform 1/N across outcomes.
    Final distribution always normalized to sum=1; the served p_yes
    (for logging / calibration) is set to dist[0].probability AFTER
    normalization so it stays in sync with what the server scores
    against (v3.14 — earlier versions used a separately-computed
    median that could disagree with dist[0]).

  Binary (2 outcomes):
  1. Fetch Kalshi market by market_ticker.
  2. Derive raw Kalshi price (depth-mid if liquid, else last trade).
  3. Tail-anchor triage: if Kalshi vol_24h ≥ $500 AND raw_p outside
     [0.05, 0.95], return market price directly with a 3% safety
     shrink. (Tails are settled; LLM disagreement costs Brier.)
  4. Cross-venue agreement gate (POLYMARKET_CATEGORIES only):
        - In safe band ([0.20, 0.80] with vol ≥ $10k): fetch Polymarket
          but skip the blend when |kalshi - poly| ≤ 0.03 (agreement
          carries no signal); blend when they disagree.
        - Outside safe band: always blend with vol-weighting. Poly
          also acts as fallback when Kalshi has no signal at all.
  5. Apply volume-weighted shrinkage toward 0.5.
  6. If no market signal at all: try category_prior() (NWS for Weather,
     yfinance for Crypto, ESPN for Sports, Manifold for Politics/World/
     Companies + Sports fallback).
  7. LLM ensemble[Opus-thinking, GPT-5-mini, Gemini-2.5-flash] with
     shared web search (Anthropic anchors search, OpenAI + Gemini
     receive its findings as `search_context`) → median → tail-aware
     non-linear LLM shrinkage with three tiers:
       - decisive (α_base=0.02): rationale mentions an outcome that
         has already resolved (markers: "already won/lost/eliminated/
         clinched", "did not win", "is impossible", etc.).
       - grounded (α_base=0.05): rationale cites current data
         ("according to", "as of", "polls", "source", etc.).
       - speculative (α_base=0.15): base-rate reasoning only.
     Plus extra tail α beyond |p - 0.5| > 0.40 to bound overconfident
     extreme outputs. Capped at α=0.50 to preserve directional signal.
     If the ensemble returns None (all 3 vendors failed), retry ONCE
     with web_search=False before falling to uniform 0.5.
     (LLM denylist is currently empty — the safe-band auto-anchor +
     tail-anchor triage handle the cases the denylist used to gate.)
  8. Market sanity guardrail: if final p deviates >0.30 from a deep
     liquid Kalshi mid (vol_24h ≥ $100k), anchor 0.6/0.4 toward market.
  9. Path-stratified calibration (binary events only). The producing
     branch is stamped into PredictionResponse.path at wrap-time
     (v3.14) — one of ~15 labels: tail-anchor, kalshi-anchor,
     kalshi+poly-blend, guardrail-anchored, prior, llm-decisive,
     llm-grounded, llm-speculative, poly-only, uniform,
     multi-outcome-kalshi, multi-outcome-poly, multi-outcome-blend,
     multi-outcome-llm, multi-outcome-uniform. log_prediction prefers
     metadata.path from the producer; classify_path regex is the
     legacy fallback for entries that didn't stamp.

     Look up that stratum's table from GCS-backed payload (60s cache),
     require n ≥ MIN_BUCKET_N_FOR_PATH (=3, lowered from 5 in v3.14
     once Beta-Bernoulli shrinkage was added) in the matching bucket —
     else fall back to the global table. Per-bucket yes-rates are
     Beta-Bernoulli shrunk toward the bucket's mean_p with prior
     strength N_0=10:
       posterior = (n * mean_actual + N_0 * mean_p) / (n + N_0)
     so a 3-event "all yes" bucket at mean_p=0.6 outputs 0.69, not 1.0.
     The final shift is bounded to ±MAX_CALIBRATION_SHIFT (=0.05) so a
     noisy small-N bucket can't yank a confident prediction wildly off.
  10. Wrap p_yes into {market: outcomes[0], probability: p},
      {market: outcomes[1], probability: 1-p}. Always clamp p to
      [0.01, 0.99] per submission contract.
  11. Log every prediction to PREDICTION_LOG_PATH (local FS) AND
      PREDICTION_LOG_GCS_PREFIX (one object per event) for durability.
      Each log entry carries metadata.path (stamped at producer) and
      metadata.version (AGENT_VERSION) for post-eval attribution.
```

Module map:

- `agent/predict.py` — pipeline orchestration + FastAPI app (`/predict`, `/health`)
- `agent/kalshi.py` — read-only Kalshi market client
- `agent/polymarket.py` — Polymarket cross-reference for politics/news markets
- `agent/priors.py` — category gate + dispatch to typed priors
- `agent/weather.py` — NWS-backed prior for "Climate and Weather"
- `agent/financials.py` — yfinance-backed prior for "Crypto"
- `agent/sports.py` — ESPN moneyline-derived prior for Sports
- `agent/manifold.py` — Manifold-backed prior for Politics/World/Companies/Sports fallback
- `agent/llm.py` — multi-vendor LLM ensemble (Anthropic / OpenAI / Google)
- `agent/calibrate.py` — path-stratified calibration table (GCS, daily refit, bounded ±0.05 shift, Beta-Bernoulli per-bucket shrinkage with N_0=10, `check_calibration_diff` guard for the daily refit cron)
- `agent/prediction_log.py` — defensive append-only log of every forecast; prefers producer-stamped `metadata.path`, falls back to `classify_path` regex for legacy entries

## Conventions

**Commits.** Lowercase prefix, present tense (`v2.7: …`, `fix: …`). **Never
add a `Co-Authored-By: Claude <…>` trailer** to commit messages on this
project — explicit user preference.

**Tests required.** Every `agent/` module has a `tests/test_*.py`. We use
`unittest.mock.patch` heavily; live API checks live in `scripts/`. Default
fixture in `tests/conftest.py` isolates `PREDICTION_LOG_PATH` per test so
runs don't accumulate state.

**Defensive degradation everywhere.** Every external call (Kalshi, NWS,
yfinance, LLM vendors) must degrade gracefully:

- Kalshi unavailable → fall through to prior → ensemble → 0.5
- Prior errors → return None, let the next tier try
- Ensemble: per-vendor failures dropped; median of survivors wins
- `log_prediction` never raises (a failed log must not break `/predict`)

**Output contract (2026-05-16 update).** The wire response is
`{"probabilities": [{"market": <outcome>, "probability": <float>}, ...]}`.
Probabilities MUST sum to 1.0 (strict, the server normalizes
otherwise but we should be exact). Each `market` MUST match one of the
event's `outcomes` exactly — typos = silent miss. `p_yes` and
`rationale` are extra fields the server ignores but we keep for our
own calibration / logging and CLI backwards-compat.

The internal pipeline still computes a binary `p_yes` for outcomes[0]
on 2-outcome events; the `_wrap_binary()` helper turns it into a
2-element distribution. Multi-outcome events build the distribution
directly from the LLM ensemble (with normalization to enforce sum=1).

**Don't shrink prior outputs.** `agent/weather.py` (sigmoid bandwidth)
and `agent/financials.py` (lognormal sigma) already model uncertainty
internally. Additional shrinkage on top would double-count.

## Gotchas

1. **Local DNS doesn't resolve Kalshi or Polymarket** on this dev machine.
   `api.elections.kalshi.com` and `gamma-api.polymarket.com` both return
   NXDOMAIN via the system resolver but resolve fine via 8.8.8.8. Live
   scripts use a `dnspython`-based override (see `scripts/check_kalshi_live.py`
   for the template). Production hosts (Cloud Run) have working DNS, so
   this only affects local smoke tests, not production behavior or unit
   tests (which mock the HTTP layer).

2. **`expiration_time` ≠ "trading ended".** Kalshi's `expiration_time`
   is the formal calendar deadline; many markets settle days or weeks
   before. For historical snapshots use `settlement_ts` instead.

3. **`updated_time` doesn't track book activity.** Active markets with
   $300k+ 24h volume can show 4-day-old `updated_time` because the
   field reflects metadata changes, not order-book changes. The
   stale-book check exists but is dormant (`APPLY_STALENESS=False`
   in `agent/predict.py`). Don't enable without a better signal.

4. **PyPI `ai-prophet-core==0.1.3` is behind upstream.** Released
   version hard-fails on missing Kalshi credentials for read-only ops;
   GitHub HEAD made them optional. We have an upstream PR open.

5. **Per-event budget is 10 minutes** (confirmed by organizers
   2026-05-16). Ensemble has a hard deadline `ENSEMBLE_HARD_DEADLINE_SECONDS`
   = 480s (8 min) — any vendor still outstanding at that point is
   abandoned and we return whatever partial responses we got. With
   shared web search (Anthropic anchors then OpenAI/Gemini run in
   parallel with `search_context` injected) typical latency is
   30-60 s; the deadline is just safety.

   **Operational rule of thumb:** healthy ensemble p99 should be
   under 90s. If Cloud Run logs show p99 climbing past 180s, one
   vendor is regressing — identify the slow member from the
   per-vendor logs and drop it from `ENSEMBLE_MODELS`. The
   survivor-median path handles partial ensembles natively, so this
   is a config change, not code.

6. **Web search in backtest = leakage.** Settled markets are public
   history; a web-searching LLM finds the actual outcome and looks
   brilliant while being useless forward-looking. `scripts/backtest.py
   --with-llm` already disables web search — don't undo that.

7. **Two coupled thresholds gate the market-anchor path** (both in
   `agent/predict.py`):

   - `MIN_VOL_24H = 10` — minimum 24h volume (USD) below which we
     normally reject the book as too noisy to trust.
   - `TIGHT_SPREAD_FOR_LOW_VOL = 0.03` — bypass: a book with a
     bid-ask spread tighter than this is trusted at *any* volume,
     even zero. Captures settled-direction markets (e.g. 0.99/1.00
     pinned book with no recent trades) where the market has
     unambiguous consensus but volume is zero because there's
     nothing left to trade against. Added v3.12 after a market-
     baseline backtest showed agent losing 0.025 Brier on this
     exact pattern (Trump-event rows in the candlestick fixture);
     bypass closed two-thirds of that gap.

   If you change either, re-run the market-baseline comparison
   (`/tmp/market_baseline_compare.py` pattern — local fixture
   comparison against raw Kalshi mid) to verify the per-category
   deltas don't regress.

8. **`_wrap_binary` requires `path=`** (v3.14). Every call site must
   pass the producing pipeline branch as a kwarg. Forgetting it raises
   TypeError at call time — which is intentional: it's the only way
   to catch missing stamps before they silently fall back to the
   `classify_path` regex and corrupt the path-stratified calibration
   table. If you add a new branch, add the path label to the same
   list `log_prediction` and `classify_path` know about, and write
   one regression test in `tests/test_predict.py` asserting
   `entry["metadata"]["path"]` matches.

9. **Calibration diff-sanity is fail-closed** (v3.14). The daily
   `scripts/fit_calibration.py` cron loads the previously-published
   table from GCS, calls `check_calibration_diff`, and exits 3 (no
   save, no GCS push) if any bucket with `n < 20` moved by more than
   `0.20` since the prior table. The live agent keeps using the
   already-published table — preferred over publishing a noisy one.
   `--skip-diff-sanity` overrides; only use it after inspecting the
   resolved-predictions for actual anomalies.

## Backtest workflow

Two fixtures live under `tests/fixtures/`:

- `resolved_markets.jsonl` — Kalshi candlestick-derived snapshots at
  75% of market lifetime + known outcome. Committed; reproducible via
  `scripts/build_backtest_fixture.py`. Weather-heavy (Kalshi settles
  short-lifetime markets most often).
- `resolved_markets_live.jsonl` — built from real-time captures
  (`scripts/capture_live_snapshots.py`) that we resolve daily
  (`scripts/resolve_captures.py`). Gitignored. Run capture daily.

Harness: `python scripts/backtest.py [fixture.jsonl]`. Reports overall
Brier + by category + by liquidity tier + calibration buckets.

**Brier deltas < 0.005 are noise** at current N. Look for 0.01+ to
claim a real win.

## What NOT to do

- **Don't enable LLM blend on liquid markets** without per-category
  validation. The pipeline structurally avoids this already: the
  safe-band auto-anchor (step 4) keeps the LLM out of the loop on
  liquid Kalshi books in [0.20, 0.80], and the market sanity
  guardrail (step 8) catches anything that does drift more than 0.30
  from a deep mid. If you're tempted to add LLM blending on liquid
  markets, validate per-category on resolved data first; the
  weather-heavy fixture isn't representative enough to A/B safely.
- **Don't add models to the ensemble** unless they're a new vendor.
  Same-family variants share architecture and training data; they're
  highly correlated and add cost without decorrelating errors.
- **Don't re-add categories to LLM_DENIED_CATEGORIES** without
  measured evidence. The denylist was emptied 2026-05-16 because
  typed priors handle their subcategories, and falling to 0.5 on
  questions the LLM ensemble could answer (e.g., hurricane track,
  IPO timing) wastes Brier. Add back only if a specific subcategory
  regresses on live calibration data.
- **Don't return naked `p_yes`** as the wire response. The server
  contract is `probabilities` (sum to 1, markets match outcomes).
  Every code path goes through `_wrap_binary` or
  `_normalize_distribution` to produce a valid distribution.
- **Don't run `git push --force`, `git reset --hard`, or `--no-verify`**
  without explicit approval.
- **Don't commit `.env`.** It holds live API keys. Already gitignored;
  don't override.

## Useful commands

```bash
.venv/bin/pytest tests/ -q                          # full test suite
.venv/bin/python scripts/check_kalshi_live.py       # live Kalshi probe
.venv/bin/python scripts/build_backtest_fixture.py  # rebuild candlestick fixture
.venv/bin/python scripts/backtest.py                # fixture backtest
.venv/bin/python scripts/capture_live_snapshots.py  # daily live capture
.venv/bin/python scripts/resolve_captures.py        # promote settled live
.venv/bin/python scripts/resolve_predictions.py     # mark logged predictions resolved
.venv/bin/python scripts/analyze_predictions.py     # inspect live calibration
uvicorn agent.predict:app --host 0.0.0.0 --port 8000  # local server
```

## Environment

See `.env.example`. Required for the full pipeline:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` (ensemble);
`PA_SERVER_API_KEY` once organizers issue it. Optional:
`PREDICTION_LOG_PATH` (default `data/predictions.jsonl`).
