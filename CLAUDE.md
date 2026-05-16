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
  Multi-outcome (3+ outcomes) → straight to LLM ensemble with explicit
                                 framing; build per-outcome distribution
                                 summing to 1; skip everything else.

  Binary (2 outcomes):
  1. Fetch Kalshi market by market_ticker.
  2. Derive raw Kalshi price (depth-mid if liquid, else last trade).
  3. Tail-anchor triage: if Kalshi vol_24h ≥ $500 AND raw_p outside
     [0.05, 0.95], return market price directly — skip Polymarket, LLM,
     shrinkage. (Tails are settled; LLM disagreement costs Brier.)
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
     non-linear LLM shrinkage. (LLM denylist is currently empty — the
     safe-band auto-anchor + tail-anchor triage handle the cases the
     denylist used to gate.)
  8. Market sanity guardrail: if final p deviates >0.30 from a deep
     liquid Kalshi mid (vol_24h ≥ $100k), anchor 0.6/0.4 toward market.
  9. Path-stratified calibration (binary events only): classify the
     rationale into one of ~12 pipeline-branch labels, look up that
     stratum's table from GCS-backed payload (60s cache), require
     n ≥ MIN_BUCKET_N_FOR_PATH (=5) in the matching bucket — else fall
     back to the global table. The final shift is bounded to
     ±MAX_CALIBRATION_SHIFT (=0.05) so a noisy small-N bucket can't
     yank a confident prediction wildly off.
  10. Wrap p_yes into {market: outcomes[0], probability: p},
      {market: outcomes[1], probability: 1-p}. Always clamp p to
      [0.01, 0.99] per submission contract.
  11. Log every prediction to PREDICTION_LOG_PATH (local FS) AND
      PREDICTION_LOG_GCS_PREFIX (one object per event) for durability.
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
- `agent/calibrate.py` — path-stratified calibration table (GCS, daily refit, bounded ±0.05 shift)
- `agent/prediction_log.py` — defensive append-only log of every forecast

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
