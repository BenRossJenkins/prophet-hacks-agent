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
  1. Fetch Kalshi market by market_ticker.
  2. Derive raw Kalshi price (depth-mid if liquid, else last trade).
  3. If category in POLYMARKET_CATEGORIES, fetch Polymarket sibling and
     volume-weighted-blend with the Kalshi price. Polymarket also acts as
     a fallback when Kalshi has no signal.
  4. Apply volume-weighted shrinkage toward 0.5.
  5. If no market signal at all: try category_prior() (NWS for Weather,
     yfinance for Crypto, Manifold for Politics/Sports/World/Companies).
  6. Else if category is on the LLM denylist: 0.5.
  7. Else ensemble[Opus-thinking, GPT-5-mini, Gemini-2.5-flash] with web
     search → median → two-tier LLM shrinkage → clamp.
  8. Always clamp output to [0.01, 0.99] (submission contract).
  9. Log every prediction to PREDICTION_LOG_PATH.
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
- `agent/calibrate.py` — binned calibration table (GCS-backed daily refit)
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

**Output is sacred.** Always clamp `p_yes` to `[0.01, 0.99]` before
returning; Pydantic 422s anything outside, which breaks the contract.

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

5. **LLM ensemble latency is ~25–30 s** with web search across four
   vendors, bounded by the slowest member. We sit at the per-event 30 s
   timeout edge. If we see timeouts in production, drop one Anthropic
   member from `agent.llm.ENSEMBLE_MODELS`.

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
  validation. We deferred this in v2.10 because naive blending of an
  LLM with a well-calibrated market price usually hurts Brier and our
  fixture is too weather-heavy to A/B safely.
- **Don't expand category-specific priors** beyond Weather + Crypto
  without first proving the existing LLM behavior is broken there.
  Climate/Crypto are denylisted because of a measured failure;
  Financials was on the denylist but came off after we saw the actual
  market shapes (IPO / CEO questions, not price thresholds).
- **Don't add models to the ensemble** unless they're a new vendor.
  Same-family variants share architecture and training data; they're
  highly correlated and add cost without decorrelating errors.
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
