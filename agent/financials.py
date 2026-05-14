"""Crypto-price prior for Kalshi "Crypto" category markets.

Handles markets like:
- "Bitcoin price on <date>" with subtitle "$X or above" (at a specific time)
- "Will Bitcoin be above $X by <date>" (by deadline)

Approach: pull current spot via yfinance, apply a log-normal price-diffusion
model from now to close_time, integrate over the resolution condition.

Returns None if anything is unclear (unknown asset, unparseable threshold,
yfinance unavailable, market already closed) so the agent falls through to
the LLM gate.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# yfinance is heavy; lazy-import inside _spot_price.
_DEFAULT_ANN_VOL = 0.50  # annualized vol fallback if history unavailable

# Map event_ticker prefix → yfinance symbol.
_ASSET_MAP = {
    "KXBTC": "BTC-USD",
    "KXETH": "ETH-USD",
    "KXSOL": "SOL-USD",
    "KXDOGE": "DOGE-USD",
    "KXXRP": "XRP-USD",
    "KXADA": "ADA-USD",
}

SUBTITLE_THRESHOLD_RE = re.compile(
    r"\$?(?P<amount>[\d,]+(?:\.\d+)?)\s*or\s*(?P<cmp>above|below|higher|lower)",
    re.IGNORECASE,
)
TICKER_THRESHOLD_RE = re.compile(r"-T(?P<amount>\d+(?:\.\d+)?)$")


def _asset_for(event_ticker: str) -> str | None:
    """Map a Kalshi event_ticker like 'KXBTCD-26MAY1413' to a yfinance symbol."""
    for prefix, symbol in _ASSET_MAP.items():
        if event_ticker.startswith(prefix):
            return symbol
    return None


def parse_market(event: dict) -> dict[str, Any] | None:
    """Extract (asset, comparison, threshold, deadline_utc) from an event dict."""
    asset = _asset_for(event.get("event_ticker", ""))
    if asset is None:
        return None

    # Threshold: prefer subtitle ("$90,300 or above"), fall back to ticker suffix.
    subtitle = event.get("subtitle") or ""
    threshold: float | None = None
    comparison: str | None = None

    m = SUBTITLE_THRESHOLD_RE.search(subtitle)
    if m:
        try:
            threshold = float(m["amount"].replace(",", ""))
        except (ValueError, TypeError):
            threshold = None
        cmp_word = m["cmp"].lower()
        comparison = "above" if cmp_word in ("above", "higher") else "below"

    if threshold is None:
        tm = TICKER_THRESHOLD_RE.search(event.get("market_ticker", ""))
        if tm:
            try:
                threshold = float(tm["amount"])
            except (ValueError, TypeError):
                threshold = None

    if threshold is None:
        return None
    if comparison is None:
        # Default to "above" if we can't tell — that's the Kalshi-common case.
        comparison = "above"

    close_str = event.get("close_time", "")
    if not close_str:
        return None
    try:
        deadline = datetime.fromisoformat(str(close_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    return {
        "asset": asset,
        "comparison": comparison,
        "threshold": threshold,
        "deadline_utc": deadline,
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lognormal_p(spot: float, strike: float, sigma_ann: float, t_years: float) -> float:
    """P(S_T > K) under geometric Brownian motion with zero drift."""
    if spot <= 0 or strike <= 0 or sigma_ann <= 0 or t_years <= 0:
        return 0.5
    z = math.log(strike / spot) / (sigma_ann * math.sqrt(t_years))
    return 1.0 - _norm_cdf(z)


_yf_cache: dict[str, tuple[float, float]] = {}


def _spot_and_vol(symbol: str) -> tuple[float, float] | None:
    """Return (current_price, annualized_vol) for a yfinance symbol.

    Cached per process; the cache is fine for one prediction round but you'd
    want to expire it in a long-running server (TTL ~60s).
    """
    if symbol in _yf_cache:
        return _yf_cache[symbol]

    try:
        import yfinance as yf  # lazy import

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="30d", interval="1d")
        if hist.empty:
            return None
        closes = hist["Close"]
        spot = float(closes.iloc[-1])
        if len(closes) < 5:
            sigma_ann = _DEFAULT_ANN_VOL
        else:
            import numpy as np

            log_ret = np.log(closes / closes.shift(1)).dropna()
            sigma_ann = float(log_ret.std() * math.sqrt(365)) or _DEFAULT_ANN_VOL
    except Exception as e:  # very broad — yfinance has many failure modes
        logger.warning("yfinance fetch failed for %s: %s", symbol, e)
        return None

    _yf_cache[symbol] = (spot, sigma_ann)
    return spot, sigma_ann


def crypto_prior(event: dict) -> tuple[float, str] | None:
    """Return (p_yes, rationale) from a yfinance-backed lognormal model, or None."""
    parsed = parse_market(event)
    if parsed is None:
        return None

    sv = _spot_and_vol(parsed["asset"])
    if sv is None:
        return None
    spot, sigma_ann = sv

    now_utc = datetime.now(UTC)
    t_seconds = (parsed["deadline_utc"] - now_utc).total_seconds()
    if t_seconds <= 0:
        return None
    t_years = t_seconds / (365 * 24 * 3600)

    p_above = lognormal_p(spot, parsed["threshold"], sigma_ann, t_years)
    p_yes = p_above if parsed["comparison"] == "above" else (1.0 - p_above)
    rationale = (
        f"yfinance spot {parsed['asset']}=${spot:,.0f}, σ_ann={sigma_ann:.2f}, "
        f"t={t_seconds / 3600:.1f}h to deadline; threshold ${parsed['threshold']:,.2f} "
        f"({parsed['comparison']}) → p_above={p_above:.3f}"
    )
    return p_yes, rationale
