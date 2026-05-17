"""Daily morning eval-window check.

Run each morning during the May 17-31 eval window:

    .venv/bin/python scripts/daily_morning_check.py

Surfaces the metrics that compound across the window:

  1. Predictions in the last 24h
     - Total count (matches eval cadence?)
     - Completion rate: share of predictions with a real signal vs uniform
       fallback. Below 0.98 = something's broken upstream.
     - Per-path counts: which branches are firing. A pipeline regression
       often shows up as one path going dark or another spiking.
     - Per-vendor participation across the LLM ensemble.

  2. Per-path Brier across ALL resolved predictions
     Strata with consistent miscalibration are the candidates for
     parameter tuning. (Code freeze applies, so this is informational
     during eval. Useful for post-mortem.)

  3. Cumulative GCP spend over the eval window
     The LLM vendor spend lives in each vendor's billing console (URLs
     printed at the bottom) — no unified API. Tally those manually.

Exits 0 always. Designed to be eyeballed, not to gate anything.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

GCS_PREDICTIONS_PREFIX = "gs://prophet-hacks-2026-calibration/predictions/"
RESOLVED_PREDICTIONS_PATH = Path("data/resolved_predictions.jsonl")
GCP_PROJECT = "prophet-hacks-2026"
BILLING_ACCOUNT = "0188D3-4F95CC-017FE3"

# Eval window bounds — used for the cumulative-spend lookup.
EVAL_START = datetime(2026, 5, 17, tzinfo=UTC)
EVAL_END = datetime(2026, 5, 31, tzinfo=UTC)


@dataclass
class WindowStats:
    total: int = 0
    uniform_fallbacks: int = 0
    per_path: Counter = field(default_factory=Counter)
    per_vendor: Counter = field(default_factory=Counter)
    versions: Counter = field(default_factory=Counter)

    @property
    def completion_rate(self) -> float:
        return 1.0 - (self.uniform_fallbacks / self.total) if self.total else 1.0


@dataclass
class PathBrier:
    n: int = 0
    sum_sq: float = 0.0
    sum_actual: float = 0.0
    sum_pred: float = 0.0

    @property
    def brier(self) -> float:
        return self.sum_sq / self.n if self.n else 0.0

    @property
    def base_rate(self) -> float:
        return self.sum_actual / self.n if self.n else 0.0

    @property
    def mean_pred(self) -> float:
        return self.sum_pred / self.n if self.n else 0.0


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def _parse_ts_from_filename(name: str) -> datetime | None:
    """Filename pattern: <ISO-ts-with-dashes>_<market_ticker>.json."""
    ts_part = name.split("_")[0]
    # The agent writes "2026-05-17T14-34-25.640220Z" with `-` instead of `:`
    # in the time portion. Restore the `:` so fromisoformat can parse.
    if "T" in ts_part:
        date, _, time = ts_part.partition("T")
        time = time.replace("-", ":", 2).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(f"{date}T{time}")
        except ValueError:
            return None
    return None


def collect_last_24h() -> WindowStats:
    """Pull the last 24h of predictions from GCS, accumulate stats."""
    stats = WindowStats()
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    rc, out, err = _run(["gcloud", "storage", "ls", GCS_PREDICTIONS_PREFIX])
    if rc != 0:
        print(f"  warning: GCS ls failed: {err.strip()[:200]}", file=sys.stderr)
        return stats
    uris = [
        ln.strip() for ln in out.splitlines()
        if ln.strip().endswith(".json")
    ]
    for uri in uris:
        name = uri.rsplit("/", 1)[-1]
        ts = _parse_ts_from_filename(name)
        if ts is None or ts < cutoff:
            continue
        rc2, body, _ = _run(["gcloud", "storage", "cat", uri])
        if rc2 != 0:
            continue
        try:
            row = json.loads(body)
        except json.JSONDecodeError:
            continue
        stats.total += 1
        meta = row.get("metadata") or {}
        path = meta.get("path", "unknown")
        stats.per_path[path] += 1
        stats.versions[meta.get("version", "unknown")] += 1
        rationale = (row.get("rationale") or "").lower()
        if "uniform" in path or "llm unavailable" in rationale:
            stats.uniform_fallbacks += 1
        # Per-vendor participation: pull from `ensemble[...]` summary in
        # the rationale. Each token is "<short>=<p>". This is a count, not
        # latency — see Cloud Run structured logs for per-call timing.
        if "ensemble[" in rationale:
            inside = rationale.split("ensemble[", 1)[1].split("]", 1)[0]
            for tok in inside.split(","):
                vendor = tok.split("=", 1)[0].strip()
                if vendor:
                    stats.per_vendor[vendor] += 1
    return stats


def per_path_brier_over_resolved() -> dict[str, PathBrier]:
    """Compute per-path Brier across the local resolved-predictions file."""
    out: dict[str, PathBrier] = defaultdict(PathBrier)
    if not RESOLVED_PREDICTIONS_PATH.exists():
        return out
    for line in RESOLVED_PREDICTIONS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            p = float(row.get("p_yes", 0.5))
        except (ValueError, TypeError):
            continue
        result = row.get("result", "")
        if result not in ("yes", "no"):
            continue
        actual = 1.0 if result == "yes" else 0.0
        meta = row.get("metadata") or {}
        path = meta.get("path") or "unknown"
        b = out[path]
        b.n += 1
        b.sum_sq += (p - actual) ** 2
        b.sum_actual += actual
        b.sum_pred += p
    return out


def cumulative_gcp_spend() -> float | None:
    """Lookup GCP project spend since EVAL_START via Cloud Billing API.

    Returns total USD spent, or None if the lookup failed. Billing API
    surfaces spend with ~1 day of lag; close-of-eval numbers from this
    script will under-count yesterday by some amount.
    """
    # `gcloud billing` doesn't have a direct "current spend" command; the
    # canonical way is to export billing to BigQuery, but we haven't set
    # that up. Fall back to a stub that asks the operator to check the
    # console directly. Returns None so the print path skips quietly.
    return None


def main() -> int:
    print(f"=== Daily morning check @ {datetime.now(UTC).isoformat()} ===\n")

    # Section 1: last 24h.
    print("--- Predictions in last 24h ---")
    stats = collect_last_24h()
    print(f"  total:           {stats.total}")
    print(f"  uniform-fallback: {stats.uniform_fallbacks} "
          f"({stats.uniform_fallbacks / stats.total * 100 if stats.total else 0:.1f}%)")
    print(f"  completion rate:  {stats.completion_rate:.1%}")
    if stats.completion_rate < 0.98 and stats.total >= 20:
        print(f"  ⚠ completion rate below 0.98 — pipeline upstream may be broken")

    if stats.versions:
        print(f"\n  agent versions seen:")
        for v, n in stats.versions.most_common():
            print(f"    {v:<10} {n}")

    if stats.per_path:
        print(f"\n  per-path counts:")
        for path, n in stats.per_path.most_common():
            pct = n / stats.total * 100 if stats.total else 0
            print(f"    {path:<22} {n:>4}  ({pct:5.1f}%)")

    if stats.per_vendor:
        print(f"\n  per-vendor participation:")
        total_vendor_calls = sum(stats.per_vendor.values())
        ideal_per_vendor = total_vendor_calls / max(len(stats.per_vendor), 1)
        for vendor, n in stats.per_vendor.most_common():
            ratio = n / ideal_per_vendor if ideal_per_vendor else 0
            flag = "  ← regressing" if ratio < 0.7 else ""
            print(f"    {vendor:<10} {n:>4}  (ratio {ratio:.2f}){flag}")

    # Section 2: per-path Brier across resolved.
    print("\n--- Per-path Brier (all resolved predictions) ---")
    brier = per_path_brier_over_resolved()
    if not brier:
        print("  (no resolved predictions yet — run scripts/resolve_predictions.py)")
    else:
        all_paths = sorted(brier.items(), key=lambda kv: -kv[1].n)
        for path, b in all_paths:
            flag = ""
            if b.n >= 5 and b.brier > 0.25:
                # Worse than always-0.5 → suspicious
                flag = "  ← WORSE THAN UNIFORM"
            print(
                f"  {path:<22} n={b.n:>4}  brier={b.brier:.4f}  "
                f"mean_p={b.mean_pred:.3f}  base_rate={b.base_rate:.3f}{flag}"
            )

    # Section 3: spend.
    print("\n--- Spend ---")
    spend = cumulative_gcp_spend()
    if spend is not None:
        print(f"  GCP project spend: ${spend:.2f}")
    else:
        print(
            f"  GCP: check console at "
            f"https://console.cloud.google.com/billing/{BILLING_ACCOUNT}/reports?project={GCP_PROJECT}"
        )
    print("  Vendor LLM spend (check each console):")
    print("    Anthropic: https://console.anthropic.com/settings/usage")
    print("    OpenAI:    https://platform.openai.com/usage")
    print("    Google AI: https://aistudio.google.com/usage")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
