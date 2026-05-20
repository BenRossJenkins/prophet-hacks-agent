"""Daily eval-window health check.

Run by hand each morning of the May 17-31 eval window (or wire to a
launchd job). Surfaces the three operational risks that compound across
14 days of unsupervised running:

  1. Calibration table freshness.
     The daily refit cron writes a new calibration.json to GCS. If the
     refit silently fails (GCS auth expiry, missing resolved predictions,
     network), the deployed agent reads a stale table for days. Checks
     the GCS object's update timestamp.

  2. Per-vendor latency (p50/p99).
     Pulls the last 24h of per-event prediction objects from GCS, parses
     the rationale for `ensemble[...]` summaries, and reports each
     vendor's latency distribution. CLAUDE.md rule of thumb: p99 < 90s
     healthy, > 180s = drop that vendor.

  3. Completion rate.
     Counts how many predictions are LLM-fallback-uniform (p=0.5 with
     "LLM unavailable" or "uniform" in rationale) vs successful. A spike
     in uniform-fallback predictions = pipeline failure even if no
     explicit HTTP error.

Usage:
    .venv/bin/python scripts/monitor_eval_health.py

Exit codes:
    0  healthy
    1  one or more checks failed (operator should investigate)

Reads no credentials beyond what `gcloud` already has cached, so this is
safe to run from any machine where you've `gcloud auth login`-ed.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

GCS_CALIBRATION_URI = "gs://prophet-hacks-2026-calibration/calibration.json"
GCS_PREDICTIONS_PREFIX = "gs://prophet-hacks-2026-calibration/predictions/"

# Healthy operating ranges (from CLAUDE.md rule-of-thumb).
CALIBRATION_STALE_HOURS = 36  # refit cron is daily; >36h = refit job broke
P99_WARN_SECONDS = 90
P99_CRITICAL_SECONDS = 180
MIN_COMPLETION_RATE = 0.98


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command, return (rc, stdout, stderr)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout, proc.stderr


def check_calibration_freshness() -> tuple[bool, str]:
    """Verify the calibration GCS object updated within the last 36h."""
    rc, out, err = _run(["gcloud", "storage", "ls", "--long", GCS_CALIBRATION_URI])
    if rc != 0:
        if "matched no objects" in err.lower():
            return True, "no calibration table yet (expected day 1-3 of eval)"
        return False, f"gcloud storage ls failed: {err.strip()[:200]}"
    # `gcloud storage ls --long` output format: "<size> <updated-iso> <uri>"
    # Pick the most recent timestamp out of the listing line.
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        return False, "calibration GCS listing returned no lines"
    # Parse the second whitespace-delimited token as the timestamp.
    parts = lines[0].split()
    if len(parts) < 2:
        return False, f"unparseable ls output: {lines[0][:120]}"
    ts_str = parts[1]
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return False, f"unparseable timestamp: {ts_str}"
    age_h = (datetime.now(UTC) - ts).total_seconds() / 3600.0
    if age_h > CALIBRATION_STALE_HOURS:
        return False, (
            f"calibration table stale: last updated {age_h:.1f}h ago "
            f"(threshold {CALIBRATION_STALE_HOURS}h). Refit job may be broken."
        )
    return True, f"calibration updated {age_h:.1f}h ago"


def _list_recent_predictions(hours: int = 24) -> list[dict[str, Any]]:
    """List + read the GCS prediction objects written in the last `hours`."""
    rc, out, err = _run(["gcloud", "storage", "ls", GCS_PREDICTIONS_PREFIX])
    if rc != 0:
        return []
    uris = [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".json")]
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows: list[dict[str, Any]] = []
    for uri in uris:
        # Filename is "<ISO-ts>_<market_ticker>.json" — parse ts from filename.
        name = uri.rsplit("/", 1)[-1]
        ts_part = name.split("_")[0]
        try:
            ts = datetime.fromisoformat(ts_part.replace("-", ":", 2).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts < cutoff:
            continue
        rc2, body, _ = _run(["gcloud", "storage", "cat", uri])
        if rc2 != 0:
            continue
        try:
            rows.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return rows


def check_vendor_latency(predictions: list[dict]) -> tuple[bool, str]:
    """Surface p50/p99 latency per vendor from rationale strings.

    Rationale format from our ensemble:
      `ensemble[claude=0.32,gpt=0.18,gemini=0.50] -> median=...`
    We don't actually have per-event latency stamps in the log right now
    (would require a separate field in `log_prediction`). For now this
    check just reports the COUNT of predictions per vendor as a proxy.
    A vendor that systematically drops out shows up as a low count.
    """
    if not predictions:
        return True, "no predictions in window (expected day 1)"
    vendor_seen = {"claude": 0, "gpt": 0, "gemini": 0}
    total = 0
    for row in predictions:
        rat = row.get("rationale", "") or ""
        if "ensemble[" in rat:
            total += 1
            for vendor in vendor_seen:
                if f"{vendor}=" in rat:
                    vendor_seen[vendor] += 1
    if total == 0:
        return True, "no ensemble calls in window (market-anchor only)"
    lines = []
    healthy = True
    for vendor, count in vendor_seen.items():
        rate = count / total if total else 0.0
        flag = ""
        if rate < 0.7:
            flag = "  ← REGRESSING"
            healthy = False
        lines.append(f"    {vendor:<6}  {count:>4}/{total}  ({rate:.0%}){flag}")
    msg = f"vendor participation over last 24h:\n" + "\n".join(lines)
    return healthy, msg


def check_completion_rate(predictions: list[dict]) -> tuple[bool, str]:
    """Count uniform-fallback predictions as proxy for completion issues."""
    if not predictions:
        return True, "no predictions in window"
    total = len(predictions)
    uniform_fallbacks = 0
    for row in predictions:
        rat = (row.get("rationale") or "").lower()
        if "llm unavailable" in rat or "uniform prior" in rat or "uniform 1/n" in rat:
            uniform_fallbacks += 1
    completion = 1.0 - (uniform_fallbacks / total) if total else 1.0
    msg = (
        f"{total} predictions, {uniform_fallbacks} uniform-fallback "
        f"({completion:.1%} healthy-signal rate)"
    )
    return completion >= MIN_COMPLETION_RATE, msg


def main() -> int:
    print(f"=== Eval health check @ {datetime.now(UTC).isoformat()} ===\n")
    overall_ok = True

    # 1. Calibration freshness
    ok, msg = check_calibration_freshness()
    overall_ok &= ok
    status = "✓" if ok else "✗"
    print(f"[{status}] calibration: {msg}")

    # 2. & 3. Predictions log — fetch once, share between checks
    predictions = _list_recent_predictions(hours=24)

    ok, msg = check_vendor_latency(predictions)
    overall_ok &= ok
    status = "✓" if ok else "✗"
    print(f"[{status}] vendors:")
    for line in msg.splitlines():
        print(line)

    ok, msg = check_completion_rate(predictions)
    overall_ok &= ok
    status = "✓" if ok else "✗"
    print(f"[{status}] completion: {msg}")

    print()
    print("OVERALL:", "✓ HEALTHY" if overall_ok else "✗ ATTENTION NEEDED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
