"""Fit and save a calibration table from resolved predictions.

Run daily during the eval window:
  python scripts/resolve_predictions.py     # mark newly-resolved
  python scripts/fit_calibration.py          # refit calibration

The live agent reads `data/calibration.json` on every predict() call
(60s in-process cache) and applies it.

Usage:
  python scripts/fit_calibration.py [--input data/resolved_predictions.jsonl]
                                    [--output data/calibration.json]
                                    [--n-bins 10]
                                    [--min-samples 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.calibrate import (
    apply_calibration,
    fit_calibration,
    get_calibration_path,
    save_calibration,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/resolved_predictions.jsonl",
        help="resolved-predictions JSONL produced by scripts/resolve_predictions.py",
    )
    p.add_argument(
        "--output",
        default=None,
        help="output path for the calibration table (defaults to $CALIBRATION_PATH)",
    )
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="refuse to save calibration if fewer than this many resolutions exist",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"No resolved-predictions file at {in_path}", file=sys.stderr)
        return 1

    rows = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(rows)} resolved predictions")
    if len(rows) < args.min_samples:
        print(
            f"Too few samples ({len(rows)} < {args.min_samples}); not saving "
            f"a calibration table to avoid overfitting noise.",
            file=sys.stderr,
        )
        return 2

    table = fit_calibration(rows, n_bins=args.n_bins)
    if not table:
        print("Fit produced an empty table (all rows missing p_yes/result?).", file=sys.stderr)
        return 1

    out_path = args.output or get_calibration_path()
    save_calibration(table, out_path)
    print(f"Calibration table → {out_path}")

    # Optionally mirror to GCS for live-agent consumption (Cloud Run reads
    # this with a 60s cache, so freshness is automatic).
    import os
    import subprocess

    gcs_uri = os.environ.get("CALIBRATION_GCS_URI")
    if gcs_uri:
        try:
            result = subprocess.run(
                ["gcloud", "storage", "cp", str(out_path), gcs_uri, "--quiet"],
                check=True,
                capture_output=True,
                timeout=30,
                text=True,
            )
            print(f"  mirrored to {gcs_uri}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  GCS upload failed (ignored): {e}", file=sys.stderr)
    print(f"{'Bucket':<14}{'N':>5}{'mean_p':>9}{'mean_actual':>13}{'bias':>9}")
    for b in table:
        bias = b["mean_actual"] - b["mean_p"]
        flag = "  ← BIAS" if abs(bias) > 0.10 else ""
        print(
            f"  [{b['bucket_lo']:.2f}-{b['bucket_hi']:.2f})"
            f"{b['n']:>5}{b['mean_p']:>9.3f}{b['mean_actual']:>13.3f}"
            f"{bias:>+8.3f}{flag}"
        )

    # Diagnostic: what change would calibration produce on the existing rows?
    if rows:
        n_changed = 0
        total_brier_delta = 0.0
        for r in rows:
            try:
                p = float(r.get("p_yes", 0.5))
            except (ValueError, TypeError):
                continue
            result = r.get("result", "")
            if result not in ("yes", "no"):
                continue
            actual = 1.0 if result == "yes" else 0.0
            new_p = apply_calibration(p, table)
            if abs(new_p - p) > 1e-6:
                n_changed += 1
                old_brier = (p - actual) ** 2
                new_brier = (new_p - actual) ** 2
                total_brier_delta += new_brier - old_brier
        if n_changed:
            print(
                f"\nOn the training rows: {n_changed} predictions changed; "
                f"in-sample Brier delta = {total_brier_delta:+.5f} "
                f"(negative = improvement; this is in-sample so optimistic)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
