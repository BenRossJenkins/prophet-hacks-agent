"""Fit and save a path-stratified calibration table from resolved predictions.

Run daily during the eval window:
  python scripts/resolve_predictions.py     # mark newly-resolved
  python scripts/fit_calibration.py         # refit calibration

The live agent reads `data/calibration.json` on every predict() call
(60s in-process cache) and applies it — per-path bucket first, falling
back to global when path data is sparse.

Usage:
  python scripts/fit_calibration.py [--input data/resolved_predictions.jsonl]
                                    [--output data/calibration.json]
                                    [--n-bins 10]
                                    [--min-samples 20]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from agent.calibrate import (
    apply_calibration_data,
    fit_calibration_by_path,
    get_calibration_path,
    save_calibration,
)
from agent.prediction_log import classify_path


def _path_for_row(row: dict) -> str | None:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        p = metadata.get("path")
        if isinstance(p, str) and p:
            return p
    rationale = row.get("rationale")
    if isinstance(rationale, str) and rationale:
        return classify_path(rationale)
    return None


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

    payload = fit_calibration_by_path(rows, n_bins=args.n_bins)
    if not payload.get("global"):
        print("Fit produced an empty table (all rows missing p_yes/result?).", file=sys.stderr)
        return 1

    out_path = args.output or get_calibration_path()
    save_calibration(payload, out_path)
    print(f"Calibration table → {out_path}")

    # Mirror to GCS so the deployed agent picks up the update without
    # redeploy (60s in-process cache + GCS-preferred load).
    gcs_uri = os.environ.get("CALIBRATION_GCS_URI")
    if gcs_uri:
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", str(out_path), gcs_uri, "--quiet"],
                check=True,
                capture_output=True,
                timeout=30,
                text=True,
            )
            print(f"  mirrored to {gcs_uri}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  GCS upload failed (ignored): {e}", file=sys.stderr)

    # Per-path summary first (this is the interesting one for analysis).
    by_path = payload.get("by_path") or {}
    if by_path:
        print("\n=== Per-path tables ===")
        for path_label in sorted(by_path):
            table = by_path[path_label]
            n_total = sum(b["n"] for b in table)
            print(f"\n[{path_label}] n={n_total} across {len(table)} buckets:")
            for b in table:
                bias = b["mean_actual"] - b["mean_p"]
                flag = "  ← BIAS" if abs(bias) > 0.10 else ""
                print(
                    f"  [{b['bucket_lo']:.2f}-{b['bucket_hi']:.2f})"
                    f"{b['n']:>5}{b['mean_p']:>9.3f}{b['mean_actual']:>13.3f}"
                    f"{bias:>+8.3f}{flag}"
                )

    # Global summary.
    print(f"\n=== Global (fallback) ===")
    print(f"{'Bucket':<14}{'N':>5}{'mean_p':>9}{'mean_actual':>13}{'bias':>9}")
    for b in payload["global"]:
        bias = b["mean_actual"] - b["mean_p"]
        flag = "  ← BIAS" if abs(bias) > 0.10 else ""
        print(
            f"  [{b['bucket_lo']:.2f}-{b['bucket_hi']:.2f})"
            f"{b['n']:>5}{b['mean_p']:>9.3f}{b['mean_actual']:>13.3f}"
            f"{bias:>+8.3f}{flag}"
        )

    # Diagnostic: in-sample Brier delta. Optimistic but useful for sanity.
    n_changed = 0
    total_brier_delta = 0.0
    for r in rows:
        try:
            p_orig = float(r.get("p_yes", 0.5))
        except (ValueError, TypeError):
            continue
        result = r.get("result", "")
        if result not in ("yes", "no"):
            continue
        actual = 1.0 if result == "yes" else 0.0
        row_path = _path_for_row(r)
        new_p = apply_calibration_data(p_orig, payload, path=row_path)
        if abs(new_p - p_orig) > 1e-6:
            n_changed += 1
            total_brier_delta += (new_p - actual) ** 2 - (p_orig - actual) ** 2
    if n_changed:
        print(
            f"\nOn the training rows: {n_changed} predictions changed; "
            f"in-sample Brier delta = {total_brier_delta:+.5f} "
            f"(negative = improvement; this is in-sample so optimistic)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
