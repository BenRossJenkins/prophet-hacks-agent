from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.calibrate import (
    apply_calibration,
    fit_calibration,
    get_calibration_table,
    load_calibration,
    save_calibration,
)


def _row(p_yes: float, result: str) -> dict:
    return {"p_yes": p_yes, "result": result}


def test_fit_returns_buckets_with_means():
    rows = [
        _row(0.05, "no"),
        _row(0.08, "no"),
        _row(0.55, "yes"),
        _row(0.58, "no"),
    ]
    table = fit_calibration(rows, n_bins=10)
    assert len(table) == 2
    b0 = next(b for b in table if b["bucket_lo"] == 0.0)
    assert b0["mean_actual"] == pytest.approx(0.0)
    b5 = next(b for b in table if b["bucket_lo"] == 0.5)
    assert b5["mean_actual"] == pytest.approx(0.5)


def test_fit_skips_rows_missing_result_or_invalid():
    rows = [
        _row(0.5, "yes"),
        {"p_yes": 0.5},  # no result
        {"p_yes": 0.5, "result": "settled"},  # bad result
        {"p_yes": "huh", "result": "yes"},  # bad p_yes
    ]
    table = fit_calibration(rows, n_bins=10)
    assert len(table) == 1
    assert table[0]["n"] == 1


def test_fit_rejects_low_n_bins():
    with pytest.raises(ValueError):
        fit_calibration([_row(0.5, "yes")], n_bins=1)


def test_apply_replaces_with_beta_bernoulli_shrunk_actual():
    """v3.14: apply_calibration shrinks mean_actual toward mean_p with
    Beta-Bernoulli posterior, N_0=10. With n=10 the shrinkage is exactly
    50/50, so (n*0.8 + 10*0.35) / (n+10) = (8 + 3.5) / 20 = 0.575."""
    table = [{"bucket_lo": 0.3, "bucket_hi": 0.4, "n": 10, "mean_p": 0.35, "mean_actual": 0.8}]
    assert apply_calibration(0.35, table) == pytest.approx(0.575)


def test_apply_passes_through_outside_buckets():
    table = [{"bucket_lo": 0.3, "bucket_hi": 0.4, "n": 10, "mean_p": 0.35, "mean_actual": 0.8}]
    assert apply_calibration(0.55, table) == pytest.approx(0.55)


def test_apply_clamps_to_contract_range():
    """Clamp still applies after B-B shrinkage. With n=1000, mean_p=0.0,
    mean_actual=0.0, the shrunk posterior is 0 → clamped to 0.01 floor."""
    table = [{"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 1000, "mean_p": 0.0, "mean_actual": 0.0}]
    assert apply_calibration(0.05, table) == 0.01


def test_apply_empty_table_passes_through():
    assert apply_calibration(0.5, []) == 0.5


def test_apply_handles_inclusive_upper_boundary():
    """Inclusive upper-boundary lookup still works. n=5 shrinks heavily:
    (5*0.85 + 10*0.95) / 15 = (4.25 + 9.5) / 15 ≈ 0.9167."""
    table = [{"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 5, "mean_p": 0.95, "mean_actual": 0.85}]
    assert apply_calibration(0.99, table) == pytest.approx(13.75 / 15)


def test_save_load_roundtrip(tmp_path: Path):
    table = [
        {"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 3, "mean_p": 0.05, "mean_actual": 0.0},
    ]
    out = tmp_path / "cal.json"
    save_calibration(table, out)
    assert load_calibration(out) == table


def test_load_returns_none_for_missing_file():
    assert load_calibration("/no/such/path/cal.json") is None


def test_load_returns_none_for_wrong_version(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"version": 999, "buckets": []}')
    assert load_calibration(p) is None


def test_load_v1_format_still_works(tmp_path: Path):
    """Legacy v1 single-bucket-list shape is coerced to v2."""
    from agent.calibrate import load_calibration_data

    p = tmp_path / "v1.json"
    p.write_text(
        '{"version": 1, "buckets": [{"bucket_lo": 0.0, "bucket_hi": 0.1, '
        '"n": 5, "mean_p": 0.05, "mean_actual": 0.0}]}'
    )
    data = load_calibration_data(p)
    assert data is not None
    assert data["by_path"] == {}
    assert len(data["global"]) == 1


# ---- path-stratified API ------------------------------------------------


def test_fit_calibration_by_path_splits_rows():
    from agent.calibrate import fit_calibration_by_path

    rows = [
        # Two tail-anchor rows in the same bucket
        {"p_yes": 0.92, "result": "yes", "metadata": {"path": "tail-anchor"}},
        {"p_yes": 0.94, "result": "yes", "metadata": {"path": "tail-anchor"}},
        # Three llm-speculative rows
        {"p_yes": 0.60, "result": "no", "metadata": {"path": "llm-speculative"}},
        {"p_yes": 0.65, "result": "no", "metadata": {"path": "llm-speculative"}},
        {"p_yes": 0.62, "result": "yes", "metadata": {"path": "llm-speculative"}},
    ]
    data = fit_calibration_by_path(rows, n_bins=10)
    # Global table has all 5 rows split into buckets.
    assert isinstance(data["global"], list)
    assert len(data["global"]) >= 1
    # by_path has both labels.
    assert "tail-anchor" in data["by_path"]
    assert "llm-speculative" in data["by_path"]


def test_apply_calibration_data_uses_path_when_n_sufficient():
    from agent.calibrate import apply_calibration_data

    # Use a small shift (≤ MAX_CALIBRATION_SHIFT = 0.05) so the cap doesn't fire.
    data = {
        "global": [
            {"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 100, "mean_p": 0.95, "mean_actual": 0.94}
        ],
        "by_path": {
            "tail-anchor": [
                {"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 20, "mean_p": 0.95, "mean_actual": 0.92}
            ]
        },
    }
    # Path bucket has n=20 ≥ min_n=3, N_0=10 →
    #   (20*0.92 + 10*0.95) / 30 = 27.9 / 30 = 0.93.
    # |0.93 - 0.95| = 0.02 < 0.05 cap → 0.93.
    assert apply_calibration_data(0.95, data, path="tail-anchor") == pytest.approx(0.93)
    # No path provided → global. n=100 →
    #   (100*0.94 + 10*0.95) / 110 = 103.5 / 110 ≈ 0.9409.
    assert apply_calibration_data(0.95, data) == pytest.approx(103.5 / 110)


def test_apply_calibration_data_falls_back_to_global_at_low_n():
    from agent.calibrate import MIN_BUCKET_N_FOR_PATH, apply_calibration_data

    # Use shifts within the ±0.05 cap so we test only the fallback logic.
    # Per-path bucket has n=2 < MIN_BUCKET_N_FOR_PATH (3 in v3.14): ignored.
    assert MIN_BUCKET_N_FOR_PATH == 3
    data = {
        "global": [
            {"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 50, "mean_p": 0.55, "mean_actual": 0.58}
        ],
        "by_path": {
            "tail-anchor": [
                {"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 2, "mean_p": 0.55, "mean_actual": 0.52}
            ]
        },
    }
    # Falls back to global. B-B with n=50: (50*0.58 + 10*0.55) / 60 = 0.575.
    assert apply_calibration_data(0.55, data, path="tail-anchor") == pytest.approx(0.575)


# ---- Beta-Bernoulli shrinkage (v3.14) -----------------------------------


def test_beta_bernoulli_small_n_shrinks_toward_mean_p():
    """At very small n, the posterior should sit close to the mean prediction
    rather than the noisy observed rate. With n=1, mean_p=0.30, mean_actual=1.0,
    raw rate is 1.0 but posterior is (1*1.0 + 10*0.30) / 11 = 4.0/11 ≈ 0.364."""
    table = [{"bucket_lo": 0.3, "bucket_hi": 0.4, "n": 1, "mean_p": 0.30, "mean_actual": 1.0}]
    assert apply_calibration(0.35, table) == pytest.approx(4.0 / 11)


def test_beta_bernoulli_large_n_converges_to_observed_rate():
    """At large n, the prior weight becomes negligible and the posterior
    approaches mean_actual. With n=1000, mean_p=0.50, mean_actual=0.30,
    posterior = (1000*0.30 + 10*0.50) / 1010 ≈ 0.302."""
    table = [{"bucket_lo": 0.4, "bucket_hi": 0.5, "n": 1000, "mean_p": 0.50, "mean_actual": 0.30}]
    out = apply_calibration(0.45, table)
    assert out == pytest.approx(305.0 / 1010)
    # And it's within 0.005 of the unshrunk rate.
    assert abs(out - 0.30) < 0.005


# ---- diff-sanity guard (v3.14) ------------------------------------------


def test_diff_sanity_passes_when_no_previous_payload():
    """First publish has no previous version to diff against. Should pass."""
    from agent.calibrate import check_calibration_diff

    new_payload = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 5, "mean_p": 0.55, "mean_actual": 0.90}],
        "by_path": {},
    }
    ok, problems = check_calibration_diff(new_payload, None)
    assert ok is True
    assert problems == []


def test_diff_sanity_passes_when_changes_are_small():
    """A small change in a small-N bucket is fine."""
    from agent.calibrate import check_calibration_diff

    prev = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 5, "mean_p": 0.55, "mean_actual": 0.50}],
        "by_path": {},
    }
    new = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 6, "mean_p": 0.55, "mean_actual": 0.55}],
        "by_path": {},
    }
    ok, problems = check_calibration_diff(new, prev)
    assert ok is True
    assert problems == []


def test_diff_sanity_blocks_big_shift_in_small_n_bucket():
    """A 0.30 shift in a small-N (n=5) bucket should fail-closed."""
    from agent.calibrate import check_calibration_diff

    prev = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 5, "mean_p": 0.55, "mean_actual": 0.50}],
        "by_path": {},
    }
    new = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 7, "mean_p": 0.55, "mean_actual": 0.80}],
        "by_path": {},
    }
    ok, problems = check_calibration_diff(new, prev)
    assert ok is False
    assert len(problems) == 1
    assert "0.500" in problems[0] and "0.800" in problems[0]


def test_diff_sanity_allows_big_shift_in_large_n_bucket():
    """A bucket with n >= small_n threshold has earned the shift — let it through."""
    from agent.calibrate import check_calibration_diff

    prev = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 50, "mean_p": 0.55, "mean_actual": 0.50}],
        "by_path": {},
    }
    new = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 60, "mean_p": 0.55, "mean_actual": 0.80}],
        "by_path": {},
    }
    ok, problems = check_calibration_diff(new, prev)
    assert ok is True
    assert problems == []


def test_diff_sanity_checks_by_path_tables_too():
    """A noisy small-N per-path bucket also fails-closed."""
    from agent.calibrate import check_calibration_diff

    prev = {
        "global": [],
        "by_path": {
            "llm-speculative": [
                {"bucket_lo": 0.6, "bucket_hi": 0.7, "n": 3, "mean_p": 0.65, "mean_actual": 0.30}
            ],
        },
    }
    new = {
        "global": [],
        "by_path": {
            "llm-speculative": [
                {"bucket_lo": 0.6, "bucket_hi": 0.7, "n": 4, "mean_p": 0.65, "mean_actual": 0.90}
            ],
        },
    }
    ok, problems = check_calibration_diff(new, prev)
    assert ok is False
    assert any("by_path[llm-speculative]" in p for p in problems)


def test_diff_sanity_skips_new_buckets_that_did_not_exist_previously():
    """A bucket present in new but not in previous can't be diffed — leave it alone."""
    from agent.calibrate import check_calibration_diff

    prev = {"global": [], "by_path": {}}
    new = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 3, "mean_p": 0.55, "mean_actual": 0.92}],
        "by_path": {},
    }
    ok, problems = check_calibration_diff(new, prev)
    assert ok is True
    assert problems == []


def test_diff_sanity_threshold_tunable():
    """Custom max_delta argument."""
    from agent.calibrate import check_calibration_diff

    prev = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 5, "mean_p": 0.55, "mean_actual": 0.50}],
        "by_path": {},
    }
    new = {
        "global": [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 6, "mean_p": 0.55, "mean_actual": 0.62}],
        "by_path": {},
    }
    # |0.62 - 0.50| = 0.12. Default 0.20 passes; strict 0.10 fails.
    assert check_calibration_diff(new, prev)[0] is True
    assert check_calibration_diff(new, prev, max_delta=0.10)[0] is False


def test_beta_bernoulli_zero_n_returns_mean_p():
    """An empty bucket (n=0) — should never happen in practice, but the
    formula divides by (n + N_0) so it's safe; posterior just equals mean_p."""
    table = [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 0, "mean_p": 0.55, "mean_actual": 0.0}]
    assert apply_calibration(0.55, table) == pytest.approx(0.55)


def test_apply_calibration_data_no_data_passes_through():
    from agent.calibrate import apply_calibration_data

    assert apply_calibration_data(0.42, None) == 0.42
    assert apply_calibration_data(0.42, {}) == 0.42


def test_apply_calibration_data_caps_shift_to_max_delta():
    """A bucket that would shift the prediction by >0.05 is bounded to that delta.

    Protects against a noisy 5-event bucket pulling a confident prediction
    wildly off. With raw=0.90 and bucket mean_actual=0.50, the unbounded
    correction would land at 0.50 (Δ=0.40). The bound keeps it at 0.85.
    """
    from agent.calibrate import MAX_CALIBRATION_SHIFT, apply_calibration_data

    data = {
        "global": [
            {"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 50, "mean_p": 0.95, "mean_actual": 0.50}
        ],
        "by_path": {},
    }
    out = apply_calibration_data(0.90, data)
    # Without cap → 0.50. With cap → 0.90 - 0.05 = 0.85.
    assert MAX_CALIBRATION_SHIFT == pytest.approx(0.05)
    assert out == pytest.approx(0.85)


def test_apply_calibration_data_small_shift_passes_unchanged():
    """When the B-B-shrunk calibration correction is within the bound, no clipping."""
    from agent.calibrate import apply_calibration_data

    data = {
        "global": [
            {"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 50, "mean_p": 0.55, "mean_actual": 0.58}
        ],
        "by_path": {},
    }
    out = apply_calibration_data(0.55, data)
    # B-B: (50*0.58 + 10*0.55) / 60 = 0.575.
    # |0.575 - 0.55| = 0.025 < 0.05 cap → no clipping.
    assert out == pytest.approx(0.575)


def test_apply_calibration_data_cap_applies_to_path_lookups_too():
    """Cap applies regardless of whether the bucket came from by_path or global."""
    from agent.calibrate import apply_calibration_data

    data = {
        "global": [
            {"bucket_lo": 0.0, "bucket_hi": 0.2, "n": 100, "mean_p": 0.10, "mean_actual": 0.10}
        ],
        "by_path": {
            "tail-anchor": [
                {"bucket_lo": 0.0, "bucket_hi": 0.2, "n": 10, "mean_p": 0.10, "mean_actual": 0.90}
            ]
        },
    }
    # path bucket says 0.90, raw=0.10, unbounded Δ=0.80 — clamp to 0.10+0.05.
    out = apply_calibration_data(0.10, data, path="tail-anchor")
    assert out == pytest.approx(0.15)


def test_save_calibration_accepts_v2_payload(tmp_path: Path):
    from agent.calibrate import load_calibration_data

    payload = {
        "global": [{"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 1, "mean_p": 0.05, "mean_actual": 0.0}],
        "by_path": {
            "kalshi-anchor": [
                {"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 10, "mean_p": 0.55, "mean_actual": 0.6}
            ]
        },
    }
    out = tmp_path / "cal.json"
    save_calibration(payload, out)
    loaded = load_calibration_data(out)
    assert loaded == payload


def test_get_calibration_table_uses_env(tmp_path: Path, monkeypatch):
    out = tmp_path / "cal.json"
    save_calibration(
        [{"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 1, "mean_p": 0.05, "mean_actual": 0.1}],
        out,
    )
    monkeypatch.setenv("CALIBRATION_PATH", str(out))
    monkeypatch.delenv("CALIBRATION_GCS_URI", raising=False)
    # Reset module-level cache
    from agent import calibrate as cal_mod
    cal_mod._cache.clear()
    table = get_calibration_table()
    assert table is not None
    assert len(table) == 1


def test_get_calibration_table_prefers_gcs_when_uri_set(tmp_path: Path, monkeypatch):
    # File on disk has bucket count 1
    out = tmp_path / "cal.json"
    save_calibration(
        [{"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 1, "mean_p": 0.05, "mean_actual": 0.1}],
        out,
    )
    monkeypatch.setenv("CALIBRATION_PATH", str(out))
    monkeypatch.setenv("CALIBRATION_GCS_URI", "gs://bucket/file.json")

    # GCS returns a 2-bucket table (wrapped in v2 payload)
    gcs_table = [
        {"bucket_lo": 0.3, "bucket_hi": 0.4, "n": 5, "mean_p": 0.35, "mean_actual": 0.7},
        {"bucket_lo": 0.6, "bucket_hi": 0.7, "n": 5, "mean_p": 0.65, "mean_actual": 0.4},
    ]
    from agent import calibrate as cal_mod
    cal_mod._cache.clear()
    gcs_payload = {"global": gcs_table, "by_path": {}}
    with patch("agent.calibrate._load_from_gcs", return_value=gcs_payload):
        table = get_calibration_table()
    assert table == gcs_table


def test_get_calibration_table_falls_back_to_disk_when_gcs_fails(tmp_path: Path, monkeypatch):
    out = tmp_path / "cal.json"
    disk_table = [
        {"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 1, "mean_p": 0.05, "mean_actual": 0.1}
    ]
    save_calibration(disk_table, out)
    monkeypatch.setenv("CALIBRATION_PATH", str(out))
    monkeypatch.setenv("CALIBRATION_GCS_URI", "gs://bucket/missing.json")
    from agent import calibrate as cal_mod
    cal_mod._cache.clear()
    with patch("agent.calibrate._load_from_gcs", return_value=None):
        table = get_calibration_table()
    assert table == disk_table


def test_predict_applies_calibration_when_table_present(tmp_path: Path, monkeypatch):
    """End-to-end: predict() applies a calibration table that's present on disk.

    Calibration shift is capped to ±MAX_CALIBRATION_SHIFT (0.05). With raw
    p=0.5425 and a bucket mean_actual=0.7 (Δ=0.16), the cap clamps the
    adjusted value to 0.5425 + 0.05 = 0.5925.
    """
    out = tmp_path / "cal.json"
    save_calibration(
        [{"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 100, "mean_p": 0.55, "mean_actual": 0.7}],
        out,
    )
    monkeypatch.setenv("CALIBRATION_PATH", str(out))
    from agent import calibrate as cal_mod
    cal_mod._cache.clear()

    from agent.predict import predict

    with patch("agent.predict.get_market", return_value=None), patch(
        "agent.predict.category_prior", return_value=None
    ), patch("agent.predict.llm_forecast_ensemble", return_value=(0.55, "test")):
        out = predict(
            {
                "event_ticker": "TEST-EVT",
                "market_ticker": "TEST-MKT",
                "title": "x",
                "category": "Politics",
                "close_time": "2026-12-31T23:59:59Z",
            }
        )
    # Speculative LLM tail-aware shrink lands at 0.5425. Calibration would push
    # to 0.7 (Δ=0.16) but cap clamps to 0.5425 + 0.05 = 0.5925.
    assert out["p_yes"] == pytest.approx(0.5925)
    assert "calibrated" in out["rationale"]
