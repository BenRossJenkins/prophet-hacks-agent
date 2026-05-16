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


def test_apply_replaces_with_actual():
    table = [{"bucket_lo": 0.3, "bucket_hi": 0.4, "n": 10, "mean_p": 0.35, "mean_actual": 0.8}]
    assert apply_calibration(0.35, table) == pytest.approx(0.8)


def test_apply_passes_through_outside_buckets():
    table = [{"bucket_lo": 0.3, "bucket_hi": 0.4, "n": 10, "mean_p": 0.35, "mean_actual": 0.8}]
    assert apply_calibration(0.55, table) == pytest.approx(0.55)


def test_apply_clamps_to_contract_range():
    table = [{"bucket_lo": 0.0, "bucket_hi": 0.1, "n": 10, "mean_p": 0.05, "mean_actual": 0.0}]
    assert apply_calibration(0.05, table) == 0.01


def test_apply_empty_table_passes_through():
    assert apply_calibration(0.5, []) == 0.5


def test_apply_handles_inclusive_upper_boundary():
    table = [{"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 5, "mean_p": 0.95, "mean_actual": 0.85}]
    assert apply_calibration(0.99, table) == pytest.approx(0.85)


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

    data = {
        "global": [
            {"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 100, "mean_p": 0.95, "mean_actual": 0.92}
        ],
        "by_path": {
            "tail-anchor": [
                {"bucket_lo": 0.9, "bucket_hi": 1.0, "n": 20, "mean_p": 0.95, "mean_actual": 0.88}
            ]
        },
    }
    # Path bucket has n=20 >= min_n: use it.
    assert apply_calibration_data(0.95, data, path="tail-anchor") == pytest.approx(0.88)
    # No path provided → global.
    assert apply_calibration_data(0.95, data) == pytest.approx(0.92)


def test_apply_calibration_data_falls_back_to_global_at_low_n():
    from agent.calibrate import apply_calibration_data

    data = {
        "global": [
            {"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 50, "mean_p": 0.55, "mean_actual": 0.70}
        ],
        "by_path": {
            "tail-anchor": [
                # n=2, below MIN_BUCKET_N_FOR_PATH (5) — should be ignored.
                {"bucket_lo": 0.5, "bucket_hi": 0.6, "n": 2, "mean_p": 0.55, "mean_actual": 0.40}
            ]
        },
    }
    # Path bucket exists but n<min — falls back to global.
    assert apply_calibration_data(0.55, data, path="tail-anchor") == pytest.approx(0.70)


def test_apply_calibration_data_no_data_passes_through():
    from agent.calibrate import apply_calibration_data

    assert apply_calibration_data(0.42, None) == 0.42
    assert apply_calibration_data(0.42, {}) == 0.42


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
    """End-to-end: predict() applies a calibration table that's present on disk."""
    out = tmp_path / "cal.json"
    # 0.5 → 0.7 calibration on bucket [0.5, 0.6)
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
    # Speculative LLM shrink: 0.55 → 0.55 * 0.85 + 0.5 * 0.15 = 0.5425. Falls in [0.5, 0.6) bucket.
    # Calibrated → 0.7.
    assert out["p_yes"] == pytest.approx(0.7)
    assert "calibrated" in out["rationale"]
