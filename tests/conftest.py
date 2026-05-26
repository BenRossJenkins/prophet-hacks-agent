"""Test-wide fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_prediction_log(tmp_path, monkeypatch):
    """Route PREDICTION_LOG_PATH at a tmp file per test so predict() calls
    don't accumulate state in the working directory.
    """
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))


@pytest.fixture(autouse=True)
def _isolate_calibration_cache():
    """Clear agent.calibrate's module-level _cache before and after each test.

    Without this, a test that populates the 60s in-process cache (directly or
    via get_calibration_data) bleeds calibration shifts into unrelated tests
    in other files — every individual file passes in isolation but the full
    suite fails because a stale payload shifts p_yes by up to ±0.05.
    """
    from agent import calibrate as cal_mod

    cal_mod._cache.clear()
    yield
    cal_mod._cache.clear()
