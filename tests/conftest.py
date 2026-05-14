"""Test-wide fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_prediction_log(tmp_path, monkeypatch):
    """Route PREDICTION_LOG_PATH at a tmp file per test so predict() calls
    don't accumulate state in the working directory.
    """
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
