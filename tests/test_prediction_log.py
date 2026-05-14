from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.prediction_log import get_log_path, log_prediction


def test_get_log_path_uses_env(monkeypatch, tmp_path: Path):
    target = tmp_path / "foo.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    assert get_log_path() == target


def test_log_prediction_appends_jsonl(tmp_path: Path, monkeypatch):
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))

    log_prediction({"market_ticker": "X-Y"}, 0.42, "test rationale")
    log_prediction({"market_ticker": "A-B"}, 0.9, "another")

    lines = target.read_text().strip().splitlines()
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["event"]["market_ticker"] == "X-Y"
    assert r0["p_yes"] == pytest.approx(0.42)
    assert r0["rationale"] == "test rationale"
    assert "ts" in r0


def test_log_prediction_creates_parent_dir(tmp_path: Path, monkeypatch):
    target = tmp_path / "nested" / "dirs" / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    log_prediction({"market_ticker": "X"}, 0.5, "r")
    assert target.exists()


def test_log_prediction_swallows_errors(monkeypatch):
    # Force an OSError on the file write — log_prediction must not raise.
    monkeypatch.setenv("PREDICTION_LOG_PATH", "/this/path/cannot/possibly/exist/preds.jsonl")
    # Patch Path.mkdir to raise so the parent-dir creation fails
    with patch("pathlib.Path.mkdir", side_effect=OSError("denied")):
        log_prediction({"market_ticker": "X"}, 0.5, "r")
