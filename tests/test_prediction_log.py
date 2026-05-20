from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.prediction_log import classify_path, get_log_path, log_prediction


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


# ---- classify_path ------------------------------------------------------


@pytest.mark.parametrize(
    "rationale,expected",
    [
        ("tail-anchor 0.97→0.956 (α=0.03)", "tail-anchor"),
        ("multi-outcome (18 options, top-4); poly event 'X'", "multi-outcome-poly"),
        ("multi-outcome (35 options, top-1); LLM unavailable; uniform", "multi-outcome-uniform"),
        ("multi-outcome (15 options, top-1); raw=0.18; ensemble", "multi-outcome-llm"),
        ("depth-mid 0.500; shrunk α=0.005 → 0.500; guardrail anchored 0.821→0.628", "guardrail-anchored"),
        ("polymarket-only (poly p=0.65); kalshi: fetch failed", "poly-only"),
        ("blend 0.45 vol-weighted (kalshi=$200000 poly=$500000)", "kalshi+poly-blend"),
        ("kalshi fetch failed; LLM (grounded, α_base=0.05, raw=0.62→0.617)", "llm-grounded"),
        ("kalshi fetch failed; LLM (speculative, α_base=0.15, raw=0.40)", "llm-speculative"),
        ("kalshi fetch failed; LLM unavailable; uniform prior", "uniform"),
        ("kalshi fetch failed; prior: yfinance spot $90000", "prior"),
        ("depth-mid 0.500; shrunk α=0.005 → 0.500", "kalshi-anchor"),
    ],
)
def test_classify_path_maps_rationales(rationale, expected):
    assert classify_path(rationale) == expected


def test_log_prediction_adds_metadata(tmp_path: Path, monkeypatch):
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    log_prediction(
        {"market_ticker": "X", "category": "Sports", "outcomes": ["A", "B"]},
        0.62,
        "tail-anchor 0.97→0.956 (α=0.03)",
    )
    r = json.loads(target.read_text().strip())
    assert r["metadata"]["path"] == "tail-anchor"
    assert r["metadata"]["category"] == "Sports"
    assert r["metadata"]["n_outcomes"] == 2


def test_log_prediction_persists_probabilities_when_passed(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    dist = [
        {"market": "Yes", "probability": 0.72},
        {"market": "No", "probability": 0.28},
    ]
    log_prediction(
        {"market_ticker": "X", "outcomes": ["Yes", "No"]},
        0.72,
        "r",
        probabilities=dist,
    )
    r = json.loads(target.read_text().strip())
    assert r["probabilities"] == dist


def test_log_prediction_omits_probabilities_when_absent(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    log_prediction({"market_ticker": "X"}, 0.5, "r")
    r = json.loads(target.read_text().strip())
    assert "probabilities" not in r


def test_log_prediction_merges_extra_metadata(tmp_path: Path, monkeypatch):
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    log_prediction(
        {"market_ticker": "X", "category": "Politics", "outcomes": ["Yes", "No"]},
        0.5,
        "kalshi fetch failed; LLM (grounded, raw=0.5)",
        metadata={"p_yes_pre_calibration": 0.62, "deadline_hit": False},
    )
    r = json.loads(target.read_text().strip())
    assert r["metadata"]["path"] == "llm-grounded"
    assert r["metadata"]["p_yes_pre_calibration"] == pytest.approx(0.62)
    assert r["metadata"]["deadline_hit"] is False


# ---- stamped path beats classify_path fallback --------------------------


def test_log_prediction_prefers_stamped_path_over_rationale(
    tmp_path: Path, monkeypatch
):
    """When the producer stamps metadata.path, log_prediction must NOT
    re-derive the path from the rationale. The producer knows which branch
    it took; the rationale regex is a fragile inverse and corrupts the
    calibration table when rationales compose (e.g. tail-anchor + guardrail).
    """
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    log_prediction(
        {"market_ticker": "X", "category": "Sports", "outcomes": ["A", "B"]},
        0.62,
        # Rationale text classify_path would map to "guardrail-anchored":
        "depth-mid 0.500; guardrail anchored 0.821→0.628",
        # …but the producer stamps the true branch:
        metadata={"path": "tail-anchor"},
    )
    r = json.loads(target.read_text().strip())
    assert r["metadata"]["path"] == "tail-anchor"


def test_log_prediction_falls_back_to_classify_path_when_absent(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    # No path in metadata → classify_path must run on the rationale.
    log_prediction(
        {"market_ticker": "X", "category": "Politics", "outcomes": ["Yes", "No"]},
        0.5,
        "kalshi fetch failed; LLM (speculative, α_base=0.15, raw=0.40)",
        metadata={"version": "v3.14"},  # other metadata, no path
    )
    r = json.loads(target.read_text().strip())
    assert r["metadata"]["path"] == "llm-speculative"
    assert r["metadata"]["version"] == "v3.14"


def test_log_prediction_falls_back_when_path_is_empty_string(
    tmp_path: Path, monkeypatch
):
    """An empty / falsy `path` field shouldn't suppress the classify_path
    fallback. Producers that pass `path=""` (e.g. unknown) get the regex."""
    target = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(target))
    log_prediction(
        {"market_ticker": "X", "category": "Sports", "outcomes": ["A", "B"]},
        0.5,
        "tail-anchor 0.97→0.956 (α=0.03)",
        metadata={"path": ""},
    )
    r = json.loads(target.read_text().strip())
    assert r["metadata"]["path"] == "tail-anchor"
