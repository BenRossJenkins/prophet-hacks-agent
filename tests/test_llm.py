from __future__ import annotations

from agent.llm import parse_response


def test_parse_plain_json():
    out = parse_response('{"p_yes": 0.42, "rationale": "weak prior"}')
    assert out == (0.42, "weak prior")


def test_parse_markdown_fenced_json():
    text = '```json\n{"p_yes": 0.6, "rationale": "evidence"}\n```'
    assert parse_response(text) == (0.6, "evidence")


def test_parse_bare_fence():
    text = '```\n{"p_yes": 0.3, "rationale": "x"}\n```'
    assert parse_response(text) == (0.3, "x")


def test_parse_embedded_json():
    text = 'Sure! Here is my forecast: {"p_yes": 0.55, "rationale": "tossup"} ok?'
    assert parse_response(text) == (0.55, "tossup")


def test_parse_clamps_to_contract_range():
    assert parse_response('{"p_yes": 1.0, "rationale": "certain"}') == (0.99, "certain")
    assert parse_response('{"p_yes": 0.0, "rationale": "no way"}') == (0.01, "no way")


def test_parse_rejects_out_of_unit_range():
    # p outside [0, 1] entirely → invalid response.
    assert parse_response('{"p_yes": 1.5, "rationale": "x"}') is None
    assert parse_response('{"p_yes": -0.1, "rationale": "x"}') is None


def test_parse_rejects_garbage():
    assert parse_response("no json here") is None
    assert parse_response('{"p_yes": "high"}') is None
    assert parse_response('{"rationale": "no number"}') is None
    assert parse_response("") is None


def test_parse_fills_default_rationale_when_missing():
    out = parse_response('{"p_yes": 0.4}')
    assert out is not None
    p, r = out
    assert p == 0.4
    assert r  # non-empty default
