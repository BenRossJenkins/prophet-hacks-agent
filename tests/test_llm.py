from __future__ import annotations

from agent.llm import _build_user_prompt, _extract_text, parse_response


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self) -> None:
        self.type = "server_tool_use"
        self.text = None


def test_extract_text_single_block():
    assert _extract_text([_TextBlock("hello")]) == "hello"


def test_extract_text_skips_tool_blocks():
    blocks = [_ToolUseBlock(), _TextBlock("answer"), _ToolUseBlock()]
    assert _extract_text(blocks) == "answer"


def test_extract_text_joins_multiple_text_blocks():
    blocks = [_TextBlock("hello"), _ToolUseBlock(), _TextBlock("world")]
    assert _extract_text(blocks) == "hello\nworld"


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


# ---- multi-outcome prompt examples ---------------------------------------


def test_multi_outcome_prompt_includes_both_favorite_and_longshot_examples():
    """The multi-outcome prompt should cover both directions: outcomes[0]
    as favorite (above uniform) AND outcomes[0] as longshot (below uniform).
    Without the longshot example the LLM tends to default to uniform when
    it doesn't recognize the candidate as a market favorite, losing real
    Brier on dark-horse / non-contender outcomes[0] cases."""
    event = {
        "title": "Who will win?",
        "category": "Sports",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["A", "B", "C", "D", "E"],  # 5 outcomes triggers multi
    }
    prompt = _build_user_prompt(event)
    # Both directions present
    assert "FAVORITE" in prompt
    assert "LONGSHOT" in prompt
    # The favorite case shows exceeding uniform (e.g. Boston 22% > 3.3%)
    assert "EXCEED" in prompt
    # The longshot case shows going below uniform
    assert "BELOW uniform" in prompt
    # Both shapes still in the marginal-probability frame
    assert "MARGINAL probability" in prompt


def test_multi_outcome_prompt_does_not_fire_on_binary():
    """Binary events should not get the multi-outcome top-K coaching."""
    event = {
        "title": "Will A beat B?",
        "category": "Sports",
        "close_time": "2026-12-31T23:59:59Z",
        "outcomes": ["A", "B"],
    }
    prompt = _build_user_prompt(event)
    assert "FAVORITE" not in prompt
    assert "LONGSHOT" not in prompt
