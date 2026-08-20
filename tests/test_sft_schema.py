import pytest

from slm_synth.sft.schema import validate_sft_row


def _metadata(**updates):
    value = {
        "task_family": "grounded_qa_and_reading", "interaction_modes": ["single_turn"],
        "output_mode": "free_text", "context_mode": "supplied_passage", "difficulty": 2,
        "template_family": "evidence_grounded_answer",
    }
    value.update(updates)
    return value


def _row():
    return {
        "id": "sft_001",
        "messages": [{"role": "user", "content": "Use the passage."}, {"role": "assistant", "content": "Grounded answer."}],
        "metadata": _metadata(),
    }


def test_validate_sft_row_accepts_new_public_metadata():
    assert validate_sft_row(_row()) == _row()


def test_validate_sft_row_accepts_multi_turn_when_labeled():
    row = _row()
    row["messages"] = [
        {"role": "user", "content": "Help me plan."}, {"role": "assistant", "content": "What is the deadline?"},
        {"role": "user", "content": "Friday."}, {"role": "assistant", "content": "Here is a plan."},
    ]
    row["metadata"] = _metadata(task_family="planning_brainstorming_recommendations", interaction_modes=["multi_turn"], context_mode="self_contained")
    assert validate_sft_row(row)["messages"][-1]["role"] == "assistant"


def test_validate_sft_row_requires_axis_message_consistency():
    row = _row()
    row["metadata"]["interaction_modes"] = ["multi_turn"]
    with pytest.raises(ValueError, match="must match"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_eval_family_and_category():
    for field in ("eval_family", "category"):
        row = _row()
        row["metadata"][field] = "legacy"
        with pytest.raises(ValueError, match="unsupported field"):
            validate_sft_row(row)


def test_validate_sft_row_rejects_adjacent_roles():
    row = _row()
    row["messages"].insert(1, {"role": "user", "content": "Second prompt"})
    with pytest.raises(ValueError, match="malformed role sequence"):
        validate_sft_row(row)


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Return weather for a resolved city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def test_validate_sft_row_accepts_system_conditioned_tool_conversation():
    row = _row()
    row["tools"] = _tools()
    row["messages"] = [
        {"role": "system", "content": "Use tools when current data is needed."},
        {"role": "user", "content": "What is the weather in Boston?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_weather",
            "type": "function",
            "function": {"name": "lookup_weather", "arguments": {"city": "Boston"}},
        }]},
        {"role": "tool", "tool_call_id": "call_weather", "content": "Cloudy, 18 C"},
        {"role": "assistant", "content": "Boston is cloudy and 18 C."},
    ]
    row["metadata"] = _metadata(
        interaction_modes=["single_turn", "system_conditioned", "tool_mediated"],
        output_mode="tool_call",
    )
    validated = validate_sft_row(row)
    assert validated["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"city": "Boston"}
    assert validated["tools"] == _tools()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda row: row["messages"].pop(2), "unresolved tool_call"),
        (lambda row: row["messages"][2].update(tool_call_id="unknown"), "unknown or duplicate"),
        (lambda row: row["messages"][1]["tool_calls"][0]["function"].update(name="missing_tool"), "undeclared tool"),
    ],
)
def test_validate_sft_row_rejects_malformed_tool_sequences(mutation, error):
    row = {
        "id": "sft_tool",
        "tools": _tools(),
        "messages": [
            {"role": "user", "content": "Check Boston."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_weather", "type": "function",
                "function": {"name": "lookup_weather", "arguments": {"city": "Boston"}},
            }]},
            {"role": "tool", "tool_call_id": "call_weather", "content": "Cloudy"},
            {"role": "assistant", "content": "It is cloudy."},
        ],
        "metadata": _metadata(interaction_modes=["single_turn", "tool_mediated"], output_mode="tool_call"),
    }
    mutation(row)
    with pytest.raises(ValueError, match=error):
        validate_sft_row(row)


def test_validate_sft_row_rejects_tool_activity_without_tool_mediated_label():
    row = {
        "id": "sft_tool", "tools": _tools(),
        "messages": [
            {"role": "user", "content": "Check Boston."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup_weather", "arguments": {"city": "Boston"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Cloudy"},
            {"role": "assistant", "content": "It is cloudy."},
        ],
        "metadata": _metadata(),
    }
    with pytest.raises(ValueError, match="tool_mediated"):
        validate_sft_row(row)
