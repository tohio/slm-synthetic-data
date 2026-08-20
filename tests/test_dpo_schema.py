import pytest

from slm_synth.dpo.schema import validate_dpo_row


def _row():
    return {
        "id": "dpo_001", "prompt": [{"role": "user", "content": "Summarize the supplied note."}],
        "chosen": [{"role": "assistant", "content": "A concise grounded summary."}],
        "rejected": [{"role": "assistant", "content": "A long response containing unsupported additions."}],
        "metadata": {
            "task_family": "summarization", "interaction_modes": ["single_turn"],
            "output_mode": "concise", "context_mode": "supplied_passage", "difficulty": 2,
            "template_family": "grounded_summary", "preference_dimension": "groundedness",
            "failure_mode": "ungrounded_response",
        },
    }


def test_validate_dpo_row_accepts_new_public_metadata():
    assert validate_dpo_row(_row()) == _row()


def test_validate_dpo_row_rejects_eval_family_and_category():
    for field in ("eval_family", "category"):
        row = _row()
        row["metadata"][field] = "legacy"
        with pytest.raises(ValueError, match="unsupported field"):
            validate_dpo_row(row)


def test_validate_dpo_row_requires_preference_dimension():
    row = _row()
    del row["metadata"]["preference_dimension"]
    with pytest.raises(ValueError, match="missing required"):
        validate_dpo_row(row)


def test_validate_dpo_row_requires_distinct_responses():
    row = _row()
    row["rejected"] = row["chosen"]
    with pytest.raises(ValueError, match="must differ"):
        validate_dpo_row(row)


def test_validate_dpo_prompt_ends_with_user():
    row = _row()
    row["prompt"].append({"role": "system", "content": "Late system message"})
    with pytest.raises(ValueError, match="system message is permitted only"):
        validate_dpo_row(row)


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search approved documents.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }]


def _tool_row():
    row = _row()
    row["tools"] = _tools()
    row["prompt"] = [
        {"role": "system", "content": "Use only approved documents."},
        {"role": "user", "content": "Find the cancellation period."},
    ]
    row["chosen"] = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "chosen_search", "type": "function", "function": {"name": "search_docs", "arguments": {"query": "cancellation period"}}}]},
        {"role": "tool", "tool_call_id": "chosen_search", "content": "Policy: notice is 30 days."},
        {"role": "assistant", "content": "The policy requires 30 days' notice."},
    ]
    row["rejected"] = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "rejected_search", "type": "function", "function": {"name": "search_docs", "arguments": {"query": "renewal period"}}}]},
        {"role": "tool", "tool_call_id": "rejected_search", "content": "Policy: renewal is annual."},
        {"role": "assistant", "content": "The cancellation period is one year."},
    ]
    row["metadata"]["interaction_modes"] = ["single_turn", "system_conditioned", "tool_mediated"]
    row["metadata"]["output_mode"] = "tool_call"
    row["metadata"]["preference_dimension"] = "tool_call_correctness"
    row["metadata"]["failure_mode"] = "incorrect_tool_call"
    return row


def test_validate_dpo_row_accepts_multi_message_tool_branches_with_shared_tools():
    validated = validate_dpo_row(_tool_row())
    assert len(validated["chosen"]) == 3
    assert len(validated["rejected"]) == 3
    assert validated["tools"] == _tools()


def test_validate_dpo_row_rejects_branch_user_or_system_messages():
    row = _tool_row()
    row["chosen"] = [{"role": "user", "content": "A different prompt."}]
    with pytest.raises(ValueError, match="branch contains unsupported role"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_branch_tool_not_in_shared_inventory():
    row = _tool_row()
    row["rejected"][0]["tool_calls"][0]["function"]["name"] = "private_search"
    with pytest.raises(ValueError, match="undeclared tool"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_unresolved_branch_tool_call():
    row = _tool_row()
    row["chosen"].pop(1)
    with pytest.raises(ValueError, match="unresolved tool_call"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_nested_prompt_or_tools_in_branch_message():
    row = _tool_row()
    row["chosen"][0]["prompt"] = row["prompt"]
    with pytest.raises(ValueError, match="unsupported field"):
        validate_dpo_row(row)
