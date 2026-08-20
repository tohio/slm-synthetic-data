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
    with pytest.raises(ValueError, match="alternate"):
        validate_sft_row(row)
