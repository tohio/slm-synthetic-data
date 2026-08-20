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
    with pytest.raises(ValueError, match="final prompt"):
        validate_dpo_row(row)
