import pytest

from slm_synth.dpo.schema import validate_dpo_row, validate_message_list


def _valid_dpo_row():
    return {
        "id": "dpo_answer_only_arithmetic_000001",
        "prompt": [
            {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
        ],
        "chosen": [
            {"role": "assistant", "content": "43"},
        ],
        "rejected": [
            {"role": "assistant", "content": "The answer is 43 because 16 plus 27 equals 43."},
        ],
        "metadata": {
            "category": "answer_only_compliance",
            "failure_mode": "extra_explanation",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
        },
    }


def test_validate_dpo_row_accepts_expected_shape():
    row = validate_dpo_row(_valid_dpo_row())

    assert row["id"] == "dpo_answer_only_arithmetic_000001"
    assert row["prompt"] == [
        {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
    ]
    assert row["chosen"] == [{"role": "assistant", "content": "43"}]
    assert row["rejected"] == [
        {"role": "assistant", "content": "The answer is 43 because 16 plus 27 equals 43."},
    ]
    assert row["metadata"]["failure_mode"] == "extra_explanation"


def test_validate_dpo_row_allows_system_prompt_prefix():
    row = _valid_dpo_row()
    row["prompt"] = [
        {"role": "system", "content": "Follow the requested format exactly."},
        {"role": "user", "content": "Repeat dog exactly four times."},
    ]
    row["chosen"] = [{"role": "assistant", "content": "dog dog dog dog"}]
    row["rejected"] = [{"role": "assistant", "content": "dog dog dog dog dog"}]
    row["metadata"] = {
        "category": "exact_output_format_control",
        "failure_mode": "format_violation",
        "difficulty": 1,
        "template_family": "repeat_word_count",
        "eval_family": "repeat_exact_n_times",
    }

    validated = validate_dpo_row(row)

    assert validated["prompt"][0]["role"] == "system"
    assert validated["prompt"][-1]["role"] == "user"


def test_validate_dpo_row_rejects_extra_fields():
    row = _valid_dpo_row()
    row["teacher_model"] = "internal-only"

    with pytest.raises(ValueError, match="unsupported field"):
        validate_dpo_row(row)


def test_validate_dpo_row_requires_failure_mode():
    row = _valid_dpo_row()
    del row["metadata"]["failure_mode"]

    with pytest.raises(ValueError, match="missing required"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_unknown_failure_mode():
    row = _valid_dpo_row()
    row["metadata"]["failure_mode"] = "random_bad_answer"

    with pytest.raises(ValueError, match="Unsupported failure_mode"):
        validate_dpo_row(row)


def test_validate_dpo_row_requires_prompt_to_end_with_user():
    row = _valid_dpo_row()
    row["prompt"] = [
        {"role": "user", "content": "What is 16 + 27?"},
        {"role": "system", "content": "Answer briefly."},
    ]

    with pytest.raises(ValueError, match="final prompt message"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_assistant_inside_prompt():
    row = _valid_dpo_row()
    row["prompt"].append({"role": "assistant", "content": "43"})

    with pytest.raises(ValueError, match="unsupported role"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_user_inside_chosen():
    row = _valid_dpo_row()
    row["chosen"] = [{"role": "user", "content": "43"}]

    with pytest.raises(ValueError, match="unsupported role"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_multiple_chosen_messages():
    row = _valid_dpo_row()
    row["chosen"] = [
        {"role": "assistant", "content": "Restated prompt."},
        {"role": "assistant", "content": "43"},
    ]

    with pytest.raises(ValueError, match="exactly 1 message"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_multiple_rejected_messages():
    row = _valid_dpo_row()
    row["rejected"] = [
        {"role": "assistant", "content": "Restated prompt."},
        {"role": "assistant", "content": "44"},
    ]

    with pytest.raises(ValueError, match="exactly 1 message"):
        validate_dpo_row(row)


def test_validate_dpo_row_rejects_identical_chosen_and_rejected():
    row = _valid_dpo_row()
    row["rejected"] = list(row["chosen"])

    with pytest.raises(ValueError, match="must differ"):
        validate_dpo_row(row)


def test_validate_message_list_rejects_empty_list():
    with pytest.raises(ValueError, match="must contain"):
        validate_message_list(
            [],
            field_name="chosen",
            allowed_roles={"assistant"},
            required_roles={"assistant"},
        )
