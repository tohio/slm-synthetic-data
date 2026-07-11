import pytest

from slm_synth.sft.schema import validate_message, validate_sft_row


def _valid_sft_row():
    return {
        "id": "sft_answer_only_arithmetic_000001",
        "messages": [
            {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
            {"role": "assistant", "content": "43"},
        ],
        "metadata": {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
        },
    }


def test_validate_sft_row_accepts_expected_shape():
    row = validate_sft_row(_valid_sft_row())

    assert row["id"] == "sft_answer_only_arithmetic_000001"
    assert row["messages"] == [
        {"role": "user", "content": "Answer with only the number: What is 16 + 27?"},
        {"role": "assistant", "content": "43"},
    ]
    assert row["metadata"] == {
        "category": "answer_only_compliance",
        "difficulty": 1,
        "template_family": "direct_qa",
        "eval_family": "basic_arithmetic_qa",
    }


def test_validate_sft_row_allows_optional_system_message():
    row = _valid_sft_row()
    row["messages"] = [
        {"role": "system", "content": "Follow the requested format exactly."},
        {"role": "user", "content": "Repeat dog exactly four times."},
        {"role": "assistant", "content": "dog dog dog dog"},
    ]
    row["metadata"] = {
        "category": "exact_output_format_control",
        "difficulty": 1,
        "template_family": "repeat_word_count",
        "eval_family": "repeat_exact_n_times",
    }

    validated = validate_sft_row(row)

    assert validated["messages"][0]["role"] == "system"
    assert validated["messages"][-1]["role"] == "assistant"


def test_validate_message_normalizes_role_and_strips_content():
    assert validate_message({"role": " USER ", "content": " hello "}) == {
        "role": "user",
        "content": "hello",
    }


def test_validate_sft_row_rejects_extra_row_fields():
    row = _valid_sft_row()
    row["teacher_model"] = "internal-only"

    with pytest.raises(ValueError, match="unsupported field"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_failure_mode_metadata():
    row = _valid_sft_row()
    row["metadata"]["failure_mode"] = "extra_explanation"

    with pytest.raises(ValueError, match="unsupported field"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_unknown_category():
    row = _valid_sft_row()
    row["metadata"]["category"] = "benchmark_answer_copying"

    with pytest.raises(ValueError, match="Unsupported category"):
        validate_sft_row(row)


def test_validate_sft_row_requires_user_assistant_role_contract():
    row = _valid_sft_row()
    row["messages"] = [{"role": "assistant", "content": "43"}]

    with pytest.raises(ValueError, match="role contract"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_multiple_user_messages():
    row = _valid_sft_row()
    row["messages"] = [
        {"role": "user", "content": "Complete the function."},
        {"role": "user", "content": "def add_numbers(a, b):"},
        {"role": "assistant", "content": "return a + b"},
    ]

    with pytest.raises(ValueError, match="role contract"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_multiple_assistant_messages():
    row = _valid_sft_row()
    row["messages"] = [
        {"role": "user", "content": "Complete the function."},
        {"role": "assistant", "content": "x = 1"},
        {"role": "assistant", "content": "return x"},
    ]

    with pytest.raises(ValueError, match="role contract"):
        validate_sft_row(row)


def test_validate_sft_row_requires_final_assistant_message():
    row = _valid_sft_row()
    row["messages"] = [
        {"role": "user", "content": "What is 16 + 27?"},
        {"role": "assistant", "content": "43"},
        {"role": "user", "content": "Thanks"},
    ]

    with pytest.raises(ValueError, match="role contract"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_bad_role():
    row = _valid_sft_row()
    row["messages"][0]["role"] = "tool"

    with pytest.raises(ValueError, match="unsupported message role"):
        validate_sft_row(row)


def test_validate_sft_row_rejects_empty_content():
    row = _valid_sft_row()
    row["messages"][1]["content"] = " "

    with pytest.raises(ValueError, match="content"):
        validate_sft_row(row)
