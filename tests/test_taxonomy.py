import pytest

from slm_synth.taxonomy import (
    CATEGORIES,
    EVAL_FAMILIES,
    FAILURE_MODES,
    validate_category,
    validate_difficulty,
    validate_eval_family,
    validate_failure_mode,
    validate_metadata,
    validate_template_family,
)


def test_taxonomy_contains_alignment_labels():
    assert "answer_only_compliance" in CATEGORIES
    assert "direct_arithmetic" in CATEGORIES
    assert "private_info_restraint" in CATEGORIES
    assert "repeat_exact_n_times" in EVAL_FAMILIES
    assert "code_expression_result" in EVAL_FAMILIES
    assert "extra_explanation" in FAILURE_MODES


def test_validate_metadata_accepts_sft_metadata():
    metadata = validate_metadata(
        {
            "category": " exact_output_format_control ",
            "difficulty": 1,
            "template_family": "repeat_word_count",
            "eval_family": "repeat_exact_n_times",
        }
    )

    assert metadata == {
        "category": "exact_output_format_control",
        "difficulty": 1,
        "template_family": "repeat_word_count",
        "eval_family": "repeat_exact_n_times",
    }


def test_validate_metadata_accepts_dpo_failure_mode():
    metadata = validate_metadata(
        {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "extra_explanation",
        },
        require_failure_mode=True,
    )

    assert metadata["failure_mode"] == "extra_explanation"


def test_validate_metadata_rejects_failure_mode_for_sft_metadata():
    with pytest.raises(ValueError, match="unsupported field"):
        validate_metadata(
            {
                "category": "answer_only_compliance",
                "difficulty": 1,
                "template_family": "direct_qa",
                "eval_family": "basic_arithmetic_qa",
                "failure_mode": "extra_explanation",
            }
        )


def test_validate_metadata_rejects_missing_dpo_failure_mode():
    with pytest.raises(ValueError, match="missing required"):
        validate_metadata(
            {
                "category": "answer_only_compliance",
                "difficulty": 1,
                "template_family": "direct_qa",
                "eval_family": "basic_arithmetic_qa",
            },
            require_failure_mode=True,
        )


def test_taxonomy_rejects_unknown_or_invalid_values():
    with pytest.raises(ValueError, match="Unsupported category"):
        validate_category("benchmark_answer_copying")

    with pytest.raises(ValueError, match="Unsupported eval_family"):
        validate_eval_family("truthfulqa_exact")

    with pytest.raises(ValueError, match="Unsupported failure_mode"):
        validate_failure_mode("random_bad_answer")

    with pytest.raises(ValueError, match="between"):
        validate_difficulty(6)

    with pytest.raises(ValueError, match="snake_case"):
        validate_template_family("Repeat Word Count")
