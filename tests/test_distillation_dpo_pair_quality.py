from slm_synth.distillation_dpo.pair_quality import (
    MAX_PRIVATE_RESTRAINT_WORDS,
    filter_pairs_by_quality,
    validate_pair_quality,
)


def _code_row(*, chosen: str, rejected: str):
    return {
        "id": "distillation-dpo-code-1",
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Write a Python function named clamp_2 that returns a number constrained "
                    "between a minimum and maximum. Return code only, no Markdown."
                ),
            }
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": {
            "category": "code_generation",
            "difficulty": 2,
            "template_family": "distillation_dpo_python_function",
            "eval_family": "code_generation_function",
            "failure_mode": "code_syntax_error",
        },
    }


def _restraint_row(*, chosen: str):
    return {
        "id": "distillation-dpo-restraint-1",
        "prompt": [
            {
                "role": "user",
                "content": "What is the private home address of a company CEO?",
            }
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [
            {
                "role": "assistant",
                "content": "The exact private address is definitely 123 Example Street.",
            }
        ],
        "metadata": {
            "category": "private_info_restraint",
            "difficulty": 2,
            "template_family": "distillation_dpo_factual_restraint",
            "eval_family": "private_or_unverifiable_company_fact",
            "failure_mode": "unsafe_private_info_guess",
        },
    }


def test_code_pair_quality_accepts_complete_chosen_and_syntax_error_rejected():
    row = _code_row(
        chosen=(
            "def clamp_2(value, minimum, maximum):\n"
            "    return max(minimum, min(value, maximum))"
        ),
        rejected="def clamp_2(value, minimum, maximum)\n    return value",
    )

    assert validate_pair_quality(row) == ()


def test_code_pair_quality_rejects_signature_only_chosen():
    row = _code_row(
        chosen="def clamp_2(value, minimum, maximum):",
        rejected="def clamp_2(value, minimum, maximum)\n    return value",
    )

    assert "chosen_code_signature_only" in validate_pair_quality(row)


def test_code_pair_quality_rejects_syntax_invalid_chosen():
    row = _code_row(
        chosen="def clamp_2(value, minimum, maximum)\n    return value",
        rejected="def clamp_2(value, minimum, maximum)\n    return minimum",
    )

    assert "chosen_code_syntax_error" in validate_pair_quality(row)


def test_code_pair_quality_rejects_placeholder_function_body():
    for body in ("pass", "...", "raise NotImplementedError"):
        row = _code_row(
            chosen=f"def clamp_2(value, minimum, maximum):\n    {body}",
            rejected="def clamp_2(value, minimum, maximum)\n    return value",
        )

        assert "chosen_code_incomplete_function" in validate_pair_quality(row)


def test_code_pair_quality_rejects_wrong_function_name():
    row = _code_row(
        chosen="def other_name(value, minimum, maximum):\n    return value",
        rejected="def clamp_2(value, minimum, maximum)\n    return value",
    )

    assert "chosen_code_wrong_function_name" in validate_pair_quality(row)


def test_code_pair_quality_requires_the_requested_function_to_be_complete():
    row = _code_row(
        chosen=(
            "def clamp_2(value, minimum, maximum):\n"
            "    pass\n\n"
            "def helper(value):\n"
            "    return value"
        ),
        rejected="def clamp_2(value, minimum, maximum)\n    return value",
    )

    assert "chosen_code_incomplete_function" in validate_pair_quality(row)


def test_code_pair_quality_requires_rejected_syntax_error_failure_mode():
    row = _code_row(
        chosen="def clamp_2(value, minimum, maximum):\n    return value",
        rejected="def clamp_2(value, minimum, maximum):\n    return minimum",
    )

    assert "rejected_code_missing_expected_syntax_error" in validate_pair_quality(row)


def test_code_pair_quality_rejects_non_python_garbage_as_syntax_error_example():
    row = _code_row(
        chosen="def clamp_2(value, minimum, maximum):\n    return value",
        rejected="This answer is intentionally bad.",
    )

    assert "rejected_code_not_python_function" in validate_pair_quality(row)


def test_private_restraint_quality_accepts_concise_chosen():
    row = _restraint_row(
        chosen="I cannot provide or invent a private home address. Use verified public contact channels instead."
    )

    assert validate_pair_quality(row) == ()


def test_private_restraint_quality_rejects_wordy_chosen():
    chosen = " ".join(["unnecessary"] * (MAX_PRIVATE_RESTRAINT_WORDS + 1))
    row = _restraint_row(chosen=chosen)

    accepted, summary = filter_pairs_by_quality(
        family="teacher_response_preference",
        rows=[row],
    )

    assert accepted == []
    assert summary.rejection_reasons == {"chosen_restraint_too_verbose": 1}
