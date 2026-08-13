from copy import deepcopy

import pytest

from slm_synth.distillation_dpo.pair_quality import validate_pair_quality
from slm_synth.distillation_dpo.spec_builders import build_production_rows


FAMILY = "teacher_response_preference"
SOURCE_ROWS = build_production_rows(family=FAMILY, count=40)


def _row(template_family):
    return deepcopy(
        next(
            row
            for row in SOURCE_ROWS
            if row["metadata"]["template_family"] == template_family
        )
    )


@pytest.mark.parametrize(
    "template_family",
    [
        "distillation_dpo_integer_subtraction",
        "distillation_dpo_inventory_word_problem",
    ],
)
def test_numeric_contract_requires_correct_chosen_and_incorrect_rejected(template_family):
    row = _row(template_family)
    correct = row["chosen"][0]["content"]
    row["chosen"] = [{"role": "assistant", "content": "not an integer"}]
    row["rejected"] = [{"role": "assistant", "content": correct}]

    reasons = validate_pair_quality(row)

    assert "chosen_wrong_numeric_answer" in reasons
    assert "rejected_matches_numeric_answer" in reasons


def test_grounded_fact_contract_requires_the_provided_one_word_status():
    row = _row("distillation_dpo_grounded_record_fact")
    correct = row["chosen"][0]["content"]
    row["chosen"] = [{"role": "assistant", "content": "unknown"}]
    row["rejected"] = [{"role": "assistant", "content": correct}]

    reasons = validate_pair_quality(row)

    assert "chosen_wrong_grounded_fact" in reasons
    assert "rejected_matches_grounded_fact" in reasons


def test_answer_only_and_exact_format_contracts_are_hard_gates():
    answer_only = _row("distillation_dpo_grounded_answer_only")
    answer_only["chosen"] = [
        {"role": "assistant", "content": "The requested region is north."}
    ]
    exact_format = _row("distillation_dpo_exact_token_repeat")
    exact_format["chosen"] = [
        {
            "role": "assistant",
            "content": exact_format["chosen"][0]["content"].replace(",", ", "),
        }
    ]

    assert "chosen_answer_only_violation" in validate_pair_quality(answer_only)
    assert "chosen_exact_format_violation" in validate_pair_quality(exact_format)


def test_clarification_sorting_and_two_bullet_contracts_are_hard_gates():
    clarification = _row("distillation_dpo_missing_transformation_context")
    clarification["chosen"] = [
        {"role": "assistant", "content": "I will assume the transformation."}
    ]
    sorting = _row("distillation_dpo_benign_task_over_refusal")
    sorting["chosen"] = [{"role": "assistant", "content": "3, 2, 1"}]
    bullets = _row("distillation_dpo_exact_two_bullets")
    bullets["chosen"] = [
        {"role": "assistant", "content": "One paragraph instead of two bullets."}
    ]

    assert "chosen_missing_clarifying_question" in validate_pair_quality(clarification)
    assert "chosen_incorrect_benign_sort" in validate_pair_quality(sorting)
    assert "chosen_two_bullet_violation" in validate_pair_quality(bullets)


def test_instruction_rewrite_enforces_its_explicit_word_limit():
    row = _row("distillation_dpo_instruction_rewrite")
    row["chosen"] = [{"role": "assistant", "content": " ".join(["word"] * 19)}]

    assert "chosen_rewrite_too_verbose" in validate_pair_quality(row)


def test_code_failure_modes_require_the_declared_rejected_structure():
    explanation = next(
        deepcopy(row)
        for row in SOURCE_ROWS
        if row["metadata"]["failure_mode"] == "code_includes_explanation"
    )
    function_name = explanation["chosen"][0]["content"].split("(", 1)[0].split()[-1]
    explanation["rejected"] = [
        {
            "role": "assistant",
            "content": f"def {function_name}(value):\n    return None",
        }
    ]
    logic_error = next(
        deepcopy(row)
        for row in SOURCE_ROWS
        if row["metadata"]["failure_mode"] == "code_logic_error"
    )
    logic_error["rejected"] = [
        {"role": "assistant", "content": "def broken(value)\n    return value"}
    ]

    assert "rejected_code_missing_expected_explanation" in validate_pair_quality(explanation)
    assert "rejected_code_not_valid_logic_error" in validate_pair_quality(logic_error)
