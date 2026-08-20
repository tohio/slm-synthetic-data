import pytest

from slm_synth.output_constraints import (
    OutputConstraintError,
    evaluate_dpo_output_constraints,
    evaluate_output_constraints,
    evaluate_sft_output_constraints,
    validate_output_constraints,
)


def test_output_constraints_reject_invalid_bounds_and_unknown_fields():
    with pytest.raises(ValueError, match="must not exceed"):
        validate_output_constraints({"min_words": 10, "max_words": 5})
    with pytest.raises(ValueError, match="unsupported"):
        validate_output_constraints({"approximately_words": 10})


def test_output_constraints_measure_words_lines_terms_headings_and_items():
    result = evaluate_output_constraints(
        "# Decision\n- Keep the pilot\n- Review Friday",
        {
            "min_words": 5,
            "max_words": 8,
            "exact_nonempty_lines": 3,
            "exact_list_items": 2,
            "required_terms": ["pilot"],
            "forbidden_terms": ["guarantee"],
            "required_headings": ["Decision"],
        },
    )
    assert result["status"] == "passed"
    assert all(check["passed"] for check in result["checks"])


def test_sft_constraints_reject_the_short_attic_scene_regression():
    spec = {
        "id": "creative-1",
        "output_constraints": {"min_words": 500, "max_words": 700},
    }
    row = {
        "id": "creative-1",
        "messages": [
            {"role": "user", "content": "Write the scene."},
            {"role": "assistant", "content": "A short scene " * 95},
        ],
    }
    with pytest.raises(OutputConstraintError, match="min_words expected=500"):
        evaluate_sft_output_constraints(specs=[spec], rows=[row])


def test_dpo_constraints_require_chosen_but_only_measure_rejected_branch():
    spec = {"id": "pair-1", "output_constraints": {"max_words": 2}}
    row = {
        "id": "pair-1",
        "chosen": [{"role": "assistant", "content": "Two words"}],
        "rejected": [
            {"role": "assistant", "content": "This branch intentionally violates brevity"}
        ],
    }
    evidence = evaluate_dpo_output_constraints(specs=[spec], rows=[row])
    assert evidence["pair-1"]["status"] == "passed"
    assert evidence["pair-1"]["chosen"]["status"] == "passed"
    assert evidence["pair-1"]["rejected"]["status"] == "failed"
