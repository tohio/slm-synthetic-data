import pytest

from slm_synth.distillation_sft.schema import validate_public_row
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS, validate_signal


def _metadata(
    *,
    category="direct_arithmetic",
    difficulty=1,
    template_family="integer_addition",
    eval_family="basic_arithmetic_qa",
):
    return {
        "category": category,
        "difficulty": difficulty,
        "template_family": template_family,
        "eval_family": eval_family,
    }


def test_distillation_signals_are_fixed_scope():
    assert DISTILLATION_SIGNALS == {
        "arithmetic",
        "code",
        "debugging",
        "database",
        "cloud",
        "data_transform",
        "educational_qa",
        "factual_restraint",
        "planning",
        "instruction",
    }
    assert validate_signal(" arithmetic ") == "arithmetic"


def test_public_row_accepts_response_only_distillation():
    row = validate_public_row(
        {
            "id": "arithmetic-000001",
            "prompt": "What is 2 + 2?",
            "reasoning": None,
            "response": "4",
            "metadata": _metadata(),
        }
    )

    assert row == {
        "id": "arithmetic-000001",
        "prompt": "What is 2 + 2?",
        "reasoning": None,
        "response": "4",
        "metadata": _metadata(),
    }


def test_public_row_rejects_step_by_step_reasoning():
    with pytest.raises(ValueError, match="reasoning.*null"):
        validate_public_row(
            {
                "id": "arithmetic-000002",
                "prompt": "What is 12 * 3?",
                "reasoning": ["12 * 3 means three groups of 12.", "12 + 12 + 12 = 36."],
                "response": "36",
                "metadata": _metadata(template_family="integer_multiplication"),
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "signal",
        "teacher_model",
        "teacher_provider",
        "generation_run",
        "difficulty",
    ],
)
def test_public_row_rejects_internal_fields(field):
    with pytest.raises(ValueError, match="forbidden field"):
        validate_public_row(
            {
                "id": "instruction-000001",
                "prompt": "Summarize this.",
                "reasoning": None,
                "response": "Summary.",
                "metadata": _metadata(
                    category="general_instruction_following",
                    template_family="instruction_rewrite",
                    eval_family=None,
                ),
                field: "internal",
            }
        )

def test_public_row_rejects_generation_only_metadata():
    metadata = _metadata()
    metadata["prompt_source"] = "production_spec"

    with pytest.raises(ValueError, match="unsupported field"):
        validate_public_row(
            {
                "id": "arithmetic-000001",
                "prompt": "What is 2 + 2?",
                "reasoning": None,
                "response": "4",
                "metadata": metadata,
            }
        )
