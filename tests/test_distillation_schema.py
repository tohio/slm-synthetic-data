import pytest

from slm_synth.distillation.schema import validate_public_row
from slm_synth.distillation.signals import DISTILLATION_SIGNALS, validate_signal
from slm_synth.distillation.validate import merge_teacher_outputs, validate_teacher_output


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
        }
    )

    assert row == {
        "id": "arithmetic-000001",
        "prompt": "What is 2 + 2?",
        "reasoning": None,
        "response": "4",
    }


def test_public_row_accepts_step_by_step_distillation():
    row = validate_public_row(
        {
            "id": "arithmetic-000002",
            "prompt": "What is 12 * 3?",
            "reasoning": ["12 * 3 means three groups of 12.", "12 + 12 + 12 = 36."],
            "response": "36",
        }
    )

    assert row["reasoning"] == ["12 * 3 means three groups of 12.", "12 + 12 + 12 = 36."]


@pytest.mark.parametrize(
    "field",
    [
        "signal",
        "metadata",
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
                field: "internal",
            }
        )


def test_teacher_output_contract_is_id_reasoning_response_only():
    output = validate_teacher_output(
        {
            "id": "planning-000001",
            "reasoning": ["Identify constraints.", "Choose a sequence."],
            "response": "Plan complete.",
        }
    )

    assert output["id"] == "planning-000001"


def test_teacher_output_rejects_unexpected_prompt_or_metadata():
    with pytest.raises(ValueError, match="unexpected field"):
        validate_teacher_output(
            {
                "id": "planning-000001",
                "prompt": "Do not return this from the teacher.",
                "reasoning": None,
                "response": "Plan complete.",
            }
        )


def test_merge_teacher_outputs_returns_public_rows_only():
    rows = merge_teacher_outputs(
        [
            {
                "id": "cloud-000001",
                "prompt": "Explain one benefit of autoscaling.",
                "signal": "cloud",
                "metadata": {"difficulty": "easy"},
            }
        ],
        [
            {
                "id": "cloud-000001",
                "reasoning": None,
                "response": "Autoscaling can add or remove capacity as demand changes.",
            }
        ],
    )

    assert rows == [
        {
            "id": "cloud-000001",
            "prompt": "Explain one benefit of autoscaling.",
            "reasoning": None,
            "response": "Autoscaling can add or remove capacity as demand changes.",
        }
    ]


def test_merge_teacher_outputs_rejects_missing_duplicate_and_unexpected_ids():
    prompt_records = [
        {"id": "code-000001", "prompt": "Write a function.", "signal": "code"},
        {"id": "code-000002", "prompt": "Write another function.", "signal": "code"},
    ]

    with pytest.raises(ValueError, match="missing id"):
        merge_teacher_outputs(
            prompt_records,
            [{"id": "code-000001", "reasoning": None, "response": "def f(): pass"}],
        )

    with pytest.raises(ValueError, match="duplicate id"):
        merge_teacher_outputs(
            [{"id": "code-000001", "prompt": "Write a function.", "signal": "code"}],
            [
                {"id": "code-000001", "reasoning": None, "response": "def f(): pass"},
                {"id": "code-000001", "reasoning": None, "response": "def g(): pass"},
            ],
        )

    with pytest.raises(ValueError, match="unexpected id"):
        merge_teacher_outputs(
            [{"id": "code-000001", "prompt": "Write a function.", "signal": "code"}],
            [
                {"id": "code-000001", "reasoning": None, "response": "def f(): pass"},
                {"id": "code-999999", "reasoning": None, "response": "def g(): pass"},
            ],
        )
