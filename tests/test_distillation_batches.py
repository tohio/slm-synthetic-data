import pytest

from slm_synth.distillation.batches import (
    TEACHER_BATCH_RESPONSE_SCHEMA,
    build_teacher_request_items,
    build_teacher_request_object,
    render_teacher_batch_prompt,
    validate_teacher_batch_response,
)


def test_build_teacher_request_items_exposes_only_id_and_prompt():
    items = build_teacher_request_items(
        [
            {
                "id": "arithmetic-000001",
                "prompt": "What is 2 + 2?",
                "signal": "arithmetic",
                "metadata": {"difficulty": "easy"},
            }
        ]
    )

    assert items == [{"id": "arithmetic-000001", "prompt": "What is 2 + 2?"}]


def test_build_teacher_request_object_wraps_items():
    payload = build_teacher_request_object(
        [{"id": "cloud-000001", "prompt": "Explain autoscaling.", "signal": "cloud"}]
    )

    assert payload == {"items": [{"id": "cloud-000001", "prompt": "Explain autoscaling."}]}


def test_build_teacher_request_rejects_duplicate_prompt_ids():
    with pytest.raises(ValueError, match="duplicate id"):
        build_teacher_request_items(
            [
                {"id": "code-000001", "prompt": "Write code.", "signal": "code"},
                {"id": "code-000001", "prompt": "Write more code.", "signal": "code"},
            ]
        )


def test_render_teacher_batch_prompt_contains_contract_and_no_local_metadata_payload():
    prompt = render_teacher_batch_prompt(
        signal="planning",
        prompt_records=[
            {
                "id": "planning-000001",
                "prompt": "Plan a three-step migration.",
                "signal": "planning",
                "metadata": {"difficulty": "medium"},
            }
        ],
    )

    assert "Signal: planning" in prompt
    assert "planning-000001" in prompt
    assert "Plan a three-step migration." in prompt
    assert "metadata" in prompt  # mentioned as a forbidden output field
    assert '"metadata": {"difficulty"' not in prompt
    assert "teacher_provider" in prompt  # mentioned as a forbidden output field


def test_validate_teacher_batch_response_accepts_items_object():
    outputs = validate_teacher_batch_response(
        {
            "items": [
                {
                    "id": "database-000001",
                    "reasoning": ["Identify the table.", "Write a filter."],
                    "response": "SELECT * FROM users WHERE active = true;",
                }
            ]
        },
        expected_count=1,
    )

    assert outputs == [
        {
            "id": "database-000001",
            "reasoning": ["Identify the table.", "Write a filter."],
            "response": "SELECT * FROM users WHERE active = true;",
        }
    ]


def test_validate_teacher_batch_response_rejects_unexpected_top_level_fields():
    with pytest.raises(ValueError, match="unexpected field"):
        validate_teacher_batch_response({"items": [], "metadata": {}})


def test_validate_teacher_batch_response_rejects_count_mismatch():
    with pytest.raises(ValueError, match="expected 2 item"):
        validate_teacher_batch_response(
            {"items": [{"id": "instruction-000001", "reasoning": None, "response": "Done."}]},
            expected_count=2,
        )


def test_teacher_batch_schema_matches_response_contract():
    assert TEACHER_BATCH_RESPONSE_SCHEMA["required"] == ["items"]
    item_schema = TEACHER_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert item_schema["required"] == ["id", "reasoning", "response"]
    assert item_schema["additionalProperties"] is False
