import pytest

from slm_synth.sft.batches import (
    SFT_BATCH_RESPONSE_SCHEMA,
    build_sft_teacher_request_items,
    build_sft_teacher_request_object,
    render_sft_batch_prompt,
    validate_sft_batch_response,
)
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec


def _sft_spec():
    return {
        "id": "sft_direct_arithmetic_000001",
        "instruction": "Create an addition question using 13 and 28. Answer concisely.",
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
        },
        "variables": {"a": 13, "b": 28},
        "constraints": ["The assistant answer should be concise."],
        "holdout_key": {"op": "add", "a": 2, "b": 2},
    }


def test_validate_sft_spec_accepts_local_generation_spec():
    spec = validate_sft_spec(_sft_spec())

    assert spec["id"] == "sft_direct_arithmetic_000001"
    assert spec["metadata"]["category"] == "direct_arithmetic"
    assert spec["variables"] == {"a": 13, "b": 28}
    assert spec["constraints"] == ["The assistant answer should be concise."]
    assert spec["holdout_key"] == {"op": "add", "a": 2, "b": 2}


def test_teacher_visible_sft_spec_hides_holdout_key():
    visible = teacher_visible_sft_spec(_sft_spec())

    assert visible["id"] == "sft_direct_arithmetic_000001"
    assert visible["variables"] == {"a": 13, "b": 28}
    assert "holdout_key" not in visible


def test_build_sft_teacher_request_items_rejects_duplicate_ids():
    spec = _sft_spec()
    with pytest.raises(ValueError, match="duplicate id"):
        build_sft_teacher_request_items([spec, spec])


def test_build_sft_teacher_request_object_wraps_items():
    payload = build_sft_teacher_request_object([_sft_spec()])

    assert set(payload) == {"items"}
    assert payload["items"][0]["id"] == "sft_direct_arithmetic_000001"
    assert "holdout_key" not in payload["items"][0]


def test_render_sft_batch_prompt_contains_llm_generation_contract():
    prompt = render_sft_batch_prompt([_sft_spec()])

    assert "synthetic supervised fine-tuning data" in prompt
    assert "sft_direct_arithmetic_000001" in prompt
    assert "Preserve metadata values" in prompt
    assert "holdout_key" in prompt  # mentioned as forbidden output/local-only field
    assert '"holdout_key":' not in prompt


def test_validate_sft_batch_response_accepts_expected_rows():
    rows = validate_sft_batch_response(
        {
            "items": [
                {
                    "id": "sft_direct_arithmetic_000001",
                    "messages": [
                        {"role": "user", "content": "What is 13 + 28?"},
                        {"role": "assistant", "content": "41"},
                    ],
                    "metadata": {
                        "category": "direct_arithmetic",
                        "difficulty": 1,
                        "template_family": "direct_qa",
                        "eval_family": "basic_arithmetic_qa",
                    },
                }
            ]
        },
        expected_ids=["sft_direct_arithmetic_000001"],
        expected_count=1,
    )

    assert rows[0]["messages"][1]["content"] == "41"


def test_validate_sft_batch_response_rejects_unexpected_top_level_fields():
    with pytest.raises(ValueError, match="unexpected field"):
        validate_sft_batch_response({"items": [], "metadata": {}})


def test_validate_sft_batch_response_rejects_missing_expected_ids():
    with pytest.raises(ValueError, match="missing expected id"):
        validate_sft_batch_response({"items": []}, expected_ids=["sft_direct_arithmetic_000001"])


def test_validate_sft_batch_response_rejects_unexpected_ids():
    with pytest.raises(ValueError, match="unexpected id"):
        validate_sft_batch_response(
            {
                "items": [
                    {
                        "id": "sft_other_000001",
                        "messages": [
                            {"role": "user", "content": "What is 13 + 28?"},
                            {"role": "assistant", "content": "41"},
                        ],
                        "metadata": {
                            "category": "direct_arithmetic",
                            "difficulty": 1,
                            "template_family": "direct_qa",
                            "eval_family": "basic_arithmetic_qa",
                        },
                    }
                ]
            },
            expected_ids=["sft_direct_arithmetic_000001"],
        )


def test_sft_batch_schema_matches_response_contract():
    assert SFT_BATCH_RESPONSE_SCHEMA["required"] == ["items"]
    item_schema = SFT_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert item_schema["required"] == ["id", "messages", "metadata"]
    assert item_schema["additionalProperties"] is False
    message_schema = item_schema["properties"]["messages"]["items"]
    assert message_schema["required"] == ["role", "content"]
    assert message_schema["properties"]["role"]["enum"] == ["user", "assistant"]
    metadata_schema = item_schema["properties"]["metadata"]
    assert metadata_schema["additionalProperties"] is False
    assert metadata_schema["required"] == ["category", "difficulty", "template_family", "eval_family"]
