import pytest

from slm_synth.dpo.batches import (
    DPO_BATCH_RESPONSE_SCHEMA,
    build_dpo_teacher_request_items,
    build_dpo_teacher_request_object,
    render_dpo_batch_prompt,
    validate_dpo_batch_response,
)
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec


def _dpo_spec():
    return {
        "id": "dpo_answer_only_arithmetic_000001",
        "instruction": (
            "Create an answer-only arithmetic prompt. The chosen answer should be only the "
            "number. The rejected answer should include extra explanation."
        ),
        "metadata": {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "extra_explanation",
        },
        "variables": {"a": 17, "b": 26},
        "constraints": ["Rejected response must be plausible but less preferred."],
        "holdout_key": {"op": "add", "a": 2, "b": 2},
    }


def _dpo_row(row_id: str = "dpo_answer_only_arithmetic_000001"):
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": "Answer with only the number: What is 17 + 26?"}],
        "chosen": [{"role": "assistant", "content": "43"}],
        "rejected": [{"role": "assistant", "content": "The answer is 43 because 17 plus 26 equals 43."}],
        "metadata": {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "extra_explanation",
        },
    }


def test_validate_dpo_spec_accepts_local_generation_spec():
    spec = validate_dpo_spec(_dpo_spec())

    assert spec["id"] == "dpo_answer_only_arithmetic_000001"
    assert spec["metadata"]["failure_mode"] == "extra_explanation"
    assert spec["variables"] == {"a": 17, "b": 26}
    assert spec["holdout_key"] == {"op": "add", "a": 2, "b": 2}


def test_teacher_visible_dpo_spec_hides_holdout_key():
    visible = teacher_visible_dpo_spec(_dpo_spec())

    assert visible["id"] == "dpo_answer_only_arithmetic_000001"
    assert visible["metadata"]["failure_mode"] == "extra_explanation"
    assert "holdout_key" not in visible


def test_build_dpo_teacher_request_items_rejects_duplicate_ids():
    spec = _dpo_spec()
    with pytest.raises(ValueError, match="duplicate id"):
        build_dpo_teacher_request_items([spec, spec])


def test_build_dpo_teacher_request_object_wraps_items():
    payload = build_dpo_teacher_request_object([_dpo_spec()])

    assert set(payload) == {"items"}
    assert payload["items"][0]["id"] == "dpo_answer_only_arithmetic_000001"
    assert "holdout_key" not in payload["items"][0]


def test_render_dpo_batch_prompt_contains_llm_generation_contract():
    prompt = render_dpo_batch_prompt([_dpo_spec()])

    assert "synthetic preference data" in prompt
    assert "dpo_answer_only_arithmetic_000001" in prompt
    assert "failure_mode" in prompt
    assert "chosen and rejected responses must differ" in prompt
    assert "variables.rejected_answer" in prompt
    assert '"holdout_key":' not in prompt


def test_validate_dpo_batch_response_accepts_expected_rows():
    rows = validate_dpo_batch_response(
        {"items": [_dpo_row()]},
        expected_ids=["dpo_answer_only_arithmetic_000001"],
        expected_count=1,
    )

    assert rows[0]["chosen"][0]["content"] == "43"
    assert rows[0]["metadata"]["failure_mode"] == "extra_explanation"


def test_validate_dpo_batch_response_rejects_unexpected_top_level_fields():
    with pytest.raises(ValueError, match="unexpected field"):
        validate_dpo_batch_response({"items": [], "metadata": {}})


def test_validate_dpo_batch_response_rejects_missing_expected_ids():
    with pytest.raises(ValueError, match="missing expected id"):
        validate_dpo_batch_response({"items": []}, expected_ids=["dpo_answer_only_arithmetic_000001"])


def test_validate_dpo_batch_response_rejects_unexpected_ids():
    with pytest.raises(ValueError, match="unexpected id"):
        validate_dpo_batch_response(
            {"items": [_dpo_row(row_id="dpo_other_000001")]},
            expected_ids=["dpo_answer_only_arithmetic_000001"],
        )


def test_validate_dpo_batch_response_rejects_same_chosen_and_rejected():
    row = _dpo_row()
    row["rejected"] = row["chosen"]

    with pytest.raises(ValueError, match="chosen and rejected"):
        validate_dpo_batch_response({"items": [row]})


def test_dpo_batch_schema_matches_response_contract():
    assert DPO_BATCH_RESPONSE_SCHEMA["required"] == ["items"]
    item_schema = DPO_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert item_schema["required"] == ["id", "prompt", "chosen", "rejected", "metadata"]
    assert item_schema["additionalProperties"] is False
    for field in ("prompt", "chosen", "rejected"):
        message_schema = item_schema["properties"][field]["items"]
        assert message_schema["required"] == ["role", "content"]
        assert message_schema["properties"]["role"]["enum"] == ["user", "assistant"]
    metadata_schema = item_schema["properties"]["metadata"]
    assert metadata_schema["additionalProperties"] is False
    assert metadata_schema["required"] == [
        "category",
        "difficulty",
        "template_family",
        "eval_family",
        "failure_mode",
    ]
