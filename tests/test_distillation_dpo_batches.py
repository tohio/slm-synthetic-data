import pytest

from slm_synth.distillation_dpo.batches import (
    DISTILLATION_DPO_BATCH_RESPONSE_SCHEMA,
    validate_distillation_dpo_batch_response,
)


def _row(row_id: str = "distillation-dpo-1"):
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "distillation_dpo_teacher_preference",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "wrong_numeric_answer",
        },
    }


def test_distillation_dpo_batch_schema_separates_prompt_and_response_roles():
    item_schema = DISTILLATION_DPO_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]

    prompt_schema = item_schema["properties"]["prompt"]["items"]
    chosen_schema = item_schema["properties"]["chosen"]["items"]
    rejected_schema = item_schema["properties"]["rejected"]["items"]

    assert prompt_schema["properties"]["role"]["enum"] == ["system", "user"]
    assert chosen_schema["properties"]["role"]["enum"] == ["assistant"]
    assert rejected_schema["properties"]["role"]["enum"] == ["assistant"]


def test_validate_distillation_dpo_batch_response_repairs_chosen_and_rejected_roles():
    row = _row()
    row["chosen"] = [{"role": "user", "content": "4"}]
    row["rejected"] = [{"role": "system", "content": "5"}]

    rows = validate_distillation_dpo_batch_response({"items": [row]})

    assert rows[0]["chosen"] == [{"role": "assistant", "content": "4"}]
    assert rows[0]["rejected"] == [{"role": "assistant", "content": "5"}]


def test_validate_distillation_dpo_batch_response_rejects_assistant_prompt_role():
    row = _row()
    row["prompt"] = [{"role": "assistant", "content": "What is 2 + 2?"}]

    with pytest.raises(ValueError, match="prompt contains unsupported role"):
        validate_distillation_dpo_batch_response({"items": [row]})
