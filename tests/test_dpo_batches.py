import pytest

from slm_synth.dpo.batches import (
    DPO_BATCH_RESPONSE_SCHEMA,
    build_dpo_teacher_request_items,
    build_dpo_teacher_request_object,
    build_exact_target_dpo_batch_response,
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
    assert item_schema["properties"]["prompt"]["items"]["properties"]["role"]["enum"] == ["system", "user"]
    assert item_schema["properties"]["chosen"]["items"]["properties"]["role"]["enum"] == ["assistant"]
    assert item_schema["properties"]["rejected"]["items"]["properties"]["role"]["enum"] == ["assistant"]
    metadata_schema = item_schema["properties"]["metadata"]
    assert metadata_schema["additionalProperties"] is False
    assert metadata_schema["required"] == [
        "category",
        "difficulty",
        "template_family",
        "eval_family",
        "failure_mode",
    ]


def test_validate_dpo_batch_response_rejects_clean_but_wrong_function_body():
    spec = {
        "id": "dpo_function_completion_body_only_000001",
        "instruction": "Complete the function body only.",
        "metadata": {
            "category": "code_generation",
            "difficulty": 2,
            "template_family": "python_function_body_only",
            "eval_family": "function_completion_body_only",
            "failure_mode": "code_includes_explanation",
        },
        "variables": {
            "function_name": "add_numbers",
            "function_signature": "def add_numbers(a, b):",
            "chosen_answer": "return a + b",
            "rejected_answer": "# Explanation: implement the requested behavior.\nreturn a + b",
        },
    }
    row = {
        "id": "dpo_function_completion_body_only_000001",
        "prompt": [{"role": "user", "content": "Complete this body: def add_numbers(a, b):"}],
        "chosen": [{"role": "assistant", "content": "return a % 2 == 0"}],
        "rejected": [{"role": "assistant", "content": "# Explanation: implement the requested behavior.\nreturn a + b"}],
        "metadata": spec["metadata"],
    }

    with pytest.raises(ValueError, match="chosen content does not match"):
        validate_dpo_batch_response({"items": [row]}, expected_specs=[spec])


def test_validate_dpo_batch_response_rejects_list_prompt_answer_leakage():
    spec = {
        "id": "dpo_list_exact_n_items_000001",
        "instruction": "List exact items.",
        "metadata": {
            "category": "exact_output_format_control",
            "difficulty": 1,
            "template_family": "list_exact_count",
            "eval_family": "list_exact_n_items",
            "failure_mode": "format_violation",
        },
        "variables": {
            "items": ["red", "green", "blue"],
            "answer": "red, green, blue",
            "chosen_answer": "red, green, blue",
            "rejected_answer": "red, green, blue, purple",
        },
    }
    row = {
        "id": "dpo_list_exact_n_items_000001",
        "prompt": [{"role": "user", "content": "List exactly 3 colors: red, green, blue."}],
        "chosen": [{"role": "assistant", "content": "red, green, blue"}],
        "rejected": [{"role": "assistant", "content": "red, green, blue, purple"}],
        "metadata": spec["metadata"],
    }

    with pytest.raises(ValueError, match="prompt leaks"):
        validate_dpo_batch_response({"items": [row]}, expected_specs=[spec])


def test_validate_dpo_batch_response_accepts_exact_targeted_spec():
    spec = {
        "id": "dpo_short_factual_stop_behavior_000001",
        "instruction": "Answer briefly.",
        "metadata": {
            "category": "controlled_verbosity",
            "difficulty": 1,
            "template_family": "short_factual_answer",
            "eval_family": "short_factual_stop_behavior",
            "failure_mode": "verbosity_mismatch",
        },
        "variables": {
            "chosen_answer": "Rome",
            "rejected_answer": "The capital of Italy is Rome.",
        },
    }
    row = {
        "id": "dpo_short_factual_stop_behavior_000001",
        "prompt": [{"role": "user", "content": "What is the capital of Italy? Answer with only the city."}],
        "chosen": [{"role": "assistant", "content": "Rome"}],
        "rejected": [{"role": "assistant", "content": "The capital of Italy is Rome."}],
        "metadata": spec["metadata"],
    }

    rows = validate_dpo_batch_response({"items": [row]}, expected_specs=[spec])

    assert rows == [row]


def test_dpo_batch_schema_separates_prompt_and_response_roles():
    item_schema = DPO_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]

    prompt_schema = item_schema["properties"]["prompt"]["items"]
    chosen_schema = item_schema["properties"]["chosen"]["items"]
    rejected_schema = item_schema["properties"]["rejected"]["items"]

    assert prompt_schema["properties"]["role"]["enum"] == ["system", "user"]
    assert chosen_schema["properties"]["role"]["enum"] == ["assistant"]
    assert rejected_schema["properties"]["role"]["enum"] == ["assistant"]


def test_validate_dpo_batch_response_repairs_chosen_and_rejected_roles():
    row = _dpo_row()
    row["chosen"] = [{"role": "user", "content": "43"}]
    row["rejected"] = [{"role": "system", "content": "The answer is 43 because 17 plus 26 equals 43."}]

    rows = validate_dpo_batch_response({"items": [row]})

    assert rows[0]["chosen"] == [{"role": "assistant", "content": "43"}]
    assert rows[0]["rejected"] == [
        {"role": "assistant", "content": "The answer is 43 because 17 plus 26 equals 43."}
    ]


def test_validate_dpo_batch_response_rejects_assistant_prompt_role():
    row = _dpo_row()
    row["prompt"] = [{"role": "assistant", "content": "Answer with only the number: What is 17 + 26?"}]

    with pytest.raises(ValueError, match="prompt contains unsupported role"):
        validate_dpo_batch_response({"items": [row]})



def test_build_exact_target_dpo_batch_response_preserves_code_generation_targets():
    spec = {
        "id": "dpo_code_generation_function_000001",
        "instruction": "Create a Python function generation request and answer with code only.",
        "metadata": {
            "category": "code_generation",
            "difficulty": 2,
            "template_family": "python_function_code_only",
            "eval_family": "code_generation_function",
            "failure_mode": "code_includes_explanation",
        },
        "variables": {
            "function_name": "add_numbers",
            "requirement": "Return the sum of two numbers.",
            "function_signature": "def add_numbers(a, b):",
            "chosen_answer": "def add_numbers(a, b):\n    return a + b",
            "rejected_answer": "Here is the function:\n```python\ndef add_numbers(a, b):\n    return a + b\n```",
        },
    }

    response = build_exact_target_dpo_batch_response([spec])
    row = response["items"][0]

    assert row["chosen"][0]["content"] == "def add_numbers(a, b):\n    return a + b"
    assert row["rejected"][0]["content"].startswith("Here is the function:")
    assert row["prompt"][0]["role"] == "user"
    assert "def add_numbers(a, b):" in row["prompt"][0]["content"]
    assert "Markdown fences" in row["prompt"][0]["content"]


def test_build_exact_target_dpo_batch_response_does_not_leak_list_answer():
    spec = {
        "id": "dpo_list_exact_n_items_000001",
        "instruction": "Create an instruction to list exact items.",
        "metadata": {
            "category": "exact_output_format_control",
            "difficulty": 1,
            "template_family": "list_exact_count",
            "eval_family": "list_exact_n_items",
            "failure_mode": "format_violation",
        },
        "variables": {
            "item_type": "colors",
            "count": 3,
            "items": ["red", "green", "blue"],
            "answer": "red, green, blue",
            "chosen_answer": "red, green, blue",
            "rejected_answer": "red, green, blue, purple",
        },
    }

    response = build_exact_target_dpo_batch_response([spec])
    prompt = response["items"][0]["prompt"][0]["content"].lower()

    assert "red" not in prompt
    assert "green" not in prompt
    assert "blue" not in prompt
    validate_dpo_batch_response(response, expected_specs=[spec], expected_count=1, expected_ids=[spec["id"]])


def test_dpo_batch_response_schema_requires_single_chosen_and_rejected_message():
    row_schema = DPO_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]["properties"]
    assert row_schema["chosen"]["minItems"] == 1
    assert row_schema["chosen"]["maxItems"] == 1
    assert row_schema["rejected"]["minItems"] == 1
    assert row_schema["rejected"]["maxItems"] == 1
