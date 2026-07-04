import pytest

from slm_synth.sft.batches import (
    SFT_BATCH_RESPONSE_SCHEMA,
    build_sft_teacher_request_items,
    build_sft_teacher_request_object,
    render_sft_batch_prompt,
    validate_sft_batch_response,
    validate_sft_rows_against_specs,
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
        "variables": {"a": 13, "b": 28, "answer": 41},
        "constraints": ["The assistant answer should be concise."],
        "holdout_key": {"op": "add", "a": 2, "b": 2},
    }


def _row(*, row_id="sft_direct_arithmetic_000001", answer="41", family="basic_arithmetic_qa"):
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "What is 13 + 28?"},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": family,
        },
    }


def _spec(*, row_id="sft_direct_arithmetic_000001", family="basic_arithmetic_qa", variables=None):
    return {
        "id": row_id,
        "instruction": "Create a controlled SFT row.",
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "direct_qa",
            "eval_family": family,
        },
        "variables": {"answer": 41} if variables is None else variables,
    }


def test_validate_sft_spec_accepts_local_generation_spec():
    spec = validate_sft_spec(_sft_spec())

    assert spec["id"] == "sft_direct_arithmetic_000001"
    assert spec["metadata"]["category"] == "direct_arithmetic"
    assert spec["variables"] == {"a": 13, "b": 28, "answer": 41}
    assert spec["constraints"] == ["The assistant answer should be concise."]
    assert spec["holdout_key"] == {"op": "add", "a": 2, "b": 2}


def test_teacher_visible_sft_spec_hides_holdout_key():
    visible = teacher_visible_sft_spec(_sft_spec())

    assert visible["id"] == "sft_direct_arithmetic_000001"
    assert visible["variables"] == {"a": 13, "b": 28, "answer": 41}
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


@pytest.mark.parametrize(
    ("family", "variables", "answer"),
    [
        ("basic_arithmetic_qa", {"answer": 41}, "41"),
        ("direct_subtraction", {"answer": 41}, "41"),
        ("direct_division", {"answer": 41}, "41"),
        ("code_expression_result", {"answer": "[1, 2, 3]"}, "[1, 2, 3]"),
        ("repeat_exact_n_times", {"answer": "dog dog dog"}, "dog dog dog"),
        ("list_exact_n_items", {"answer": "red, green, blue"}, "red, green, blue"),
        ("capital_city_qa", {"capital": "Ottawa"}, "Ottawa"),
        ("clear_sky_color_qa", {"answer": "blue"}, "blue"),
        ("short_factual_stop_behavior", {"capital": "Rome"}, "Rome"),
    ],
)
def test_validate_sft_rows_against_specs_accepts_exact_answer_families(family, variables, answer):
    validate_sft_rows_against_specs(
        [_row(answer=answer, family=family)],
        [_spec(family=family, variables=variables)],
    )


def test_validate_sft_rows_against_specs_rejects_answer_only_prose():
    with pytest.raises(ValueError, match="exactly match expected answer"):
        validate_sft_rows_against_specs(
            [_row(answer="The answer is 41.")],
            [_spec(variables={"answer": 41})],
        )


def test_validate_sft_rows_against_specs_rejects_metadata_mismatch():
    with pytest.raises(ValueError, match="metadata does not match"):
        validate_sft_rows_against_specs(
            [_row()],
            [
                _spec(
                    variables={"answer": 41},
                    family="direct_subtraction",
                )
            ],
        )


def test_validate_sft_rows_against_specs_rejects_private_metric_fabrication():
    with pytest.raises(ValueError, match="must not invent private numeric metrics"):
        validate_sft_rows_against_specs(
            [_row(answer="The private revenue last month was $12,000,000.", family="private_or_unverifiable_company_fact")],
            [_spec(family="private_or_unverifiable_company_fact", variables={"company": "OpenAI"})],
        )


def test_validate_sft_rows_against_specs_accepts_private_metric_restraint():
    validate_sft_rows_against_specs(
        [_row(answer="I cannot verify that private metric from public information.", family="private_or_unverifiable_company_fact")],
        [_spec(family="private_or_unverifiable_company_fact", variables={"company": "OpenAI"})],
    )


def test_validate_sft_rows_against_specs_rejects_code_fences():
    with pytest.raises(ValueError, match="Markdown fences"):
        validate_sft_rows_against_specs(
            [_row(answer="```python\ndef add_numbers(a, b):\n    return a + b\n```", family="code_generation_function")],
            [_spec(family="code_generation_function", variables={"function_name": "add_numbers"})],
        )


def test_validate_sft_rows_against_specs_accepts_code_only_function():
    validate_sft_rows_against_specs(
        [_row(answer="def add_numbers(a, b):\n    return a + b", family="code_generation_function")],
        [_spec(family="code_generation_function", variables={"function_name": "add_numbers"})],
    )


def test_validate_sft_rows_against_specs_rejects_function_body_signature():
    with pytest.raises(ValueError, match="must not repeat a function signature"):
        validate_sft_rows_against_specs(
            [_row(answer="def add_numbers(a, b):\n    return a + b", family="function_completion_body_only")],
            [_spec(family="function_completion_body_only", variables={"function_name": "add_numbers"})],
        )


def test_validate_sft_rows_against_specs_accepts_code_explanation_plain_text():
    validate_sft_rows_against_specs(
        [_row(answer="The multiplication happens first, so the result is 14.", family="code_explanation_no_code")],
        [_spec(family="code_explanation_no_code", variables={"snippet": "result = 2 + 3 * 4", "expected_result": "14"})],
    )


def test_validate_sft_rows_against_specs_rejects_ai_concept_wrong_topic():
    with pytest.raises(ValueError, match="must mention the target concept"):
        validate_sft_rows_against_specs(
            [_row(answer="A gear transfers motion between machine parts.", family="ai_concept_explanation")],
            [_spec(family="ai_concept_explanation", variables={"concept": "embedding"})],
        )


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
