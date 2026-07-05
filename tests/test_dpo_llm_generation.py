import json

import pytest

from slm_synth.dpo.generation import (
    materialize_llm_batch,
    materialize_llm_batch_from_files,
    read_specs_jsonl,
)
from slm_synth.taxonomy.holdouts import HoldoutRegistry


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
        "variables": {"a": 17, "b": 26, "answer": 43, "rejected_answer": "44"},
        "holdout_key": {"op": "add", "a": 17, "b": 26},
    }


def _teacher_response(row_id: str = "dpo_answer_only_arithmetic_000001"):
    return {
        "items": [
            {
                "id": row_id,
                "prompt": [{"role": "user", "content": "Answer with only the number: What is 17 + 26?"}],
                "chosen": [{"role": "assistant", "content": "43"}],
                "rejected": [
                    {"role": "assistant", "content": "The answer is 43 because 17 plus 26 equals 43."}
                ],
                "metadata": {
                    "category": "answer_only_compliance",
                    "difficulty": 1,
                    "template_family": "direct_qa",
                    "eval_family": "basic_arithmetic_qa",
                    "failure_mode": "extra_explanation",
                },
            }
        ]
    }


def test_materialize_llm_batch_writes_dpo_dataset_and_manifest(tmp_path):
    result = materialize_llm_batch(
        specs=[_dpo_spec()],
        teacher_response=_teacher_response(),
        output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "dpo.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-llm-smoke-001",
    )

    assert result.row_count == 1
    row = json.loads((tmp_path / "dpo.jsonl").read_text().strip())
    assert set(row) == {"id", "prompt", "chosen", "rejected", "metadata"}
    assert row["chosen"][0]["content"] == "43"
    assert row["metadata"]["failure_mode"] == "extra_explanation"

    manifest = json.loads((tmp_path / "dpo.manifest.json").read_text())
    assert manifest["dataset_type"] == "dpo"
    assert manifest["row_count"] == 1
    assert manifest["metadata"]["generation_mode"] == "llm_batch"
    assert manifest["metadata"]["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["teacher_model"] == "openai/gpt-4.1-mini"


def test_materialize_llm_batch_rejects_non_openrouter_provider(tmp_path):
    with pytest.raises(ValueError, match="Unsupported teacher_provider"):
        materialize_llm_batch(
            specs=[_dpo_spec()],
            teacher_response=_teacher_response(),
            output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "dpo.manifest.json",
            teacher_model="some/model",
            teacher_provider="groq",
            generation_run="dpo-llm-smoke-001",
        )


def test_materialize_llm_batch_rejects_teacher_id_mismatch(tmp_path):
    with pytest.raises(ValueError, match="missing expected id"):
        materialize_llm_batch(
            specs=[_dpo_spec()],
            teacher_response=_teacher_response(row_id="dpo_other_000001"),
            output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "dpo.manifest.json",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-llm-smoke-001",
        )


def test_materialize_llm_batch_repairs_copied_rejected_answer(tmp_path):
    response = _teacher_response()
    response["items"][0]["rejected"] = response["items"][0]["chosen"]

    result = materialize_llm_batch(
        specs=[_dpo_spec()],
        teacher_response=response,
        output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "dpo.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-llm-smoke-001",
    )

    assert result.row_count == 1
    row = json.loads((tmp_path / "dpo.jsonl").read_text().strip())
    assert row["chosen"][0]["content"] == "43"
    assert row["rejected"][0]["content"] == "44"


def test_materialize_llm_batch_rejects_copied_rejected_without_repair_target(tmp_path):
    spec = _dpo_spec()
    del spec["variables"]["rejected_answer"]
    response = _teacher_response()
    response["items"][0]["rejected"] = response["items"][0]["chosen"]

    with pytest.raises(ValueError, match="chosen and rejected"):
        materialize_llm_batch(
            specs=[spec],
            teacher_response=response,
            output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "dpo.manifest.json",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-llm-smoke-001",
        )


def test_materialize_llm_batch_rejects_holdout_prompt_match(tmp_path):
    registry = HoldoutRegistry.from_mapping(
        {
            "basic_arithmetic_qa": [
                {
                    "id": "holdout_add_17_26",
                    "prompt": "Answer with only the number: What is 17 + 26?",
                    "answer": "43",
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="eval holdout prompt"):
        materialize_llm_batch(
            specs=[_dpo_spec()],
            teacher_response=_teacher_response(),
            output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "dpo.manifest.json",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-llm-smoke-001",
            holdout_registry=registry,
        )


def test_materialize_llm_batch_from_files(tmp_path):
    specs_path = tmp_path / "specs.jsonl"
    response_path = tmp_path / "response.json"
    specs_path.write_text(json.dumps(_dpo_spec()) + "\n")
    response_path.write_text(json.dumps(_teacher_response()))

    result = materialize_llm_batch_from_files(
        specs_path=specs_path,
        teacher_response_path=response_path,
        output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "dpo.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-llm-smoke-001",
    )

    assert result.row_count == 1
    assert read_specs_jsonl(specs_path)[0]["id"] == "dpo_answer_only_arithmetic_000001"
