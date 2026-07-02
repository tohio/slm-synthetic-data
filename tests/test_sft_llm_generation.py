import json

import pytest

from slm_synth.sft.generation import (
    materialize_llm_batch,
    materialize_llm_batch_from_files,
    read_specs_jsonl,
)
from slm_synth.taxonomy.holdouts import HoldoutRegistry


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
        "holdout_key": {"op": "add", "a": 13, "b": 28},
    }


def _teacher_response(row_id: str = "sft_direct_arithmetic_000001"):
    return {
        "items": [
            {
                "id": row_id,
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
    }


def test_materialize_llm_batch_writes_sft_dataset_and_manifest(tmp_path):
    result = materialize_llm_batch(
        specs=[_sft_spec()],
        teacher_response=_teacher_response(),
        output_path=tmp_path / "sft.jsonl",
        manifest_path=tmp_path / "sft.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-llm-smoke-001",
    )

    assert result.row_count == 1
    row = json.loads((tmp_path / "sft.jsonl").read_text().strip())
    assert set(row) == {"id", "messages", "metadata"}
    assert row["messages"][1]["content"] == "41"

    manifest = json.loads((tmp_path / "sft.manifest.json").read_text())
    assert manifest["dataset_type"] == "sft"
    assert manifest["row_count"] == 1
    assert manifest["metadata"]["generation_mode"] == "llm_batch"
    assert manifest["metadata"]["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["teacher_model"] == "openai/gpt-4.1-mini"


def test_materialize_llm_batch_rejects_non_openrouter_provider(tmp_path):
    with pytest.raises(ValueError, match="Unsupported teacher_provider"):
        materialize_llm_batch(
            specs=[_sft_spec()],
            teacher_response=_teacher_response(),
            output_path=tmp_path / "sft.jsonl",
            manifest_path=tmp_path / "sft.manifest.json",
            teacher_model="some/model",
            teacher_provider="groq",
            generation_run="sft-llm-smoke-001",
        )


def test_materialize_llm_batch_rejects_teacher_id_mismatch(tmp_path):
    with pytest.raises(ValueError, match="missing expected id"):
        materialize_llm_batch(
            specs=[_sft_spec()],
            teacher_response=_teacher_response(row_id="sft_other_000001"),
            output_path=tmp_path / "sft.jsonl",
            manifest_path=tmp_path / "sft.manifest.json",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="sft-llm-smoke-001",
        )


def test_materialize_llm_batch_rejects_holdout_prompt_match(tmp_path):
    registry = HoldoutRegistry.from_mapping(
        {
            "basic_arithmetic_qa": [
                {
                    "id": "holdout_add_13_28",
                    "prompt": "What is 13 + 28?",
                    "answer": "41",
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="eval holdout prompt"):
        materialize_llm_batch(
            specs=[_sft_spec()],
            teacher_response=_teacher_response(),
            output_path=tmp_path / "sft.jsonl",
            manifest_path=tmp_path / "sft.manifest.json",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="sft-llm-smoke-001",
            holdout_registry=registry,
        )


def test_materialize_llm_batch_from_files(tmp_path):
    specs_path = tmp_path / "specs.jsonl"
    response_path = tmp_path / "response.json"
    specs_path.write_text(json.dumps(_sft_spec()) + "\n")
    response_path.write_text(json.dumps(_teacher_response()))

    result = materialize_llm_batch_from_files(
        specs_path=specs_path,
        teacher_response_path=response_path,
        output_path=tmp_path / "sft.jsonl",
        manifest_path=tmp_path / "sft.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-llm-smoke-001",
    )

    assert result.row_count == 1
    assert read_specs_jsonl(specs_path)[0]["id"] == "sft_direct_arithmetic_000001"
