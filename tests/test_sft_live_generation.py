import json

import pytest

from slm_synth.sft.generation import generate_llm_batch, generate_teacher_batch_response


class FakeBackend:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        return {"data": self.data, "telemetry": {"usage": {"total_tokens": 12}}}


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
        "holdout_key": {"op": "add", "a": 13, "b": 28},
    }


def _teacher_data():
    return {
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
    }


def test_generate_teacher_batch_response_sends_specs_to_backend():
    backend = FakeBackend(_teacher_data())

    response = generate_teacher_batch_response(specs=[_sft_spec()], backend=backend)

    assert response == _teacher_data()
    call = backend.calls[0]
    assert call["schema_name"] == "sft_batch"
    assert call["schema"]["required"] == ["items"]
    assert "sft_direct_arithmetic_000001" in call["prompt"]
    assert '"variables"' in call["prompt"]
    assert '"holdout_key":' not in call["prompt"]


def test_generate_teacher_batch_response_rejects_non_object_data():
    backend = FakeBackend([])

    with pytest.raises(ValueError, match="SFT teacher backend returned non-object data"):
        generate_teacher_batch_response(specs=[_sft_spec()], backend=backend)


def test_generate_llm_batch_writes_dataset_and_manifest(tmp_path):
    backend = FakeBackend(_teacher_data())

    result = generate_llm_batch(
        specs=[_sft_spec()],
        output_path=tmp_path / "sft.jsonl",
        manifest_path=tmp_path / "sft.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-live-smoke-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 1
    row = json.loads((tmp_path / "sft.jsonl").read_text().strip())
    assert row["messages"][1]["content"] == "41"

    manifest = json.loads((tmp_path / "sft.manifest.json").read_text())
    assert manifest["metadata"]["generation_mode"] == "live_llm_batch"
    assert manifest["metadata"]["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["spec_count"] == 1


def test_generate_llm_batch_rejects_non_openrouter_provider(tmp_path):
    backend = FakeBackend(_teacher_data())

    with pytest.raises(ValueError, match="Unsupported teacher_provider"):
        generate_llm_batch(
            specs=[_sft_spec()],
            output_path=tmp_path / "sft.jsonl",
            manifest_path=tmp_path / "sft.manifest.json",
            teacher_model="some/model",
            teacher_provider="groq",
            generation_run="sft-live-smoke-001",
            max_tokens=1024,
            backend=backend,
        )
