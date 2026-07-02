import json

import pytest

from slm_synth.dpo.generation import generate_llm_batch, generate_teacher_batch_response


class FakeBackend:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        return {"data": self.data, "telemetry": {"usage": {"total_tokens": 12}}}


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
        "holdout_key": {"op": "add", "a": 17, "b": 26},
    }


def _teacher_data():
    return {
        "items": [
            {
                "id": "dpo_answer_only_arithmetic_000001",
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


def test_generate_teacher_batch_response_sends_specs_to_backend():
    backend = FakeBackend(_teacher_data())

    response = generate_teacher_batch_response(specs=[_dpo_spec()], backend=backend)

    assert response == _teacher_data()
    call = backend.calls[0]
    assert call["schema_name"] == "dpo_batch"
    assert call["schema"]["required"] == ["items"]
    assert "dpo_answer_only_arithmetic_000001" in call["prompt"]
    assert "failure_mode" in call["prompt"]
    assert '"holdout_key":' not in call["prompt"]


def test_generate_teacher_batch_response_rejects_non_object_data():
    backend = FakeBackend([])

    with pytest.raises(ValueError, match="DPO teacher backend returned non-object data"):
        generate_teacher_batch_response(specs=[_dpo_spec()], backend=backend)


def test_generate_llm_batch_writes_dataset_and_manifest(tmp_path):
    backend = FakeBackend(_teacher_data())

    result = generate_llm_batch(
        specs=[_dpo_spec()],
        output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "dpo.manifest.json",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-smoke-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 1
    row = json.loads((tmp_path / "dpo.jsonl").read_text().strip())
    assert row["chosen"][0]["content"] == "43"

    manifest = json.loads((tmp_path / "dpo.manifest.json").read_text())
    assert manifest["metadata"]["generation_mode"] == "live_llm_batch"
    assert manifest["metadata"]["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["spec_count"] == 1


def test_generate_llm_batch_rejects_non_openrouter_provider(tmp_path):
    backend = FakeBackend(_teacher_data())

    with pytest.raises(ValueError, match="Unsupported teacher_provider"):
        generate_llm_batch(
            specs=[_dpo_spec()],
            output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "dpo.manifest.json",
            teacher_model="some/model",
            teacher_provider="groq",
            generation_run="dpo-live-smoke-001",
            max_tokens=1024,
            backend=backend,
        )
