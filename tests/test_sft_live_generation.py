import json

import pytest

from slm_synth.sft.generation import generate_llm_batch, generate_teacher_batch_response
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend


class FakeBackend:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def generate_text_with_metadata(self, *, prompt, system_prompt):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return {"text": json.dumps(self.data), "telemetry": {"usage": {"total_tokens": 12}}}


def _sft_spec():
    return {
        "id": "sft_direct_arithmetic_000001",
        "instruction": "Create an addition question using 13 and 28. Answer concisely.",
        "metadata": {
            "task_family": "grounded_qa_and_reading",
            "interaction_modes": ["single_turn"],
            "output_mode": "free_text",
            "context_mode": "supplied_passage",
            "difficulty": 1,
            "template_family": "direct_qa",
        },
        "variables": {"a": 13, "b": 28, "answer": 41},
        "holdout_key": {"op": "add", "a": 13, "b": 28},
    }


def _teacher_data():
    return {
        "items": [
            {
                "messages": [
                    {"role": "user", "content": "What is 13 + 28?"},
                    {"role": "assistant", "content": "41"},
                ],
            }
        ]
    }


def test_generate_teacher_batch_response_sends_specs_to_backend():
    backend = FakeBackend(_teacher_data())

    response = generate_teacher_batch_response(specs=[_sft_spec()], backend=backend)

    assert response["items"][0]["messages"] == _teacher_data()["items"][0]["messages"]
    assert response["items"][0]["id"] == _sft_spec()["id"]
    assert response["items"][0]["metadata"] == _sft_spec()["metadata"]
    call = backend.calls[0]
    assert "JSON object" in call["system_prompt"]
    assert "sft_direct_arithmetic_000001" in call["prompt"]
    assert '"variables"' in call["prompt"]
    assert '"holdout_key":' not in call["prompt"]


def test_generate_teacher_batch_response_rejects_non_object_data():
    backend = FakeBackend([])

    with pytest.raises(ValueError, match="JSON object"):
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
        adjudicator_backend=AcceptingAdjudicatorBackend(),
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
            adjudicator_backend=AcceptingAdjudicatorBackend(),
        )
