import json

import pytest

from slm_synth.distillation_sft.generation import (
    generate_and_materialize_signal_batch,
    generate_teacher_batch_response,
)
from slm_synth.distillation_sft.seeds import build_seed_prompt_records


class FakeBackend:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        return {"data": self.data, "telemetry": {"usage": {"total_tokens": 12}}}


def test_generate_teacher_batch_response_sends_only_prompt_items():
    records = build_seed_prompt_records(signal="arithmetic", count=1)
    backend = FakeBackend({"items": [{"id": "arithmetic-000001", "reasoning": None, "response": "4"}]})

    response = generate_teacher_batch_response(signal="arithmetic", prompt_records=records, backend=backend)

    assert response == {
        "items": [{"id": "arithmetic-000001", "reasoning": None, "response": "4"}]
    }
    rendered = backend.calls[0]["prompt"]
    assert '"id": "arithmetic-000001"' in rendered
    assert '"prompt"' in rendered
    assert '"metadata"' not in rendered
    assert '"teacher_model"' not in rendered
    assert backend.calls[0]["schema_name"] == "arithmetic_distillation_batch"


def test_generate_teacher_batch_response_rejects_non_object_data():
    records = build_seed_prompt_records(signal="instruction", count=1)
    backend = FakeBackend([])

    with pytest.raises(ValueError, match="teacher backend returned non-object data"):
        generate_teacher_batch_response(signal="instruction", prompt_records=records, backend=backend)


def test_generate_and_materialize_signal_batch_writes_public_dataset_and_manifest(tmp_path):
    records = build_seed_prompt_records(signal="planning", count=1)
    backend = FakeBackend(
        {
            "items": [
                {
                    "id": "planning-000001",
                    "reasoning": None,
                    "response": "Start by identifying dependencies, then order tasks by risk and unblockers.",
                }
            ]
        }
    )

    result = generate_and_materialize_signal_batch(
        signal="planning",
        prompt_records=records,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="smoke-001",
        max_tokens=512,
        token_target="100K",
        backend=backend,
    )

    assert result.row_count == 1
    public_row = json.loads(result.dataset_path.read_text(encoding="utf-8").strip())
    assert public_row == {
        "id": "planning-000001",
        "prompt": "Create a concise plan for preparing a small dataset for model fine-tuning.",
        "reasoning": None,
        "response": "Start by identifying dependencies, then order tasks by risk and unblockers.",
        "metadata": {
            "category": "general_instruction_following",
            "difficulty": 2,
            "template_family": "operational_planning_checklist",
            "eval_family": None,
        },
    }
    assert "signal" not in public_row

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert manifest["metadata"]["prompt_count"] == 1
