import json

import pytest

from slm_synth.dpo.generation import (
    DPOBatchAcceptanceError, generate_llm_batch, generate_teacher_batch_response_with_metadata,
    materialize_llm_batch, materialize_llm_batch_from_files,
)
from slm_synth.dpo.spec_builders import build_specs


class Backend:
    def __init__(self):
        self.ids = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.ids = [spec["id"] for spec in specs]
        return {
            "data": {"items": [self._row(spec) for spec in specs]},
            "telemetry": {"usage": {"total_tokens": 10}},
        }

    @staticmethod
    def _row(spec):
        modes = spec["metadata"]["interaction_modes"]
        prompt = [{"role": "user", "content": spec["instruction"]}]
        if "system_conditioned" in modes:
            prompt.insert(0, {"role": "system", "content": "Follow the requested constraints."})
        if "multi_turn" in modes:
            prompt.extend([
                {"role": "assistant", "content": "What constraint matters most?"},
                {"role": "user", "content": "Accuracy matters most."},
            ])
        row = {
            "id": spec["id"], "prompt": prompt,
            "chosen": [{"role": "assistant", "content": "A correct and complete response."}],
            "rejected": [{"role": "assistant", "content": "A plausible but incomplete response."}],
            "metadata": spec["metadata"],
        }
        if "tool_mediated" in modes:
            row["tools"] = [{"type": "function", "function": {
                "name": "lookup", "description": "Look up an approved record.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }}]
            row["chosen"] = [
                {"role": "assistant", "content": None, "tool_calls": [{"id": "chosen_call", "type": "function", "function": {"name": "lookup", "arguments": {"query": "correct"}}}]},
                {"role": "tool", "tool_call_id": "chosen_call", "content": "verified result"},
                {"role": "assistant", "content": "A correct and complete response."},
            ]
            row["rejected"] = [
                {"role": "assistant", "content": None, "tool_calls": [{"id": "rejected_call", "type": "function", "function": {"name": "lookup", "arguments": {"query": "wrong"}}}]},
                {"role": "tool", "tool_call_id": "rejected_call", "content": "irrelevant result"},
                {"role": "assistant", "content": "A plausible but incorrect response."},
            ]
        return row


def test_all_generic_dpo_specs_use_teacher_generation():
    specs = [
        build_specs(family="factual_accuracy", count=1)[0],
        build_specs(family="helpfulness_and_completeness", count=1)[0],
    ]
    backend = Backend()
    response, telemetry = generate_teacher_batch_response_with_metadata(specs=specs, backend=backend)
    assert backend.ids == [spec["id"] for spec in specs]
    assert len(response["items"]) == 2
    assert telemetry["usage"]["total_tokens"] == 10


def test_generate_llm_batch_writes_teacher_generated_pairs(tmp_path):
    specs = build_specs(family="code_correctness", count=2)
    result = generate_llm_batch(
        specs=specs, output_path=tmp_path / "dpo.jsonl", manifest_path=tmp_path / "dpo.manifest.json",
        teacher_model="teacher/model", generation_run="dpo-001", max_tokens=1024, backend=Backend(),
    )
    assert result.row_count == 2
    assert len((tmp_path / "dpo.jsonl").read_text().splitlines()) == 2


def test_materialize_rejects_id_mismatch(tmp_path):
    spec = build_specs(family="style_and_tone", count=1)[0]
    response = Backend().generate_structured_object_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [dict(spec, id="wrong")]}), schema={}, schema_name="dpo"
    )["data"]
    with pytest.raises(DPOBatchAcceptanceError, match="id mismatch"):
        materialize_llm_batch(
            specs=[spec], teacher_response=response, output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model", generation_run="dpo-001",
        )


def test_materialize_from_files_round_trips(tmp_path):
    spec = build_specs(family="groundedness", count=1)[0]
    backend = Backend()
    response = backend.generate_structured_object_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [spec]}), schema={}, schema_name="dpo"
    )["data"]
    specs_path = tmp_path / "specs.jsonl"
    response_path = tmp_path / "response.json"
    specs_path.write_text(json.dumps(spec) + "\n")
    response_path.write_text(json.dumps(response))
    result = materialize_llm_batch_from_files(
        specs_path=specs_path, teacher_response_path=response_path, output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model", generation_run="dpo-001",
    )
    assert result.row_count == 1
