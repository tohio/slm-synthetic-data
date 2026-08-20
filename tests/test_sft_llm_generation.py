import json

import pytest

from slm_synth.sft.generation import SFTBatchAcceptanceError, generate_llm_batch, materialize_llm_batch_from_files
from slm_synth.sft.spec_builders import build_specs
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend


class Backend:
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        return {"data": {"items": [{
            "id": spec["id"],
            "messages": [
                {"role": "user", "content": "Please complete this task."},
                {"role": "assistant", "content": "A correct, grounded response."},
            ],
            "metadata": spec["metadata"],
        } for spec in specs]}, "telemetry": {"usage": {"total_tokens": 12}}}


def test_generate_sft_llm_batch_writes_rows(tmp_path):
    specs = build_specs(family="grounded_qa_and_reading", count=2)
    result = generate_llm_batch(
        specs=specs, output_path=tmp_path / "sft.jsonl", manifest_path=tmp_path / "manifest.json",
        teacher_model="teacher/model", generation_run="sft-001", max_tokens=1024, backend=Backend(),
        adjudicator_backend=AcceptingAdjudicatorBackend(),
    )
    assert result.row_count == 2


def test_materialize_sft_from_files_round_trips(tmp_path):
    spec = build_specs(family="rewriting_and_editing", count=1)[0]
    response = Backend().generate_structured_object_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [spec]}), schema={}, schema_name="sft"
    )["data"]
    specs_path, response_path = tmp_path / "specs.jsonl", tmp_path / "response.json"
    specs_path.write_text(json.dumps(spec) + "\n")
    response_path.write_text(json.dumps(response))
    result = materialize_llm_batch_from_files(
        specs_path=specs_path, teacher_response_path=response_path, output_path=tmp_path / "sft.jsonl",
        manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model", generation_run="sft-001",
    )
    assert result.row_count == 1


def test_sft_batch_rejects_metadata_drift(tmp_path):
    spec = build_specs(family="rewriting_and_editing", count=1)[0]
    response = Backend().generate_structured_object_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [spec]}), schema={}, schema_name="sft"
    )["data"]
    response["items"][0]["metadata"]["task_family"] = "summarization"
    with pytest.raises(SFTBatchAcceptanceError, match="metadata does not match"):
        generate_llm_batch(
            specs=[spec], output_path=tmp_path / "sft.jsonl", manifest_path=tmp_path / "manifest.json",
            teacher_model="teacher/model", generation_run="sft-001", max_tokens=1024,
            backend=type("DriftBackend", (), {"generate_structured_object_with_metadata": lambda self, **kwargs: {"data": response, "telemetry": {}}})(),
            adjudicator_backend=AcceptingAdjudicatorBackend(),
        )
