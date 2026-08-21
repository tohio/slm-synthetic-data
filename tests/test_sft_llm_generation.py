import json

import pytest

from slm_synth.sft.generation import SFTBatchAcceptanceError, generate_llm_batch, materialize_llm_batch
from slm_synth.sft.spec_builders import build_specs
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend


class Backend:
    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        return {"text": json.dumps({"items": [{
            "messages": [
                {"role": "user", "content": "Please complete this task."},
                {"role": "assistant", "content": "A correct, grounded response."},
            ],
        } for spec in specs]}), "telemetry": {"usage": {"total_tokens": 12}}}


def test_generate_sft_llm_batch_writes_rows(tmp_path):
    specs = build_specs(family="grounded_qa_and_reading", count=2)
    result = generate_llm_batch(
        specs=specs, output_path=tmp_path / "sft.jsonl", manifest_path=tmp_path / "manifest.json",
        teacher_model="teacher/model", generation_run="sft-001", max_tokens=1024, backend=Backend(),
        adjudicator_backend=AcceptingAdjudicatorBackend(),
    )
    assert result.row_count == 2


def test_materialize_sft_batch_round_trips(tmp_path):
    spec = build_specs(family="rewriting_and_editing", count=1)[0]
    generated = json.loads(Backend().generate_text_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [spec]}), system_prompt=""
    )["text"])
    response = {"items": [{"id": spec["id"], **generated["items"][0], "metadata": spec["metadata"]}]}
    result = materialize_llm_batch(
        specs=[spec], teacher_response=response, output_path=tmp_path / "sft.jsonl",
        manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model", generation_run="sft-001",
    )
    assert result.row_count == 1


def test_sft_batch_rejects_model_owned_metadata(tmp_path):
    spec = build_specs(family="rewriting_and_editing", count=1)[0]
    response = json.loads(Backend().generate_text_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [spec]}), system_prompt=""
    )["text"])
    response["items"][0]["metadata"] = {"task_family": "summarization"}
    class DriftBackend:
        def generate_text_with_metadata(self, **kwargs): return {"text": json.dumps(response), "telemetry": {}}
    with pytest.raises(SFTBatchAcceptanceError, match="only messages") as error:
        generate_llm_batch(
            specs=[spec], output_path=tmp_path / "sft.jsonl", manifest_path=tmp_path / "manifest.json",
            teacher_model="teacher/model", generation_run="sft-001", max_tokens=1024,
            backend=DriftBackend(),
            adjudicator_backend=AcceptingAdjudicatorBackend(),
        )
    assert error.value.failure_type == "renderer_response_error"


def test_sft_batch_rejects_failed_deterministic_constraint_before_adjudication(tmp_path):
    spec = build_specs(family="creative_writing", count=1)[0]
    with pytest.raises(SFTBatchAcceptanceError, match="min_words expected=500") as error:
        generate_llm_batch(
            specs=[spec], output_path=tmp_path / "sft.jsonl",
            manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model",
            generation_run="sft-001", max_tokens=1024, backend=Backend(),
            adjudicator_backend=AcceptingAdjudicatorBackend(),
        )
    assert error.value.failure_type == "deterministic_constraint_error"
