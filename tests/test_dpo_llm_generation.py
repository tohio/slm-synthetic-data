import json

import pytest

from slm_synth.dpo.generation import (
    DPOBatchAcceptanceError, generate_llm_batch, materialize_llm_batch,
)
from slm_synth.dpo.spec_builders import build_specs
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend, StagedDPOBackend


class Backend:
    def __init__(self):
        self.ids = []

    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.ids = [spec["id"] for spec in specs]
        return {
            "text": json.dumps({"items": [self._row(spec) for spec in specs]}),
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
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": "A correct and complete response."}],
            "rejected": [{"role": "assistant", "content": "A plausible but incomplete response."}],
        }
        if "tool_mediated" in modes:
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


def test_generate_llm_batch_writes_teacher_generated_pairs(tmp_path):
    specs = build_specs(family="code_correctness", count=2)
    result = generate_llm_batch(
        specs=specs, output_path=tmp_path / "dpo.jsonl", manifest_path=tmp_path / "dpo.manifest.json",
        teacher_model="teacher/model", generation_run="dpo-001", max_tokens=1024,
        backend=StagedDPOBackend(Backend()), adjudicator_backend=AcceptingAdjudicatorBackend(),
    )
    assert result.row_count == 2
    assert len((tmp_path / "dpo.jsonl").read_text().splitlines()) == 2


def test_materialize_rejects_id_mismatch(tmp_path):
    spec = build_specs(family="style_and_tone", count=1)[0]
    generated = json.loads(Backend().generate_text_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [dict(spec, id="wrong")]}), system_prompt=""
    )["text"])
    response = {"items": [{"id": "wrong", **generated["items"][0], "metadata": spec["metadata"]}]}
    with pytest.raises(DPOBatchAcceptanceError, match="id mismatch"):
        materialize_llm_batch(
            specs=[spec], teacher_response=response, output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model", generation_run="dpo-001",
        )


def test_materialize_batch_round_trips(tmp_path):
    spec = build_specs(family="groundedness", count=1)[0]
    backend = Backend()
    generated = json.loads(backend.generate_text_with_metadata(
        prompt="Input specs:\n" + json.dumps({"items": [spec]}), system_prompt=""
    )["text"])
    response = {"items": [{"id": spec["id"], **generated["items"][0], "metadata": spec["metadata"]}]}
    result = materialize_llm_batch(
        specs=[spec], teacher_response=response, output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "manifest.json", teacher_model="teacher/model", generation_run="dpo-001",
    )
    assert result.row_count == 1
