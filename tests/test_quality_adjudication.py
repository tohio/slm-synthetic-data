import copy
import json

import pytest

from slm_synth.dpo.adjudication import DPO_ADJUDICATION_SCHEMA, adjudicate_dpo_rows
from slm_synth.dpo.generation import DPOBatchAcceptanceError, generate_llm_batch as generate_dpo_batch
from slm_synth.dpo.spec_builders import build_specs as build_dpo_specs
from slm_synth.quality_adjudication import (
    QUALITY_ADJUDICATION_SCHEMA, render_quality_adjudication_prompt,
    validate_quality_adjudication,
)
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend, StagedDPOBackend
from tests.test_dpo_llm_generation import Backend as CompleteDPOBackend


def test_quality_adjudication_rejects_a_failed_source_constraint():
    spec = {
        "id": "sft_example",
        "constraints": ["Use only supplied facts."],
    }
    response = {"items": [{
        "id": "sft_example",
        "accepted": True,
        "scores": {
            "correctness": 4, "grounding": 4, "instruction_adherence": 4,
            "completeness": 4, "coherence": 4,
        },
        "constraint_results": [{
            "constraint_index": 0, "passed": False, "reason": "Invented a date."
        }],
        "reasons": ["Invented a date."],
    }]}
    with pytest.raises(ValueError, match="semantic quality adjudication rejected"):
        validate_quality_adjudication(response, specs=[spec])


def test_quality_adjudication_references_constraints_by_stable_index():
    spec = {"id": "sft_example", "constraints": ["Use only supplied facts."]}
    prompt = render_quality_adjudication_prompt(dataset_type="SFT", specs=[spec], rows=[{"id": "sft_example"}])
    result_schema = QUALITY_ADJUDICATION_SCHEMA["properties"]["items"]["items"]["properties"][
        "constraint_results"
    ]["items"]

    assert "constraint_index" in result_schema["required"]
    assert "constraint" not in result_schema["properties"]
    assert "Do not copy or paraphrase the constraint text" in prompt


def test_quality_adjudication_rejects_missing_constraint_index_coverage():
    spec = {"id": "sft_example", "constraints": ["First.", "Second."]}
    response = {"items": [{
        "id": "sft_example",
        "accepted": True,
        "scores": {
            "correctness": 4, "grounding": 4, "instruction_adherence": 4,
            "completeness": 4, "coherence": 4,
        },
        "constraint_results": [
            {"constraint_index": 0, "passed": True, "reason": "satisfied"},
            {"constraint_index": 0, "passed": True, "reason": "satisfied"},
        ],
        "reasons": [],
    }]}
    with pytest.raises(ValueError, match="constraint index coverage"):
        validate_quality_adjudication(response, specs=[spec])


def test_dpo_adjudication_uses_and_validates_stable_constraint_indexes():
    result_schema = DPO_ADJUDICATION_SCHEMA["properties"]["items"]["items"]["properties"][
        "constraint_results"
    ]["items"]
    assert "constraint_index" in result_schema["required"]
    assert "constraint" not in result_schema["properties"]

    spec = build_dpo_specs(family="factual_accuracy", count=1)[0]

    class DuplicateIndexBackend:
        def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
            assert "Do not copy or paraphrase the constraint text" in prompt
            return {"data": {"items": [{
                "id": spec["id"],
                "accepted": True,
                "preference_dimension": spec["metadata"]["preference_dimension"],
                "failure_mode": spec["metadata"]["failure_mode"],
                "observed_weakness": spec["metadata"]["failure_mode"],
                "scores": {
                    "chosen_quality": 4,
                    "rejected_plausibility": 4,
                    "weakness_match": 4,
                    "preference_separation": 4,
                    "collateral_preservation": 4,
                },
                "constraint_results": [
                    {"constraint_index": 0, "passed": True, "reason": "satisfied"},
                    {"constraint_index": 0, "passed": True, "reason": "satisfied"},
                ],
                "reasons": [],
            }]}}

    with pytest.raises(ValueError, match="constraint index coverage"):
        adjudicate_dpo_rows(specs=[spec], rows=[{"id": spec["id"]}], backend=DuplicateIndexBackend())


def test_dpo_live_generation_uses_chosen_rejected_adjudication_order(tmp_path):
    class RecordingBackend(CompleteDPOBackend):
        def __init__(self):
            super().__init__()
            self.calls = []

        def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
            self.calls.append(schema_name)
            return super().generate_structured_object_with_metadata(
                prompt=prompt, schema=schema, schema_name=schema_name
            )

    complete = RecordingBackend()
    renderer = StagedDPOBackend(complete)
    adjudicator = AcceptingAdjudicatorBackend()
    generate_dpo_batch(
        specs=build_dpo_specs(family="factual_accuracy", count=1),
        output_path=tmp_path / "dpo.jsonl",
        manifest_path=tmp_path / "manifest.json",
        teacher_model="teacher/model",
        generation_run="staged-dpo",
        max_tokens=1024,
        backend=renderer,
        adjudicator_backend=adjudicator,
    )
    assert complete.calls == ["dpo_chosen_batch"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["metadata"]["llm_stage_telemetry"]) == {
        "chosen_renderer", "rejected_renderer", "adjudicator"
    }


def test_dpo_identical_branches_are_rejected_without_local_repair(tmp_path):
    spec = build_dpo_specs(family="factual_accuracy", count=1)[0]

    class IdenticalBackend(CompleteDPOBackend):
        def _row(self, value):
            row = super()._row(value)
            row["rejected"] = copy.deepcopy(row["chosen"])
            return row

    with pytest.raises(DPOBatchAcceptanceError, match="must differ"):
        generate_dpo_batch(
            specs=[spec],
            output_path=tmp_path / "dpo.jsonl",
            manifest_path=tmp_path / "manifest.json",
            teacher_model="teacher/model",
            generation_run="no-repair",
            max_tokens=1024,
            backend=StagedDPOBackend(IdenticalBackend()),
            adjudicator_backend=AcceptingAdjudicatorBackend(),
        )
