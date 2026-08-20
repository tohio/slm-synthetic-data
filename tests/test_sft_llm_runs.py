import json

import pytest

from slm_synth.sft.runs import generate_llm_run, resolve_spec_families
from slm_synth.sft.spec_builders import unique_capacity
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend


class FakeSFTBackend:
    def __init__(self): self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append(len(specs))
        return {"data": {"items": [self._item(spec) for spec in specs]}, "telemetry": {"usage": {"total_tokens": 12}}}

    def _item(self, spec):
        return {
            "id": spec["id"],
            "messages": [
                {"role": "user", "content": f"Answer this generated item: {spec['id']}"},
                {"role": "assistant", "content": str(spec.get("variables", {}).get("answer", "Correct."))},
            ],
            "metadata": spec["metadata"],
        }


class SplitOnLargeSFTBackend(FakeSFTBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append(len(specs))
        if len(specs) > 1: raise ValueError("batch too large")
        return {"data": {"items": [self._item(spec) for spec in specs]}}


class DuplicatePromptSFTBackend(FakeSFTBackend):
    def _item(self, spec):
        item = super()._item(spec)
        item["messages"][0]["content"] = "Answer the repeated prompt."
        return item


class RejectSecondSFTBackend(FakeSFTBackend):
    def _item(self, spec):
        item = super()._item(spec)
        if int(spec["id"].rsplit("_", 1)[1]) == 2:
            item["metadata"] = {**item["metadata"], "output_mode": "concise"}
        return item


class RejectAllSFTBackend(FakeSFTBackend):
    def _item(self, spec):
        item = super()._item(spec)
        item["metadata"] = {**item["metadata"], "task_family": "summarization"}
        return item


class SplitFirstFamilySFTBackend(FakeSFTBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        family = specs[0]["metadata"]["task_family"]
        self.calls.append((family, len(specs)))
        if family == "grounded_qa_and_reading" and len(specs) > 1:
            raise ValueError("split the first family")
        return {"data": {"items": [self._item(spec) for spec in specs]}}


def generate(tmp_path, backend, **planning):
    return generate_llm_run(
        families=planning.pop("families", ["grounded_qa_and_reading"]),
        batch_size=planning.pop("batch_size", 2),
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run=planning.pop("generation_run", "sft-run-001"),
        max_tokens=1024,
        backend=backend,
        adjudicator_backend=AcceptingAdjudicatorBackend(),
        **planning,
    )


def test_generate_sft_llm_run_writes_candidate_manifest(tmp_path):
    result = generate(tmp_path, FakeSFTBackend(), candidate_counts_by_family={"grounded_qa_and_reading": 3})
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 3
    assert manifest["metadata"]["candidate_rows"] == 3
    assert manifest["metadata"]["accepted_rows"] == 3
    assert manifest["metadata"]["generation_status"] == "complete"
    assert manifest["metadata"]["publish_ready"] is True


def test_generate_sft_llm_run_supports_explicit_candidate_counts(tmp_path):
    result = generate(
        tmp_path,
        FakeSFTBackend(),
        families=["grounded_qa_and_reading", "rewriting_and_editing"],
        candidate_counts_by_family={"grounded_qa_and_reading": 2, "rewriting_and_editing": 1},
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 3
    assert manifest["metadata"]["planning_mode"] == "candidate_counts_by_family"
    assert manifest["metadata"]["candidate_rows_per_family"] == {"grounded_qa_and_reading": 2, "rewriting_and_editing": 1}


def test_generate_sft_llm_run_does_not_replace_duplicate_candidates(tmp_path):
    result = generate(tmp_path, DuplicatePromptSFTBackend(), candidate_counts_by_family={"grounded_qa_and_reading": 2})
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 1
    assert manifest["metadata"]["candidate_rows"] == 2
    assert manifest["metadata"]["attempted_rows"] == 2
    assert manifest["metadata"]["accepted_rows"] == 1
    assert manifest["metadata"]["duplicate_rows"] == 1


def test_generate_sft_llm_run_does_not_replace_rejected_candidates(tmp_path):
    result = generate(tmp_path, RejectSecondSFTBackend(), candidate_counts_by_family={"grounded_qa_and_reading": 2})
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 1
    assert manifest["metadata"]["attempted_rows"] == 2
    assert manifest["metadata"]["accepted_rows"] == 1
    assert manifest["metadata"]["rejected_rows"] == 1
    assert manifest["metadata"]["rejection_reason_counts"] == {
        "render_validation_error": 1
    }
    assert manifest["metadata"]["rejection_diagnostics"][0]["id"].endswith("000002")


def test_generate_sft_llm_run_blocks_an_empty_requested_family(tmp_path):
    result = generate(
        tmp_path,
        RejectAllSFTBackend(),
        candidate_counts_by_family={"grounded_qa_and_reading": 1},
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 0
    assert manifest["metadata"]["publish_ready"] is False
    assert manifest["metadata"]["empty_families"] == ["grounded_qa_and_reading"]


def test_generate_sft_llm_run_resets_adaptive_batch_size_per_family(tmp_path):
    backend = SplitFirstFamilySFTBackend()
    result = generate(
        tmp_path,
        backend,
        families=["grounded_qa_and_reading", "rewriting_and_editing"],
        candidate_counts_by_family={
            "grounded_qa_and_reading": 2,
            "rewriting_and_editing": 2,
        },
        batch_size=2,
    )
    assert result.row_count == 4
    assert ("rewriting_and_editing", 2) in backend.calls


def test_generate_sft_llm_run_reduces_batch_size_after_failure(tmp_path):
    backend = SplitOnLargeSFTBackend()
    result = generate(tmp_path, backend, candidate_counts_by_family={"grounded_qa_and_reading": 3}, batch_size=3)
    assert result.row_count == 3
    assert backend.calls == [3, 1, 1, 1]


def test_generate_sft_llm_run_requires_exact_candidate_mapping(tmp_path):
    with pytest.raises(ValueError, match="exactly the requested"):
        generate(tmp_path, FakeSFTBackend(), families=["grounded_qa_and_reading", "summarization"], candidate_counts_by_family={"grounded_qa_and_reading": 1})


def test_generate_sft_llm_run_checks_capacity_before_backend_construction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("slm_synth.sft.runs.build_openrouter_backend", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="finite source capacity"):
        generate_llm_run(
            families=["summarization"], candidate_counts_by_family={"summarization": 2}, batch_size=1,
            output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
            teacher_model="teacher/model", generation_run="too-large", max_tokens=100,
            start_index=unique_capacity("summarization"), concurrency=1,
        )
    assert calls == []


def test_resolve_sft_spec_families_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate SFT spec family"):
        resolve_spec_families(["grounded_qa_and_reading", "grounded_qa_and_reading"])
