import json

import pytest

from slm_synth.sft.runs import generate_llm_run, resolve_spec_families
from slm_synth.sft.spec_builders import unique_capacity


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
        **planning,
    )


def test_generate_sft_llm_run_writes_candidate_manifest(tmp_path):
    result = generate(tmp_path, FakeSFTBackend(), count_per_family=3)
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
    result = generate(tmp_path, DuplicatePromptSFTBackend(), count_per_family=2)
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 1
    assert manifest["metadata"]["candidate_rows"] == 2
    assert manifest["metadata"]["attempted_rows"] == 2
    assert manifest["metadata"]["accepted_rows"] == 1
    assert manifest["metadata"]["duplicate_rows"] == 1


def test_generate_sft_llm_run_does_not_replace_rejected_candidates(tmp_path):
    result = generate(tmp_path, RejectSecondSFTBackend(), count_per_family=2)
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 1
    assert manifest["metadata"]["attempted_rows"] == 2
    assert manifest["metadata"]["accepted_rows"] == 1
    assert manifest["metadata"]["rejected_rows"] == 1


def test_generate_sft_llm_run_reduces_batch_size_after_failure(tmp_path):
    backend = SplitOnLargeSFTBackend()
    result = generate(tmp_path, backend, count_per_family=3, batch_size=3)
    assert result.row_count == 3
    assert backend.calls == [3, 1, 1, 1]


def test_generate_sft_llm_run_rejects_multiple_planning_strategies(tmp_path):
    with pytest.raises(ValueError, match="provide only one"):
        generate(tmp_path, FakeSFTBackend(), count_per_family=1, candidate_counts_by_family={"grounded_qa_and_reading": 1})


def test_generate_sft_llm_run_checks_capacity_before_backend_construction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("slm_synth.sft.runs.build_openrouter_backend", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="finite source capacity"):
        generate_llm_run(
            families=["summarization"], count_per_family=2, batch_size=1,
            output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
            teacher_model="teacher/model", generation_run="too-large", max_tokens=100,
            start_index=unique_capacity("summarization"), concurrency=1,
        )
    assert calls == []


def test_resolve_sft_spec_families_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate SFT spec family"):
        resolve_spec_families(["grounded_qa_and_reading", "grounded_qa_and_reading"])
