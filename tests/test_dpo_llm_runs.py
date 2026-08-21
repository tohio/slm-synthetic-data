import json

import pytest

from slm_synth.dpo.runs import generate_llm_run, resolve_preference_dimensions
from slm_synth.dpo.spec_builders import DPO_SPEC_CAPACITIES
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend, StagedDPOBackend


def _fake_row(spec, *, prompt=None):
    return {"prompt": [{"role": "user", "content": prompt or f"Answer this generated item: {spec['id']}"}], "chosen": [{"role": "assistant", "content": "A strong grounded answer."}], "rejected": [{"role": "assistant", "content": "A plausible but weaker answer."}]}


class FakeDPOBackend:
    def __init__(self):
        self.calls = []

    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append([spec["id"] for spec in specs])
        return {"text": json.dumps({"items": [_fake_row(spec) for spec in specs]}), "telemetry": {"usage": {"total_tokens": 12}}}


class DuplicatePromptDPOBackend(FakeDPOBackend):
    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append([spec["id"] for spec in specs])
        return {"text": json.dumps({"items": [_fake_row(spec, prompt="Answer the repeated prompt.") for spec in specs]}), "telemetry": {"usage": {"total_tokens": 12}}}


class SplitOnLargeDPOBackend(FakeDPOBackend):
    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append(len(specs))
        if len(specs) > 1:
            raise ValueError("batch too large")
        return {"text": json.dumps({"items": [_fake_row(spec) for spec in specs]}), "telemetry": {"usage": {"total_tokens": 12}}}


class RejectAllDPOBackend(FakeDPOBackend):
    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        rows = [_fake_row(spec) for spec in specs]
        for row in rows:
            row["prompt"] = []
        return {"text": json.dumps({"items": rows}), "telemetry": {"usage": {"total_tokens": 12}}}


class SplitFirstDimensionDPOBackend(FakeDPOBackend):
    def generate_text_with_metadata(self, *, prompt, system_prompt):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        dimension = specs[0]["metadata"]["preference_dimension"]
        self.calls.append((dimension, len(specs)))
        if dimension == "helpfulness_and_completeness" and len(specs) > 1:
            raise ValueError("split the first dimension")
        return {"text": json.dumps({"items": [_fake_row(spec) for spec in specs]})}


def _backends(backend):
    return {"backend": StagedDPOBackend(backend), "adjudicator_backend": AcceptingAdjudicatorBackend()}


def _generate(tmp_path, backend, *, preference_dimensions=None, counts=None, batch_size=2, **kwargs):
    dimensions = preference_dimensions or ["helpfulness_and_completeness"]
    counts = counts or {dimension: 3 for dimension in dimensions}
    return generate_llm_run(preference_dimensions=dimensions, candidate_counts_by_dimension=counts, batch_size=batch_size, output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests", teacher_model="openai/gpt-4.1-mini", generation_run="dpo-candidate-run-001", max_tokens=1024, **_backends(backend), **kwargs)


def test_generate_dpo_llm_run_records_candidate_outcome_and_tokens(tmp_path):
    result = _generate(tmp_path, FakeDPOBackend(), counts={"helpfulness_and_completeness": 3})
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 3
    assert manifest["metadata"]["planning_mode"] == "candidate_counts_by_dimension"
    assert manifest["metadata"]["candidate_pairs"] == 3
    assert manifest["metadata"]["candidate_pairs_per_dimension"] == {"helpfulness_and_completeness": 3}
    assert manifest["metadata"]["accepted_pairs_per_dimension"] == {"helpfulness_and_completeness": 3}
    assert manifest["metadata"]["accepted_pairs"] == 3
    assert manifest["metadata"]["estimated_tokens"] > 0
    assert "accepted_target" not in manifest["metadata"]
    assert "remaining_pairs" not in manifest["metadata"]


def test_generate_dpo_llm_run_supports_dimension_specific_counts(tmp_path):
    families = ["helpfulness_and_completeness", "safe_refusal_calibration"]
    result = _generate(tmp_path, FakeDPOBackend(), preference_dimensions=families, counts={"helpfulness_and_completeness": 2, "safe_refusal_calibration": 1})
    assert result.row_count == 3
    assert result.preference_dimensions == tuple(families)


def test_generate_dpo_llm_run_does_not_replace_duplicate_candidates(tmp_path):
    backend = DuplicatePromptDPOBackend()
    result = _generate(tmp_path, backend, counts={"helpfulness_and_completeness": 2})
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 1
    assert len(backend.calls) == 1
    assert manifest["metadata"]["attempted_pairs"] == 2
    assert manifest["metadata"]["accepted_pairs"] == 1
    assert manifest["metadata"]["duplicate_pairs"] == 1
    assert manifest["metadata"]["generation_status"] == "complete"


def test_generate_dpo_llm_run_blocks_an_empty_requested_dimension(tmp_path):
    result = _generate(
        tmp_path,
        RejectAllDPOBackend(),
        counts={"helpfulness_and_completeness": 1},
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert result.row_count == 0
    assert manifest["metadata"]["publish_ready"] is False
    assert manifest["metadata"]["empty_preference_dimensions"] == [
        "helpfulness_and_completeness"
    ]
    assert manifest["metadata"]["rejection_reason_counts"] == {
        "chosen_render_error": 1
    }


def test_generate_dpo_llm_run_resets_adaptive_batch_size_per_dimension(tmp_path):
    backend = SplitFirstDimensionDPOBackend()
    result = _generate(
        tmp_path,
        backend,
        preference_dimensions=["helpfulness_and_completeness", "factual_accuracy"],
        counts={"helpfulness_and_completeness": 2, "factual_accuracy": 2},
        batch_size=2,
    )
    assert result.row_count == 4
    assert ("factual_accuracy", 2) in backend.calls


def test_generate_dpo_llm_run_only_splits_failed_requests(tmp_path):
    backend = SplitOnLargeDPOBackend()
    result = _generate(tmp_path, backend, counts={"helpfulness_and_completeness": 3}, batch_size=3)
    assert result.row_count == 3
    assert backend.calls == [3, 1, 1, 1]


def test_generate_dpo_llm_run_requires_exact_dimension_mapping(tmp_path):
    with pytest.raises(ValueError, match="exactly the requested"):
        _generate(tmp_path, FakeDPOBackend(), preference_dimensions=["factual_accuracy", "groundedness"], counts={"factual_accuracy": 1})


def test_generate_dpo_llm_run_preflights_capacity_before_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("slm_synth.dpo.runs.build_openrouter_backend", lambda **kwargs: (_ for _ in ()).throw(AssertionError("backend constructed")))
    family = "groundedness"
    with pytest.raises(ValueError, match="finite source capacity"):
        generate_llm_run(preference_dimensions=[family], candidate_counts_by_dimension={family: 2}, start_index=DPO_SPEC_CAPACITIES[family], batch_size=1, output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests", teacher_model="unused/model", generation_run="dpo-capacity-001", max_tokens=1024)


def test_generate_dpo_llm_run_rejects_bad_batch_and_concurrency(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        _generate(tmp_path, FakeDPOBackend(), batch_size=0)
    with pytest.raises(ValueError, match="concurrency"):
        _generate(tmp_path, FakeDPOBackend(), concurrency=0)


def test_resolve_dpo_preference_dimensions_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate DPO preference dimension"):
        resolve_preference_dimensions(["factual_accuracy", "factual_accuracy"])
