import json

import pytest

from slm_synth.accepted_target import UnderfilledRunError
from slm_synth.dpo.runs import generate_llm_run, resolve_spec_families
from slm_synth.dpo.spec_builders import DPO_SPEC_CAPACITIES
from tests.alignment_backend_fakes import AcceptingAdjudicatorBackend, StagedDPOBackend


def generation_backends(backend):
    return {
        "backend": StagedDPOBackend(backend),
        "adjudicator_backend": AcceptingAdjudicatorBackend(),
    }


def _fake_row_from_spec(spec):
    variables = spec.get("variables", {})
    chosen = str(variables.get("chosen_answer") or variables.get("answer") or "Correct.")
    rejected = str(variables.get("rejected_answer") or "Incorrect, with a realistic failure.")
    return {
        "id": spec["id"],
        "prompt": [{"role": "user", "content": f"Answer this generated item: {spec['id']}"}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "metadata": spec["metadata"],
    }


class FakeDPOBackend:
    def __init__(self):
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        return {
            "data": {
                "items": [_fake_row_from_spec(spec) for spec in specs]
            },
            "telemetry": {"usage": {"total_tokens": 12}},
        }


class SplitOnLargeDPOBackend(FakeDPOBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append({"batch_size": len(specs)})
        if len(specs) > 1:
            raise ValueError("batch too large")
        return {
            "data": {
                "items": [_fake_row_from_spec(spec) for spec in specs]
            },
            "telemetry": {"usage": {"total_tokens": 12}},
        }


class DuplicatePromptDPOBackend(FakeDPOBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append([spec["id"] for spec in specs])
        return {
            "data": {"items": [
                {
                    "id": spec["id"],
                    "prompt": [{"role": "user", "content": "Answer the repeated prompt."}],
                    "chosen": [{"role": "assistant", "content": f"Correct {spec['id']}"}],
                    "rejected": [{"role": "assistant", "content": f"Wrong {spec['id']}"}],
                    "metadata": spec["metadata"],
                }
                for spec in specs
            ]},
            "telemetry": {"usage": {"total_tokens": 12}},
        }


class OneRoundBackfillDPOBackend(FakeDPOBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append([spec["id"] for spec in specs])
        items = []
        for spec in specs:
            index = int(spec["id"].rsplit("_", 1)[1])
            user_prompt = "Answer the repeated initial prompt." if index <= 2 else f"Answer unique item {index}."
            items.append(
                {
                    "id": spec["id"],
                    "prompt": [{"role": "user", "content": user_prompt}],
                    "chosen": [{"role": "assistant", "content": f"Correct {index}"}],
                    "rejected": [{"role": "assistant", "content": f"Wrong {index}"}],
                    "metadata": spec["metadata"],
                }
            )
        return {"data": {"items": items}, "telemetry": {"usage": {"total_tokens": 12}}}


class RejectSecondDPOBackend(OneRoundBackfillDPOBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        result = super().generate_structured_object_with_metadata(
            prompt=prompt, schema=schema, schema_name=schema_name
        )
        for item in result["data"]["items"]:
            if item["id"].endswith("000002"):
                item["metadata"] = {**item["metadata"], "failure_mode": "extra_explanation"}
        return result


def test_generate_dpo_llm_run_writes_batches_and_run_manifest(tmp_path):
    backend = FakeDPOBackend()

    result = generate_llm_run(
        families=["helpfulness_and_completeness"],
        count_per_family=3,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-run-001",
        max_tokens=1024,
        concurrency=2,
        max_backfill_rounds=0,
        **generation_backends(backend),
    )

    assert result.row_count == 3
    assert result.families == ("helpfulness_and_completeness",)
    assert len(result.results) == 2
    assert len(backend.calls) == 2
    assert (tmp_path / "datasets" / "helpfulness_and_completeness.jsonl").exists()
    assert not (tmp_path / "datasets" / "helpfulness_and_completeness.batch000001.jsonl").exists()
    assert (tmp_path / "batches" / "helpfulness_and_completeness.batch000001.jsonl").exists()
    assert (tmp_path / "batches" / "helpfulness_and_completeness.batch000002.jsonl").exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_type"] == "dpo"
    assert manifest["generation_mode"] == "live_llm_run"
    assert manifest["total_rows"] == 3
    assert manifest["total_pairs"] == 3
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["batch_size"] == 2
    assert manifest["metadata"]["concurrency"] == 2
    assert manifest["metadata"]["adaptive_maximum_in_flight"] == 2
    assert manifest["metadata"]["adaptive_initial_in_flight"] == 8
    assert manifest["metadata"]["llm_telemetry"]["batch_count"] == 6
    assert manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 40
    assert manifest["metadata"]["attempted_pairs"] == 3
    assert manifest["metadata"]["accepted_pairs"] == 3
    assert manifest["metadata"]["duplicate_pairs"] == 0
    assert manifest["metadata"]["attempted_pairs_per_family"] == {"helpfulness_and_completeness": 3}
    assert [item["row_count"] for item in manifest["datasets"]] == [3]
    assert manifest["datasets"][0]["dataset_path"] == str(tmp_path / "datasets" / "helpfulness_and_completeness.jsonl")
    assert manifest["datasets"][0]["batch_count"] == 2
    assert len(manifest["datasets"][0]["batch_manifests"]) == 2
    batch_manifest = json.loads((tmp_path / "manifests" / "helpfulness_and_completeness.batch000001.dpo-live-run-001.manifest.json").read_text())
    assert batch_manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 20


def test_generate_dpo_llm_run_supports_multiple_families(tmp_path):
    backend = FakeDPOBackend()

    result = generate_llm_run(
        families=["helpfulness_and_completeness", "safe_refusal_calibration"],
        count_per_family=1,
        batch_size=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-run-001",
        max_tokens=1024,
        **generation_backends(backend),
    )

    assert result.row_count == 2
    assert result.families == ("helpfulness_and_completeness", "safe_refusal_calibration")
    assert len(backend.calls) == 2
    assert (tmp_path / "datasets" / "helpfulness_and_completeness.jsonl").exists()
    assert (tmp_path / "datasets" / "safe_refusal_calibration.jsonl").exists()


def test_generate_dpo_llm_run_reduces_batch_size_after_failure(tmp_path):
    backend = SplitOnLargeDPOBackend()

    result = generate_llm_run(
        families=["helpfulness_and_completeness"],
        count_per_family=3,
        batch_size=3,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-run-001",
        max_tokens=1024,
        **generation_backends(backend),
    )

    assert result.row_count == 3
    assert [call["batch_size"] for call in backend.calls] == [3, 1, 1, 1]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["adaptive_batch_size_observed_minimum"] == 1
    assert manifest["metadata"]["adaptive_batch_size_decreases"] == 1


def test_resolve_dpo_spec_families_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate DPO spec family"):
        resolve_spec_families(["factual_accuracy", "factual_accuracy"])


def test_generate_dpo_llm_run_rejects_bad_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        generate_llm_run(
            families=["factual_accuracy"],
            count_per_family=1,
            batch_size=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-live-run-001",
            max_tokens=1024,
            **generation_backends(FakeDPOBackend()),
        )


def test_generate_dpo_llm_run_rejects_bad_concurrency(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        generate_llm_run(
            families=["factual_accuracy"],
            count_per_family=1,
            batch_size=1,
            concurrency=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-live-run-001",
            max_tokens=1024,
            **generation_backends(FakeDPOBackend()),
        )


def test_generate_dpo_llm_run_preflights_capacity_before_backend_construction(tmp_path, monkeypatch):
    def unexpected_backend(**kwargs):
        raise AssertionError("provider backend must not be constructed")

    monkeypatch.setattr("slm_synth.dpo.runs.build_openrouter_backend", unexpected_backend)
    family = "groundedness"
    with pytest.raises(ValueError, match="finite source capacity"):
        generate_llm_run(
            families=[family],
            count_per_family=2,
            start_index=DPO_SPEC_CAPACITIES[family],
            batch_size=1,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="unused/model",
            generation_run="dpo-capacity-001",
            max_tokens=1024,
        )


def test_generate_dpo_llm_run_accepts_target_pairs_and_records_planning(tmp_path):
    backend = FakeDPOBackend()

    result = generate_llm_run(
        families=["helpfulness_and_completeness", "safe_refusal_calibration"],
        target_pairs=3,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-target-run-001",
        max_tokens=1024,
        **generation_backends(backend),
    )

    assert result.row_count == 3
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["planning_mode"] == "target_pairs"
    assert manifest["metadata"]["target_pairs"] == 3
    assert manifest["metadata"]["planned_pairs"] == 3
    assert manifest["metadata"]["accepted_pairs"] == 3
    assert manifest["metadata"]["rejected_pairs"] == 0
    assert manifest["metadata"]["pairs_per_family"] == {
        "helpfulness_and_completeness": 2,
        "safe_refusal_calibration": 1,
    }
    assert manifest["metadata"]["count_per_family"] is None


def test_generate_dpo_llm_run_backfills_duplicates_with_new_source_indexes(tmp_path):
    backend = OneRoundBackfillDPOBackend()

    result = generate_llm_run(
        families=["helpfulness_and_completeness"],
        count_per_family=2,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-backfill-run-001",
        max_tokens=1024,
        max_backfill_rounds=1,
        **generation_backends(backend),
    )

    assert result.row_count == 2
    assert backend.calls == [
        [
            "dpo_helpfulness_and_completeness_planning_brainstorming_recommendations_000001",
            "dpo_helpfulness_and_completeness_planning_brainstorming_recommendations_000002",
        ],
        ["dpo_helpfulness_and_completeness_everyday_conversation_000003"],
    ]
    public_rows = [
        json.loads(line)
        for line in (tmp_path / "datasets" / "helpfulness_and_completeness.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in public_rows] == [
        "dpo_helpfulness_and_completeness_planning_brainstorming_recommendations_000001",
        "dpo_helpfulness_and_completeness_everyday_conversation_000003",
    ]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["attempted_pairs"] == 3
    assert manifest["metadata"]["accepted_pairs"] == 2
    assert manifest["metadata"]["duplicate_pairs"] == 1
    assert manifest["metadata"]["backfill_rounds"] == 1
    assert manifest["metadata"]["next_start_index_per_family"] == {"helpfulness_and_completeness": 4}
    assert manifest["metadata"]["publish_ready"] is True


def test_generate_dpo_llm_run_exhausts_backfill_budget_without_counting_duplicates(tmp_path):
    with pytest.raises(UnderfilledRunError, match="remaining=1"):
        generate_llm_run(
            families=["helpfulness_and_completeness"],
            count_per_family=2,
            batch_size=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-backfill-exhausted-001",
            max_tokens=1024,
            max_backfill_rounds=1,
            **generation_backends(DuplicatePromptDPOBackend()),
        )

    manifest = json.loads((tmp_path / "manifests" / "dpo-backfill-exhausted-001.manifest.json").read_text())
    assert manifest["metadata"]["attempted_pairs"] == 3
    assert manifest["metadata"]["accepted_pairs"] == 1
    assert manifest["metadata"]["duplicate_pairs"] == 2
    assert manifest["metadata"]["remaining_pairs"] == 1
    assert manifest["metadata"]["accepted_target"]["backfill_budget_exhausted"] is True


def test_generate_dpo_llm_run_backfills_terminal_validation_rejections(tmp_path):
    result = generate_llm_run(
        families=["helpfulness_and_completeness"],
        count_per_family=2,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-rejection-backfill-001",
        max_tokens=1024,
        max_backfill_rounds=1,
        **generation_backends(RejectSecondDPOBackend()),
    )

    assert result.row_count == 2
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["attempted_pairs"] == 3
    assert manifest["metadata"]["accepted_pairs"] == 2
    assert manifest["metadata"]["rejected_pairs"] == 1
    assert manifest["metadata"]["duplicate_pairs"] == 0
    assert manifest["metadata"]["rejection_reason_counts"] == {"batch_acceptance_error": 1}
    assert manifest["metadata"]["backfill_rounds"] == 1


def test_generate_dpo_llm_run_resumes_without_repeating_prior_indexes(tmp_path):
    with pytest.raises(UnderfilledRunError):
        generate_llm_run(
            families=["helpfulness_and_completeness"], count_per_family=2, batch_size=2,
            output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini", generation_run="dpo-resume-run-001",
            max_tokens=1024, max_backfill_rounds=0, **generation_backends(OneRoundBackfillDPOBackend()),
        )

    resume_backend = OneRoundBackfillDPOBackend()
    result = generate_llm_run(
        families=["helpfulness_and_completeness"], count_per_family=2, batch_size=2,
        output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini", generation_run="dpo-resume-run-001",
        max_tokens=1024, max_backfill_rounds=1, resume=True, **generation_backends(resume_backend),
    )

    assert result.row_count == 2
    assert resume_backend.calls == [["dpo_helpfulness_and_completeness_everyday_conversation_000003"]]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["attempted_pairs"] == 3
    assert manifest["metadata"]["duplicate_pairs"] == 1
    assert manifest["datasets"][0]["batch_count"] == 2


def test_generate_dpo_llm_run_complete_resume_does_not_construct_backend(tmp_path, monkeypatch):
    generate_llm_run(
        families=["helpfulness_and_completeness"], count_per_family=2, batch_size=2,
        output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini", generation_run="dpo-complete-resume-001",
        max_tokens=1024, **generation_backends(FakeDPOBackend()),
    )
    monkeypatch.setattr(
        "slm_synth.dpo.runs.build_openrouter_backend",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("backend must not be constructed")),
    )

    resumed = generate_llm_run(
        families=["helpfulness_and_completeness"], count_per_family=2, batch_size=2,
        output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini", generation_run="dpo-complete-resume-001",
        max_tokens=1024, resume=True,
    )

    assert resumed.row_count == 2
    assert resumed.results == ()


def test_generate_dpo_llm_run_resume_rejects_modified_accepted_data(tmp_path):
    with pytest.raises(UnderfilledRunError):
        generate_llm_run(
            families=["helpfulness_and_completeness"], count_per_family=2, batch_size=2,
            output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini", generation_run="dpo-tampered-run-001",
            max_tokens=1024, max_backfill_rounds=0, **generation_backends(DuplicatePromptDPOBackend()),
        )
    dataset_path = tmp_path / "datasets" / "helpfulness_and_completeness.jsonl"
    row = json.loads(dataset_path.read_text())
    row["prompt"][0]["content"] = "Modified after the failed run."
    dataset_path.write_text(json.dumps(row) + "\n")
    backend = OneRoundBackfillDPOBackend()

    with pytest.raises(ValueError, match="content fingerprint"):
        generate_llm_run(
            families=["helpfulness_and_completeness"], count_per_family=2, batch_size=2,
            output_dir=tmp_path / "datasets", manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini", generation_run="dpo-tampered-run-001",
            max_tokens=1024, max_backfill_rounds=1, resume=True, **generation_backends(backend),
        )
    assert backend.calls == []


def test_generate_dpo_llm_run_rejects_multiple_planning_strategies(tmp_path):
    with pytest.raises(ValueError, match="provide exactly one"):
        generate_llm_run(
            families=["factual_accuracy"],
            count_per_family=1,
            target_pairs=1,
            batch_size=1,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-live-run-001",
            max_tokens=1024,
            **generation_backends(FakeDPOBackend()),
        )


def test_generate_dpo_llm_run_fails_when_public_pairs_underfill_after_budget(tmp_path, monkeypatch):
    def write_underfilled_public_family_files(*, jobs, output_dir, families, **kwargs):
        dataset_path = output_dir / "helpfulness_and_completeness.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("", encoding="utf-8")
        accepted_rows = {family: [] for family in families}
        return ([{
                "family": "helpfulness_and_completeness",
                "dataset_path": dataset_path,
                "row_count": 1,
                "batch_count": len(jobs),
                "batch_manifests": [job["result"].manifest_path for job in jobs],
            }], {
                "attempted_pairs": 2,
                "accepted_pairs": 1,
                "rejected_pairs": 0,
                "duplicate_pairs": 1,
                "duplicate_reason_counts": {"duplicate_prompt": 1},
                "rejection_reason_counts": {},
                "attempted_pairs_per_family": {"helpfulness_and_completeness": 2},
                "accepted_pairs_per_family": {"helpfulness_and_completeness": 1},
                "rejected_pairs_per_family": {"helpfulness_and_completeness": 0},
                "duplicate_pairs_per_family": {"helpfulness_and_completeness": 1},
            }, accepted_rows)

    monkeypatch.setattr(
        "slm_synth.dpo.runs._write_public_family_files",
        write_underfilled_public_family_files,
    )

    with pytest.raises(UnderfilledRunError, match="DPO.*underfilled.*remaining=1"):
        generate_llm_run(
            families=["helpfulness_and_completeness"],
            count_per_family=2,
            batch_size=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-underfilled-001",
            max_tokens=1024,
            max_backfill_rounds=0,
            **generation_backends(FakeDPOBackend()),
        )

    manifest = json.loads((tmp_path / "manifests" / "dpo-underfilled-001.manifest.json").read_text())
    assert manifest["metadata"]["generation_status"] == "underfilled"
    assert manifest["metadata"]["failure_status"] == "failed"
    assert manifest["metadata"]["run_failed"] is True
    assert manifest["metadata"]["remaining_pairs"] == 1
    assert manifest["metadata"]["duplicate_pairs"] == 1
