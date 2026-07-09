import json

import pytest

from slm_synth.accepted_target import UnderfilledRunError
from slm_synth.sft.runs import generate_llm_run, resolve_spec_families


class FakeSFTBackend:
    def __init__(self):
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        return {
            "data": {
                "items": [
                    {
                        "id": spec["id"],
                        "messages": [
                            {"role": "user", "content": f"Answer this generated item: {spec['id']}"},
                            {"role": "assistant", "content": str(spec.get("variables", {}).get("answer", "Correct."))},
                        ],
                        "metadata": spec["metadata"],
                    }
                    for spec in specs
                ]
            },
            "telemetry": {"usage": {"total_tokens": 12}},
        }


class SplitOnLargeSFTBackend(FakeSFTBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        specs = json.loads(prompt.split("Input specs:\n", 1)[1])["items"]
        self.calls.append({"batch_size": len(specs)})
        if len(specs) > 1:
            raise ValueError("batch too large")
        return {
            "data": {
                "items": [
                    {
                        "id": spec["id"],
                        "messages": [
                            {"role": "user", "content": f"Answer this generated item: {spec['id']}"},
                            {"role": "assistant", "content": str(spec.get("variables", {}).get("answer", "Correct."))},
                        ],
                        "metadata": spec["metadata"],
                    }
                    for spec in specs
                ]
            },
            "telemetry": {"usage": {"total_tokens": 12}},
        }


def test_generate_sft_llm_run_writes_batches_and_run_manifest(tmp_path):
    backend = FakeSFTBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa"],
        count_per_family=3,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-live-run-001",
        max_tokens=1024,
        concurrency=2,
        backend=backend,
    )

    assert result.row_count == 3
    assert result.families == ("basic_arithmetic_qa",)
    assert len(result.results) == 2
    assert len(backend.calls) == 2
    assert (tmp_path / "datasets" / "basic_arithmetic_qa.jsonl").exists()
    assert not (tmp_path / "datasets" / "basic_arithmetic_qa.batch000001.jsonl").exists()
    assert (tmp_path / "batches" / "basic_arithmetic_qa.batch000001.jsonl").exists()
    assert (tmp_path / "batches" / "basic_arithmetic_qa.batch000002.jsonl").exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_type"] == "sft"
    assert manifest["generation_mode"] == "live_llm_run"
    assert manifest["total_rows"] == 3
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["batch_size"] == 2
    assert manifest["metadata"]["concurrency"] == 2
    assert manifest["metadata"]["adaptive_maximum_in_flight"] == 2
    assert manifest["metadata"]["adaptive_initial_in_flight"] == 8
    assert manifest["metadata"]["llm_telemetry"]["batch_count"] == 2
    assert manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 24
    assert [item["row_count"] for item in manifest["datasets"]] == [3]
    assert manifest["datasets"][0]["dataset_path"] == str(tmp_path / "datasets" / "basic_arithmetic_qa.jsonl")
    assert manifest["datasets"][0]["batch_count"] == 2
    assert len(manifest["datasets"][0]["batch_manifests"]) == 2
    batch_manifest = json.loads((tmp_path / "manifests" / "basic_arithmetic_qa.batch000001.sft-live-run-001.manifest.json").read_text())
    assert batch_manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 12


def test_generate_sft_llm_run_supports_multiple_families(tmp_path):
    backend = FakeSFTBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa", "repeat_exact_n_times"],
        count_per_family=1,
        batch_size=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-live-run-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 2
    assert result.families == ("basic_arithmetic_qa", "repeat_exact_n_times")
    assert len(backend.calls) == 2
    assert (tmp_path / "datasets" / "basic_arithmetic_qa.jsonl").exists()
    assert (tmp_path / "datasets" / "repeat_exact_n_times.jsonl").exists()


def test_generate_sft_llm_run_reduces_batch_size_after_failure(tmp_path):
    backend = SplitOnLargeSFTBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa"],
        count_per_family=3,
        batch_size=3,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-live-run-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 3
    assert [call["batch_size"] for call in backend.calls] == [3, 1, 1, 1]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["adaptive_batch_size_observed_minimum"] == 1
    assert manifest["metadata"]["adaptive_batch_size_decreases"] == 1


def test_resolve_sft_spec_families_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate SFT spec family"):
        resolve_spec_families(["basic_arithmetic_qa", "basic_arithmetic_qa"])


def test_generate_sft_llm_run_rejects_bad_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        generate_llm_run(
            families=["basic_arithmetic_qa"],
            count_per_family=1,
            batch_size=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="sft-live-run-001",
            max_tokens=1024,
            backend=FakeSFTBackend(),
        )


def test_generate_sft_llm_run_rejects_bad_concurrency(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        generate_llm_run(
            families=["basic_arithmetic_qa"],
            count_per_family=1,
            batch_size=1,
            concurrency=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="sft-live-run-001",
            max_tokens=1024,
            backend=FakeSFTBackend(),
        )


def test_generate_sft_llm_run_accepts_target_rows_and_records_planning(tmp_path):
    backend = FakeSFTBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa", "repeat_exact_n_times"],
        target_rows=3,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="sft-target-run-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 3
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["planning_mode"] == "target_rows"
    assert manifest["metadata"]["target_rows"] == 3
    assert manifest["metadata"]["planned_rows"] == 3
    assert manifest["metadata"]["accepted_rows"] == 3
    assert manifest["metadata"]["rejected_rows"] == 0
    assert manifest["metadata"]["rows_per_family"] == {
        "basic_arithmetic_qa": 2,
        "repeat_exact_n_times": 1,
    }
    assert manifest["metadata"]["count_per_family"] is None


def test_generate_sft_llm_run_rejects_multiple_planning_strategies(tmp_path):
    with pytest.raises(ValueError, match="provide exactly one"):
        generate_llm_run(
            families=["basic_arithmetic_qa"],
            count_per_family=1,
            target_rows=1,
            batch_size=1,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="sft-live-run-001",
            max_tokens=1024,
            backend=FakeSFTBackend(),
        )


def test_generate_sft_llm_run_fails_when_public_rows_underfill_after_budget(tmp_path, monkeypatch):
    def write_underfilled_public_family_files(*, jobs, output_dir):
        dataset_path = output_dir / "basic_arithmetic_qa.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("", encoding="utf-8")
        return [
            {
                "family": "basic_arithmetic_qa",
                "dataset_path": dataset_path,
                "row_count": 1,
                "batch_count": len(jobs),
                "batch_manifests": [job["result"].manifest_path for job in jobs],
            }
        ]

    monkeypatch.setattr(
        "slm_synth.sft.runs._write_public_family_files",
        write_underfilled_public_family_files,
    )

    with pytest.raises(UnderfilledRunError, match="SFT.*underfilled.*remaining=1"):
        generate_llm_run(
            families=["basic_arithmetic_qa"],
            count_per_family=2,
            batch_size=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="sft-underfilled-001",
            max_tokens=1024,
            max_backfill_rounds=0,
            backend=FakeSFTBackend(),
        )

    manifest = json.loads((tmp_path / "manifests" / "sft-underfilled-001.manifest.json").read_text())
    assert manifest["metadata"]["generation_status"] == "underfilled"
    assert manifest["metadata"]["failure_status"] == "failed"
    assert manifest["metadata"]["run_failed"] is True
    assert manifest["metadata"]["remaining_rows"] == 1
