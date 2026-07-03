import json

import pytest

from slm_synth.dpo.runs import generate_llm_run, resolve_spec_families


class FakeDPOBackend:
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
                        "prompt": [{"role": "user", "content": f"Answer this generated item: {spec['id']}"}],
                        "chosen": [{"role": "assistant", "content": "Correct."}],
                        "rejected": [{"role": "assistant", "content": "Incorrect, with a realistic failure."}],
                        "metadata": spec["metadata"],
                    }
                    for spec in specs
                ]
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
                "items": [
                    {
                        "id": spec["id"],
                        "prompt": [{"role": "user", "content": f"Answer this generated item: {spec['id']}"}],
                        "chosen": [{"role": "assistant", "content": "Correct."}],
                        "rejected": [{"role": "assistant", "content": "Incorrect, with a realistic failure."}],
                        "metadata": spec["metadata"],
                    }
                    for spec in specs
                ]
            },
            "telemetry": {"usage": {"total_tokens": 12}},
        }


def test_generate_dpo_llm_run_writes_batches_and_run_manifest(tmp_path):
    backend = FakeDPOBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa"],
        count_per_family=3,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-run-001",
        max_tokens=1024,
        concurrency=2,
        backend=backend,
    )

    assert result.row_count == 3
    assert result.families == ("basic_arithmetic_qa",)
    assert len(result.results) == 2
    assert len(backend.calls) == 2
    assert (tmp_path / "datasets" / "basic_arithmetic_qa.batch000001.jsonl").exists()
    assert (tmp_path / "datasets" / "basic_arithmetic_qa.batch000002.jsonl").exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_type"] == "dpo"
    assert manifest["generation_mode"] == "live_llm_run"
    assert manifest["total_rows"] == 3
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["batch_size"] == 2
    assert manifest["metadata"]["concurrency"] == 2
    assert manifest["metadata"]["adaptive_maximum_in_flight"] == 2
    assert manifest["metadata"]["adaptive_initial_in_flight"] == 8
    assert [item["row_count"] for item in manifest["datasets"]] == [2, 1]
    assert [item["batch_number"] for item in manifest["datasets"]] == [1, 2]
    batch_manifest = json.loads((tmp_path / "manifests" / "basic_arithmetic_qa.batch000001.dpo-live-run-001.manifest.json").read_text())
    assert batch_manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 12


def test_generate_dpo_llm_run_supports_multiple_families(tmp_path):
    backend = FakeDPOBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa", "repeat_exact_n_times"],
        count_per_family=1,
        batch_size=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-run-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 2
    assert result.families == ("basic_arithmetic_qa", "repeat_exact_n_times")
    assert len(backend.calls) == 2


def test_generate_dpo_llm_run_reduces_batch_size_after_failure(tmp_path):
    backend = SplitOnLargeDPOBackend()

    result = generate_llm_run(
        families=["basic_arithmetic_qa"],
        count_per_family=3,
        batch_size=3,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="dpo-live-run-001",
        max_tokens=1024,
        backend=backend,
    )

    assert result.row_count == 3
    assert [call["batch_size"] for call in backend.calls] == [3, 1, 1, 1]
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["metadata"]["adaptive_batch_size_observed_minimum"] == 1
    assert manifest["metadata"]["adaptive_batch_size_decreases"] == 1


def test_resolve_dpo_spec_families_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate DPO spec family"):
        resolve_spec_families(["basic_arithmetic_qa", "basic_arithmetic_qa"])


def test_generate_dpo_llm_run_rejects_bad_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        generate_llm_run(
            families=["basic_arithmetic_qa"],
            count_per_family=1,
            batch_size=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-live-run-001",
            max_tokens=1024,
            backend=FakeDPOBackend(),
        )


def test_generate_dpo_llm_run_rejects_bad_concurrency(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        generate_llm_run(
            families=["basic_arithmetic_qa"],
            count_per_family=1,
            batch_size=1,
            concurrency=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="dpo-live-run-001",
            max_tokens=1024,
            backend=FakeDPOBackend(),
        )
