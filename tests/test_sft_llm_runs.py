import json

import pytest

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
                            {"role": "assistant", "content": "Correct."},
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
        backend=backend,
    )

    assert result.row_count == 3
    assert result.families == ("basic_arithmetic_qa",)
    assert len(result.results) == 2
    assert len(backend.calls) == 2
    assert (tmp_path / "datasets" / "basic_arithmetic_qa.batch000001.jsonl").exists()
    assert (tmp_path / "datasets" / "basic_arithmetic_qa.batch000002.jsonl").exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_type"] == "sft"
    assert manifest["generation_mode"] == "live_llm_run"
    assert manifest["total_rows"] == 3
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["metadata"]["batch_size"] == 2
    assert [item["row_count"] for item in manifest["datasets"]] == [2, 1]
    assert [item["batch_number"] for item in manifest["datasets"]] == [1, 2]


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
