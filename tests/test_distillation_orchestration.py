import json
import re

import pytest

from slm_synth.distillation.orchestration import (
    generate_seed_multi_signal_run,
    normalize_signal_counts,
    normalize_signal_sequence,
)


class SignalAwareBackend:
    def __init__(self, signal):
        self.signal = signal
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        return {
            "data": {
                "items": [
                    {
                        "id": f"{self.signal}-000001",
                        "reasoning": None,
                        "response": f"Response for {self.signal}.",
                    }
                ]
            }
        }


class PromptIdBackend:
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        ids = re.findall(r'"id": "([^"]+)"', prompt)
        return {
            "data": {
                "items": [
                    {
                        "id": item_id,
                        "reasoning": None,
                        "response": f"Response for {item_id}.",
                    }
                    for item_id in ids
                ]
            }
        }


def test_normalize_signal_sequence_defaults_to_all_supported_signals_sorted():
    signals = normalize_signal_sequence()

    assert signals == tuple(sorted(signals))
    assert "arithmetic" in signals
    assert "instruction" in signals


def test_normalize_signal_sequence_rejects_duplicate_explicit_signals():
    with pytest.raises(ValueError, match="duplicate signal"):
        normalize_signal_sequence(["cloud", "cloud"])


def test_normalize_signal_counts_uses_fixed_count_per_signal():
    counts = normalize_signal_counts(signals=["cloud", "database"], count_per_signal=3)

    assert counts == {"cloud": 3, "database": 3}


def test_generate_seed_multi_signal_run_writes_one_dataset_and_manifest_per_signal(tmp_path):
    backends = {}

    def backend_factory(signal):
        backend = SignalAwareBackend(signal)
        backends[signal] = backend
        return backend

    result = generate_seed_multi_signal_run(
        signals=["cloud", "database"],
        count_per_signal=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="smoke-001",
        max_tokens=512,
        token_target="100K",
        concurrency=2,
        backend_factory=backend_factory,
    )

    assert result.generation_run == "smoke-001"
    assert result.signals == ("cloud", "database")
    assert result.row_count == 2
    assert [item.row_count for item in result.results] == [1, 1]
    assert result.manifest_path == tmp_path / "manifests" / "smoke-001.manifest.json"

    cloud_row = json.loads((tmp_path / "datasets" / "cloud.jsonl").read_text(encoding="utf-8").strip())
    database_row = json.loads((tmp_path / "datasets" / "database.jsonl").read_text(encoding="utf-8").strip())
    assert cloud_row["id"] == "cloud-000001"
    assert cloud_row["response"] == "Response for cloud."
    assert database_row["id"] == "database-000001"
    assert database_row["response"] == "Response for database."
    assert "signal" not in cloud_row
    assert "metadata" not in cloud_row

    cloud_manifest = json.loads((tmp_path / "manifests" / "cloud.smoke-001.manifest.json").read_text())
    database_manifest = json.loads((tmp_path / "manifests" / "database.smoke-001.manifest.json").read_text())
    run_manifest = json.loads((tmp_path / "manifests" / "smoke-001.manifest.json").read_text())
    assert cloud_manifest["teacher_provider"] == "openrouter"
    assert database_manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert cloud_manifest["metadata"]["prompt_count"] == 1
    assert database_manifest["metadata"]["prompt_count"] == 1
    assert run_manifest["generation_run"] == "smoke-001"
    assert run_manifest["signals"] == ["cloud", "database"]
    assert run_manifest["total_rows"] == 2
    assert run_manifest["datasets"][0]["signal"] == "cloud"
    assert run_manifest["datasets"][1]["signal"] == "database"
    assert run_manifest["metadata"]["signal_count"] == 2
    assert run_manifest["metadata"]["concurrency"] == 2

    assert backends["cloud"].calls[0]["schema_name"] == "cloud_distillation_batch"
    assert backends["database"].calls[0]["schema_name"] == "database_distillation_batch"


def test_generate_seed_multi_signal_run_rejects_missing_count_strategy(tmp_path):
    with pytest.raises(ValueError, match="count_per_signal is required"):
        generate_seed_multi_signal_run(
            signals=["cloud"],
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="smoke-001",
            max_tokens=512,
        )


def test_generate_seed_multi_signal_run_rejects_bad_concurrency(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        generate_seed_multi_signal_run(
            signals=["cloud"],
            count_per_signal=1,
            concurrency=0,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="smoke-001",
            max_tokens=512,
        )


def test_generate_seed_multi_signal_run_splits_large_signal_batches(tmp_path):
    result = generate_seed_multi_signal_run(
        signals=["arithmetic"],
        count_per_signal=3,
        batch_size=2,
        concurrency=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="smoke-001",
        max_tokens=512,
        backend_factory=lambda signal: PromptIdBackend(),
    )

    assert result.row_count == 3
    assert result.results[0].dataset_path == tmp_path / "datasets" / "arithmetic.jsonl"
    assert (tmp_path / "datasets" / "arithmetic.batch000001.jsonl").exists()
    assert (tmp_path / "datasets" / "arithmetic.batch000002.jsonl").exists()

    rows = [
        json.loads(line)
        for line in (tmp_path / "datasets" / "arithmetic.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["arithmetic-000001", "arithmetic-000002", "arithmetic-000003"]

    manifest = json.loads((tmp_path / "manifests" / "arithmetic.smoke-001.manifest.json").read_text())
    assert manifest["row_count"] == 3
    assert manifest["metadata"]["batch_count"] == 2
    assert manifest["metadata"]["batch_size"] == 2
    assert manifest["metadata"]["concurrency"] == 2
