import json

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
        backend_factory=backend_factory,
    )

    assert result.generation_run == "smoke-001"
    assert result.signals == ("cloud", "database")
    assert result.row_count == 2
    assert [item.row_count for item in result.results] == [1, 1]

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
    assert cloud_manifest["teacher_provider"] == "openrouter"
    assert database_manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert cloud_manifest["metadata"]["prompt_count"] == 1
    assert database_manifest["metadata"]["prompt_count"] == 1

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
