import json
import re

import pytest

from slm_synth.distillation.orchestration import (
    generate_prompt_spec_multi_signal_run,
    generate_seed_multi_signal_run,
    normalize_signal_counts,
    normalize_signal_sequence,
)
from slm_synth.distillation.prompts import build_prompt_record


class SignalAwareBackend:
    def __init__(self, signal):
        self.signal = signal
        self.calls = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        self.calls.append({"prompt": prompt, "schema": schema, "schema_name": schema_name})
        response = f"Response for {self.signal}."
        if self.signal == "database":
            response = "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id."
        return {
            "data": {
                "items": [
                    {
                        "id": f"{self.signal}-000001",
                        "reasoning": None,
                        "response": response,
                    }
                ]
            },
            "telemetry": {"adaptive_peak_in_flight_limit": 2},
        }


class SignalTelemetryBackend(SignalAwareBackend):
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        result = super().generate_structured_object_with_metadata(prompt=prompt, schema=schema, schema_name=schema_name)
        if self.signal == "cloud":
            result["data"]["items"][0]["response"] = "Use autoscaling to add capacity during traffic spikes."
            result["telemetry"] = {
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.01},
                "retry_count": 1,
                "retryable_provider_retries": 2,
                "retry_sleep_seconds": 1.25,
                "adaptive_window_increases": 1,
                "adaptive_window_decreases": 0,
                "adaptive_admission_wait_seconds": 0.5,
                "adaptive_peak_in_flight_limit": 8,
                "adaptive_min_in_flight_limit": 4,
                "max_adaptive_cooldown_seconds": 0.25,
                "elapsed_seconds": 2.5,
            }
        else:
            result["telemetry"] = {
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.02},
                "retry_count": 3,
                "retryable_provider_retries": 4,
                "retry_sleep_seconds": 2.75,
                "adaptive_window_increases": 2,
                "adaptive_window_decreases": 1,
                "adaptive_admission_wait_seconds": 1.5,
                "adaptive_peak_in_flight_limit": 16,
                "adaptive_min_in_flight_limit": 2,
                "max_adaptive_cooldown_seconds": 0.75,
                "elapsed_seconds": 4.5,
            }
        return result


class PromptIdBackend:
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        items = self._request_items(prompt)
        ids = [item["id"] for item in items]
        return {
            "data": {
                "items": [
                    {
                        "id": item["id"],
                        "reasoning": None,
                        "response": self._response_for(item),
                    }
                    for item in items
                ]
            },
            "telemetry": {
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(ids), "total_tokens": len(ids) * 2},
                "retryable_provider_retries": len(ids),
                "retry_sleep_seconds": float(len(ids)),
                "adaptive_peak_in_flight_limit": len(ids) * 8,
                "adaptive_min_in_flight_limit": 8,
                "elapsed_seconds": float(len(ids)),
            },
        }

    def _request_items(self, prompt):
        request_json = prompt.split("Input items:\n", 1)[1]
        return json.loads(request_json)["items"]

    def _response_for(self, item):
        item_id = item["id"]
        item_prompt = item["prompt"]
        if item_id.startswith("arithmetic-"):
            match = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", item_prompt)
            if match:
                left = int(match.group(1))
                right = int(match.group(3))
                op = match.group(2)
                if op == "+":
                    return str(left + right)
                if op == "-":
                    return str(left - right)
                if op == "*":
                    return str(left * right)
                if op == "/" and right:
                    return str(left // right)
            if "96 pencils" in item_prompt:
                return "8"
        if item_id.startswith("code-"):
            return "def generated(value):\n    return value"
        if item_id.startswith("database-") and "query" in item_prompt.casefold():
            return "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"
        if item_id.startswith("factual_restraint-"):
            return "I cannot verify that, so I should not invent a specific answer."
        return f"Response for {item_id}."


class SplitOnLargePromptIdBackend(PromptIdBackend):
    def __init__(self):
        self.calls: list[int] = []

    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        items = self._request_items(prompt)
        self.calls.append(len(items))
        if len(items) > 1:
            raise ValueError("batch too large")
        return {
            "data": {
                "items": [
                    {
                        "id": item["id"],
                        "reasoning": None,
                        "response": self._response_for(item),
                    }
                    for item in items
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


def test_normalize_signal_counts_splits_target_rows_deterministically():
    counts = normalize_signal_counts(signals=["cloud", "database", "debugging"], target_rows=8)

    assert counts == {"cloud": 3, "database": 3, "debugging": 2}


def test_normalize_signal_counts_rejects_conflicting_planning_strategies():
    with pytest.raises(ValueError, match="only one"):
        normalize_signal_counts(signals=["cloud"], count_per_signal=1, target_rows=1)


def test_normalize_signal_counts_requires_target_rows_for_each_signal():
    with pytest.raises(ValueError, match="at least the number of requested signals"):
        normalize_signal_counts(signals=["cloud", "database"], target_rows=1)


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
    assert database_row["response"] == "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id."
    assert "signal" not in cloud_row
    assert "metadata" not in cloud_row

    cloud_manifest = json.loads((tmp_path / "manifests" / "cloud.smoke-001.manifest.json").read_text())
    database_manifest = json.loads((tmp_path / "manifests" / "database.smoke-001.manifest.json").read_text())
    run_manifest = json.loads((tmp_path / "manifests" / "smoke-001.manifest.json").read_text())
    assert cloud_manifest["teacher_provider"] == "openrouter"
    assert database_manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert cloud_manifest["metadata"]["prompt_count"] == 1
    assert cloud_manifest["metadata"]["llm_telemetry"]["adaptive_peak_in_flight_limit"] == 2
    assert database_manifest["metadata"]["prompt_count"] == 1
    assert run_manifest["generation_run"] == "smoke-001"
    assert run_manifest["signals"] == ["cloud", "database"]
    assert run_manifest["total_rows"] == 2
    assert run_manifest["datasets"][0]["signal"] == "cloud"
    assert run_manifest["datasets"][1]["signal"] == "database"
    assert run_manifest["metadata"]["signal_count"] == 2
    assert run_manifest["metadata"]["concurrency"] == 2
    assert run_manifest["metadata"]["adaptive_maximum_in_flight"] == 2
    assert run_manifest["metadata"]["adaptive_initial_in_flight"] == 8
    assert run_manifest["metadata"]["prompt_source"] == "builtin_seed"
    assert cloud_manifest["metadata"]["prompt_source"] == "builtin_seed"

    run_telemetry = run_manifest["metadata"]["llm_telemetry"]
    assert run_telemetry["batch_count"] == 2
    assert run_telemetry["adaptive_peak_in_flight_limit"] == 2

    assert backends["cloud"].calls[0]["schema_name"] == "cloud_distillation_batch"
    assert backends["database"].calls[0]["schema_name"] == "database_distillation_batch"


def test_generate_seed_multi_signal_run_aggregates_llm_telemetry_across_signals(tmp_path):
    result = generate_seed_multi_signal_run(
        signals=["cloud", "database"],
        count_per_signal=1,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="smoke-001",
        max_tokens=512,
        backend_factory=lambda signal: SignalTelemetryBackend(signal),
    )

    assert result.row_count == 2
    run_manifest = json.loads((tmp_path / "manifests" / "smoke-001.manifest.json").read_text())
    telemetry = run_manifest["metadata"]["llm_telemetry"]
    assert telemetry["batch_count"] == 2
    assert telemetry["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 22,
        "total_tokens": 33,
        "cost": 0.03,
    }
    assert telemetry["retry_count"] == 4
    assert telemetry["retryable_provider_retries"] == 6
    assert telemetry["retry_sleep_seconds"] == 4.0
    assert telemetry["adaptive_window_increases"] == 3
    assert telemetry["adaptive_window_decreases"] == 1
    assert telemetry["adaptive_admission_wait_seconds"] == 2.0
    assert telemetry["adaptive_peak_in_flight_limit"] == 16
    assert telemetry["adaptive_min_in_flight_limit"] == 2
    assert telemetry["max_adaptive_cooldown_seconds"] == 0.75
    assert telemetry["elapsed_seconds"] == 7.0


def test_generate_prompt_spec_multi_signal_run_uses_production_prompt_specs(tmp_path):
    result = generate_prompt_spec_multi_signal_run(
        signals=["arithmetic"],
        target_rows=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="target-001",
        max_tokens=512,
        backend_factory=lambda signal: PromptIdBackend(),
    )

    assert result.row_count == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "datasets" / "arithmetic.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["arithmetic-000001", "arithmetic-000002"]
    assert "Answer with only the integer result" in rows[0]["prompt"]

    signal_manifest = json.loads((tmp_path / "manifests" / "arithmetic.target-001.manifest.json").read_text())
    run_manifest = json.loads((tmp_path / "manifests" / "target-001.manifest.json").read_text())
    assert signal_manifest["metadata"]["prompt_source"] == "production_spec"
    assert signal_manifest["metadata"]["prompt_preflight"]["require_unique_prompt_text"] is True
    assert signal_manifest["metadata"]["prompt_preflight"]["near_duplicate_prompt_count"] == 0
    assert signal_manifest["metadata"]["planned_prompt_rows"] == 2
    assert signal_manifest["metadata"]["accepted_rows"] == 2
    assert signal_manifest["metadata"]["rejected_rows"] == 0
    assert signal_manifest["metadata"]["rejection_reasons"] == {}
    assert signal_manifest["metadata"]["response_quality"]["accepted_rows"] == 2
    assert run_manifest["metadata"]["prompt_source"] == "production_spec"
    assert run_manifest["metadata"]["target_rows"] == 2
    assert run_manifest["metadata"]["planned_prompt_rows"] == 2
    assert run_manifest["metadata"]["accepted_rows"] == 2
    assert run_manifest["metadata"]["rejected_rows"] == 0
    assert run_manifest["metadata"]["rejection_reasons"] == {}
    assert run_manifest["metadata"]["response_quality"]["accepted_rows"] == 2
    assert run_manifest["metadata"]["rows_per_signal"] == {"arithmetic": 2}
    assert run_manifest["metadata"]["signals"] == ["arithmetic"]
    assert run_manifest["metadata"]["prompt_preflight"]["prompt_count"] == 2
    assert run_manifest["metadata"]["prompt_preflight"]["require_unique_prompt_text"] is True


def test_generate_prompt_spec_multi_signal_run_rejects_near_duplicate_prompts_before_backend_calls(
    tmp_path, monkeypatch
):
    backend_calls = []

    def duplicate_prompt_builder(*, signal, count, start_index=1):
        return [
            build_prompt_record(signal=signal, prompt="Explain autoscaling in one sentence.", index=start_index),
            build_prompt_record(signal=signal, prompt=" explain   autoscaling in one sentence! ", index=start_index + 1),
        ]

    monkeypatch.setattr(
        "slm_synth.distillation.orchestration.build_prompt_spec_records",
        duplicate_prompt_builder,
    )

    with pytest.raises(ValueError, match="near-duplicate prompt text"):
        generate_prompt_spec_multi_signal_run(
            signals=["cloud"],
            count_per_signal=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="target-001",
            max_tokens=512,
            backend_factory=lambda signal: backend_calls.append(signal) or PromptIdBackend(),
        )

    assert backend_calls == []
    assert not (tmp_path / "datasets" / "cloud.jsonl").exists()


def test_generate_seed_multi_signal_run_rejects_missing_count_strategy(tmp_path):
    with pytest.raises(ValueError, match="one of count_per_signal"):
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
    assert not (tmp_path / "datasets" / "arithmetic.batch000001.jsonl").exists()
    assert (tmp_path / "batches" / "arithmetic.batch000001.jsonl").exists()
    assert (tmp_path / "batches" / "arithmetic.batch000002.jsonl").exists()

    rows = [
        json.loads(line)
        for line in (tmp_path / "datasets" / "arithmetic.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["arithmetic-000001", "arithmetic-000002", "arithmetic-000003"]

    manifest = json.loads((tmp_path / "manifests" / "arithmetic.smoke-001.manifest.json").read_text())
    run_manifest = json.loads((tmp_path / "manifests" / "smoke-001.manifest.json").read_text())
    assert manifest["row_count"] == 3
    assert manifest["metadata"]["batch_count"] == 2
    assert manifest["metadata"]["batch_size"] == 2
    assert manifest["metadata"]["concurrency"] == 2
    assert manifest["metadata"]["llm_telemetry"]["batch_count"] == 2
    assert manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 6
    assert manifest["metadata"]["llm_telemetry"]["retryable_provider_retries"] == 3
    assert manifest["metadata"]["llm_telemetry"]["adaptive_peak_in_flight_limit"] == 16
    assert run_manifest["metadata"]["llm_telemetry"]["batch_count"] == 2
    assert run_manifest["metadata"]["llm_telemetry"]["usage"]["total_tokens"] == 6
    assert run_manifest["metadata"]["llm_telemetry"]["retryable_provider_retries"] == 3
    assert run_manifest["metadata"]["llm_telemetry"]["adaptive_peak_in_flight_limit"] == 16


def test_generate_seed_multi_signal_run_reduces_batch_size_after_failure(tmp_path):
    backend = SplitOnLargePromptIdBackend()

    result = generate_seed_multi_signal_run(
        signals=["arithmetic"],
        count_per_signal=3,
        batch_size=3,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="smoke-001",
        max_tokens=512,
        backend_factory=lambda signal: backend,
    )

    assert result.row_count == 3
    assert backend.calls == [3, 1, 1, 1]
    manifest = json.loads((tmp_path / "manifests" / "arithmetic.smoke-001.manifest.json").read_text())
    assert manifest["metadata"]["adaptive_batch_size_observed_minimum"] == 1
    assert manifest["metadata"]["adaptive_batch_size_decreases"] == 1


class OneRejectedCloudBackend(PromptIdBackend):
    def _response_for(self, item):
        if item["id"].endswith("000002"):
            return "ok"
        return "Use autoscaling to add capacity during traffic spikes."


def test_generate_prompt_spec_multi_signal_run_records_response_rejections(tmp_path):
    result = generate_prompt_spec_multi_signal_run(
        signals=["cloud"],
        target_rows=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="target-001",
        max_tokens=512,
        backend_factory=lambda signal: OneRejectedCloudBackend(),
    )

    assert result.row_count == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "datasets" / "cloud.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["cloud-000001"]

    signal_manifest = json.loads((tmp_path / "manifests" / "cloud.target-001.manifest.json").read_text())
    run_manifest = json.loads((tmp_path / "manifests" / "target-001.manifest.json").read_text())
    assert signal_manifest["metadata"]["planned_prompt_rows"] == 2
    assert signal_manifest["metadata"]["accepted_rows"] == 1
    assert signal_manifest["metadata"]["rejected_rows"] == 1
    assert signal_manifest["metadata"]["rejection_reasons"] == {"too_short_response": 1}
    assert signal_manifest["metadata"]["response_quality"]["checked_rows"] == 2
    assert run_manifest["metadata"]["accepted_rows"] == 1
    assert run_manifest["metadata"]["rejected_rows"] == 1
    assert run_manifest["metadata"]["rejection_reasons"] == {"too_short_response": 1}
