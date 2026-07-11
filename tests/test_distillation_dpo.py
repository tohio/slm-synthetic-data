import json
from pathlib import Path

import pytest

from slm_synth.accepted_target import UnderfilledRunError
from slm_synth.distillation_dpo.card import write_dataset_card
from slm_synth.distillation_dpo.io import read_jsonl
from slm_synth.distillation_dpo.push_hf import (
    count_and_validate_jsonl,
    discover_jsonl_files,
    normalize_repo_id,
    push_distillation_dpo_run,
    repo_id_for_family,
)
from slm_synth.distillation_dpo.report import build_coverage_report
from slm_synth.distillation_dpo.pair_quality import filter_pairs_by_quality
from slm_synth.distillation_dpo.runs import generate_llm_run, normalize_family_pair_counts
from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row


def _row(row_id="distillation-dpo-1"):
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "distillation_dpo_teacher_preference",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "wrong_numeric_answer",
        },
    }


class _EchoDistillationDPOBackend:
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        request = json.loads(prompt.split("Input specs:\n", 1)[1])
        return {
            "data": {
                "items": [
                    {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "chosen": item["reference_chosen"],
                        "rejected": item["reference_rejected"],
                        "metadata": item["metadata"],
                    }
                    for item in request["items"]
                ]
            },
            "telemetry": {
                "model": "fake-teacher",
                "provider": "fake-provider",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.01,
                },
                "retry_count": 0,
                "retryable_provider_retries": 0,
                "retry_sleep_seconds": 0.0,
                "adaptive_peak_in_flight_limit": 1,
                "adaptive_min_in_flight_limit": 1,
                "elapsed_seconds": 0.1,
            },
        }


class _BadDistillationDPOBackend:
    def generate_structured_object_with_metadata(self, *, prompt, schema, schema_name):
        request = json.loads(prompt.split("Input specs:\n", 1)[1])
        return {
            "data": {
                "items": [
                    {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "chosen": [{"role": "assistant", "content": item["prompt"][-1]["content"]}],
                        "rejected": [{"role": "assistant", "content": "different but still rejected"}],
                        "metadata": item["metadata"],
                    }
                    for item in request["items"]
                ]
            },
            "telemetry": {
                "model": "fake-teacher",
                "provider": "fake-provider",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.01,
                },
                "retry_count": 0,
                "retryable_provider_retries": 0,
                "retry_sleep_seconds": 0.0,
                "adaptive_peak_in_flight_limit": 1,
                "adaptive_min_in_flight_limit": 1,
                "elapsed_seconds": 0.1,
            },
        }


def test_distillation_dpo_schema_keeps_lineage_out_of_rows():
    row = _row()
    row["teacher_model"] = "deepseek/deepseek-v4-flash"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_distillation_dpo_row(row)


def test_generate_llm_run_writes_live_distillation_dpo_outputs(tmp_path):
    result = generate_llm_run(
        families=["teacher_response_preference"],
        count_per_family=3,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="deepseek/deepseek-v4-flash",
        generation_run="distillation-dpo-llm-001",
        max_tokens=2048,
        concurrency=2,
        adaptive_initial_in_flight=1,
        adaptive_initial_batch_size=1,
        adaptive_batch_increase_successes=1,
        backend=_EchoDistillationDPOBackend(),
    )

    dataset_path = tmp_path / "datasets" / "teacher_response_preference.jsonl"
    rows = read_jsonl(dataset_path)
    manifest = json.loads(result.manifest_path.read_text())

    assert result.row_count == 3
    assert result.accepted_pairs == 3
    assert len(rows) == 3
    assert manifest["dataset_type"] == "distillation-dpo"
    assert manifest["metadata"]["generation_mode"] == "live_llm_run"
    assert manifest["metadata"]["accepted_pairs"] == 3
    assert manifest["metadata"]["llm_telemetry"]["batch_count"] >= 1
    assert manifest["metadata"]["source_contract"]["target_consumer"] == "slm-distillation"
    assert "teacher_model" not in rows[0]
    assert rows[0]["chosen"] != rows[0]["rejected"]


def test_distillation_dpo_pair_quality_rejects_bad_pairs():
    identical = _row("identical")
    identical["rejected"] = list(identical["chosen"])
    prompt_copy = _row("prompt-copy")
    prompt_copy["chosen"] = [{"role": "assistant", "content": "What is 2 + 2?"}]

    accepted, summary = filter_pairs_by_quality(
        family="teacher_response_preference",
        rows=[_row("good"), identical, prompt_copy],
    )

    assert [row["id"] for row in accepted] == ["good"]
    assert summary.accepted_pairs == 1
    assert summary.rejected_pairs == 2
    assert summary.rejection_reasons["malformed_row"] == 1
    assert summary.rejection_reasons["prompt_copy_pair"] == 1


def test_normalize_family_pair_counts_requires_target_for_each_family():
    assert normalize_family_pair_counts(families=["teacher_response_preference"], target_pairs=3) == {
        "teacher_response_preference": 3
    }
    with pytest.raises(ValueError, match="target_pairs must be a positive integer"):
        normalize_family_pair_counts(families=["teacher_response_preference"], target_pairs=0)


def test_distillation_dpo_report_and_card(tmp_path):
    result = generate_llm_run(
        families=["teacher_response_preference"],
        count_per_family=2,
        batch_size=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="deepseek/deepseek-v4-flash",
        generation_run="distillation-dpo-smoke-001",
        max_tokens=2048,
        concurrency=1,
        adaptive_initial_in_flight=1,
        adaptive_initial_batch_size=1,
        adaptive_batch_increase_successes=1,
        backend=_EchoDistillationDPOBackend(),
    )

    report = build_coverage_report([tmp_path / "datasets"])
    assert report["dataset_type"] == "distillation-dpo"
    assert report["row_count"] == 2
    assert report["failure_modes"]

    card_path = write_dataset_card(
        run_manifest_path=result.manifest_path,
        output_path=tmp_path / "README.md",
        dataset_name="SLM Synthetic Distillation DPO",
    )
    card_text = card_path.read_text()
    assert "Distillation-DPO" in card_text
    assert "controlled_weak" in card_text
    assert "Target pairs" in card_text


def test_distillation_dpo_push_discovers_public_files_only(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    public_path = dataset_dir / "teacher_response_preference.jsonl"
    batch_path = dataset_dir / "teacher_response_preference.batch000001.jsonl"
    scratch_path = dataset_dir / "scratch" / "teacher_response_preference.jsonl"
    public_path.write_text(json.dumps(_row("public")) + "\n", encoding="utf-8")
    batch_path.write_text(json.dumps(_row("batch")) + "\n", encoding="utf-8")
    scratch_path.parent.mkdir()
    scratch_path.write_text(json.dumps(_row("scratch")) + "\n", encoding="utf-8")

    assert discover_jsonl_files(dataset_dir) == [public_path]
    assert count_and_validate_jsonl(public_path) == 1
    assert (
        repo_id_for_family(
            repo_owner="tohio",
            repo_prefix="slm-synthetic-distillation-dpo",
            family="teacher_response_preference",
        )
        == "tohio/slm-synthetic-distillation-dpo-teacher-response-preference"
    )
    assert normalize_repo_id("/tohio/slm-synthetic-distillation-dpo/") == "tohio/slm-synthetic-distillation-dpo"


def test_push_distillation_dpo_run_uploads_exact_repo_id(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "teacher_response_preference.jsonl").write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# Distillation DPO\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "distillation-dpo-smoke-001.manifest.json").write_text("{}", encoding="utf-8")

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs["repo_id"]))
            return ["coverage.json", "manifests/old.manifest.json", "README.md"]

        def create_commit(self, **kwargs):
            calls.append(("commit", kwargs["repo_id"], [operation.path_in_repo for operation in kwargs["operations"]]))

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.distillation_dpo.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.distillation_dpo.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_distillation_dpo_run(
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        repo_id="tohio/slm-synthetic-distillation-dpo",
    )

    repo_id = "tohio/slm-synthetic-distillation-dpo"
    assert result["repo_count"] == 1
    assert result["rows"] == 1
    assert result["repos"]["default"]["repo_id"] == repo_id
    commit_calls = [call for call in calls if call[0] == "commit"]
    assert len(commit_calls) == 1
    paths = commit_calls[0][2]
    assert "data/teacher_response_preference.jsonl" in paths
    assert "README.md" in paths
    assert "artifacts/coverage.json" in paths
    assert "artifacts/manifests/distillation-dpo-smoke-001.manifest.json" in paths
    assert "coverage.json" in paths
    assert "manifests/old.manifest.json" in paths


def test_distillation_dpo_make_targets_are_not_generic_dpo_wrappers():
    makefile = Path("Makefile").read_text()
    block = makefile.split("distillation-dpo-smoke:", 1)[1].split("sft-smoke:", 1)[0]

    assert "slm_synth.distillation_dpo" in block
    assert "generate-llm-run" in block
    assert "--target-pairs $(DISTILLATION_DPO_TARGET_PAIRS)" in block
    assert "$(OPENROUTER_ENV)" in block
    assert "slm_synth.dpo" not in block


def test_distillation_dpo_push_target_uses_exact_repo_id():
    makefile = Path("Makefile").read_text()
    block = makefile.split("distillation-dpo-push:", 1)[1].split("sft-smoke:", 1)[0]

    assert "DISTILLATION_DPO_HF_REPO ?= $(DISTILLATION_DPO_HF_NAMESPACE)/slm-synthetic-distillation-dpo" in makefile
    assert "--repo-id $(DISTILLATION_DPO_HF_REPO)" in block
    assert "--repo-prefix $(DISTILLATION_DPO_HF_PREFIX)" not in block


def test_distillation_dpo_llm_run_fails_underfilled_after_backfill_budget(tmp_path):
    with pytest.raises(UnderfilledRunError, match="distillation-dpo.*underfilled.*remaining=2"):
        generate_llm_run(
            families=["teacher_response_preference"],
            count_per_family=2,
            batch_size=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="deepseek/deepseek-v4-flash",
            generation_run="distillation-dpo-underfilled-001",
            max_tokens=2048,
            concurrency=1,
            adaptive_initial_in_flight=1,
            adaptive_initial_batch_size=1,
            adaptive_batch_increase_successes=1,
            max_backfill_rounds=0,
            backend=_BadDistillationDPOBackend(),
        )

    family_manifest_path = (
        tmp_path
        / "manifests"
        / "teacher_response_preference.distillation-dpo-underfilled-001.manifest.json"
    )
    family_manifest = json.loads(family_manifest_path.read_text())
    run_manifest = json.loads((tmp_path / "manifests" / "distillation-dpo-underfilled-001.manifest.json").read_text())
    assert family_manifest["metadata"]["generation_status"] == "underfilled"
    assert family_manifest["metadata"]["failure_status"] == "failed"
    assert family_manifest["metadata"]["run_failed"] is True
    assert run_manifest["metadata"]["generation_status"] == "underfilled"
    assert run_manifest["metadata"]["failure_status"] == "failed"
    assert run_manifest["metadata"]["run_failed"] is True
    assert run_manifest["metadata"]["remaining_pairs"] == 2
