import json
from pathlib import Path

import pytest

from slm_synth.accepted_target import UnderfilledRunError
from slm_synth.distillation_dpo.card import write_dataset_card
from slm_synth.distillation_dpo.io import read_jsonl
from slm_synth.distillation_dpo.push_hf import (
    count_and_validate_jsonl,
    discover_jsonl_files,
    push_distillation_dpo_run,
    repo_id_for_family,
)
from slm_synth.distillation_dpo.report import build_coverage_report
from slm_synth.distillation_dpo.pair_quality import filter_pairs_by_quality
from slm_synth.distillation_dpo.runs import materialize_production_run, materialize_seed_run, normalize_family_pair_counts
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


def test_distillation_dpo_schema_keeps_lineage_out_of_rows():
    row = _row()
    row["teacher_model"] = "deepseek/deepseek-v4-flash"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_distillation_dpo_row(row)


def test_materialize_seed_run_writes_isolated_manifest_and_public_rows(tmp_path):
    result = materialize_seed_run(
        families=["teacher_response_preference"],
        count_per_family=3,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="deepseek/deepseek-v4-flash",
        generation_run="distillation-dpo-smoke-001",
    )

    assert result.row_count == 3
    assert result.families == ("teacher_response_preference",)
    dataset_path = tmp_path / "datasets" / "teacher_response_preference.jsonl"
    assert result.results[0].dataset_path == dataset_path
    rows = read_jsonl(dataset_path)
    assert len(rows) == 3
    assert "teacher_model" not in rows[0]
    assert rows[0]["chosen"] != rows[0]["rejected"]

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_type"] == "distillation-dpo"
    assert manifest["chosen_source"] == "teacher"
    assert manifest["rejected_source"] == "controlled_weak"
    assert manifest["teacher_model"] == "deepseek/deepseek-v4-flash"
    assert manifest["target_consumer"] == "slm-distillation"
    assert manifest["datasets"][0]["dataset_path"] == str(dataset_path)


def test_materialize_production_run_uses_target_pairs_and_controlled_weak_contract(tmp_path):
    result = materialize_production_run(
        families=["teacher_response_preference"],
        target_pairs=12,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="deepseek/deepseek-v4-flash",
        generation_run="distillation-dpo-target-001",
    )

    dataset_path = tmp_path / "datasets" / "teacher_response_preference.jsonl"
    rows = read_jsonl(dataset_path)
    manifest = json.loads(result.manifest_path.read_text())

    assert result.target_pairs == 12
    assert result.planned_pairs == 12
    assert result.accepted_pairs == 12
    assert result.rejected_pairs == 0
    assert len(rows) == 12
    assert manifest["dataset_type"] == "distillation-dpo"
    assert manifest["chosen_source"] == "teacher"
    assert manifest["rejected_source"] == "controlled_weak"
    assert manifest["target_consumer"] == "slm-distillation"
    assert manifest["metadata"]["generation_mode"] == "production_controlled_weak_run"
    assert manifest["metadata"]["target_pairs"] == 12
    assert manifest["metadata"]["planned_pairs"] == 12
    assert manifest["metadata"]["accepted_pairs"] == 12
    assert manifest["metadata"]["rejected_pairs"] == 0
    assert manifest["metadata"]["source_contract"]["rejected_source"] == "controlled_weak"
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
    result = materialize_seed_run(
        families=["teacher_response_preference"],
        count_per_family=2,
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="deepseek/deepseek-v4-flash",
        generation_run="distillation-dpo-smoke-001",
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
            repo_prefix="distillation-dpo",
            family="teacher_response_preference",
        )
        == "tohio/distillation-dpo-teacher-response-preference"
    )


def test_push_distillation_dpo_run_uploads_family_repo(tmp_path, monkeypatch):
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

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs["repo_id"], kwargs["path_in_repo"]))

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.distillation_dpo.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.distillation_dpo.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_distillation_dpo_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_owner="tohio")

    repo_id = "tohio/distillation-dpo-teacher-response-preference"
    assert result["repo_count"] == 1
    assert result["rows"] == 1
    assert result["repos"]["teacher_response_preference"]["repo_id"] == repo_id
    assert ("upload", repo_id, "data/teacher_response_preference.jsonl") in calls
    assert ("upload", repo_id, "README.md") in calls
    assert ("upload", repo_id, "coverage.json") in calls
    assert ("upload", repo_id, "manifests/distillation-dpo-smoke-001.manifest.json") in calls


def test_distillation_dpo_make_targets_are_not_generic_dpo_wrappers():
    makefile = Path("Makefile").read_text()
    block = makefile.split("distillation-dpo-smoke:", 1)[1].split("sft-smoke:", 1)[0]

    assert "slm_synth.distillation_dpo" in block
    assert "materialize-production-run" in block
    assert "--target-pairs $(DISTILLATION_DPO_TARGET_PAIRS)" in block
    assert "slm_synth.dpo" not in block


def test_distillation_dpo_seed_run_fails_underfilled_after_backfill_budget(tmp_path, monkeypatch):
    def build_bad_rows(*, family, count, start_index):
        rows = []
        for offset in range(count):
            row = _row(f"bad-{start_index + offset:06d}")
            row["chosen"] = [{"role": "assistant", "content": "same answer"}]
            row["rejected"] = [{"role": "assistant", "content": "same answer"}]
            rows.append(row)
        return rows

    monkeypatch.setattr("slm_synth.distillation_dpo.runs.build_seed_rows", build_bad_rows)

    with pytest.raises(UnderfilledRunError, match="distillation-dpo.*underfilled.*remaining=2"):
        materialize_seed_run(
            families=["teacher_response_preference"],
            count_per_family=2,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="deepseek/deepseek-v4-flash",
            generation_run="distillation-dpo-underfilled-001",
            max_backfill_rounds=0,
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
