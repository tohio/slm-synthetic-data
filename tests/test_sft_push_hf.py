import json
from pathlib import Path

import pytest

from slm_synth.cards import build_dataset_card
from slm_synth.sft.push_hf import count_and_validate_jsonl, discover_jsonl_files, push_sft_run
from slm_synth.sft.report import build_coverage_report, write_coverage_report
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def _sft_row(row_id="sft-1"):
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": f"What is 2 + 2? Request {row_id}."},
            {"role": "assistant", "content": f"Answer for {row_id}."},
        ],
        "metadata": {
            "task_family": "grounded_qa_and_reading",
            "interaction_modes": ["single_turn"],
            "output_mode": "free_text",
            "context_mode": "supplied_passage",
            "difficulty": 1,
            "template_family": "direct_addition",
        },
    }


def test_count_and_validate_sft_jsonl_rejects_bad_public_row(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    row = _sft_row()
    row["teacher_model"] = "openai/gpt-4.1-mini"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field"):
        count_and_validate_jsonl(dataset)


def test_discover_sft_jsonl_rejects_stale_batch_shards(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    final_path = dataset_dir / "grounded_qa_and_reading.jsonl"
    stale_batch_path = dataset_dir / "grounded_qa_and_reading.batch000001.jsonl"
    final_path.write_text(json.dumps(_sft_row("sft-1")) + "\n", encoding="utf-8")
    stale_batch_path.write_text(json.dumps(_sft_row("sft-2")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flat final"):
        discover_jsonl_files(dataset_dir)


def test_discover_sft_jsonl_rejects_batch_only_export(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    batch_path = dataset_dir / "grounded_qa_and_reading.batch000001.jsonl"
    batch_path.write_text(json.dumps(_sft_row("sft-1")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flat final"):
        discover_jsonl_files(dataset_dir)


@pytest.mark.parametrize(
    "dirname",
    ["scratch", "batches", "partials", "partial", "rejected", "retries", "retry", "provider", "provider_internal", "tmp"],
)
def test_discover_sft_jsonl_rejects_nested_compatibility_artifacts(tmp_path, dirname):
    dataset_dir = tmp_path / "datasets"
    internal_dir = dataset_dir / dirname
    internal_dir.mkdir(parents=True)
    public_path = dataset_dir / "grounded_qa_and_reading.jsonl"
    internal_path = internal_dir / "grounded_qa_and_reading.jsonl"
    public_path.write_text(json.dumps(_sft_row("sft-1")) + "\n", encoding="utf-8")
    internal_path.write_text(json.dumps(_sft_row("sft-2")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flat final"):
        discover_jsonl_files(dataset_dir)


def test_count_and_validate_sft_jsonl_requires_filename_family_binding(tmp_path):
    dataset = tmp_path / "creative_writing.jsonl"
    dataset.write_text(json.dumps(_sft_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'creative_writing'"):
        count_and_validate_jsonl(dataset, expected_task_family="creative_writing")


def test_push_sft_run_blocks_before_hf_when_acceptance_report_is_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "grounded_qa_and_reading.jsonl").write_text(
        json.dumps(_sft_row()) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "sft-run.manifest.json").write_text(
        json.dumps(
            {
                "dataset_type": "sft",
                "families": ["grounded_qa_and_reading"],
                "datasets": [],
                "metadata": {
                    "generation_status": "complete",
                    "publish_ready": True,
                    "candidate_rows": 1,
                    "attempted_rows": 1,
                    "accepted_rows": 1,
                    "rejected_rows": 0,
                    "duplicate_rows": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    def unexpected_hf_api(*args, **kwargs):
        raise AssertionError("Hugging Face client must not be created")

    monkeypatch.setattr("slm_synth.sft.push_hf.HfApi", unexpected_hf_api)

    with pytest.raises(FileNotFoundError, match="acceptance report does not exist"):
        push_sft_run(
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            repo_id="tohio/slm-synthetic-sft",
        )


def test_push_sft_run_uploads_all_families_in_one_atomic_commit(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "grounded_qa_and_reading.jsonl").write_text(
        json.dumps(_sft_row("sft-1")) + "\n",
        encoding="utf-8",
    )
    ai_row = _sft_row("sft-2")
    ai_row["metadata"]["task_family"] = "creative_writing"
    ai_row["metadata"]["template_family"] = "character_vignette"
    ai_row["messages"] = [
        {"role": "user", "content": "Write a quiet vignette about a lighthouse keeper finding a brass key."},
        {"role": "assistant", "content": "At dawn, the keeper found a warm brass key beneath the silent lens."},
    ]
    (dataset_dir / "creative_writing.jsonl").write_text(
        json.dumps(ai_row) + "\n",
        encoding="utf-8",
    )
    families = ["creative_writing", "grounded_qa_and_reading"]
    (run_dir / "README.md").write_text(
        build_dataset_card("sft", total=2, signals=families),
        encoding="utf-8",
    )
    grounded_batch_manifest = manifest_dir / "grounded_qa_and_reading.batch000001.sft-run.manifest.json"
    creative_batch_manifest = manifest_dir / "creative_writing.batch000001.sft-run.manifest.json"
    scores = {
        "correctness": 4,
        "grounding": 4,
        "instruction_adherence": 4,
        "completeness": 4,
        "coherence": 4,
    }
    grounded_batch_manifest.write_text(
        json.dumps(
            {
                "metadata": {
                    "quality_adjudication": {
                        "sft-1": {"accepted": True, "scores": scores, "constraint_results": []}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    creative_batch_manifest.write_text(
        json.dumps(
            {
                "metadata": {
                    "quality_adjudication": {
                        "sft-2": {"accepted": True, "scores": scores, "constraint_results": []}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    run_manifest = manifest_dir / "sft-run.manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "dataset_type": "sft",
                "families": families,
                "datasets": [
                    {"batch_manifests": [str(creative_batch_manifest)]},
                    {"batch_manifests": [str(grounded_batch_manifest)]},
                ],
                "metadata": {
                    "generation_status": "complete",
                    "publish_ready": True,
                    "attempted_rows": 2,
                    "accepted_rows": 2,
                    "rejected_rows": 0,
                    "duplicate_rows": 0,
                    "candidate_rows": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    coverage = build_coverage_report(
        [dataset_dir],
        holdout_registry=HoldoutRegistry([]),
        run_manifest=run_manifest,
    )
    write_coverage_report(report=coverage, path=run_dir / "coverage.json")

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs["repo_id"]))
            return [
                "coverage.json",
                "manifests/old.manifest.json",
                "README.md",
                "data/removed_family.jsonl",
                "artifacts/manifests/obsolete.manifest.json",
            ]

        def create_commit(self, **kwargs):
            readme_operation = next(
                operation
                for operation in kwargs["operations"]
                if operation.path_in_repo == "README.md"
            )
            calls.append(
                (
                    "commit",
                    kwargs["repo_id"],
                    [operation.path_in_repo for operation in kwargs["operations"]],
                    readme_operation.path_or_fileobj.decode("utf-8"),
                )
            )

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.sft.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.sft.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_sft_run(
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        repo_id="tohio/slm-synthetic-sft",
    )

    assert result == {
        "repo_id": "tohio/slm-synthetic-sft",
        "files": [
            "data/creative_writing.jsonl",
            "data/grounded_qa_and_reading.jsonl",
        ],
        "families": families,
        "family_count": 2,
        "rows": 2,
    }
    assert ("repo", {"repo_id": "tohio/slm-synthetic-sft", "repo_type": "dataset", "private": False, "exist_ok": True}) in calls
    commit_calls = [call for call in calls if call[0] == "commit"]
    assert len(commit_calls) == 1
    assert commit_calls[0][1] == "tohio/slm-synthetic-sft"
    operations = commit_calls[0][2]
    assert "data/grounded_qa_and_reading.jsonl" in operations
    assert "data/creative_writing.jsonl" in operations
    assert "data/removed_family.jsonl" in operations
    assert "artifacts/manifests/obsolete.manifest.json" in operations
    assert "README.md" in operations
    assert "artifacts/coverage.json" in operations
    assert "artifacts/manifests/sft-run.manifest.json" in operations
    assert "artifacts/manifests/grounded_qa_and_reading.batch000001.sft-run.manifest.json" in operations
    assert "artifacts/manifests/creative_writing.batch000001.sft-run.manifest.json" in operations
    assert "coverage.json" in operations
    assert "manifests/old.manifest.json" in operations
    uploaded_readme = commit_calls[0][3]
    assert "config_name: default" in uploaded_readme
    assert "config_name: creative_writing" in uploaded_readme
    assert "path: data/creative_writing.jsonl" in uploaded_readme


def test_sft_push_make_target_uses_one_exact_repository():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    block = makefile.split("\nsft-push:", 1)[1].split("\ndpo-smoke:", 1)[0]

    assert "SFT_HF_REPO" in block
    assert "--repo-id $(SFT_HF_REPO)" in block
    assert "--repo-owner" not in block
    assert "--repo-prefix" not in block
    assert "SFT_HF_REPO ?=" in makefile
    assert "/slm-synthetic-sft" in makefile
