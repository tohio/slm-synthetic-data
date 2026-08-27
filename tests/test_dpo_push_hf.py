import json
from pathlib import Path

import pytest

from slm_synth.cards import build_dataset_card
from slm_synth.runtime.reporting import estimate_dpo_tokens
from slm_synth.dpo.push_hf import count_and_validate_jsonl, discover_jsonl_files, push_dpo_run
from slm_synth.dpo.report import build_coverage_report, write_coverage_report
from slm_synth.taxonomy.holdouts import HoldoutRegistry


def _dpo_row(row_id="dpo-1"):
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": f"What is the answer for {row_id}?"}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
        "metadata": {
            "task_family": "applied_math_and_reasoning",
            "interaction_modes": ["single_turn"],
            "output_mode": "concise",
            "context_mode": "self_contained",
            "preference_dimension": "factual_accuracy",
            "difficulty": 1,
            "template_family": "direct_addition",
            "failure_mode": "wrong_numeric_answer",
        },
    }


def test_count_and_validate_dpo_jsonl_rejects_bad_public_row(tmp_path):
    dataset = tmp_path / "dpo.jsonl"
    row = _dpo_row()
    row["teacher_model"] = "openai/gpt-4.1-mini"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field"):
        count_and_validate_jsonl(dataset)


def test_discover_dpo_jsonl_rejects_stale_batch_shards(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    final_path = dataset_dir / "factual_accuracy.jsonl"
    stale_batch_path = dataset_dir / "factual_accuracy.batch000001.jsonl"
    final_path.write_text(json.dumps(_dpo_row("dpo-1")) + "\n", encoding="utf-8")
    stale_batch_path.write_text(json.dumps(_dpo_row("dpo-2")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flat final"):
        discover_jsonl_files(dataset_dir)


def test_discover_dpo_jsonl_rejects_batch_only_export(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    batch_path = dataset_dir / "factual_accuracy.batch000001.jsonl"
    batch_path.write_text(json.dumps(_dpo_row("dpo-1")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flat final"):
        discover_jsonl_files(dataset_dir)


@pytest.mark.parametrize(
    "dirname",
    ["scratch", "batches", "partials", "partial", "rejected", "retries", "retry", "provider", "provider_internal", "tmp"],
)
def test_discover_dpo_jsonl_rejects_nested_compatibility_artifacts(tmp_path, dirname):
    dataset_dir = tmp_path / "datasets"
    internal_dir = dataset_dir / dirname
    internal_dir.mkdir(parents=True)
    public_path = dataset_dir / "factual_accuracy.jsonl"
    internal_path = internal_dir / "factual_accuracy.jsonl"
    public_path.write_text(json.dumps(_dpo_row("dpo-1")) + "\n", encoding="utf-8")
    internal_path.write_text(json.dumps(_dpo_row("dpo-2")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="flat final"):
        discover_jsonl_files(dataset_dir)


def test_count_and_validate_dpo_jsonl_requires_filename_dimension_binding(tmp_path):
    dataset = tmp_path / "groundedness.jsonl"
    dataset.write_text(json.dumps(_dpo_row()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'groundedness'"):
        count_and_validate_jsonl(
            dataset, expected_preference_dimension="groundedness"
        )


def test_push_dpo_run_uploads_all_families_in_one_atomic_commit(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "factual_accuracy.jsonl").write_text(
        json.dumps(_dpo_row("dpo-1")) + "\n",
        encoding="utf-8",
    )
    ai_row = _dpo_row("dpo-2")
    ai_row["metadata"]["preference_dimension"] = "helpfulness_and_completeness"
    (dataset_dir / "helpfulness_and_completeness.jsonl").write_text(
        json.dumps(ai_row) + "\n",
        encoding="utf-8",
    )
    families = ["factual_accuracy", "helpfulness_and_completeness"]
    (run_dir / "README.md").write_text(
        build_dataset_card("dpo", total=2, signals=families), encoding="utf-8"
    )
    factual_batch_manifest = manifest_dir / "factual_accuracy.batch000001.dpo-run.manifest.json"
    helpful_batch_manifest = manifest_dir / "helpfulness_and_completeness.batch000001.dpo-run.manifest.json"
    for path, row_id in (
        (factual_batch_manifest, "dpo-1"),
        (helpful_batch_manifest, "dpo-2"),
    ):
        path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "deterministic_output_validation": {
                            row_id: {
                                "status": "passed",
                                "declared_constraint_count": 0,
                                "chosen": {
                                    "status": "passed",
                                    "declared_constraint_count": 0,
                                    "checks": [],
                                },
                                "rejected": {
                                    "status": "passed",
                                    "declared_constraint_count": 0,
                                    "checks": [],
                                },
                            }
                        },
                        "quality_adjudication": {
                            row_id: {
                                "id": row_id,
                                "assessable": True,
                                "judge_accepted": True,
                                "reviewed": True,
                                "reviewer_agreed": True,
                                "accepted": True,
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
    run_manifest = manifest_dir / "dpo-run.manifest.json"
    run_manifest.write_text(json.dumps({
        "dataset_type": "dpo",
        "preference_dimensions": families,
        "datasets": [
            {"batch_manifests": [str(factual_batch_manifest)]},
            {"batch_manifests": [str(helpful_batch_manifest)]},
        ],
        "metadata": {
            "publish_ready": True,
            "candidate_pairs": 2,
            "attempted_pairs": 2,
            "accepted_pairs": 2,
            "estimated_tokens": estimate_dpo_tokens(_dpo_row("dpo-1")) + estimate_dpo_tokens(ai_row),
            "rejected_pairs": 0,
            "duplicate_pairs": 0,
            "candidate_pairs_per_dimension": {"factual_accuracy": 1, "helpfulness_and_completeness": 1},
        },
    }), encoding="utf-8")
    coverage = build_coverage_report(
        [dataset_dir], holdout_registry=HoldoutRegistry([]), run_manifest=run_manifest,
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
                operation for operation in kwargs["operations"] if operation.path_in_repo == "README.md"
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
    monkeypatch.setattr("slm_synth.dpo.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.dpo.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_dpo_run(
        dataset_dir=dataset_dir, run_dir=run_dir, repo_id="tohio/slm-synthetic-dpo"
    )

    assert result == {
        "repo_id": "tohio/slm-synthetic-dpo",
        "files": ["data/factual_accuracy.jsonl", "data/helpfulness_and_completeness.jsonl"],
        "preference_dimensions": families,
        "preference_dimension_count": 2,
        "pairs": 2,
    }
    assert ("repo", {"repo_id": "tohio/slm-synthetic-dpo", "repo_type": "dataset", "private": False, "exist_ok": True}) in calls
    commit_calls = [call for call in calls if call[0] == "commit"]
    assert len(commit_calls) == 1
    assert commit_calls[0][1] == "tohio/slm-synthetic-dpo"
    operations = commit_calls[0][2]
    assert "data/factual_accuracy.jsonl" in operations
    assert "data/helpfulness_and_completeness.jsonl" in operations
    assert "data/removed_family.jsonl" in operations
    assert "artifacts/manifests/obsolete.manifest.json" in operations
    assert "README.md" in operations
    assert "artifacts/coverage.json" in operations
    assert "artifacts/manifests/dpo-run.manifest.json" in operations
    assert "artifacts/manifests/factual_accuracy.batch000001.dpo-run.manifest.json" in operations
    assert "artifacts/manifests/helpfulness_and_completeness.batch000001.dpo-run.manifest.json" in operations
    assert "coverage.json" in operations
    assert "manifests/old.manifest.json" in operations
    uploaded_readme = commit_calls[0][3]
    assert "config_name: default" in uploaded_readme
    assert "config_name: helpfulness_and_completeness" in uploaded_readme
    assert "path: data/helpfulness_and_completeness.jsonl" in uploaded_readme


def test_push_dpo_run_blocks_missing_acceptance_report_before_token_lookup(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "factual_accuracy.jsonl").write_text(
        json.dumps(_dpo_row()) + "\n", encoding="utf-8",
    )
    (manifest_dir / "run.manifest.json").write_text(json.dumps({
        "dataset_type": "dpo",
        "preference_dimensions": ["factual_accuracy"],
        "metadata": {
            "publish_ready": True,
            "candidate_pairs": 1,
            "attempted_pairs": 1,
            "accepted_pairs": 1,
            "estimated_tokens": estimate_dpo_tokens(_dpo_row()),
        },
    }), encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    with pytest.raises(FileNotFoundError, match="acceptance report"):
        push_dpo_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_id="tohio/slm-synthetic-dpo")


def test_dpo_push_make_target_uses_one_exact_repository():
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")
    block = makefile.split("\ndpo-push:", 1)[1].split("\nmodel-qualify:", 1)[0]

    assert "DPO_HF_REPO" in block
    assert "--repo-id $(DPO_HF_REPO)" in block
    assert "--repo-owner" not in block
    assert "--repo-prefix" not in block
    assert "DPO_HF_REPO ?=" in makefile
    assert "/slm-synthetic-dpo" in makefile
