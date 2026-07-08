import json

import pytest

from slm_synth.sft.push_hf import count_and_validate_jsonl, discover_jsonl_files, push_sft_run, repo_id_for_family


def _sft_row(row_id="sft-1"):
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "direct_addition",
            "eval_family": "basic_arithmetic_qa",
        },
    }


def test_count_and_validate_sft_jsonl_rejects_bad_public_row(tmp_path):
    dataset = tmp_path / "sft.jsonl"
    row = _sft_row()
    row["teacher_model"] = "openai/gpt-4.1-mini"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field"):
        count_and_validate_jsonl(dataset)


def test_repo_id_for_sft_family_uses_slm_sft_prefix():
    assert (
        repo_id_for_family(repo_owner="tohio", repo_prefix="slm-sft", family="basic_arithmetic_qa")
        == "tohio/slm-sft-basic-arithmetic-qa"
    )


def test_discover_sft_jsonl_prefers_final_files_over_stale_batches(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    final_path = dataset_dir / "basic_arithmetic_qa.jsonl"
    stale_batch_path = dataset_dir / "basic_arithmetic_qa.batch000001.jsonl"
    final_path.write_text(json.dumps(_sft_row("sft-1")) + "\n", encoding="utf-8")
    stale_batch_path.write_text(json.dumps(_sft_row("sft-2")) + "\n", encoding="utf-8")

    assert discover_jsonl_files(dataset_dir) == [final_path]


def test_discover_sft_jsonl_keeps_batch_shards_without_final_file(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    batch_path = dataset_dir / "basic_arithmetic_qa.batch000001.jsonl"
    batch_path.write_text(json.dumps(_sft_row("sft-1")) + "\n", encoding="utf-8")

    assert discover_jsonl_files(dataset_dir) == [batch_path]


@pytest.mark.parametrize(
    "dirname",
    ["scratch", "batches", "partials", "partial", "rejected", "retries", "retry", "provider", "provider_internal", "tmp"],
)
def test_discover_sft_jsonl_ignores_internal_dirs(tmp_path, dirname):
    dataset_dir = tmp_path / "datasets"
    internal_dir = dataset_dir / dirname
    internal_dir.mkdir(parents=True)
    public_path = dataset_dir / "basic_arithmetic_qa.jsonl"
    internal_path = internal_dir / "basic_arithmetic_qa.jsonl"
    public_path.write_text(json.dumps(_sft_row("sft-1")) + "\n", encoding="utf-8")
    internal_path.write_text(json.dumps(_sft_row("sft-2")) + "\n", encoding="utf-8")

    assert discover_jsonl_files(dataset_dir) == [public_path]


def test_push_sft_run_uploads_one_repo_per_family(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "basic_arithmetic_qa.batch000001.jsonl").write_text(
        json.dumps(_sft_row("sft-1")) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "basic_arithmetic_qa.batch000002.jsonl").write_text(
        json.dumps(_sft_row("sft-2")) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "ai_concept_explanation.batch000001.jsonl").write_text(
        json.dumps(_sft_row("sft-3")) + "\n",
        encoding="utf-8",
    )
    (run_dir / "README.md").write_text("# SFT dataset\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "basic_arithmetic_qa.batch000001.sft-run.manifest.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "sft-run.manifest.json").write_text("{}", encoding="utf-8")

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs["repo_id"], kwargs["path_in_repo"]))

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.sft.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.sft.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_sft_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_owner="tohio")

    assert result["repo_count"] == 2
    assert result["rows"] == 3
    assert result["repos"]["basic_arithmetic_qa"]["repo_id"] == "tohio/slm-sft-basic-arithmetic-qa"
    assert result["repos"]["ai_concept_explanation"]["repo_id"] == "tohio/slm-sft-ai-concept-explanation"
    assert ("repo", {"repo_id": "tohio/slm-sft-basic-arithmetic-qa", "repo_type": "dataset", "private": False, "exist_ok": True}) in calls
    assert ("upload", "tohio/slm-sft-basic-arithmetic-qa", "data/basic_arithmetic_qa.batch000001.jsonl") in calls
    assert ("upload", "tohio/slm-sft-basic-arithmetic-qa", "data/basic_arithmetic_qa.batch000002.jsonl") in calls
    assert ("upload", "tohio/slm-sft-ai-concept-explanation", "data/ai_concept_explanation.batch000001.jsonl") in calls
    assert ("upload", "tohio/slm-sft-basic-arithmetic-qa", "README.md") in calls
    assert ("upload", "tohio/slm-sft-basic-arithmetic-qa", "coverage.json") in calls
    assert (
        "upload",
        "tohio/slm-sft-basic-arithmetic-qa",
        "manifests/basic_arithmetic_qa.batch000001.sft-run.manifest.json",
    ) in calls
    assert ("upload", "tohio/slm-sft-basic-arithmetic-qa", "manifests/sft-run.manifest.json") in calls
