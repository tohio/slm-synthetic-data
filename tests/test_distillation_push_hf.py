import json

import pytest

from slm_synth.distillation.push_hf import count_and_validate_jsonl, discover_jsonl_files, push_distillation_run


def _distillation_row(row_id="distill-1"):
    return {
        "id": row_id,
        "prompt": "What is 2 + 2?",
        "reasoning": None,
        "response": "4",
    }


def test_count_and_validate_distillation_jsonl_rejects_bad_public_row(tmp_path):
    dataset = tmp_path / "distill.jsonl"
    row = _distillation_row()
    row["teacher_model"] = "openai/gpt-4.1-mini"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected field"):
        count_and_validate_jsonl(dataset)


def test_discover_distillation_jsonl_prefers_final_files_over_batch_shards(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    final_path = dataset_dir / "arithmetic.jsonl"
    batch_path = dataset_dir / "arithmetic.batch000001.jsonl"
    final_path.write_text(json.dumps(_distillation_row("distill-1")) + "\n", encoding="utf-8")
    batch_path.write_text(json.dumps(_distillation_row("distill-2")) + "\n", encoding="utf-8")

    assert discover_jsonl_files(dataset_dir) == [final_path]


@pytest.mark.parametrize(
    "dirname",
    ["scratch", "batches", "partials", "partial", "rejected", "retries", "retry", "provider", "provider_internal", "tmp"],
)
def test_discover_distillation_jsonl_ignores_internal_dirs(tmp_path, dirname):
    dataset_dir = tmp_path / "datasets"
    internal_dir = dataset_dir / dirname
    internal_dir.mkdir(parents=True)
    public_path = dataset_dir / "arithmetic.jsonl"
    internal_path = internal_dir / "arithmetic.jsonl"
    public_path.write_text(json.dumps(_distillation_row("distill-1")) + "\n", encoding="utf-8")
    internal_path.write_text(json.dumps(_distillation_row("distill-2")) + "\n", encoding="utf-8")

    assert discover_jsonl_files(dataset_dir) == [public_path]


def test_push_distillation_run_uploads_dataset_and_run_files(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "arithmetic.jsonl").write_text(json.dumps(_distillation_row()) + "\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# Distillation dataset\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "run.manifest.json").write_text("{}", encoding="utf-8")

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs["path_in_repo"]))

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.distillation.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.distillation.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_distillation_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_id="org/distill")

    assert result == {"repo_id": "org/distill", "files": ["data/arithmetic.jsonl"], "rows": 1}
    assert ("upload", "data/arithmetic.jsonl") in calls
    assert ("upload", "README.md") in calls
    assert ("upload", "coverage.json") in calls
    assert ("upload", "manifests/run.manifest.json") in calls
