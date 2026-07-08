import json

import pytest

from slm_synth.distillation.push_hf import (
    count_and_validate_jsonl,
    discover_jsonl_files,
    discover_run_manifest,
    push_distillation_run,
)


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

    with pytest.raises(ValueError, match="forbidden field"):
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


def test_discover_run_manifest_selects_only_run_level_manifest(tmp_path):
    run_dir = tmp_path / "run"
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    run_manifest = manifest_dir / "run.manifest.json"
    run_manifest.write_text(json.dumps({"datasets": []}), encoding="utf-8")
    (manifest_dir / "arithmetic.run.manifest.json").write_text(json.dumps({"signal": "arithmetic"}), encoding="utf-8")
    (manifest_dir / "arithmetic.batch000001.run.manifest.json").write_text(
        json.dumps({"signal": "arithmetic", "batch_number": 1}), encoding="utf-8"
    )

    assert discover_run_manifest(run_dir) == run_manifest


def test_push_distillation_run_uploads_only_public_surface_files(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    batch_dir = run_dir / "batches"
    rejected_dir = run_dir / "rejected"
    retry_dir = run_dir / "retries"
    provider_dir = run_dir / "provider_internal"
    tmp_dir = run_dir / "tmp"
    for path in [dataset_dir, manifest_dir, batch_dir, rejected_dir, retry_dir, provider_dir, tmp_dir]:
        path.mkdir(parents=True)
    (dataset_dir / "arithmetic.jsonl").write_text(json.dumps(_distillation_row()) + "\n", encoding="utf-8")
    (batch_dir / "arithmetic.batch000001.jsonl").write_text(
        json.dumps(_distillation_row("distill-batch")) + "\n", encoding="utf-8"
    )
    (rejected_dir / "bad.jsonl").write_text(json.dumps(_distillation_row("distill-rejected")) + "\n", encoding="utf-8")
    (retry_dir / "retry.jsonl").write_text(json.dumps(_distillation_row("distill-retry")) + "\n", encoding="utf-8")
    (provider_dir / "payload.jsonl").write_text(json.dumps(_distillation_row("distill-provider")) + "\n", encoding="utf-8")
    (tmp_dir / "tmp.jsonl").write_text(json.dumps(_distillation_row("distill-tmp")) + "\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# Distillation dataset\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "run.manifest.json").write_text(json.dumps({"datasets": []}), encoding="utf-8")
    (manifest_dir / "arithmetic.run.manifest.json").write_text(json.dumps({"signal": "arithmetic"}), encoding="utf-8")
    (manifest_dir / "arithmetic.batch000001.run.manifest.json").write_text(
        json.dumps({"signal": "arithmetic", "batch_number": 1}), encoding="utf-8"
    )

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
    assert [call for call in calls if call[0] == "upload"] == [
        ("upload", "data/arithmetic.jsonl"),
        ("upload", "README.md"),
        ("upload", "coverage.json"),
        ("upload", "manifests/run.manifest.json"),
    ]


@pytest.mark.parametrize(
    ("missing_path", "message"),
    [
        ("README.md", "README.md"),
        ("coverage.json", "coverage.json"),
        ("manifests/run.manifest.json", "run.manifest.json"),
    ],
)
def test_push_distillation_run_requires_public_surface_files(tmp_path, monkeypatch, missing_path, message):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "arithmetic.jsonl").write_text(json.dumps(_distillation_row()) + "\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# Distillation dataset\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "run.manifest.json").write_text(json.dumps({"datasets": []}), encoding="utf-8")
    (run_dir / missing_path).unlink()

    class FakeApi:
        def __init__(self, token):
            pass

        def upload_file(self, **kwargs):
            pass

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.distillation.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.distillation.push_hf.create_repo", lambda **kwargs: None)

    with pytest.raises(FileNotFoundError, match=message):
        push_distillation_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_id="org/distill")
