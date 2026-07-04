import json

import pytest

from slm_synth.sft.push_hf import count_and_validate_jsonl, push_sft_run


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


def test_push_sft_run_uploads_dataset_and_run_files(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "basic_arithmetic_qa.jsonl").write_text(json.dumps(_sft_row()) + "\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# SFT dataset\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "run.manifest.json").write_text("{}", encoding="utf-8")

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs["path_in_repo"]))

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.sft.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.sft.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_sft_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_id="org/sft")

    assert result == {"repo_id": "org/sft", "files": ["data/basic_arithmetic_qa.jsonl"], "rows": 1}
    assert ("upload", "data/basic_arithmetic_qa.jsonl") in calls
    assert ("upload", "README.md") in calls
    assert ("upload", "coverage.json") in calls
    assert ("upload", "manifests/run.manifest.json") in calls
