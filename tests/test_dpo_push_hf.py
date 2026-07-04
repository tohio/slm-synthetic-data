import json

import pytest

from slm_synth.dpo.push_hf import count_and_validate_jsonl, push_dpo_run


def _dpo_row(row_id="dpo-1"):
    return {
        "id": row_id,
        "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "direct_addition",
            "eval_family": "basic_arithmetic_qa",
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


def test_push_dpo_run_uploads_dataset_and_run_files(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    dataset_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    (dataset_dir / "basic_arithmetic_qa.jsonl").write_text(json.dumps(_dpo_row()) + "\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# DPO dataset\n", encoding="utf-8")
    (run_dir / "coverage.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "run.manifest.json").write_text("{}", encoding="utf-8")

    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("api", token))

        def upload_file(self, **kwargs):
            calls.append(("upload", kwargs["path_in_repo"]))

    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr("slm_synth.dpo.push_hf.HfApi", FakeApi)
    monkeypatch.setattr("slm_synth.dpo.push_hf.create_repo", lambda **kwargs: calls.append(("repo", kwargs)))

    result = push_dpo_run(dataset_dir=dataset_dir, run_dir=run_dir, repo_id="org/dpo")

    assert result == {"repo_id": "org/dpo", "files": ["data/basic_arithmetic_qa.jsonl"], "rows": 1}
    assert ("upload", "data/basic_arithmetic_qa.jsonl") in calls
    assert ("upload", "README.md") in calls
    assert ("upload", "coverage.json") in calls
    assert ("upload", "manifests/run.manifest.json") in calls
