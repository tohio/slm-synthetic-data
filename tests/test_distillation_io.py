import json

import pytest

from slm_synth.distillation.io import write_jsonl, write_manifest, write_run_manifest, write_signal_dataset


def test_write_jsonl_writes_public_rows_only(tmp_path):
    path = tmp_path / "arithmetic.jsonl"
    count = write_jsonl(
        [
            {
                "id": "arithmetic-000001",
                "prompt": "What is 2 + 2?",
                "reasoning": None,
                "response": "4",
            }
        ],
        path,
    )

    assert count == 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "id": "arithmetic-000001",
            "prompt": "What is 2 + 2?",
            "reasoning": None,
            "response": "4",
        }
    ]


def test_write_jsonl_rejects_internal_public_fields(tmp_path):
    with pytest.raises(ValueError, match="forbidden field"):
        write_jsonl(
            [
                {
                    "id": "cloud-000001",
                    "prompt": "Explain autoscaling.",
                    "reasoning": None,
                    "response": "Autoscaling adjusts capacity.",
                    "teacher_model": "internal-only",
                }
            ],
            tmp_path / "cloud.jsonl",
        )


def test_write_signal_dataset_uses_signal_filename(tmp_path):
    path = write_signal_dataset(
        signal=" educational_qa ",
        rows=[
            {
                "id": "educational_qa-000001",
                "prompt": "What is photosynthesis?",
                "reasoning": None,
                "response": "Photosynthesis is how plants convert light into chemical energy.",
            }
        ],
        output_dir=tmp_path,
    )

    assert path == tmp_path / "educational_qa.jsonl"
    assert path.exists()


def test_write_manifest_keeps_teacher_details_outside_public_rows(tmp_path):
    dataset_path = tmp_path / "instruction.jsonl"
    write_jsonl(
        [
            {
                "id": "instruction-000001",
                "prompt": "Rewrite this clearly.",
                "reasoning": None,
                "response": "Clear rewrite.",
            }
        ],
        dataset_path,
    )

    manifest_path = write_manifest(
        manifest_path=tmp_path / "manifests" / "instruction.run-001.json",
        signal="instruction",
        dataset_path=dataset_path,
        row_count=1,
        teacher_model="deepseek/deepseek-v4-flash",
        teacher_provider="openrouter",
        generation_run="run-001",
        token_target="smoke-100k",
        metadata={"retry_count": 0},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["signal"] == "instruction"
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["teacher_model"] == "deepseek/deepseek-v4-flash"
    assert manifest["generation_run"] == "run-001"
    assert manifest["metadata"] == {"retry_count": 0}

    public_row = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert "teacher_model" not in public_row
    assert "teacher_provider" not in public_row
    assert "generation_run" not in public_row


def test_write_manifest_rejects_non_openrouter_provider(tmp_path):
    with pytest.raises(ValueError, match="openrouter"):
        write_manifest(
            manifest_path=tmp_path / "manifest.json",
            signal="code",
            dataset_path=tmp_path / "code.jsonl",
            row_count=0,
            teacher_model="some-model",
            teacher_provider="groq",
            generation_run="run-001",
        )


def test_write_run_manifest_summarizes_signal_outputs(tmp_path):
    manifest_path = write_run_manifest(
        manifest_path=tmp_path / "manifests" / "smoke-001.manifest.json",
        generation_run="smoke-001",
        teacher_model="openai/gpt-4.1-mini",
        teacher_provider="openrouter",
        token_target="100K",
        datasets=[
            {
                "signal": "cloud",
                "dataset_path": tmp_path / "datasets" / "cloud.jsonl",
                "manifest_path": tmp_path / "manifests" / "cloud.smoke-001.manifest.json",
                "row_count": 2,
            },
            {
                "signal": "database",
                "dataset_path": tmp_path / "datasets" / "database.jsonl",
                "manifest_path": tmp_path / "manifests" / "database.smoke-001.manifest.json",
                "row_count": 3,
            },
        ],
        metadata={"signal_count": 2},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation_run"] == "smoke-001"
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert manifest["token_target"] == "100K"
    assert manifest["signals"] == ["cloud", "database"]
    assert manifest["total_rows"] == 5
    assert manifest["metadata"] == {"signal_count": 2}
    assert manifest["datasets"][0]["dataset_path"] == str(tmp_path / "datasets" / "cloud.jsonl")
    assert manifest["datasets"][0]["manifest_path"] == str(tmp_path / "manifests" / "cloud.smoke-001.manifest.json")


def test_write_run_manifest_rejects_duplicate_signals(tmp_path):
    with pytest.raises(ValueError, match="duplicate signal"):
        write_run_manifest(
            manifest_path=tmp_path / "manifest.json",
            generation_run="smoke-001",
            teacher_model="openai/gpt-4.1-mini",
            teacher_provider="openrouter",
            token_target="100K",
            datasets=[
                {
                    "signal": "cloud",
                    "dataset_path": tmp_path / "cloud-a.jsonl",
                    "manifest_path": tmp_path / "cloud-a.manifest.json",
                    "row_count": 1,
                },
                {
                    "signal": "cloud",
                    "dataset_path": tmp_path / "cloud-b.jsonl",
                    "manifest_path": tmp_path / "cloud-b.manifest.json",
                    "row_count": 1,
                },
            ],
        )
