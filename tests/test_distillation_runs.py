import json

import pytest

from slm_synth.distillation.runs import default_manifest_path, materialize_teacher_batch


def test_default_manifest_path_uses_signal_and_generation_run(tmp_path):
    path = default_manifest_path(
        manifest_dir=tmp_path / "manifests",
        signal="arithmetic",
        generation_run="smoke-001",
    )

    assert path == tmp_path / "manifests" / "arithmetic.smoke-001.manifest.json"


def test_materialize_teacher_batch_writes_public_dataset_and_manifest(tmp_path):
    result = materialize_teacher_batch(
        signal="arithmetic",
        prompt_records=[
            {
                "id": "arithmetic-000001",
                "prompt": "What is 2 + 2?",
                "signal": "arithmetic",
                "metadata": {"difficulty": "easy"},
            }
        ],
        teacher_response={
            "items": [
                {
                    "id": "arithmetic-000001",
                    "reasoning": None,
                    "response": "4",
                }
            ]
        },
        output_dir=tmp_path / "datasets",
        manifest_dir=tmp_path / "manifests",
        teacher_model="openai/gpt-4.1-mini",
        generation_run="smoke-001",
        token_target="100K",
        metadata={"batch_size": 1},
    )

    assert result.signal == "arithmetic"
    assert result.row_count == 1
    assert result.dataset_path == tmp_path / "datasets" / "arithmetic.jsonl"
    assert result.manifest_path == tmp_path / "manifests" / "arithmetic.smoke-001.manifest.json"

    public_row = json.loads(result.dataset_path.read_text(encoding="utf-8").strip())
    assert public_row == {
        "id": "arithmetic-000001",
        "prompt": "What is 2 + 2?",
        "reasoning": None,
        "response": "4",
    }
    assert "signal" not in public_row
    assert "metadata" not in public_row
    assert "teacher_model" not in public_row
    assert "teacher_provider" not in public_row
    assert "generation_run" not in public_row

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["signal"] == "arithmetic"
    assert manifest["row_count"] == 1
    assert manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["generation_run"] == "smoke-001"
    assert manifest["token_target"] == "100K"
    assert manifest["metadata"] == {"prompt_count": 1, "batch_size": 1}


def test_materialize_teacher_batch_rejects_prompt_signal_mismatch(tmp_path):
    with pytest.raises(ValueError, match="signal mismatch"):
        materialize_teacher_batch(
            signal="cloud",
            prompt_records=[{"id": "arithmetic-000001", "prompt": "What is 2 + 2?", "signal": "arithmetic"}],
            teacher_response={"items": [{"id": "arithmetic-000001", "reasoning": None, "response": "4"}]},
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="smoke-001",
        )


def test_materialize_teacher_batch_rejects_missing_teacher_id(tmp_path):
    with pytest.raises(ValueError, match="expected 1 item"):
        materialize_teacher_batch(
            signal="instruction",
            prompt_records=[{"id": "instruction-000001", "prompt": "Summarize this.", "signal": "instruction"}],
            teacher_response={"items": []},
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="openai/gpt-4.1-mini",
            generation_run="smoke-001",
        )


def test_materialize_teacher_batch_rejects_non_openrouter_provider(tmp_path):
    with pytest.raises(ValueError, match="teacher_provider must be 'openrouter'"):
        materialize_teacher_batch(
            signal="database",
            prompt_records=[{"id": "database-000001", "prompt": "Write a SELECT.", "signal": "database"}],
            teacher_response={"items": [{"id": "database-000001", "reasoning": None, "response": "SELECT 1;"}]},
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="some-model",
            teacher_provider="groq",
            generation_run="smoke-001",
        )
