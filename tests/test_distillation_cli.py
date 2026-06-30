import json

import pytest

from slm_synth.distillation.cli import main


def test_build_seed_prompts_cli_writes_internal_prompt_records(tmp_path, capsys):
    output = tmp_path / "prompts.jsonl"

    assert main(["build-seed-prompts", "--signal", "arithmetic", "--count", "2", "--output", str(output)]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["arithmetic-000001", "arithmetic-000002"]
    assert all(row["signal"] == "arithmetic" for row in rows)
    assert all("metadata" in row for row in rows)

    captured = capsys.readouterr()
    assert "wrote 2 arithmetic prompt record" in captured.out


def test_render_teacher_prompt_cli_writes_prompt_with_only_teacher_request_items(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    teacher_prompt = tmp_path / "teacher_prompt.txt"
    main(["build-seed-prompts", "--signal", "instruction", "--count", "1", "--output", str(prompts)])

    assert main(["render-teacher-prompt", "--signal", "instruction", "--prompts", str(prompts), "--output", str(teacher_prompt)]) == 0

    text = teacher_prompt.read_text(encoding="utf-8")
    assert "instruction-000001" in text
    assert '"prompt"' in text
    assert '"metadata"' not in text
    assert '"teacher_model"' not in text
    assert '"teacher_provider"' not in text


def test_materialize_batch_cli_writes_public_dataset_and_manifest(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    teacher_response = tmp_path / "teacher_response.json"
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"

    main(["build-seed-prompts", "--signal", "cloud", "--count", "1", "--output", str(prompts)])
    teacher_response.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "cloud-000001",
                        "reasoning": None,
                        "response": "Use autoscaling to add capacity when traffic increases.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "materialize-batch",
                "--signal",
                "cloud",
                "--prompts",
                str(prompts),
                "--teacher-response",
                str(teacher_response),
                "--output-dir",
                str(output_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "smoke-001",
                "--token-target",
                "100K",
            ]
        )
        == 0
    )

    public_row = json.loads((output_dir / "cloud.jsonl").read_text(encoding="utf-8").strip())
    assert public_row == {
        "id": "cloud-000001",
        "prompt": "Explain one practical use of autoscaling in a cloud application.",
        "reasoning": None,
        "response": "Use autoscaling to add capacity when traffic increases.",
    }
    assert "signal" not in public_row
    assert "metadata" not in public_row

    manifest = json.loads((manifest_dir / "cloud.smoke-001.manifest.json").read_text(encoding="utf-8"))
    assert manifest["teacher_provider"] == "openrouter"
    assert manifest["teacher_model"] == "openai/gpt-4.1-mini"
    assert manifest["metadata"]["source_prompt_file"] == str(prompts)


def test_materialize_batch_cli_rejects_non_object_teacher_response(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    teacher_response = tmp_path / "teacher_response.json"
    main(["build-seed-prompts", "--signal", "database", "--count", "1", "--output", str(prompts)])
    teacher_response.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="teacher response file must contain a JSON object"):
        main(
            [
                "materialize-batch",
                "--signal",
                "database",
                "--prompts",
                str(prompts),
                "--teacher-response",
                str(teacher_response),
                "--output-dir",
                str(tmp_path / "datasets"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "smoke-001",
            ]
        )
