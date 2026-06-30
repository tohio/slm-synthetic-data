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


def test_generate_batch_cli_uses_live_generation_wrapper(tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    main(["build-seed-prompts", "--signal", "debugging", "--count", "1", "--output", str(prompts)])

    calls = []

    def fake_generate_and_materialize_signal_batch(**kwargs):
        calls.append(kwargs)

        class Result:
            signal = "debugging"
            row_count = 1
            dataset_path = output_dir / "debugging.jsonl"
            manifest_path = manifest_dir / "debugging.smoke-001.manifest.json"

        return Result()

    monkeypatch.setattr(
        "slm_synth.distillation.cli.generate_and_materialize_signal_batch",
        fake_generate_and_materialize_signal_batch,
    )

    assert (
        main(
            [
                "generate-batch",
                "--signal",
                "debugging",
                "--prompts",
                str(prompts),
                "--output-dir",
                str(output_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "smoke-001",
                "--max-tokens",
                "512",
                "--token-target",
                "100K",
            ]
        )
        == 0
    )

    assert calls[0]["signal"] == "debugging"
    assert calls[0]["teacher_model"] == "openai/gpt-4.1-mini"
    assert calls[0]["generation_run"] == "smoke-001"
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["token_target"] == "100K"
    assert calls[0]["prompt_records"][0]["id"] == "debugging-000001"
    captured = capsys.readouterr()
    assert "generated and materialized 1 debugging row" in captured.out


def test_generate_seed_run_cli_uses_multi_signal_orchestrator(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    calls = []

    def fake_generate_seed_multi_signal_run(**kwargs):
        calls.append(kwargs)

        class Result:
            generation_run = "smoke-001"
            row_count = 4
            results = [object(), object()]
            signals = ("cloud", "database")
            manifest_path = manifest_dir / "smoke-001.manifest.json"

        return Result()

    monkeypatch.setattr(
        "slm_synth.distillation.cli.generate_seed_multi_signal_run",
        fake_generate_seed_multi_signal_run,
    )

    assert (
        main(
            [
                "generate-seed-run",
                "--signals",
                "cloud",
                "database",
                "--count-per-signal",
                "2",
                "--output-dir",
                str(output_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "smoke-001",
                "--max-tokens",
                "512",
                "--token-target",
                "100K",
                "--run-manifest-filename",
                "smoke-001.manifest.json",
            ]
        )
        == 0
    )

    assert calls[0]["signals"] == ["cloud", "database"]
    assert calls[0]["count_per_signal"] == 2
    assert calls[0]["teacher_model"] == "openai/gpt-4.1-mini"
    assert calls[0]["generation_run"] == "smoke-001"
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["token_target"] == "100K"
    assert calls[0]["run_manifest_filename"] == "smoke-001.manifest.json"
    captured = capsys.readouterr()
    assert "generated and materialized 4 row(s) across 2 signal(s): cloud, database" in captured.out
    assert "run manifest:" in captured.out


def test_plan_token_target_cli_prints_json_plan(capsys):
    assert (
        main(
            [
                "plan-token-target",
                "--target",
                "100K",
                "--signals",
                "cloud",
                "database",
                "--estimated-tokens-per-row",
                "25000",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["target_tokens"] == 100_000
    assert payload["target_label"] == "100K"
    assert payload["estimated_tokens_per_row"] == 25_000
    assert payload["counts_by_signal"] == {"cloud": 2, "database": 2}


def test_generate_seed_run_cli_uses_target_preset_counts(tmp_path, monkeypatch):
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    calls = []

    def fake_generate_seed_multi_signal_run(**kwargs):
        calls.append(kwargs)

        class Result:
            generation_run = "pilot-001"
            row_count = 4
            results = [object(), object()]
            signals = ("cloud", "database")
            manifest_path = manifest_dir / "pilot-001.manifest.json"

        return Result()

    monkeypatch.setattr(
        "slm_synth.distillation.cli.generate_seed_multi_signal_run",
        fake_generate_seed_multi_signal_run,
    )

    assert (
        main(
            [
                "generate-seed-run",
                "--signals",
                "cloud",
                "database",
                "--target-preset",
                "100K",
                "--estimated-tokens-per-row",
                "25000",
                "--output-dir",
                str(output_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "pilot-001",
                "--max-tokens",
                "512",
            ]
        )
        == 0
    )

    assert calls[0]["count_per_signal"] is None
    assert calls[0]["counts_by_signal"] == {"cloud": 2, "database": 2}
    assert calls[0]["token_target"] == "100K"
