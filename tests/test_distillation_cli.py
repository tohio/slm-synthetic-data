import json
from pathlib import Path

import pytest

from slm_synth.distillation_sft.cli import main
from slm_synth.distillation_sft.response_diversity import build_response_diversity_summary

def test_build_dataset_card_cli_writes_markdown(tmp_path, capsys):
    run_manifest = tmp_path / "smoke-001.manifest.json"
    output = tmp_path / "README.md"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_run": "smoke-001",
                "teacher_model": "openai/gpt-4.1-mini",
                "teacher_provider": "openrouter",
                "signals": ["cloud"],
                "datasets": [
                    {
                        "signal": "cloud",
                        "dataset_path": "data/distillation/datasets/cloud.jsonl",
                        "manifest_path": "data/distillation/manifests/cloud.smoke-001.manifest.json",
                        "row_count": 1,
                    }
                ],
                "total_rows": 1,
                "metadata": {"signal_count": 1},
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "build-dataset-card",
                "--run-manifest",
                str(run_manifest),
                "--output",
                str(output),
                "--dataset-name",
                "SLM Synthetic Distillation Smoke",
                "--license",
                "mit",
            ]
        )
        == 0
    )

    text = output.read_text(encoding="utf-8")
    assert "# SLM Synthetic Distillation Smoke" in text
    assert "- Teacher model: `openai/gpt-4.1-mini`" in text
    assert "| cloud | 1 | `data/distillation/datasets/cloud.jsonl` |" in text
    assert "wrote dataset card" in capsys.readouterr().out


def test_report_coverage_cli_prints_json(tmp_path, capsys):
    run_manifest = tmp_path / "smoke-001.manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_run": "smoke-001",
                "teacher_model": "openai/gpt-4.1-mini",
                "teacher_provider": "openrouter",
                "datasets": [
                    {
                        "signal": "cloud",
                        "dataset_path": "data/distillation/datasets/cloud.jsonl",
                        "manifest_path": "data/distillation/manifests/cloud.smoke-001.manifest.json",
                        "row_count": 1,
                    }
                ],
                "total_rows": 1,
            }
        ),
        encoding="utf-8",
    )

    assert main(["report-coverage", "--run-manifest", str(run_manifest)]) == 0

    output = capsys.readouterr().out
    assert '"dataset_type": "distillation"' in output
    assert '"row_count": 1' in output
    assert '"cloud": 1' in output


def test_report_coverage_cli_writes_json(tmp_path, capsys):
    run_manifest = tmp_path / "smoke-001.manifest.json"
    output = tmp_path / "coverage.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_run": "smoke-001",
                "teacher_model": "openai/gpt-4.1-mini",
                "teacher_provider": "openrouter",
                "datasets": [
                    {
                        "signal": "cloud",
                        "dataset_path": "data/distillation/datasets/cloud.jsonl",
                        "manifest_path": "data/distillation/manifests/cloud.smoke-001.manifest.json",
                        "row_count": 1,
                    }
                ],
                "total_rows": 1,
            }
        ),
        encoding="utf-8",
    )

    assert main(["report-coverage", "--run-manifest", str(run_manifest), "--output", str(output)]) == 0

    assert json.loads(output.read_text(encoding="utf-8"))["signals"] == {"cloud": 1}
    assert "wrote distillation coverage report" in capsys.readouterr().out


def test_report_coverage_cli_reports_low_response_diversity_without_failing(tmp_path):
    dataset = tmp_path / "debugging.jsonl"
    run_manifest = tmp_path / "smoke-001.manifest.json"
    output = tmp_path / "coverage.json"
    rows = [
        {
            "id": f"debugging-{index:06d}",
            "prompt": f"Unique debugging prompt {index}",
            "reasoning": None,
            "response": "Repeated debugging response.",
            "metadata": {
                "category": "general_instruction_following",
                "difficulty": 2,
                "template_family": "python_optional_key_bug",
                "eval_family": None,
            },
        }
        for index in range(1, 5)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_run": "smoke-001",
                "teacher_model": "openai/gpt-4.1-mini",
                "teacher_provider": "openrouter",
                "datasets": [
                    {
                        "signal": "debugging",
                        "dataset_path": str(dataset),
                        "manifest_path": str(tmp_path / "debugging.manifest.json"),
                        "row_count": 4,
                    }
                ],
                "total_rows": 4,
            }
        ),
        encoding="utf-8",
    )

    assert main(["report-coverage", "--run-manifest", str(run_manifest), "--output", str(output)]) == 0
    diversity = json.loads(output.read_text(encoding="utf-8"))["response_diversity"]
    assert diversity["unique_response_ratio"] == 0.25
    assert diversity["signals"]["debugging"]["unique_response_ratio"] == 0.25


@pytest.mark.parametrize(
    "command",
    [
        "build-seed-prompts",
        "build-prompt-specs",
        "render-teacher-prompt",
        "materialize-batch",
        "generate-batch",
        "generate-seed-run",
        "generate-production-run",
        "apply-response-cluster-adjudications",
    ],
)
def test_legacy_distillation_sft_cli_commands_are_unavailable(command):
    with pytest.raises(SystemExit):
        main([command])
