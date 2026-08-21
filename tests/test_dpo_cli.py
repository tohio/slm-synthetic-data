import pytest

from slm_synth.dpo.cli import main
from slm_synth.dpo.io import write_jsonl


def _sample_dpo_rows(count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "id": f"dpo-sample-{index}",
            "prompt": [{"role": "user", "content": "What is 2 + 2? Answer with only the integer."}],
            "chosen": [{"role": "assistant", "content": "4"}],
            "rejected": [{"role": "assistant", "content": "The answer is 4."}],
            "metadata": {
                "task_family": "applied_math_and_reasoning",
                "interaction_modes": ["single_turn"],
                "output_mode": "concise",
                "context_mode": "self_contained",
                "preference_dimension": "factual_accuracy",
                "difficulty": 1,
                "template_family": "direct_qa",
                "failure_mode": "extra_explanation",
            },
        }
        for index in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    "command", ["build-specs", "materialize-llm-batch", "generate-llm-batch"]
)
def test_dpo_cli_has_no_standalone_legacy_generation_paths(command):
    with pytest.raises(SystemExit):
        main([command])


def test_dpo_generate_llm_run_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 4
            preference_dimensions = ("factual_accuracy", "instruction_adherence")
            generation_run = "dpo-live-run-001"
            manifest_path = tmp_path / "manifests" / "dpo-live-run-001.manifest.json"

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.generate_llm_run", fake_generate_llm_run)
    monkeypatch.setattr("slm_synth.dpo.cli.print_dpo_run_summary", lambda manifest_path: None)

    assert (
        main(
            [
                "generate-llm-run",
                "--preference-dimensions",
                "factual_accuracy",
                "instruction_adherence",
                "--candidate-counts",
                "factual_accuracy=2",
                "instruction_adherence=2",
                "--batch-size",
                "1",
                "--output-dir",
                str(tmp_path / "datasets"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "dpo-live-run-001",
                "--max-tokens",
                "1024",
                "--start-index",
                "5",
                "--run-manifest-filename",
                "custom.manifest.json",
                "--concurrency",
                "2",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "preference_dimensions": ["factual_accuracy", "instruction_adherence"],
            "candidate_counts_by_dimension": {"factual_accuracy": 2, "instruction_adherence": 2},
            "batch_size": 1,
            "output_dir": str(tmp_path / "datasets"),
            "manifest_dir": str(tmp_path / "manifests"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "dpo-live-run-001",
                "max_tokens": 1024,
                "adjudicator_model": None,
                "adjudicator_max_tokens": None,
                "reviewer_model": None,
                "reviewer_max_tokens": None,
            "start_index": 5,
            "temperature": None,
            "top_p": None,
            "request_timeout": None,
            "max_request_retries": 3,
            "max_retryable_request_attempts": 20,
            "retry_max_elapsed_seconds": 1800.0,
            "adaptive_maximum_in_flight": 2,
            "adaptive_initial_in_flight": 8,
            "adaptive_initial_batch_size": 4,
            "adaptive_batch_increase_successes": 4,
            "concurrency": 2,
                "run_manifest_filename": "custom.manifest.json",
                "holdout_registry": None,
        }
    ]
    captured = capsys.readouterr()
    assert "generated 4 LLM-generated DPO row" in captured.out


def test_dpo_generate_llm_run_cli_accepts_explicit_dimension_counts(tmp_path, monkeypatch):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 3
            preference_dimensions = ("factual_accuracy", "instruction_adherence")
            generation_run = "dpo-target-001"
            manifest_path = tmp_path / "manifests" / "dpo-target-001.manifest.json"

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.generate_llm_run", fake_generate_llm_run)
    monkeypatch.setattr("slm_synth.dpo.cli.print_dpo_run_summary", lambda manifest_path: None)

    assert (
        main(
            [
                "generate-llm-run",
                "--preference-dimensions",
                "factual_accuracy",
                "instruction_adherence",
                "--candidate-counts",
                "factual_accuracy=2",
                "instruction_adherence=1",
                "--batch-size",
                "1",
                "--output-dir",
                str(tmp_path / "datasets"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "dpo-target-001",
                "--max-tokens",
                "1024",
            ]
        )
        == 0
    )

    assert calls[0]["candidate_counts_by_dimension"] == {
        "factual_accuracy": 2,
        "instruction_adherence": 1,
    }


def test_dpo_report_coverage_cli_prints_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    write_jsonl(_sample_dpo_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path)]) == 0

    captured = capsys.readouterr()
    assert '"dataset_type": "dpo"' in captured.out
    assert '"row_count": 1' in captured.out
    assert '"preference_dimension_counts"' in captured.out
    assert '"factual_accuracy": 1' in captured.out
    assert '"extra_explanation": 1' in captured.out


def test_dpo_report_coverage_cli_writes_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    report_path = tmp_path / "coverage.json"
    write_jsonl(_sample_dpo_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert "wrote DPO coverage report" in captured.out
    assert report_path.exists()
