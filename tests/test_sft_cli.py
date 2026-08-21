import pytest

from slm_synth.sft.cli import main
from slm_synth.sft.io import write_jsonl


def _sample_sft_rows(count: int = 1) -> list[dict[str, object]]:
    return [
        {
            "id": f"sft-sample-{index}",
            "messages": [
                {"role": "user", "content": "What is 2 + 2? Answer with only the integer."},
                {"role": "assistant", "content": "4"},
            ],
            "metadata": {
                "task_family": "grounded_qa_and_reading",
                "interaction_modes": ["single_turn"],
                "output_mode": "free_text",
                "context_mode": "supplied_passage",
                "difficulty": 1,
                "template_family": "direct_qa",
            },
        }
        for index in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    "command", ["build-specs", "materialize-llm-batch", "generate-llm-batch"]
)
def test_sft_cli_has_no_standalone_legacy_generation_paths(command):
    with pytest.raises(SystemExit):
        main([command])


def test_sft_generate_llm_run_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 4
            families = ("grounded_qa_and_reading", "rewriting_and_editing")
            generation_run = "sft-live-run-001"
            manifest_path = tmp_path / "manifests" / "sft-live-run-001.manifest.json"

        return Result()

    monkeypatch.setattr("slm_synth.sft.cli.generate_llm_run", fake_generate_llm_run)
    monkeypatch.setattr("slm_synth.sft.cli.print_sft_run_summary", lambda manifest_path: None)

    assert (
        main(
            [
                "generate-llm-run",
                "--families",
                "grounded_qa_and_reading",
                "rewriting_and_editing",
                "--candidate-counts",
                "grounded_qa_and_reading=2",
                "rewriting_and_editing=2",
                "--batch-size",
                "1",
                "--output-dir",
                str(tmp_path / "datasets"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "sft-live-run-001",
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
            "families": ["grounded_qa_and_reading", "rewriting_and_editing"],
            "candidate_counts_by_family": {"grounded_qa_and_reading": 2, "rewriting_and_editing": 2},
            "batch_size": 1,
            "output_dir": str(tmp_path / "datasets"),
            "manifest_dir": str(tmp_path / "manifests"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "sft-live-run-001",
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
    assert "generated 4 LLM-generated SFT row" in captured.out


def test_sft_generate_llm_run_cli_accepts_explicit_candidate_counts(tmp_path, monkeypatch):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 3
            families = ("grounded_qa_and_reading", "rewriting_and_editing")
            generation_run = "sft-target-001"
            manifest_path = tmp_path / "manifests" / "sft-target-001.manifest.json"

        return Result()

    monkeypatch.setattr("slm_synth.sft.cli.generate_llm_run", fake_generate_llm_run)
    monkeypatch.setattr("slm_synth.sft.cli.print_sft_run_summary", lambda manifest_path: None)

    assert (
        main(
            [
                "generate-llm-run",
                "--families",
                "grounded_qa_and_reading",
                "rewriting_and_editing",
                "--candidate-counts",
                "grounded_qa_and_reading=2",
                "rewriting_and_editing=1",
                "--batch-size",
                "1",
                "--output-dir",
                str(tmp_path / "datasets"),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--generation-run",
                "sft-target-001",
                "--max-tokens",
                "1024",
            ]
        )
        == 0
    )

    assert calls[0]["candidate_counts_by_family"] == {
        "grounded_qa_and_reading": 2,
        "rewriting_and_editing": 1,
    }


def test_sft_report_coverage_cli_prints_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    write_jsonl(_sample_sft_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path)]) == 0

    captured = capsys.readouterr()
    assert '"dataset_type": "sft"' in captured.out
    assert '"row_count": 1' in captured.out
    assert '"grounded_qa_and_reading": 1' in captured.out


def test_sft_report_coverage_cli_writes_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    report_path = tmp_path / "coverage.json"
    write_jsonl(_sample_sft_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert "wrote SFT coverage report" in captured.out
    assert report_path.exists()
