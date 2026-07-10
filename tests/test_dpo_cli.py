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
                "category": "answer_only_compliance",
                "difficulty": 1,
                "template_family": "direct_qa",
                "eval_family": "basic_arithmetic_qa",
                "failure_mode": "extra_explanation",
            },
        }
        for index in range(1, count + 1)
    ]


def test_dpo_build_specs_cli_calls_builder(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_build_and_write_specs(**kwargs):
        calls.append(kwargs)
        return 3

    monkeypatch.setattr("slm_synth.dpo.cli.build_and_write_specs", fake_build_and_write_specs)

    assert (
        main(
            [
                "build-specs",
                "--family",
                "basic_arithmetic_qa",
                "--count",
                "3",
                "--output",
                str(tmp_path / "dpo.specs.jsonl"),
                "--start-index",
                "7",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "family": "basic_arithmetic_qa",
            "count": 3,
            "output_path": str(tmp_path / "dpo.specs.jsonl"),
            "start_index": 7,
        }
    ]
    captured = capsys.readouterr()
    assert "wrote 3 DPO task spec" in captured.out


def test_dpo_materialize_llm_batch_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_materialize_llm_batch_from_files(**kwargs):
        calls.append(kwargs)

        class Result:
            dataset_path = tmp_path / "dpo.jsonl"
            manifest_path = tmp_path / "dpo.manifest.json"
            row_count = 2

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.materialize_llm_batch_from_files", fake_materialize_llm_batch_from_files)

    assert (
        main(
            [
                "materialize-llm-batch",
                "--specs",
                str(tmp_path / "specs.jsonl"),
                "--teacher-response",
                str(tmp_path / "teacher_response.json"),
                "--output",
                str(tmp_path / "dpo.jsonl"),
                "--manifest",
                str(tmp_path / "dpo.manifest.json"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--teacher-provider",
                "openrouter",
                "--generation-run",
                "dpo-llm-smoke-001",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "specs_path": str(tmp_path / "specs.jsonl"),
            "teacher_response_path": str(tmp_path / "teacher_response.json"),
            "output_path": str(tmp_path / "dpo.jsonl"),
            "manifest_path": str(tmp_path / "dpo.manifest.json"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "dpo-llm-smoke-001",
        }
    ]
    captured = capsys.readouterr()
    assert "materialized 2 LLM-generated DPO row" in captured.out


def test_dpo_generate_llm_batch_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_generate_llm_batch_from_files(**kwargs):
        calls.append(kwargs)

        class Result:
            dataset_path = tmp_path / "dpo.jsonl"
            manifest_path = tmp_path / "dpo.manifest.json"
            row_count = 2

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.generate_llm_batch_from_files", fake_generate_llm_batch_from_files)

    assert (
        main(
            [
                "generate-llm-batch",
                "--specs",
                str(tmp_path / "specs.jsonl"),
                "--output",
                str(tmp_path / "dpo.jsonl"),
                "--manifest",
                str(tmp_path / "dpo.manifest.json"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--teacher-provider",
                "openrouter",
                "--generation-run",
                "dpo-live-smoke-001",
                "--max-tokens",
                "1024",
                "--temperature",
                "0.1",
                "--top-p",
                "0.9",
                "--request-timeout",
                "30",
                "--max-request-retries",
                "2",
                "--max-retryable-request-attempts",
                "5",
                "--retry-max-elapsed-seconds",
                "120",
                "--adaptive-maximum-in-flight",
                "3",
                "--adaptive-initial-in-flight",
                "1",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "specs_path": str(tmp_path / "specs.jsonl"),
            "output_path": str(tmp_path / "dpo.jsonl"),
            "manifest_path": str(tmp_path / "dpo.manifest.json"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "dpo-live-smoke-001",
            "max_tokens": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
            "request_timeout": 30.0,
            "max_request_retries": 2,
            "max_retryable_request_attempts": 5,
            "retry_max_elapsed_seconds": 120.0,
            "adaptive_maximum_in_flight": 3,
            "adaptive_initial_in_flight": 1,
        }
    ]
    captured = capsys.readouterr()
    assert "generated 2 LLM-generated DPO row" in captured.out


def test_dpo_generate_llm_run_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 4
            families = ("basic_arithmetic_qa", "repeat_exact_n_times")
            generation_run = "dpo-live-run-001"
            manifest_path = tmp_path / "manifests" / "dpo-live-run-001.manifest.json"

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.generate_llm_run", fake_generate_llm_run)
    monkeypatch.setattr("slm_synth.dpo.cli.print_dpo_run_summary", lambda manifest_path: None)

    assert (
        main(
            [
                "generate-llm-run",
                "--families",
                "basic_arithmetic_qa",
                "repeat_exact_n_times",
                "--count-per-family",
                "2",
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
            "families": ["basic_arithmetic_qa", "repeat_exact_n_times"],
            "count_per_family": 2,
            "target_pairs": None,
            "batch_size": 1,
            "output_dir": str(tmp_path / "datasets"),
            "manifest_dir": str(tmp_path / "manifests"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "dpo-live-run-001",
            "max_tokens": 1024,
            "start_index": 5,
            "temperature": 0.2,
            "top_p": 0.95,
            "request_timeout": None,
            "max_request_retries": 3,
            "max_retryable_request_attempts": 20,
            "retry_max_elapsed_seconds": 1800.0,
            "adaptive_maximum_in_flight": 2,
            "adaptive_initial_in_flight": 8,
            "adaptive_initial_batch_size": 4,
            "adaptive_batch_increase_successes": 4,
            "concurrency": 2,
            "max_backfill_rounds": 2,
            "run_manifest_filename": "custom.manifest.json",
        }
    ]
    captured = capsys.readouterr()
    assert "generated 4 LLM-generated DPO row" in captured.out


def test_dpo_generate_llm_run_cli_accepts_target_pairs(tmp_path, monkeypatch):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 3
            families = ("basic_arithmetic_qa", "repeat_exact_n_times")
            generation_run = "dpo-target-001"
            manifest_path = tmp_path / "manifests" / "dpo-target-001.manifest.json"

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.generate_llm_run", fake_generate_llm_run)
    monkeypatch.setattr("slm_synth.dpo.cli.print_dpo_run_summary", lambda manifest_path: None)

    assert (
        main(
            [
                "generate-llm-run",
                "--families",
                "basic_arithmetic_qa",
                "repeat_exact_n_times",
                "--target-pairs",
                "3",
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

    assert calls[0]["count_per_family"] is None
    assert calls[0]["target_pairs"] == 3


def test_dpo_report_coverage_cli_prints_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    write_jsonl(_sample_dpo_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path)]) == 0

    captured = capsys.readouterr()
    assert '"dataset_type": "dpo"' in captured.out
    assert '"row_count": 1' in captured.out
    assert '"answer_only_compliance": 1' in captured.out
    assert '"extra_explanation": 1' in captured.out


def test_dpo_report_coverage_cli_writes_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    report_path = tmp_path / "coverage.json"
    write_jsonl(_sample_dpo_rows(), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert "wrote DPO coverage report" in captured.out
    assert report_path.exists()
