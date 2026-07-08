from pathlib import Path

from slm_synth.sft.cli import main
from slm_synth.sft.io import write_jsonl
from slm_synth.sft.seeds import build_seed_rows


def test_sft_materialize_seed_dataset_cli_calls_runner(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    calls = []

    def fake_materialize_seed_dataset(**kwargs):
        calls.append(kwargs)

        class Result:
            dataset_path = output_dir / "answer_only_arithmetic.jsonl"
            manifest_path = manifest_dir / "answer_only_arithmetic.sft-smoke-001.manifest.json"
            row_count = 2
            family = "answer_only_arithmetic"

        return Result()

    monkeypatch.setattr("slm_synth.sft.cli.materialize_seed_dataset", fake_materialize_seed_dataset)

    assert (
        main(
            [
                "materialize-seed-dataset",
                "--family",
                "answer_only_arithmetic",
                "--count",
                "2",
                "--output-dir",
                str(output_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--generation-run",
                "sft-smoke-001",
                "--start-index",
                "5",
                "--dataset-filename",
                "sft.jsonl",
                "--manifest-filename",
                "sft.manifest.json",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "family": "answer_only_arithmetic",
            "count": 2,
            "output_dir": str(output_dir),
            "manifest_dir": str(manifest_dir),
            "generation_run": "sft-smoke-001",
            "start_index": 5,
            "dataset_filename": "sft.jsonl",
            "manifest_filename": "sft.manifest.json",
        }
    ]
    captured = capsys.readouterr()
    assert "materialized 2 SFT row" in captured.out
    assert str(Path(output_dir) / "answer_only_arithmetic.jsonl") in captured.out


def test_sft_build_specs_cli_calls_builder(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_build_and_write_specs(**kwargs):
        calls.append(kwargs)
        return 3

    monkeypatch.setattr("slm_synth.sft.cli.build_and_write_specs", fake_build_and_write_specs)

    assert (
        main(
            [
                "build-specs",
                "--family",
                "basic_arithmetic_qa",
                "--count",
                "3",
                "--output",
                str(tmp_path / "sft.specs.jsonl"),
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
            "output_path": str(tmp_path / "sft.specs.jsonl"),
            "start_index": 7,
        }
    ]
    captured = capsys.readouterr()
    assert "wrote 3 SFT task spec" in captured.out


def test_sft_materialize_seed_run_cli_calls_runner(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    calls = []

    def fake_materialize_seed_run(**kwargs):
        calls.append(kwargs)

        class FamilyResult:
            family = "answer_only_arithmetic"
            row_count = 2
            dataset_path = output_dir / "answer_only_arithmetic.jsonl"
            manifest_path = manifest_dir / "answer_only_arithmetic.sft-smoke-001.manifest.json"

        class Result:
            row_count = 2
            families = ("answer_only_arithmetic",)
            generation_run = "sft-smoke-001"
            manifest_path = manifest_dir / "sft-smoke-001.manifest.json"
            results = (FamilyResult(),)

        return Result()

    monkeypatch.setattr("slm_synth.sft.cli.materialize_seed_run", fake_materialize_seed_run)

    assert (
        main(
            [
                "materialize-seed-run",
                "--families",
                "answer_only_arithmetic",
                "--count-per-family",
                "2",
                "--output-dir",
                str(output_dir),
                "--manifest-dir",
                str(manifest_dir),
                "--generation-run",
                "sft-smoke-001",
                "--start-index",
                "5",
                "--run-manifest-filename",
                "sft-smoke-001.manifest.json",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "families": ["answer_only_arithmetic"],
            "count_per_family": 2,
            "output_dir": str(output_dir),
            "manifest_dir": str(manifest_dir),
            "generation_run": "sft-smoke-001",
            "start_index": 5,
            "run_manifest_filename": "sft-smoke-001.manifest.json",
        }
    ]
    captured = capsys.readouterr()
    assert "materialized 2 SFT row" in captured.out
    assert "run manifest:" in captured.out
    assert "- answer_only_arithmetic: 2 row" in captured.out


def test_sft_materialize_llm_batch_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_materialize_llm_batch_from_files(**kwargs):
        calls.append(kwargs)

        class Result:
            dataset_path = tmp_path / "sft.jsonl"
            manifest_path = tmp_path / "sft.manifest.json"
            row_count = 2

        return Result()

    monkeypatch.setattr("slm_synth.sft.cli.materialize_llm_batch_from_files", fake_materialize_llm_batch_from_files)

    assert (
        main(
            [
                "materialize-llm-batch",
                "--specs",
                str(tmp_path / "specs.jsonl"),
                "--teacher-response",
                str(tmp_path / "teacher_response.json"),
                "--output",
                str(tmp_path / "sft.jsonl"),
                "--manifest",
                str(tmp_path / "sft.manifest.json"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--teacher-provider",
                "openrouter",
                "--generation-run",
                "sft-llm-smoke-001",
            ]
        )
        == 0
    )

    assert calls == [
        {
            "specs_path": str(tmp_path / "specs.jsonl"),
            "teacher_response_path": str(tmp_path / "teacher_response.json"),
            "output_path": str(tmp_path / "sft.jsonl"),
            "manifest_path": str(tmp_path / "sft.manifest.json"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "sft-llm-smoke-001",
        }
    ]
    captured = capsys.readouterr()
    assert "materialized 2 LLM-generated SFT row" in captured.out


def test_sft_generate_llm_batch_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_generate_llm_batch_from_files(**kwargs):
        calls.append(kwargs)

        class Result:
            dataset_path = tmp_path / "sft.jsonl"
            manifest_path = tmp_path / "sft.manifest.json"
            row_count = 2

        return Result()

    monkeypatch.setattr("slm_synth.sft.cli.generate_llm_batch_from_files", fake_generate_llm_batch_from_files)

    assert (
        main(
            [
                "generate-llm-batch",
                "--specs",
                str(tmp_path / "specs.jsonl"),
                "--output",
                str(tmp_path / "sft.jsonl"),
                "--manifest",
                str(tmp_path / "sft.manifest.json"),
                "--teacher-model",
                "openai/gpt-4.1-mini",
                "--teacher-provider",
                "openrouter",
                "--generation-run",
                "sft-live-smoke-001",
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
            "output_path": str(tmp_path / "sft.jsonl"),
            "manifest_path": str(tmp_path / "sft.manifest.json"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "sft-live-smoke-001",
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
    assert "generated 2 LLM-generated SFT row" in captured.out


def test_sft_generate_llm_run_cli_calls_runner(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_generate_llm_run(**kwargs):
        calls.append(kwargs)

        class Result:
            row_count = 4
            families = ("basic_arithmetic_qa", "repeat_exact_n_times")
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
            "families": ["basic_arithmetic_qa", "repeat_exact_n_times"],
            "count_per_family": 2,
            "batch_size": 1,
            "output_dir": str(tmp_path / "datasets"),
            "manifest_dir": str(tmp_path / "manifests"),
            "teacher_model": "openai/gpt-4.1-mini",
            "teacher_provider": "openrouter",
            "generation_run": "sft-live-run-001",
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
            "adaptive_batch_increase_successes": 16,
            "concurrency": 2,
            "run_manifest_filename": "custom.manifest.json",
        }
    ]
    captured = capsys.readouterr()
    assert "generated 4 LLM-generated SFT row" in captured.out


def test_sft_report_coverage_cli_prints_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    write_jsonl(build_seed_rows(family="answer_only_arithmetic", count=1), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path)]) == 0

    captured = capsys.readouterr()
    assert '"dataset_type": "sft"' in captured.out
    assert '"row_count": 1' in captured.out
    assert '"answer_only_compliance": 1' in captured.out


def test_sft_report_coverage_cli_writes_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    report_path = tmp_path / "coverage.json"
    write_jsonl(build_seed_rows(family="answer_only_arithmetic", count=1), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert "wrote SFT coverage report" in captured.out
    assert report_path.exists()
