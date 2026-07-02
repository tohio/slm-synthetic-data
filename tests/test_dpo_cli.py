from pathlib import Path

from slm_synth.dpo.cli import main
from slm_synth.dpo.io import write_jsonl
from slm_synth.dpo.seeds import build_seed_rows


def test_dpo_materialize_seed_dataset_cli_calls_runner(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    calls = []

    def fake_materialize_seed_dataset(**kwargs):
        calls.append(kwargs)

        class Result:
            dataset_path = output_dir / "answer_only_arithmetic.jsonl"
            manifest_path = manifest_dir / "answer_only_arithmetic.dpo-smoke-001.manifest.json"
            row_count = 2
            family = "answer_only_arithmetic"

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.materialize_seed_dataset", fake_materialize_seed_dataset)

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
                "dpo-smoke-001",
                "--start-index",
                "5",
                "--dataset-filename",
                "dpo.jsonl",
                "--manifest-filename",
                "dpo.manifest.json",
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
            "generation_run": "dpo-smoke-001",
            "start_index": 5,
            "dataset_filename": "dpo.jsonl",
            "manifest_filename": "dpo.manifest.json",
        }
    ]
    captured = capsys.readouterr()
    assert "materialized 2 DPO row" in captured.out
    assert str(Path(output_dir) / "answer_only_arithmetic.jsonl") in captured.out


def test_dpo_materialize_seed_run_cli_calls_runner(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "datasets"
    manifest_dir = tmp_path / "manifests"
    calls = []

    def fake_materialize_seed_run(**kwargs):
        calls.append(kwargs)

        class FamilyResult:
            family = "answer_only_arithmetic"
            row_count = 2
            dataset_path = output_dir / "answer_only_arithmetic.jsonl"
            manifest_path = manifest_dir / "answer_only_arithmetic.dpo-smoke-001.manifest.json"

        class Result:
            row_count = 2
            families = ("answer_only_arithmetic",)
            generation_run = "dpo-smoke-001"
            results = (FamilyResult(),)

        return Result()

    monkeypatch.setattr("slm_synth.dpo.cli.materialize_seed_run", fake_materialize_seed_run)

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
                "dpo-smoke-001",
                "--start-index",
                "5",
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
            "generation_run": "dpo-smoke-001",
            "start_index": 5,
        }
    ]
    captured = capsys.readouterr()
    assert "materialized 2 DPO row" in captured.out
    assert "- answer_only_arithmetic: 2 row" in captured.out


def test_dpo_report_coverage_cli_prints_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    write_jsonl(build_seed_rows(family="answer_only_arithmetic", count=1), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path)]) == 0

    captured = capsys.readouterr()
    assert '"dataset_type": "dpo"' in captured.out
    assert '"row_count": 1' in captured.out
    assert '"answer_only_compliance": 1' in captured.out
    assert '"extra_explanation": 1' in captured.out


def test_dpo_report_coverage_cli_writes_json(tmp_path, capsys):
    dataset_path = tmp_path / "answer_only_arithmetic.jsonl"
    report_path = tmp_path / "coverage.json"
    write_jsonl(build_seed_rows(family="answer_only_arithmetic", count=1), dataset_path)

    assert main(["report-coverage", "--input", str(dataset_path), "--output", str(report_path)]) == 0

    captured = capsys.readouterr()
    assert "wrote DPO coverage report" in captured.out
    assert report_path.exists()
