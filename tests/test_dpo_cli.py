from pathlib import Path

from slm_synth.dpo.cli import main


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
