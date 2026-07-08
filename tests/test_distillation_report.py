import json

import pytest

from slm_synth.distillation.report import build_coverage_report, write_coverage_report


def _write_run_manifest(path, datasets):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_run": "smoke-001",
                "teacher_model": "openai/gpt-4.1-mini",
                "teacher_provider": "openrouter",
                "token_target": "100K",
                "datasets": datasets,
                "total_rows": sum(dataset["row_count"] for dataset in datasets),
                "metadata": {
                    "target_rows": 5,
                    "planned_prompt_rows": 5,
                    "accepted_rows": 5,
                    "rejected_rows": 0,
                    "rows_per_signal": {"cloud": 2, "database": 3},
                },
            },
            default=str,
        ),
        encoding="utf-8",
    )


def test_build_distillation_coverage_report_from_run_manifest(tmp_path):
    run_manifest = tmp_path / "smoke-001.manifest.json"
    _write_run_manifest(
        run_manifest,
        [
            {
                "signal": "cloud",
                "dataset_path": tmp_path / "datasets" / "cloud.jsonl",
                "manifest_path": tmp_path / "manifests" / "cloud.smoke-001.manifest.json",
                "row_count": 2,
            },
            {
                "signal": "database",
                "dataset_path": tmp_path / "datasets" / "database.jsonl",
                "manifest_path": tmp_path / "manifests" / "database.smoke-001.manifest.json",
                "row_count": 3,
            },
        ],
    )

    report = build_coverage_report(run_manifest)

    assert report["dataset_type"] == "distillation"
    assert report["generation_run"] == "smoke-001"
    assert report["teacher_provider"] == "openrouter"
    assert report["teacher_model"] == "openai/gpt-4.1-mini"
    assert report["token_target"] == "100K"
    assert report["target_rows"] == 5
    assert report["planned_prompt_rows"] == 5
    assert report["accepted_rows"] == 5
    assert report["rejected_rows"] == 0
    assert report["row_count"] == 5
    assert report["signals"] == {"cloud": 2, "database": 3}
    assert report["rows_per_signal"] == {"cloud": 2, "database": 3}
    assert report["dataset_paths"]["cloud"] == str(tmp_path / "datasets" / "cloud.jsonl")
    assert report["manifest_paths"]["database"] == str(tmp_path / "manifests" / "database.smoke-001.manifest.json")


def test_write_distillation_coverage_report_writes_json(tmp_path):
    output = tmp_path / "coverage.json"
    report = {
        "dataset_type": "distillation",
        "generation_run": "smoke-001",
        "teacher_model": "openai/gpt-4.1-mini",
        "teacher_provider": "openrouter",
        "token_target": "100K",
        "target_rows": None,
        "planned_prompt_rows": None,
        "accepted_rows": 0,
        "rejected_rows": None,
        "row_count": 0,
        "signals": {},
        "rows_per_signal": {},
        "dataset_paths": {},
        "manifest_paths": {},
    }

    path = write_coverage_report(report=report, path=output)

    assert path == output
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_build_distillation_coverage_report_rejects_duplicate_signals(tmp_path):
    run_manifest = tmp_path / "smoke-001.manifest.json"
    _write_run_manifest(
        run_manifest,
        [
            {"signal": "cloud", "dataset_path": "cloud-a.jsonl", "manifest_path": "cloud-a.json", "row_count": 1},
            {"signal": "cloud", "dataset_path": "cloud-b.jsonl", "manifest_path": "cloud-b.json", "row_count": 1},
        ],
    )

    with pytest.raises(ValueError, match="duplicate signal"):
        build_coverage_report(run_manifest)
