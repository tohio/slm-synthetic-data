import json
from pathlib import Path

from slm_synth.pretrain.manifest import (
    build_run_manifest,
    default_manifest_path,
    generate_run_manifest,
    write_run_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_pretrain_run_manifest_counts_stage_files_and_signals(tmp_path):
    output_dir = tmp_path / "runs" / "pretrain-smoke"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        f"""
run_name: pretrain-smoke
output_dir: "{output_dir}"
target_total_tokens: 1000
mix:
  arithmetic:
    share: 0.5
  task_code:
    share: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "raw" / "arithmetic.jsonl", [{"id": 1}, {"id": 2}])
    _write_jsonl(output_dir / "raw" / "task_code.jsonl", [{"id": 3}])
    _write_jsonl(output_dir / "validated" / "arithmetic.jsonl", [{"id": 1}])
    _write_jsonl(output_dir / "deduped" / "arithmetic.jsonl", [{"id": 1}])
    _write_jsonl(output_dir / "rejected" / "arithmetic.jsonl", [{"id": 2}])

    manifest = build_run_manifest(config_path=config_path)

    assert manifest["dataset_type"] == "pretrain"
    assert manifest["generation_run"] == "pretrain-smoke"
    assert manifest["config_path"] == str(config_path)
    assert manifest["output_dir"] == str(output_dir)
    assert manifest["stages"]["raw"]["file_count"] == 2
    assert manifest["stages"]["raw"]["row_count"] == 3
    assert manifest["stages"]["validated"]["row_count"] == 1
    assert manifest["stages"]["deduped"]["row_count"] == 1
    assert manifest["stages"]["rejected"]["row_count"] == 1
    assert manifest["signals"]["arithmetic"] == {
        "deduped_rows": 1,
        "raw_rows": 2,
        "rejected_rows": 1,
        "validated_rows": 1,
    }
    assert manifest["signals"]["task_code"] == {"raw_rows": 1}


def test_build_pretrain_run_manifest_includes_grounded_telemetry(tmp_path):
    output_dir = tmp_path / "runs" / "pretrain-smoke"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        f"""
run_name: pretrain-smoke
output_dir: "{output_dir}"
target_total_tokens: 1000
mix:
  arithmetic:
    share: 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "raw" / "arithmetic.jsonl", [{"id": 1}])
    batch_dir = output_dir / "manifests" / "grounded" / "arithmetic" / "batches"
    batch_dir.mkdir(parents=True)
    (batch_dir / "batch_000000000.json").write_text(
        json.dumps(
            {
                "records": [{"id": 1}],
                "telemetry": {
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
                    "retry_count": 1,
                    "retryable_provider_retries": 2,
                    "retry_sleep_seconds": 3.0,
                    "adaptive_window_increases": 1,
                    "adaptive_window_decreases": 0,
                    "adaptive_admission_wait_seconds": 4.0,
                    "adaptive_peak_in_flight_limit": 32,
                    "adaptive_min_in_flight_limit": 8,
                    "max_adaptive_cooldown_seconds": 0.5,
                    "adaptive_batch_size_observed_minimum": 16,
                    "adaptive_batch_size_observed_peak": 16,
                    "adaptive_batch_size_increases": 0,
                    "adaptive_batch_size_decreases": 0,
                    "adaptive_batch_size_failures": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_run_manifest(config_path=config_path)

    telemetry = manifest["metadata"]["telemetry"]
    assert telemetry["signals"]["arithmetic"]["adaptive_peak_in_flight_limit"] == 32
    assert telemetry["totals"]["total_tokens"] == 15
    assert telemetry["totals"]["retryable_provider_retries"] == 2
    assert telemetry["totals"]["adaptive_min_in_flight_limit"] == 8
    assert telemetry["totals"]["adaptive_batch_size_observed_minimum"] == 16


def test_default_pretrain_manifest_path():
    assert default_manifest_path(output_dir="data/runs/pretrain-smoke", generation_run="pretrain-smoke") == (
        Path("data/runs/pretrain-smoke") / "manifests" / "pretrain-smoke.manifest.json"
    )


def test_write_pretrain_run_manifest_writes_json(tmp_path):
    path = write_run_manifest(
        manifest_path=tmp_path / "manifests" / "pretrain.manifest.json",
        manifest={
            "dataset_type": "pretrain",
            "generation_run": "pretrain-smoke",
            "stages": {},
            "signals": {},
        },
    )

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["dataset_type"] == "pretrain"


def test_generate_pretrain_run_manifest_uses_default_output_path(tmp_path):
    output_dir = tmp_path / "runs" / "pretrain-smoke"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        f"""
run_name: pretrain-smoke
output_dir: "{output_dir}"
target_total_tokens: 1000
mix: {{}}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    path = generate_run_manifest(config_path=config_path)

    assert path == output_dir / "manifests" / "pretrain-smoke.manifest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["generation_run"] == "pretrain-smoke"


def test_generate_pretrain_run_manifest_supports_overrides(tmp_path):
    output_dir = tmp_path / "runs" / "from-config"
    override_output_dir = tmp_path / "runs" / "override"
    config_path = tmp_path / "synthetic.yaml"
    custom_manifest = tmp_path / "custom.manifest.json"
    config_path.write_text(
        f"""
run_name: from-config
output_dir: "{output_dir}"
target_total_tokens: 1000
mix: {{}}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    path = generate_run_manifest(
        config_path=config_path,
        output=custom_manifest,
        output_dir=override_output_dir,
        generation_run="manual-run",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert path == custom_manifest
    assert manifest["generation_run"] == "manual-run"
    assert manifest["output_dir"] == str(override_output_dir)


def test_pretrain_manifest_cli_prints_run_summary(tmp_path, capsys):
    output_dir = tmp_path / "runs" / "pretrain-smoke"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        f"""
run_name: pretrain-smoke
output_dir: "{output_dir}"
target_total_tokens: 1000
mix:
  arithmetic:
    share: 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "raw" / "arithmetic.jsonl", [{"id": 1}])
    _write_jsonl(output_dir / "validated" / "arithmetic.jsonl", [{"id": 1}])
    _write_jsonl(output_dir / "deduped" / "arithmetic.jsonl", [{"id": 1}])

    from slm_synth.pretrain.manifest import cli

    assert cli(["--config", str(config_path)]) == 0

    output = capsys.readouterr().out
    assert "wrote pretrain run manifest" in output
    assert "[generate] Completed pretrain run: rows=1" in output
    assert '[generate] pretrain signals={"arithmetic": 1}' in output
