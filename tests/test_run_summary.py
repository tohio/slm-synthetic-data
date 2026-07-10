import json

from slm_synth.run_summary import (
    print_distillation_run_summary,
    print_dpo_run_summary,
    print_pretrain_run_summary,
    print_sft_run_summary,
)


def test_print_pretrain_run_summary_emits_common_stats(tmp_path, capsys):
    manifest = tmp_path / "pretrain-run.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_type": "pretrain",
                "generation_run": "pretrain-smoke",
                "stages": {
                    "raw": {"row_count": 12},
                    "validated": {"row_count": 11},
                    "deduped": {"row_count": 10},
                    "rejected": {"row_count": 1},
                },
                "signals": {
                    "arithmetic": {"deduped_rows": 6},
                    "task_code": {"deduped_rows": 4},
                },
                "metadata": {
                    "telemetry": {
                        "totals": {
                            "batches": 5,
                            "retry_count": 1,
                            "retryable_provider_retries": 2,
                            "retry_sleep_seconds": 0.5,
                            "adaptive_window_increases": 3,
                            "adaptive_window_decreases": 1,
                            "adaptive_admission_wait_seconds": 9.25,
                            "adaptive_peak_in_flight_limit": 16,
                            "adaptive_min_in_flight_limit": 8,
                            "max_adaptive_cooldown_seconds": 2.0,
                            "adaptive_batch_size_observed_minimum": 4,
                            "adaptive_batch_size_observed_peak": 8,
                            "adaptive_batch_size_increases": 2,
                            "adaptive_batch_size_decreases": 1,
                            "adaptive_batch_size_failures": 1,
                            "cost": 0.125,
                            "total_tokens": 4096,
                            "elapsed_seconds": 12.5,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    print_pretrain_run_summary(manifest)

    output = capsys.readouterr().out
    assert "[generate] Completed pretrain run: rows=10" in output
    assert "raw_rows=12" in output
    assert "validated_rows=11" in output
    assert "rejected_rows=1" in output
    assert "signals=2" in output
    assert "adaptive_batch_size_observed_minimum=4" in output
    assert "adaptive_batch_size_observed_peak=8" in output
    assert "adaptive_batch_size_failures=1" in output
    assert "batches=5" in output
    assert "provider_retries=2" in output
    assert "adaptive_peak_in_flight_limit=16" in output
    assert "cost=0.12500000" in output
    assert "request_tokens=4096" in output
    assert '[generate] pretrain signals={"arithmetic": 6, "task_code": 4}' in output


def test_print_sft_run_summary_emits_pretrain_style_stats(tmp_path, capsys):
    manifest = tmp_path / "sft-run.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "total_rows": 128,
                "datasets": [
                    {"family": "basic_arithmetic_qa", "row_count": 64},
                    {"family": "code_generation_function", "row_count": 64},
                ],
                "metadata": {
                    "batch_size": 64,
                    "concurrency": 1024,
                    "adaptive_maximum_in_flight": 1024,
                    "adaptive_initial_in_flight": 8,
                    "adaptive_batch_size_observed_minimum": 8,
                    "adaptive_batch_size_observed_peak": 64,
                    "adaptive_batch_size_increases": 3,
                    "adaptive_batch_size_decreases": 2,
                    "adaptive_batch_size_failures": 2,
                    "llm_telemetry": {
                        "batch_count": 12,
                        "retry_count": 1,
                        "retryable_provider_retries": 4,
                        "retry_sleep_seconds": 7.25,
                        "adaptive_window_increases": 5,
                        "adaptive_window_decreases": 1,
                        "adaptive_admission_wait_seconds": 13.5,
                        "adaptive_peak_in_flight_limit": 128,
                        "adaptive_min_in_flight_limit": 8,
                        "max_adaptive_cooldown_seconds": 2.0,
                        "elapsed_seconds": 90.5,
                        "usage": {"cost": 1.25, "total_tokens": 12345},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    print_sft_run_summary(manifest)

    output = capsys.readouterr().out
    assert "[generate] Completed SFT run: rows=128" in output
    assert "families=2" in output
    assert "batch_size=64" in output
    assert "concurrency=1024" in output
    assert "adaptive_batch_size_observed_minimum=8" in output
    assert "adaptive_batch_size_failures=2" in output
    assert "batches=12" in output
    assert "adaptive_peak_in_flight_limit=128" in output
    assert "cost=1.25000000" in output
    assert "request_tokens=12345" in output
    assert '[generate] SFT families={"basic_arithmetic_qa": 64, "code_generation_function": 64}' in output


def test_print_dpo_run_summary_emits_same_shape(tmp_path, capsys):
    manifest = tmp_path / "dpo-run.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "total_rows": 64,
                "datasets": [{"family": "basic_arithmetic_qa", "row_count": 64}],
                "metadata": {
                    "batch_size": 64,
                    "concurrency": 1024,
                    "adaptive_maximum_in_flight": 1024,
                    "adaptive_initial_in_flight": 8,
                    "adaptive_batch_size_observed_minimum": 1,
                    "adaptive_batch_size_observed_peak": 64,
                    "adaptive_batch_size_increases": 7,
                    "adaptive_batch_size_decreases": 8,
                    "adaptive_batch_size_failures": 8,
                    "llm_telemetry": {
                        "batch_count": 32,
                        "adaptive_peak_in_flight_limit": 1024,
                        "usage": {"cost": 0.5, "total_tokens": 6789},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    print_dpo_run_summary(manifest)

    output = capsys.readouterr().out
    assert "[generate] Completed DPO run: rows=64" in output
    assert "adaptive_batch_size_observed_minimum=1" in output
    assert "adaptive_batch_size_decreases=8" in output
    assert "batches=32" in output
    assert "adaptive_peak_in_flight_limit=1024" in output
    assert '[generate] DPO families={"basic_arithmetic_qa": 64}' in output


def test_print_distillation_run_summary_emits_run_and_signal_stats(tmp_path, capsys):
    signal_manifest = tmp_path / "arithmetic.run.manifest.json"
    signal_manifest.write_text(
        json.dumps(
            {
                "signal": "arithmetic",
                "row_count": 64,
                "metadata": {
                    "batch_count": 4,
                    "batch_size": 64,
                    "concurrency": 1024,
                    "adaptive_batch_size_observed_minimum": 16,
                    "adaptive_batch_size_observed_peak": 64,
                    "adaptive_batch_size_increases": 2,
                    "adaptive_batch_size_decreases": 1,
                    "adaptive_batch_size_failures": 1,
                    "llm_telemetry": {
                        "batch_count": 4,
                        "adaptive_window_decreases": 1,
                        "adaptive_peak_in_flight_limit": 32,
                        "usage": {"cost": 0.25, "total_tokens": 4567},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    run_manifest = tmp_path / "distill-run.manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "total_rows": 64,
                "datasets": [
                    {
                        "signal": "arithmetic",
                        "row_count": 64,
                        "manifest_path": str(signal_manifest),
                    }
                ],
                "metadata": {
                    "batch_size": 64,
                    "concurrency": 1024,
                    "adaptive_maximum_in_flight": 1024,
                    "adaptive_initial_in_flight": 8,
                    "llm_telemetry": {
                        "batch_count": 4,
                        "adaptive_peak_in_flight_limit": 32,
                        "usage": {"cost": 0.25, "total_tokens": 4567},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    print_distillation_run_summary(run_manifest)

    output = capsys.readouterr().out
    assert "[generate] Completed distillation run: rows=64" in output
    assert "signals=1" in output
    assert "batch_size=64" in output
    assert "adaptive_peak_in_flight_limit=32" in output
    assert '[generate] distillation signals={"arithmetic": 64}' in output
    assert "[generate] Completed distillation signal: arithmetic rows=64" in output
    assert "adaptive_batch_size_observed_minimum=16" in output
    assert "adaptive_batch_size_failures=1" in output
