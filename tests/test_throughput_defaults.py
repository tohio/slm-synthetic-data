from pathlib import Path

from configs import configure_synthetic
from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    DEFAULT_OPENROUTER_BATCH_SIZE,
    DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    DEFAULT_OPENROUTER_TARGET_CONCURRENCY,
    MAX_OPENROUTER_BATCH_SIZE,
    MAX_OPENROUTER_CONCURRENCY,
)


def test_grounded_pretrain_config_uses_shared_throughput_bounds():
    assert configure_synthetic.MIN_BATCH_SIZE == 1
    assert configure_synthetic.MAX_BATCH_SIZE == MAX_OPENROUTER_BATCH_SIZE
    assert configure_synthetic.MIN_CONCURRENCY == 1
    assert configure_synthetic.MAX_CONCURRENCY == MAX_OPENROUTER_CONCURRENCY


def test_make_openrouter_backed_defaults_match_pretrain_posture():
    makefile = Path("Makefile").read_text()

    assert f"PRETRAIN_BATCH_SIZE ?= {DEFAULT_OPENROUTER_BATCH_SIZE}" in makefile
    assert f"PRETRAIN_CONCURRENCY ?= {DEFAULT_OPENROUTER_SMOKE_CONCURRENCY}" in makefile
    assert f"PRETRAIN_TARGET_CONCURRENCY ?= {DEFAULT_OPENROUTER_TARGET_CONCURRENCY}" in makefile

    for name in ("DISTILLATION_SFT", "SFT", "DPO"):
        assert f"{name}_BATCH_SIZE ?= $(PRETRAIN_BATCH_SIZE)" in makefile
        assert f"{name}_CONCURRENCY ?= $(PRETRAIN_CONCURRENCY)" in makefile
        assert f"{name}_TARGET_CONCURRENCY ?= $(PRETRAIN_TARGET_CONCURRENCY)" in makefile
        assert f"{name}_BATCH_INCREASE_SUCCESSES ?= {DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES}" in makefile

    assert "--concurrency $(DISTILLATION_SFT_TARGET_CONCURRENCY)" in makefile
    assert "--concurrency $(SFT_TARGET_CONCURRENCY)" in makefile
    assert "--concurrency $(DPO_TARGET_CONCURRENCY)" in makefile


def test_plain_batch_cli_defaults_do_not_clamp_adaptive_initial_in_flight_to_one():
    from slm_synth.dpo.cli import build_parser as build_dpo_parser
    from slm_synth.sft.cli import build_parser as build_sft_parser
    from slm_synth.distillation_sft.cli import build_parser as build_distillation_parser

    common_sft_dpo_args = [
        "generate-llm-batch",
        "--specs",
        "specs.jsonl",
        "--output",
        "rows.jsonl",
        "--manifest",
        "manifest.json",
        "--teacher-model",
        "openai/gpt-4.1-mini",
        "--generation-run",
        "batch-smoke",
        "--max-tokens",
        "1024",
    ]
    for parser_builder in (build_sft_parser, build_dpo_parser):
        args = parser_builder().parse_args(common_sft_dpo_args)
        assert args.adaptive_initial_in_flight == DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT
        assert args.adaptive_maximum_in_flight == DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT

    distillation_args = build_distillation_parser().parse_args(
        [
            "generate-batch",
            "--signal",
            "debugging",
            "--prompts",
            "prompts.jsonl",
            "--output-dir",
            "datasets",
            "--manifest-dir",
            "manifests",
            "--teacher-model",
            "openai/gpt-4.1-mini",
            "--generation-run",
            "batch-smoke",
            "--max-tokens",
            "1024",
        ]
    )
    assert distillation_args.adaptive_initial_in_flight == DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT
    assert distillation_args.adaptive_maximum_in_flight == DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT
