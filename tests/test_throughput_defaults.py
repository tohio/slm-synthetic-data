from pathlib import Path

from configs import configure_synthetic
from slm_synth.model_support import is_supported_model
from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    DEFAULT_OPENROUTER_BATCH_SIZE,
    DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    DEFAULT_OPENROUTER_TARGET_CONCURRENCY,
    MAX_OPENROUTER_BATCH_SIZE,
    MAX_OPENROUTER_CONCURRENCY,
)


def test_make_live_generation_paths_inherit_a_validated_default_model():
    makefile = Path("Makefile").read_text()
    default_model = next(
        line.split("?=", 1)[1].strip()
        for line in makefile.splitlines()
        if line.startswith("MODEL ?=")
    )

    assert is_supported_model(default_model)
    for name in ("PRETRAIN", "DISTILLATION_SFT", "DISTILLATION_DPO", "SFT", "DPO"):
        assert f"{name}_MODEL ?= $(MODEL)" in makefile


def test_make_live_generation_targets_preserve_openrouter_routing_controls():
    makefile = Path("Makefile").read_text()

    assert "OPENROUTER_ROUTING_MODE ?= auto" in makefile
    assert "OPENROUTER_ROUTING_ARGS := --openrouter-routing-mode $(OPENROUTER_ROUTING_MODE)" in makefile
    targets = {
        "pretrain-smoke": False,
        "pretrain-generate": False,
        "distillation-sft-smoke": True,
        "distillation-sft-generate": True,
        "distillation-dpo-smoke": False,
        "distillation-dpo-generate": False,
        "sft-smoke": True,
        "sft-generate": True,
        "dpo-smoke": True,
        "dpo-generate": True,
    }
    for target, uses_cli_routing_args in targets.items():
        recipe = makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
        assert "$(OPENROUTER_ENV)" in recipe
        if uses_cli_routing_args:
            assert "$(OPENROUTER_ROUTING_ARGS)" in recipe


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
