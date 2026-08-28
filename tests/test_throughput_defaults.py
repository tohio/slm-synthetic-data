from pathlib import Path
from configs import configure_synthetic
from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_BATCH_SIZE, DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    DEFAULT_OPENROUTER_TARGET_CONCURRENCY, MAX_OPENROUTER_BATCH_SIZE,
    MAX_OPENROUTER_CONCURRENCY,
)

def test_make_uses_role_specific_model_defaults():
    makefile = Path("Makefile").read_text()
    expected = {
        "PRETRAIN_MODEL": "openai/gpt-5.6-luna-pro",
        "PRETRAIN_JUDGE_MODEL": "google/gemma-4-31b-it",
        "PRETRAIN_REVIEWER_MODEL": "openai/gpt-5.6-luna-pro",
        "SFT_DERIVATION_MODEL": "openai/gpt-5.6-luna-pro",
        "SFT_TASK_MODEL": "deepseek/deepseek-v4-flash",
        "SFT_ANSWER_MODEL": "deepseek/deepseek-v4-flash",
        "SFT_JUDGE_MODEL": "nvidia/nemotron-3.5-lightning",
        "SFT_REVIEWER_MODEL": "google/gemma-4-31b-it",
        "DPO_DERIVATION_MODEL": "openai/gpt-5.6-luna-pro",
        "DPO_TASK_MODEL": "deepseek/deepseek-v4-flash",
        "DPO_PAIR_MODEL": "deepseek/deepseek-v4-flash",
        "DPO_JUDGE_MODEL": "nvidia/nemotron-3.5-lightning",
        "DPO_REVIEWER_MODEL": "google/gemma-4-31b-it",
    }
    for name, model in expected.items():
        assert f"{name} ?= {model}" in makefile

def test_make_live_generation_targets_preserve_openrouter_routing_controls():
    makefile = Path("Makefile").read_text()
    assert "OPENROUTER_ROUTING_MODE ?= auto" in makefile
    assert "OPENROUTER_ENV :=" in makefile
    for target in ("pretrain-smoke", "pretrain-generate", "distillation-sft-smoke", "distillation-sft-generate", "distillation-dpo-smoke", "distillation-dpo-generate", "sft-smoke", "sft-generate", "dpo-smoke", "dpo-generate"):
        recipe = makefile.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
        assert "$(OPENROUTER_ENV)" in recipe
        assert ("--routing-mode $(OPENROUTER_ROUTING_MODE)" in recipe or "$(OPENROUTER_ROUTING_ARGS)" in recipe)

def test_grounded_pretrain_config_uses_shared_throughput_bounds():
    assert configure_synthetic.MIN_BATCH_SIZE == 1
    assert configure_synthetic.MAX_BATCH_SIZE == MAX_OPENROUTER_BATCH_SIZE
    assert configure_synthetic.MIN_CONCURRENCY == 1
    assert configure_synthetic.MAX_CONCURRENCY == MAX_OPENROUTER_CONCURRENCY

def test_pretrain_make_defaults_match_configure_synthetic_throughput_posture():
    makefile = Path("Makefile").read_text()
    assert f"PRETRAIN_BATCH_SIZE ?= {DEFAULT_OPENROUTER_BATCH_SIZE}" in makefile
    assert f"PRETRAIN_CONCURRENCY ?= {DEFAULT_OPENROUTER_SMOKE_CONCURRENCY}" in makefile
    assert f"PRETRAIN_TARGET_CONCURRENCY ?= {DEFAULT_OPENROUTER_TARGET_CONCURRENCY}" in makefile
    for name in ("SFT", "DPO", "DISTILLATION_SFT", "DISTILLATION_DPO"):
        assert f"{name}_CONCURRENCY ?= 8" in makefile
