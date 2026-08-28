from pathlib import Path


def test_distillation_dpo_make_defaults_match_pipeline_planning():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "DISTILLATION_DPO_DIMENSIONS ?= all" in makefile
    assert "DISTILLATION_DPO_SMOKE_DIMENSIONS ?= factual_accuracy" in makefile
    assert "DISTILLATION_DPO_SEEDS ?= 1" in makefile
    assert "DISTILLATION_DPO_DERIVATIONS_PER_SEED ?= 30" in makefile
    assert "DISTILLATION_DPO_TASKS_PER_DERIVATION ?= 15" in makefile
    assert "DISTILLATION_DPO_SMOKE_DERIVATIONS_PER_SEED ?= 1" in makefile
    assert "DISTILLATION_DPO_SMOKE_TASKS_PER_DERIVATION ?= 2" in makefile
    assert "DISTILLATION_DPO_TARGET_PAIRS" not in makefile
    assert "DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY" not in makefile


def test_distillation_dpo_documentation_matches_pipeline_planning():
    commands = Path("docs/COMMANDS.md").read_text(encoding="utf-8")
    workflow = Path("docs/GENERATION_WORKFLOW.md").read_text(encoding="utf-8")
    package_readme = Path("slm_synth/distillation_dpo/README.md").read_text(encoding="utf-8")

    for variable in (
        "DISTILLATION_DPO_DIMENSIONS",
        "DISTILLATION_DPO_SEEDS",
        "DISTILLATION_DPO_DERIVATIONS_PER_SEED",
        "DISTILLATION_DPO_TASKS_PER_DERIVATION",
    ):
        assert f"`{variable}`" in commands
    assert "DISTILLATION_DPO_DERIVATIONS_PER_SEED=30" in workflow
    assert "five-gate Gemma judge" in workflow
    assert "five-gate Gemma judge" in package_readme
    assert "DISTILLATION_DPO_TARGET_PAIRS" not in commands + workflow + package_readme
