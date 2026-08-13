from pathlib import Path


def test_distillation_dpo_make_defaults_match_approved_targets():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY ?= 1000" in makefile
    assert "DISTILLATION_DPO_TARGET_PAIRS ?= 15000" in makefile
    assert "DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY ?= 2" not in makefile
    assert "DISTILLATION_DPO_TARGET_PAIRS ?= 50000" not in makefile


def test_distillation_dpo_documentation_matches_approved_targets():
    commands = Path("docs/COMMANDS.md").read_text(encoding="utf-8")
    purpose = Path("docs/DATASET_PURPOSE.md").read_text(encoding="utf-8")
    workflow = Path("docs/GENERATION_WORKFLOW.md").read_text(encoding="utf-8")
    package_readme = Path("slm_synth/distillation_dpo/README.md").read_text(
        encoding="utf-8"
    )
    workflow_section = workflow.split("## Distillation DPO", 1)[1].split(
        "## Pretraining", 1
    )[0]

    assert "`DISTILLATION_DPO_SMOKE_COUNT_PER_FAMILY` | `1000`" in commands
    assert "`DISTILLATION_DPO_TARGET_PAIRS` | `15000`" in commands
    assert "Distillation DPO | `DISTILLATION_DPO_TARGET_PAIRS` | `15000`" in purpose
    assert "DISTILLATION_DPO_TARGET_PAIRS=15000" in workflow_section
    assert "DISTILLATION_DPO_TARGET_PAIRS=50000" not in workflow_section
    assert "DISTILLATION_DPO_TARGET_PAIRS=100" not in workflow_section
    assert "The smoke target is 1,000 accepted pairs" in package_readme
    assert "The production target is 15,000" in package_readme
