from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")

def test_distillation_sft_pipeline_planning_is_consistent():
    makefile = _read("Makefile")
    commands = _read("docs/COMMANDS.md")
    workflow = _read("docs/GENERATION_WORKFLOW.md")

    assert "DISTILLATION_SFT_SIGNALS ?= all" in makefile
    assert "DISTILLATION_SFT_SEEDS ?= 1" in makefile
    assert "DISTILLATION_SFT_DERIVATIONS_PER_SEED ?= 30" in makefile
    assert "DISTILLATION_SFT_TASKS_PER_DERIVATION ?= 15" in makefile
    assert "DISTILLATION_SFT_SMOKE_DERIVATIONS_PER_SEED ?= 1" in makefile
    assert "DISTILLATION_SFT_SMOKE_TASKS_PER_DERIVATION ?= 2" in makefile
    for variable in ("DISTILLATION_SFT_SEEDS", "DISTILLATION_SFT_DERIVATIONS_PER_SEED", "DISTILLATION_SFT_TASKS_PER_DERIVATION"):
        assert f"`{variable}`" in commands
    assert "DISTILLATION_SFT_DERIVATIONS_PER_SEED=30" in workflow
    assert "DISTILLATION_SFT_CANDIDATE_COUNTS" not in makefile + commands + workflow

def test_distillation_sft_supported_command_boundaries_are_documented():
    makefile = _read("Makefile")
    commands = _read("docs/COMMANDS.md")
    workflow = _read("docs/GENERATION_WORKFLOW.md")
    package_readme = _read("slm_synth/distillation_sft/README.md")

    for target in ("distillation-sft-smoke", "distillation-sft-generate", "distillation-sft-inspect", "distillation-sft-report", "distillation-sft-push"):
        assert f"{target}:" in makefile
        assert target in commands
        assert target in workflow
    assert "distillation-sft-adjudicate" not in makefile + commands + workflow
    assert "Manual post-run adjudication is not a supported path" in package_readme
