from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_distillation_sft_candidate_planning_is_consistent():
    makefile = _read("Makefile")
    commands = _read("docs/COMMANDS.md")
    workflow = _read("docs/GENERATION_WORKFLOW.md")
    purpose = _read("docs/DATASET_PURPOSE.md")

    assert "DISTILLATION_SFT_SMOKE_COUNT_PER_SIGNAL ?= 2" in makefile
    assert "DISTILLATION_SFT_CANDIDATE_COUNTS ?=" in makefile
    assert "DISTILLATION_SFT_CANDIDATE_COUNTS" in commands
    assert "DISTILLATION_SFT_CANDIDATE_COUNTS" in workflow
    assert "candidate generation budget" in purpose


def test_distillation_sft_review_workflow_documents_every_command_boundary():
    makefile = _read("Makefile")
    commands = _read("docs/COMMANDS.md")
    workflow = _read("docs/GENERATION_WORKFLOW.md")
    package_readme = _read("slm_synth/distillation_sft/README.md")

    for target in (
        "distillation-sft-report",
        "distillation-sft-adjudicate",
        "distillation-sft-push",
    ):
        assert f"{target}:" in makefile
        assert target in commands
        assert target in workflow

    assert "member_fingerprint" in commands
    assert '"decision": "keep"' in commands
    assert "rejected candidates are not" in commands
    assert "block publication while unresolved" in package_readme
