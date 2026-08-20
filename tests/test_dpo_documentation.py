from pathlib import Path

from slm_synth.dpo.spec_builders import DPO_PREFERENCE_DIMENSIONS


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dpo_package_readme_describes_implemented_boundaries():
    readme = _read("slm_synth/dpo/README.md")

    for module in ("acceptance.py", "runs.py", "report.py", "card.py", "push_hf.py"):
        assert module in readme
    for behavior in (
        "normalized ID, prompt, and `(prompt, chosen, rejected)` triple",
        "exits nonzero",
        "DPO_HF_REPO",
        "one complete run",
    ):
        assert behavior in readme
    assert "distillation-specific pairs" in readme


def test_dpo_command_reference_documents_resume_holdouts_and_one_repo():
    commands = _read("docs/COMMANDS.md")

    for variable in (
        "DPO_MAX_BACKFILL_ROUNDS",
        "DPO_RESUME",
        "DPO_HOLDOUT_REGISTRY",
        "DPO_HF_REPO",
    ):
        assert f"`{variable}`" in commands
    assert "DPO_RESUME=true" in commands
    assert "one atomic version in `DPO_HF_REPO`" in commands


def test_dpo_workflow_documents_acceptance_and_consolidated_publication():
    workflow = _read("docs/GENERATION_WORKFLOW.md")

    for heading in (
        "### DPO acceptance and backfill",
        "### DPO reporting and publish blockers",
        "### DPO consolidated publication",
    ):
        assert heading in workflow
    assert "DPO_HF_REPO=tohio/slm-synthetic-dpo" in workflow
    assert "Similarity and repeated negative patterns" not in workflow
    assert "Chosen/rejected similarity and repeated negative patterns are reported" in workflow


def test_documented_dpo_family_table_covers_every_supported_family():
    document = _read("docs/GENERATION_FAMILIES.md")
    dpo_section = document.split("## DPO Preference Dimensions", 1)[1].split("## Distillation SFT Signals", 1)[0]

    for family in DPO_PREFERENCE_DIMENSIONS:
        assert f"`{family}`" in dpo_section
    assert "unique source capacity" in dpo_section
    assert "inspection signals rather than automatic rejection thresholds" in dpo_section
