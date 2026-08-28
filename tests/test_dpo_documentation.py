from pathlib import Path
from slm_synth.dpo.pipeline import DPO_DIMENSIONS

ROOT = Path(__file__).resolve().parents[1]

def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")

def test_dpo_package_readme_describes_current_pipeline():
    readme = _read("slm_synth/dpo/README.md")
    for stage in ("derivation", "concrete task", "chosen/rejected pair generation", "deterministic pair validation", "Nemotron judge", "Gemma reviewer", "final exact preference-triple dedup"):
        assert stage in readme
    assert "one consolidated DPO repository" in readme
    assert "runs.py" not in readme

def test_dpo_command_reference_documents_current_planning_variables():
    commands = _read("docs/COMMANDS.md")
    for variable in ("DPO_GENERATION_RUN", "DPO_PREFERENCE_DIMENSIONS", "DPO_SEEDS", "DPO_DERIVATIONS_PER_SEED", "DPO_TASKS_PER_DERIVATION", "DPO_PAIR_BATCH_SIZE"):
        assert f"`{variable}`" in commands
    assert "DPO_CANDIDATE_COUNTS" not in commands

def test_dpo_workflow_documents_current_pipeline_and_publication():
    workflow = _read("docs/GENERATION_WORKFLOW.md")
    section = workflow.split("## Generic DPO", 1)[1].split("## Distillation SFT", 1)[0]
    assert "DPO_DERIVATIONS_PER_SEED=30" in section
    assert "DPO_TASKS_PER_DERIVATION=15" in section
    assert "Nemotron judge" in section
    assert "Gemma reviewer" in section
    assert "final preference-triple dedup" in section

def test_documented_dpo_dimension_table_covers_every_supported_dimension():
    document = _read("docs/GENERATION_FAMILIES.md")
    section = document.split("## Generic DPO Preference Dimensions", 1)[1].split("## Distillation SFT Signals", 1)[0]
    for dimension in DPO_DIMENSIONS:
        assert f"`{dimension}`" in section
    assert "plain-text prompt/chosen/rejected semantics" in section
