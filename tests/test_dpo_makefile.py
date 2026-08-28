from pathlib import Path

def _target_block(makefile: str, target: str, next_target: str) -> str:
    return makefile.split(f"\n{target}:\n", 1)[1].split(f"\n{next_target}:", 1)[0]

def test_dpo_generation_targets_use_pipeline_planning_without_legacy_counts():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    smoke = _target_block(makefile, "dpo-smoke", "dpo-generate")
    generate = _target_block(makefile, "dpo-generate", "dpo-report")

    assert "slm_synth.dpo.pipeline" in smoke + generate
    assert "--dimensions $(DPO_SMOKE_PREFERENCE_DIMENSIONS_EFFECTIVE)" in smoke
    assert "--dimensions $(DPO_PREFERENCE_DIMENSIONS)" in generate
    assert "--seeds $(DPO_SEEDS)" in smoke + generate
    assert "--derivations-per-seed $(DPO_SMOKE_DERIVATIONS_PER_SEED)" in smoke
    assert "--tasks-per-derivation $(DPO_SMOKE_TASKS_PER_DERIVATION)" in smoke
    assert "--derivations-per-seed $(DPO_DERIVATIONS_PER_SEED)" in generate
    assert "--tasks-per-derivation $(DPO_TASKS_PER_DERIVATION)" in generate
    for removed in ("--candidate-counts", "--target-pairs", "--count-per-family", "--max-backfill-rounds", "--resume"):
        assert removed not in smoke + generate

def test_dpo_report_uses_configured_run_root_and_acceptance_inputs():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(makefile, "dpo-report", "dpo-inspect")
    assert "--holdout-registry $(DPO_HOLDOUT_REGISTRY)" in block
    assert "--run-manifest $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/manifests/$(DPO_REPORT_RUN).manifest.json" in block
    assert block.count("--run-dir $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)") == 2
