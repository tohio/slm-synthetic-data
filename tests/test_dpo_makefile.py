from pathlib import Path


def _target_block(makefile: str, target: str, next_target: str) -> str:
    return makefile.split(f"\n{target}:\n", 1)[1].split(f"\n{next_target}:", 1)[0]


def test_dpo_generation_targets_require_explicit_candidate_plans_without_backfill():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    smoke = _target_block(makefile, "dpo-smoke", "dpo-generate")
    generate = _target_block(makefile, "dpo-generate", "dpo-report")
    assert "--candidate-counts $(DPO_SMOKE_CANDIDATE_COUNTS_EFFECTIVE)" in smoke
    assert "--candidate-counts $(DPO_CANDIDATE_COUNTS)" in generate
    assert "--preference-dimensions $(DPO_SMOKE_PREFERENCE_DIMENSIONS_EFFECTIVE)" in smoke
    assert "--preference-dimensions $(DPO_PREFERENCE_DIMENSIONS)" in generate
    assert "--families" not in smoke + generate
    for removed in ("--target-pairs", "--count-per-family", "--max-backfill-rounds", "--resume"):
        assert removed not in smoke + generate
    assert not any(line.startswith("DPO_TARGET_PAIRS ?=") for line in makefile.splitlines())
    assert not any(line.startswith("DPO_COUNT_PER_FAMILY ?=") for line in makefile.splitlines())
    assert "$(if $(filter file,$(origin DPO_PREFERENCE_DIMENSIONS)),$(DPO_SMOKE_PREFERENCE_DIMENSIONS),$(DPO_PREFERENCE_DIMENSIONS))" in makefile
    assert "$(if $(filter file,$(origin DPO_CANDIDATE_COUNTS)),$(DPO_SMOKE_CANDIDATE_COUNTS),$(DPO_CANDIDATE_COUNTS))" in makefile


def test_dpo_report_uses_configured_run_root_and_acceptance_inputs():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    block = _target_block(makefile, "dpo-report", "dpo-inspect")

    assert "--holdout-registry $(DPO_HOLDOUT_REGISTRY)" in block
    assert "--run-manifest $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)/manifests/$(DPO_REPORT_RUN).manifest.json" in block
    assert block.count("--run-dir $(DPO_RUN_ROOT)/$(DPO_REPORT_RUN)") == 2
    assert "data/dpo/runs/$(DPO_REPORT_RUN)" not in block
