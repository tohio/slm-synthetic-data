from pathlib import Path


def _makefile() -> str:
    return (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")


def _target_block(makefile: str, target: str, next_target: str) -> str:
    return makefile.split(f"\n{target}:", 1)[1].split(f"\n{next_target}:", 1)[0]


def test_sft_generation_targets_enforce_holdouts_and_candidate_plans():
    makefile = _makefile()
    smoke = _target_block(makefile, "sft-smoke", "sft-generate")
    generate = _target_block(makefile, "sft-generate", "sft-report")

    for block in (smoke, generate):
        assert "--holdout-registry $(SFT_HOLDOUT_REGISTRY)" in block
    assert "--count-per-family $(SFT_SMOKE_COUNT_PER_FAMILY)" in smoke
    assert "--candidate-counts $(SFT_CANDIDATE_COUNTS)" in generate


def test_sft_report_uses_run_manifest_and_required_holdout_registry():
    makefile = _makefile()
    report = _target_block(makefile, "sft-report", "sft-inspect")

    assert "--input $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/datasets" in report
    assert "--run-manifest $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/manifests/$(SFT_REPORT_RUN).manifest.json" in report
    assert "--holdout-registry $(SFT_HOLDOUT_REGISTRY)" in report
    assert "--output $(SFT_RUN_ROOT)/$(SFT_REPORT_RUN)/coverage.json" in report


def test_sft_generation_rebuilds_acceptance_report_after_completion():
    makefile = _makefile()
    smoke = _target_block(makefile, "sft-smoke", "sft-generate")
    generate = _target_block(makefile, "sft-generate", "sft-report")

    assert "$(MAKE) sft-report SFT_REPORT_RUN=$(SFT_RUN)" in smoke
    assert "$(MAKE) sft-report SFT_REPORT_RUN=$(SFT_GENERATION_RUN)" in generate
