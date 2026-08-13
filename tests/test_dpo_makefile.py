from pathlib import Path


def _target_block(makefile: str, target: str, next_target: str) -> str:
    return makefile.split(f"{target}:\n", 1)[1].split(f"\n{next_target}:", 1)[0]


def test_dpo_generation_targets_support_explicit_resume():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target, next_target in (("dpo-smoke", "dpo-generate"), ("dpo-generate", "dpo-report")):
        block = _target_block(makefile, target, next_target)
        assert "--max-backfill-rounds $(DPO_MAX_BACKFILL_ROUNDS)" in block
        assert "$(DPO_RESUME_ARG)" in block

    assert "DPO_RESUME ?= false" in makefile
    assert "DPO_RESUME_ARG := $(if $(filter true yes 1,$(DPO_RESUME)),--resume,)" in makefile
