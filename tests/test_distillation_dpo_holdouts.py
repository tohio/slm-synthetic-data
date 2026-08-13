import json
from pathlib import Path

import pytest

from slm_synth.distillation_dpo.report import (
    build_coverage_report,
    require_publish_ready_report,
)
from slm_synth.distillation_dpo.runs import generate_llm_run
from slm_synth.distillation_dpo.spec_builders import build_production_rows
from slm_synth.taxonomy.holdouts import HoldoutRecord, HoldoutRegistry


FAMILY = "teacher_response_preference"


def _registry_for_prompt(prompt):
    return HoldoutRegistry(
        [
            HoldoutRecord(
                id="protected-eval-item",
                eval_family="direct_subtraction",
                prompt=prompt,
                answer=None,
            )
        ]
    )


def test_generation_rejects_candidate_holdout_before_provider_setup(tmp_path, monkeypatch):
    row = build_production_rows(family=FAMILY, count=1)[0]
    prompt = row["prompt"][-1]["content"]
    monkeypatch.setattr(
        "slm_synth.distillation_dpo.runs.build_openrouter_backend",
        lambda **_kwargs: pytest.fail("provider setup must not run"),
    )

    with pytest.raises(ValueError, match="candidate prompt matches eval holdout prompt"):
        generate_llm_run(
            families=[FAMILY],
            count_per_family=1,
            batch_size=1,
            output_dir=tmp_path / "datasets",
            manifest_dir=tmp_path / "manifests",
            teacher_model="unused",
            generation_run="holdout-collision",
            max_tokens=1,
            concurrency=1,
            holdout_registry=_registry_for_prompt(prompt),
        )


def test_report_requires_holdout_check_and_blocks_collisions(tmp_path):
    row = build_production_rows(family=FAMILY, count=1)[0]
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    unchecked = build_coverage_report([dataset_path])
    assert unchecked["holdouts"]["status"] == "not_checked"
    assert unchecked["dataset_acceptance"]["publish_blockers"] == [
        "holdouts_not_checked"
    ]

    clean = build_coverage_report(
        [dataset_path],
        holdout_registry=HoldoutRegistry([]),
    )
    assert clean["holdouts"] == {
        "status": "checked",
        "collision_count": 0,
        "collision_ids": [],
    }
    require_publish_ready_report(clean)

    prompt = row["prompt"][-1]["content"]
    collision = build_coverage_report(
        [dataset_path],
        holdout_registry=_registry_for_prompt(prompt),
    )
    assert collision["holdouts"]["collision_ids"] == [row["id"]]
    assert collision["dataset_acceptance"]["publish_blockers"] == [
        "eval_holdout_collisions"
    ]
    with pytest.raises(ValueError, match="eval_holdout_collisions"):
        require_publish_ready_report(collision)


def test_distillation_dpo_make_targets_use_holdout_registry():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    block = makefile.split("distillation-dpo-smoke:", 1)[1].split("sft-smoke:", 1)[0]

    assert "DISTILLATION_DPO_HOLDOUT_REGISTRY ?= configs/eval_holdouts.yaml" in makefile
    assert block.count("--holdout-registry $(DISTILLATION_DPO_HOLDOUT_REGISTRY)") == 3
