import pytest

from slm_synth.dpo.spec_builders import (
    DPO_PREFERENCE_DIMENSIONS, DPO_SPEC_CAPACITIES, build_specs, unique_capacity,
)
from slm_synth.dpo.specs import dpo_source_fingerprint, teacher_visible_dpo_spec
from slm_synth.taxonomy import PREFERENCE_DIMENSIONS, TASK_FAMILIES


def test_dpo_selector_is_exactly_the_preference_dimension_taxonomy():
    assert DPO_PREFERENCE_DIMENSIONS == PREFERENCE_DIMENSIONS
    assert len(DPO_PREFERENCE_DIMENSIONS) == 10


@pytest.mark.parametrize("dimension", sorted(DPO_PREFERENCE_DIMENSIONS))
def test_build_dpo_specs_for_each_preference_dimension(dimension):
    specs = build_specs(family=dimension, count=2)
    for spec in specs:
        metadata = spec["metadata"]
        assert metadata["preference_dimension"] == dimension
        assert metadata["task_family"] in TASK_FAMILIES
        assert metadata["failure_mode"]
        assert "eval_family" not in metadata


def test_dpo_holdout_key_stays_internal():
    spec = build_specs(family="factual_accuracy", count=1)[0]
    assert "holdout_key" in spec
    assert "holdout_key" not in teacher_visible_dpo_spec(spec)


def test_dpo_exact_constraints_are_teacher_visible_and_machine_checkable():
    spec = build_specs(family="instruction_adherence", count=3)[2]
    assert spec["output_constraints"] == {
        "min_words": 120,
        "max_words": 150,
        "forbidden_terms": ["star", "planet", "alone"],
    }
    assert teacher_visible_dpo_spec(spec)["output_constraints"] == spec["output_constraints"]


def test_dpo_sources_are_finite_and_unique():
    for dimension in DPO_PREFERENCE_DIMENSIONS:
        specs = build_specs(family=dimension, count=unique_capacity(dimension))
        assert unique_capacity(dimension) == DPO_SPEC_CAPACITIES[dimension]
        assert len({dpo_source_fingerprint(spec) for spec in specs}) == len(specs)


def test_dpo_capacity_rejects_range_beyond_source_limit():
    capacity = unique_capacity("groundedness")
    with pytest.raises(ValueError, match="finite source capacity"):
        build_specs(family="groundedness", count=2, start_index=capacity)


def test_dpo_specs_require_semantic_teacher_preference_generation():
    spec = build_specs(family="code_correctness", count=1)[0]
    assert "chosen_answer" not in spec["variables"]
    assert "rejected_answer" not in spec["variables"]
