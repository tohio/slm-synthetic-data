import json

import pytest

from slm_synth.sft.generation import read_specs_jsonl
from slm_synth.sft.spec_builders import (
    SFT_SPEC_CAPACITIES, SFT_SPEC_FAMILIES, build_and_write_specs, build_specs,
    unique_capacity, validate_spec_range, write_specs_jsonl,
)
from slm_synth.sft.specs import require_unique_sft_sources, sft_source_fingerprint, teacher_visible_sft_spec
from slm_synth.taxonomy import TASK_FAMILIES


def test_sft_selector_is_exactly_the_task_family_taxonomy():
    assert SFT_SPEC_FAMILIES == TASK_FAMILIES
    assert len(SFT_SPEC_FAMILIES) == 10


@pytest.mark.parametrize("family", sorted(SFT_SPEC_FAMILIES))
def test_build_sft_specs_for_each_task_family(family):
    specs = build_specs(family=family, count=2)
    assert [spec["id"] for spec in specs] == [f"sft_{family}_000001", f"sft_{family}_000002"]
    for spec in specs:
        metadata = spec["metadata"]
        assert metadata["task_family"] == family
        assert set(metadata) == {
            "task_family", "interaction_modes", "output_mode", "context_mode",
            "difficulty", "template_family",
        }


def test_sft_holdout_key_is_internal_and_not_part_of_source_fingerprint():
    spec = build_specs(family="applied_math_and_reasoning", count=1)[0]
    visible = teacher_visible_sft_spec(spec)
    assert "holdout_key" in spec
    assert "holdout_key" not in visible
    assert "eval_family" not in json.dumps(spec)


def test_sft_specs_are_finite_and_materially_distinct():
    for family in SFT_SPEC_FAMILIES:
        specs = build_specs(family=family, count=unique_capacity(family))
        assert unique_capacity(family) == SFT_SPEC_CAPACITIES[family]
        assert len({sft_source_fingerprint(spec) for spec in specs}) == len(specs)


def test_sft_range_rejects_crossing_finite_capacity():
    capacity = unique_capacity("summarization")
    validate_spec_range(family="summarization", count=1, start_index=capacity)
    with pytest.raises(ValueError, match="finite source capacity"):
        build_specs(family="summarization", count=2, start_index=capacity)


def test_sft_source_fingerprint_ignores_id():
    first = build_specs(family="programming", count=1)[0]
    renamed = dict(first, id="different-id")
    assert sft_source_fingerprint(first) == sft_source_fingerprint(renamed)
    with pytest.raises(ValueError, match="repeated teacher-visible"):
        require_unique_sft_sources([first, renamed])


def test_sft_spec_jsonl_round_trip(tmp_path):
    path = tmp_path / "nested" / "sft.specs.jsonl"
    assert build_and_write_specs(family="rewriting_and_editing", count=2, output_path=path) == 2
    rows = read_specs_jsonl(path)
    assert len(rows) == 2
    assert write_specs_jsonl(rows, tmp_path / "copy.jsonl") == 2
