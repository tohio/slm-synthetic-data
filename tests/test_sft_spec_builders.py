import json

import pytest

from slm_synth.sft.generation import read_specs_jsonl
from slm_synth.sft.spec_builders import (
    SFT_SPEC_CAPACITIES,
    SFT_SPEC_FAMILIES,
    build_and_write_specs,
    build_specs,
    unique_capacity,
    validate_spec_range,
    write_specs_jsonl,
)
from slm_synth.sft.specs import (
    require_unique_sft_sources,
    sft_source_fingerprint,
    teacher_visible_sft_spec,
    validate_sft_spec,
)


@pytest.mark.parametrize("family", sorted(SFT_SPEC_FAMILIES))
def test_build_sft_specs_for_each_family(family):
    specs = build_specs(family=family, count=2, start_index=3)

    assert len(specs) == 2
    assert specs[0]["id"] == f"sft_{family}_000003"
    assert specs[1]["id"] == f"sft_{family}_000004"
    for spec in specs:
        validated = validate_sft_spec(spec)
        assert validated["metadata"]["eval_family"] == family
        assert "instruction" in validated
        assert "variables" in validated


def test_sft_spec_builder_keeps_holdout_key_local_only():
    spec = build_specs(family="basic_arithmetic_qa", count=1)[0]
    visible = teacher_visible_sft_spec(spec)

    assert "holdout_key" in spec
    assert "holdout_key" not in visible
    assert visible["metadata"]["category"] == "direct_arithmetic"


def test_write_sft_specs_jsonl_round_trips_through_generation_reader(tmp_path):
    specs = build_specs(family="repeat_exact_n_times", count=2)
    path = tmp_path / "sft.specs.jsonl"

    count = write_specs_jsonl(specs, path)

    assert count == 2
    rows = read_specs_jsonl(path)
    assert [row["id"] for row in rows] == [
        "sft_repeat_exact_n_times_000001",
        "sft_repeat_exact_n_times_000002",
    ]


def test_build_and_write_sft_specs_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "sft.specs.jsonl"

    count = build_and_write_specs(
        family="code_expression_result",
        count=1,
        output_path=path,
        start_index=8,
    )

    assert count == 1
    row = json.loads(path.read_text().strip())
    assert row["id"] == "sft_code_expression_result_000008"


def test_build_sft_specs_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unsupported SFT spec family"):
        build_specs(family="unknown_family", count=1)


def test_build_sft_specs_rejects_bad_count():
    with pytest.raises(ValueError, match="count"):
        build_specs(family="basic_arithmetic_qa", count=0)


@pytest.mark.parametrize("family", sorted(SFT_SPEC_FAMILIES))
def test_sft_family_declares_capacity_above_current_target(family):
    assert unique_capacity(family) == SFT_SPEC_CAPACITIES[family]
    assert unique_capacity(family) >= 1000


@pytest.mark.parametrize("family", sorted(SFT_SPEC_FAMILIES))
def test_sft_production_target_has_unique_teacher_visible_sources(family):
    specs = build_specs(family=family, count=1000)
    fingerprints = {sft_source_fingerprint(spec) for spec in specs}

    assert len(fingerprints) == 1000


def test_sft_source_fingerprint_does_not_treat_id_as_content():
    first = build_specs(family="basic_arithmetic_qa", count=1)[0]
    renamed = dict(first, id="different-id")

    assert sft_source_fingerprint(first) == sft_source_fingerprint(renamed)
    with pytest.raises(ValueError, match="repeated teacher-visible source content"):
        require_unique_sft_sources([first, renamed])


def test_sft_spec_range_allows_final_capacity_index():
    capacity = unique_capacity("direct_division")

    validate_spec_range(family="direct_division", count=1, start_index=capacity)
    spec = build_specs(family="direct_division", count=1, start_index=capacity)[0]
    assert spec["id"] == f"sft_direct_division_{capacity:06d}"


def test_sft_spec_range_rejects_first_index_beyond_capacity():
    capacity = unique_capacity("direct_division")

    with pytest.raises(ValueError, match="exceeds declared unique source capacity"):
        build_specs(family="direct_division", count=1, start_index=capacity + 1)


def test_sft_spec_range_rejects_range_crossing_capacity():
    capacity = unique_capacity("repeat_exact_n_times")

    with pytest.raises(ValueError, match="exceeds declared unique source capacity"):
        validate_spec_range(family="repeat_exact_n_times", count=2, start_index=capacity)


def test_sft_start_index_is_stable_across_partitioned_builds():
    complete = build_specs(family="ai_concept_explanation", count=20, start_index=991)
    partitioned = [
        *build_specs(family="ai_concept_explanation", count=7, start_index=991),
        *build_specs(family="ai_concept_explanation", count=13, start_index=998),
    ]

    assert complete == partitioned