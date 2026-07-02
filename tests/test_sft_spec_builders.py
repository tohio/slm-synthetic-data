import json

import pytest

from slm_synth.sft.generation import read_specs_jsonl
from slm_synth.sft.spec_builders import (
    SFT_SPEC_FAMILIES,
    build_and_write_specs,
    build_specs,
    write_specs_jsonl,
)
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec


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
