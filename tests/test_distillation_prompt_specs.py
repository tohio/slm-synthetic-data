import json

import pytest

from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS
from slm_synth.distillation_sft.prompt_quality import validate_prompt_preflight
from slm_synth.distillation_sft.seeds import build_seed_prompt_records
from slm_synth.distillation_sft.spec_builders import (
    build_and_write_prompt_specs,
    build_prompt_spec_records,
)


def test_production_prompt_specs_cover_all_distillation_signals():
    for signal in sorted(DISTILLATION_SIGNALS):
        rows = build_prompt_spec_records(signal=signal, count=2)
        assert [row["id"] for row in rows] == [f"{signal}-000001", f"{signal}-000002"]
        assert all(row["signal"] == signal for row in rows)
        assert all(row["metadata"]["prompt_source"] == "production_spec" for row in rows)
        assert all("seed_source" not in row["metadata"] for row in rows)
        assert all(
            set(row["metadata"]) >= {
                "category",
                "difficulty",
                "template_family",
                "eval_family",
            }
            for row in rows
        )
        assert len({row["prompt"] for row in rows}) == 2


def test_production_prompt_specs_are_deterministic_and_not_builtin_seed_records():
    first = build_prompt_spec_records(signal="arithmetic", count=4, start_index=10)
    second = build_prompt_spec_records(signal="arithmetic", count=4, start_index=10)

    assert first == second
    assert [row["id"] for row in first] == [
        "arithmetic-000010",
        "arithmetic-000011",
        "arithmetic-000012",
        "arithmetic-000013",
    ]
    assert all(row["metadata"]["template_family"].startswith("integer_") for row in first)
    assert {row["metadata"]["category"] for row in first} == {"direct_arithmetic"}
    assert {row["metadata"]["eval_family"] for row in first} == {
        "basic_arithmetic_qa",
        "direct_division",
        "direct_subtraction",
    }


def test_factual_restraint_specs_expose_filterable_categories():
    rows = build_prompt_spec_records(signal="factual_restraint", count=5)

    assert {row["metadata"]["category"] for row in rows} == {
        "future_event_restraint",
        "private_info_restraint",
    }
    assert {row["metadata"]["template_family"] for row in rows} == {"factual_restraint"}
    assert {row["metadata"]["eval_family"] for row in rows} == {None}


def test_distillation_smoke_target_has_unique_prompt_text():
    records = [
        row
        for signal in sorted(DISTILLATION_SIGNALS)
        for row in build_seed_prompt_records(signal=signal, count=200)
    ]

    summary = validate_prompt_preflight(records, require_unique_prompt_text=True)

    assert summary.prompt_count == 2_000
    assert summary.duplicate_prompt_text_count == 0
    assert summary.near_duplicate_prompt_count == 0


def test_distillation_production_target_has_unique_prompt_text():
    records = [
        row
        for signal in sorted(DISTILLATION_SIGNALS)
        for row in build_prompt_spec_records(signal=signal, count=3_000)
    ]

    summary = validate_prompt_preflight(records, require_unique_prompt_text=True)

    assert summary.prompt_count == 30_000
    assert summary.duplicate_prompt_text_count == 0
    assert summary.near_duplicate_prompt_count == 0

    backfill_records = [
        row
        for signal in sorted(DISTILLATION_SIGNALS)
        for row in build_prompt_spec_records(signal=signal, count=5, start_index=3_001)
    ]
    backfill_summary = validate_prompt_preflight(
        records + backfill_records,
        require_unique_prompt_text=True,
    )

    assert backfill_summary.prompt_count == 30_050
    assert backfill_summary.duplicate_prompt_text_count == 0
    assert backfill_summary.near_duplicate_prompt_count == 0


def test_build_and_write_prompt_specs_writes_valid_jsonl(tmp_path):
    output = tmp_path / "cloud_specs.jsonl"

    count = build_and_write_prompt_specs(signal="cloud", count=2, output_path=output)

    assert count == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["metadata"]["prompt_source"] for row in rows] == ["production_spec", "production_spec"]


@pytest.mark.parametrize("count", [-1, "2"])
def test_production_prompt_specs_reject_bad_count(count):
    with pytest.raises(ValueError, match="non-negative integer"):
        build_prompt_spec_records(signal="cloud", count=count)
