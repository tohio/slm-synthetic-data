from collections import Counter

import pytest

from slm_synth.distillation_dpo.pair_quality import validate_pair_quality
from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row
from slm_synth.distillation_dpo.spec_builders import (
    CATEGORY_WEIGHTS,
    SOURCE_CAPACITY_CEILING,
    build_production_rows,
    build_source_capacity_summary,
    require_source_capacity,
)


FAMILY = "teacher_response_preference"


def test_approved_category_allocation_is_exact_at_15000_pairs():
    summary = build_source_capacity_summary(family=FAMILY, count=15_000)

    assert summary["categories"] == {
        "answer_only_compliance": 1_125,
        "code_generation": 2_250,
        "concise_factual_qa": 1_500,
        "controlled_verbosity": 375,
        "direct_arithmetic": 1_125,
        "exact_output_format_control": 1_125,
        "future_event_restraint": 750,
        "general_instruction_following": 3_000,
        "incomplete_prompt_handling": 375,
        "no_persona_fabrication": 375,
        "private_info_restraint": 750,
        "refusal_calibration": 375,
        "unknown_fact_restraint": 750,
        "word_problem_arithmetic": 1_125,
    }


@pytest.mark.parametrize("count", [15_000, 50_000, 100_000])
def test_source_prompts_and_preference_triples_scale_without_duplicates(count):
    summary = build_source_capacity_summary(family=FAMILY, count=count)

    assert summary["row_count"] == count
    assert summary["unique_prompt_count"] == count
    assert summary["unique_triple_count"] == count


def test_100000_target_preflights_all_300000_candidate_indexes():
    summary = require_source_capacity(
        family=FAMILY,
        target_pairs=100_000,
        max_backfill_rounds=2,
    )

    assert summary["row_count"] == SOURCE_CAPACITY_CEILING
    assert summary["unique_prompt_count"] == SOURCE_CAPACITY_CEILING
    assert summary["unique_triple_count"] == SOURCE_CAPACITY_CEILING
    assert summary["next_start_index"] == SOURCE_CAPACITY_CEILING + 1


def test_backfill_range_does_not_recreate_initial_source_content():
    initial = build_production_rows(family=FAMILY, count=1_000, start_index=1)
    replacement = build_production_rows(family=FAMILY, count=2_000, start_index=1_001)

    initial_prompts = {row["prompt"][0]["content"] for row in initial}
    replacement_prompts = {row["prompt"][0]["content"] for row in replacement}
    initial_triples = {
        (
            row["prompt"][0]["content"],
            row["chosen"][0]["content"],
            row["rejected"][0]["content"],
        )
        for row in initial
    }
    replacement_triples = {
        (
            row["prompt"][0]["content"],
            row["chosen"][0]["content"],
            row["rejected"][0]["content"],
        )
        for row in replacement
    }
    assert initial_prompts.isdisjoint(replacement_prompts)
    assert initial_triples.isdisjoint(replacement_triples)


def test_start_index_rebuilds_the_same_stable_slice():
    complete = build_production_rows(family=FAMILY, count=80, start_index=1)
    rebuilt_slice = build_production_rows(family=FAMILY, count=20, start_index=41)

    assert rebuilt_slice == complete[40:60]


def test_every_schedule_slot_is_schema_valid_and_passes_existing_pair_gates():
    rows = build_production_rows(family=FAMILY, count=sum(CATEGORY_WEIGHTS.values()))

    assert Counter(row["metadata"]["category"] for row in rows) == Counter(CATEGORY_WEIGHTS)
    for row in rows:
        validate_distillation_dpo_row(row)
        assert validate_pair_quality(row) == ()


def test_capacity_overflow_fails_locally():
    with pytest.raises(ValueError, match="source capacity exceeded"):
        require_source_capacity(
            family=FAMILY,
            target_pairs=100_001,
            max_backfill_rounds=2,
        )
