import json
import re
from collections import Counter, defaultdict

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
    assert {row["metadata"]["template_family"] for row in rows} == {
        "restraint_account_credential",
        "restraint_future_event_prediction",
        "restraint_future_private_financial",
        "restraint_private_address",
        "restraint_private_medical",
    }
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
    _assert_template_scale(rows=records, count_per_signal=3_000)

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


def test_debugging_and_factual_specs_use_substantive_30k_scale_variation():
    debugging = build_prompt_spec_records(signal="debugging", count=3_000)
    factual = build_prompt_spec_records(signal="factual_restraint", count=3_000)

    debugging_without_case_ids = {
        re.sub(r"^Diagnostic case \d+: ", "", row["prompt"])
        for row in debugging
    }
    factual_without_case_ids = {
        re.sub(r"^Restraint case \d+: ", "", row["prompt"])
        for row in factual
    }

    assert len(debugging_without_case_ids) == 3_000
    assert len(factual_without_case_ids) == 3_000
    assert all("def " in row["prompt"] for row in debugging)
    assert all("Keep the answer under 60 words." in row["prompt"] for row in factual)
    assert set(Counter(row["metadata"]["template_family"] for row in debugging).values()) == {600}
    assert set(Counter(row["metadata"]["template_family"] for row in factual).values()) == {600}


def test_distillation_inventory_scales_to_100k_rows():
    records = [
        row
        for signal in sorted(DISTILLATION_SIGNALS)
        for row in build_prompt_spec_records(signal=signal, count=10_000)
    ]

    summary = validate_prompt_preflight(records, require_unique_prompt_text=True)

    assert summary.prompt_count == 100_000
    assert summary.duplicate_prompt_text_count == 0
    assert summary.near_duplicate_prompt_count == 0
    _assert_template_scale(rows=records, count_per_signal=10_000)

    backfill_records = [
        row
        for signal in sorted(DISTILLATION_SIGNALS)
        for row in build_prompt_spec_records(signal=signal, count=100, start_index=10_001)
    ]
    backfill_summary = validate_prompt_preflight(
        records + backfill_records,
        require_unique_prompt_text=True,
    )

    assert backfill_summary.prompt_count == 101_000
    assert backfill_summary.near_duplicate_prompt_count == 0


def _assert_template_scale(*, rows, count_per_signal):
    counts_by_signal = defaultdict(Counter)
    all_templates = set()
    for row in rows:
        signal = row["signal"]
        template_family = row["metadata"]["template_family"]
        counts_by_signal[signal][template_family] += 1
        all_templates.add(template_family)

    assert len(all_templates) == 50
    assert set(counts_by_signal) == DISTILLATION_SIGNALS
    for counts in counts_by_signal.values():
        assert len(counts) >= 4
        assert max(counts.values()) / count_per_signal <= 0.30


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
