import json

import pytest

from slm_synth.distillation.signals import DISTILLATION_SIGNALS
from slm_synth.distillation.spec_builders import (
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
