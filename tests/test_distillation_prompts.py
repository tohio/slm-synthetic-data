import pytest

from slm_synth.distillation.prompts import build_prompt_record, build_prompt_records, format_prompt_id
from slm_synth.distillation.seeds import (
    DISTILLATION_PROMPT_SEEDS,
    build_seed_prompt_records,
    iter_seed_prompts,
)
from slm_synth.distillation.signals import DISTILLATION_SIGNALS


def test_format_prompt_id_uses_normalized_signal_and_padded_index():
    assert format_prompt_id(" Arithmetic ", 12) == "arithmetic-000012"


@pytest.mark.parametrize("index", [0, -1])
def test_format_prompt_id_rejects_non_positive_index(index):
    with pytest.raises(ValueError, match="positive integer"):
        format_prompt_id("code", index)


def test_build_prompt_record_tracks_internal_signal_and_metadata():
    record = build_prompt_record(
        signal="cloud",
        prompt="Explain autoscaling.",
        index=3,
        metadata={"source": "unit-test"},
    )

    assert record == {
        "id": "cloud-000003",
        "prompt": "Explain autoscaling.",
        "signal": "cloud",
        "metadata": {"source": "unit-test"},
    }


def test_build_prompt_records_uses_contiguous_ids():
    rows = build_prompt_records(
        signal="planning",
        prompts=["Plan A.", "Plan B."],
        start_index=10,
    )

    assert [row["id"] for row in rows] == ["planning-000010", "planning-000011"]
    assert all(row["signal"] == "planning" for row in rows)


def test_builtin_seed_prompts_cover_exact_supported_signal_set():
    assert set(DISTILLATION_PROMPT_SEEDS) == DISTILLATION_SIGNALS
    assert all(DISTILLATION_PROMPT_SEEDS[signal] for signal in DISTILLATION_SIGNALS)


def test_iter_seed_prompts_validates_signal():
    prompts = list(iter_seed_prompts("instruction"))
    assert len(prompts) >= 1

    with pytest.raises(ValueError, match="Unsupported distillation signal"):
        list(iter_seed_prompts("unknown"))


def test_build_seed_prompt_records_cycles_prompts_and_keeps_metadata_internal():
    rows = build_seed_prompt_records(signal="arithmetic", count=5, start_index=7)

    assert [row["id"] for row in rows] == [
        "arithmetic-000007",
        "arithmetic-000008",
        "arithmetic-000009",
        "arithmetic-000010",
        "arithmetic-000011",
    ]
    assert [row["metadata"]["seed_index"] for row in rows] == [0, 1, 2, 0, 1]
    assert all(row["metadata"]["seed_source"] == "builtin" for row in rows)
    assert all("response" not in row for row in rows)
