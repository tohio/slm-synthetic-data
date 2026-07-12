import pytest

from slm_synth.distillation_sft.prompts import build_prompt_record, build_prompt_records, format_prompt_id
from slm_synth.distillation_sft.seeds import (
    DISTILLATION_PROMPT_SEEDS,
    MIN_SEED_PROMPTS_PER_SIGNAL,
    build_seed_prompt_records,
    iter_seed_prompts,
)
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS


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
        "metadata": {
            "source": "unit-test",
            "category": "general_instruction_following",
            "difficulty": 2,
            "template_family": "cloud_architecture_explanation",
            "eval_family": None,
        },
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


def test_builtin_seed_prompts_have_validation_sized_unique_inventory():
    for signal in DISTILLATION_SIGNALS:
        prompts = DISTILLATION_PROMPT_SEEDS[signal]
        assert len(prompts) >= MIN_SEED_PROMPTS_PER_SIGNAL
        normalized = {" ".join(prompt.casefold().strip().split()).strip(" .!?;:") for prompt in prompts}
        assert len(normalized) == len(prompts)


def test_iter_seed_prompts_validates_signal():
    prompts = list(iter_seed_prompts("instruction"))
    assert len(prompts) >= 1

    with pytest.raises(ValueError, match="Unsupported distillation signal"):
        list(iter_seed_prompts("unknown"))


def test_build_seed_prompt_records_uses_unique_validation_sized_prompt_window():
    rows = build_seed_prompt_records(
        signal="arithmetic",
        count=MIN_SEED_PROMPTS_PER_SIGNAL,
        start_index=7,
    )

    assert rows[0]["id"] == "arithmetic-000007"
    assert rows[-1]["id"] == f"arithmetic-{6 + MIN_SEED_PROMPTS_PER_SIGNAL:06d}"
    builtin_rows = [row for row in rows if row["metadata"].get("seed_source") == "builtin"]
    parameterized_rows = [
        row for row in rows if row["metadata"].get("seed_source") == "parameterized_spec"
    ]
    assert [row["metadata"]["seed_index"] for row in builtin_rows] == list(
        range(6, MIN_SEED_PROMPTS_PER_SIGNAL)
    )
    assert len(parameterized_rows) == 6
    assert len({row["prompt"] for row in rows}) == MIN_SEED_PROMPTS_PER_SIGNAL
    assert all("response" not in row for row in rows)
