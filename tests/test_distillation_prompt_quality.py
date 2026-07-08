import pytest

from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text, validate_prompt_preflight
from slm_synth.distillation_sft.prompts import build_prompt_record
from slm_synth.distillation_sft.seeds import build_seed_prompt_records


def test_normalize_prompt_text_catches_case_spacing_and_terminal_punctuation():
    assert normalize_prompt_text("  Answer  with ONLY the integer result: 2 + 2. ") == normalize_prompt_text(
        "answer with only the integer result: 2+2"
    )


def test_prompt_preflight_rejects_duplicate_ids_before_teacher_calls():
    records = [
        build_prompt_record(signal="cloud", prompt="Explain autoscaling.", index=1),
        build_prompt_record(signal="cloud", prompt="Explain object storage.", index=1),
    ]

    with pytest.raises(ValueError, match="duplicate id"):
        validate_prompt_preflight(records, require_unique_prompt_text=True)


def test_prompt_preflight_rejects_near_duplicate_prompt_text():
    records = [
        build_prompt_record(signal="cloud", prompt="Explain autoscaling in a cloud application.", index=1),
        build_prompt_record(signal="cloud", prompt="  explain   autoscaling in a cloud application! ", index=2),
    ]

    with pytest.raises(ValueError, match="near-duplicate prompt text"):
        validate_prompt_preflight(records, require_unique_prompt_text=True)


def test_prompt_preflight_allows_cycled_smoke_seed_text_when_prompt_uniqueness_is_not_required():
    records = build_seed_prompt_records(signal="cloud", count=4)

    summary = validate_prompt_preflight(records, require_unique_prompt_text=False)

    assert summary.prompt_count == 4
    assert summary.duplicate_id_count == 0
    assert summary.duplicate_prompt_text_count == 1
    assert summary.near_duplicate_prompt_count == 1
    assert summary.to_dict()["checks"] == ["id"]
