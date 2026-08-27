from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text


def test_normalize_prompt_text_collapses_cosmetic_variation():
    assert normalize_prompt_text("  What is 2 + 2 ?  ") == normalize_prompt_text("what is 2+2?")


def test_normalize_prompt_text_preserves_task_content():
    assert normalize_prompt_text("What is 2 + 2?") != normalize_prompt_text("What is 3 + 2?")
