from __future__ import annotations

from slm_synth.dpo.batches import (
    DPO_ASSISTANT_CONTENT_MAX_LENGTH,
    DPO_ASSISTANT_MESSAGE_SCHEMA,
    render_dpo_batch_prompt,
)
from slm_synth.dpo.spec_builders import build_specs


def test_dpo_assistant_response_schema_has_bounded_content() -> None:
    content_schema = DPO_ASSISTANT_MESSAGE_SCHEMA["properties"]["content"]

    assert content_schema["minLength"] == 1
    assert content_schema["maxLength"] == DPO_ASSISTANT_CONTENT_MAX_LENGTH
    assert DPO_ASSISTANT_CONTENT_MAX_LENGTH <= 1200


def test_code_explanation_no_code_specs_are_compact_single_line_contracts() -> None:
    spec = build_specs(family="code_explanation_no_code", count=1)[0]
    contract = spec["instruction"] + "\n" + "\n".join(spec["constraints"])

    assert "compact single-line assistant messages" in contract
    assert "under 220 characters" in contract
    assert "literal newlines" in contract
    assert "quotes, or backslashes" in contract


def test_dpo_batch_prompt_includes_code_explanation_compaction_rules() -> None:
    spec = build_specs(family="code_explanation_no_code", count=1)[0]

    prompt = render_dpo_batch_prompt([spec])

    assert "For metadata.eval_family=code_explanation_no_code" in prompt
    assert "under 220 characters" in prompt
    assert "single-line strings" in prompt
    assert "No Markdown fences" in prompt
