import json

import pytest

from slm_synth.dpo.batches import (
    DPO_BATCH_RESPONSE_SCHEMA, DPO_METADATA_SCHEMA, build_dpo_teacher_request_items,
    render_dpo_batch_prompt, validate_dpo_batch_response,
)
from slm_synth.dpo.spec_builders import build_specs


def _response(spec):
    return {"items": [{
        "id": spec["id"], "prompt": [{"role": "user", "content": "Please rewrite this note."}],
        "chosen": [{"role": "assistant", "content": "A clear professional rewrite."}],
        "rejected": [{"role": "assistant", "content": "A rude rewrite."}],
        "metadata": spec["metadata"],
    }]}


def test_dpo_batch_schema_uses_new_metadata_only():
    required = set(DPO_METADATA_SCHEMA["required"])
    assert {"task_family", "interaction_modes", "output_mode", "context_mode", "preference_dimension"} <= required
    assert "eval_family" not in json.dumps(DPO_BATCH_RESPONSE_SCHEMA)
    assert "category" not in DPO_METADATA_SCHEMA["properties"]


def test_dpo_teacher_request_hides_holdout_key():
    spec = build_specs(family="factual_accuracy", count=1)[0]
    item = build_dpo_teacher_request_items([spec])[0]
    assert "holdout_key" in spec
    assert "holdout_key" not in item


def test_dpo_prompt_names_preference_contract_without_internal_fields():
    spec = build_specs(family="style_and_tone", count=1)[0]
    prompt = render_dpo_batch_prompt([spec])
    assert "preference_dimension" in prompt
    assert "holdout_key" in prompt  # appears only in the prohibition rule
    assert json.dumps(spec.get("holdout_key")) not in prompt


def test_validate_dpo_batch_response_matches_spec_metadata():
    spec = build_specs(family="style_and_tone", count=1)[0]
    rows = validate_dpo_batch_response(_response(spec), expected_ids=[spec["id"]], expected_count=1, expected_specs=[spec])
    assert rows[0]["metadata"]["preference_dimension"] == "style_and_tone"


def test_validate_dpo_batch_response_rejects_metadata_drift():
    spec = build_specs(family="style_and_tone", count=1)[0]
    response = _response(spec)
    response["items"][0]["metadata"] = dict(spec["metadata"], preference_dimension="organization")
    with pytest.raises(ValueError, match="metadata does not match"):
        validate_dpo_batch_response(response, expected_specs=[spec])
