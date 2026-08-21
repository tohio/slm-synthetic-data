import json

import pytest

from slm_synth.dpo.batches import (
    DPO_CHOSEN_BATCH_RESPONSE_SCHEMA, DPO_METADATA_SCHEMA, build_dpo_teacher_request_items,
    render_dpo_chosen_prompt, validate_dpo_batch_response,
)
from slm_synth.dpo.spec_builders import build_specs


def _response(spec):
    prompt = [{"role": "user", "content": "Please rewrite this note."}]
    if "system_conditioned" in spec["metadata"]["interaction_modes"]:
        prompt.insert(0, {"role": "system", "content": "Use a respectful professional tone."})
    return {"items": [{
        "id": spec["id"], "prompt": prompt,
        "chosen": [{"role": "assistant", "content": "A clear professional rewrite."}],
        "rejected": [{"role": "assistant", "content": "A rude rewrite."}],
        "metadata": spec["metadata"],
    }]}


def test_dpo_batch_schema_uses_new_metadata_only():
    required = set(DPO_METADATA_SCHEMA["required"])
    assert {"task_family", "interaction_modes", "output_mode", "context_mode", "preference_dimension"} <= required
    assert "eval_family" not in json.dumps(DPO_CHOSEN_BATCH_RESPONSE_SCHEMA)
    assert "category" not in DPO_METADATA_SCHEMA["properties"]
    item_schema = DPO_CHOSEN_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert set(item_schema["properties"]) == {"prompt", "chosen"}
    assert "metadata" not in item_schema["properties"]
    assert "maxItems" not in item_schema["properties"]["chosen"]
    assert "content" in item_schema["properties"]["prompt"]["items"]["required"]
    assert item_schema["properties"]["prompt"]["items"]["allOf"]


def test_dpo_teacher_request_hides_holdout_key():
    spec = build_specs(family="factual_accuracy", count=1)[0]
    item = build_dpo_teacher_request_items([spec])[0]
    assert "holdout_key" in spec
    assert "holdout_key" not in item


def test_dpo_prompt_names_preference_contract_without_internal_fields():
    spec = build_specs(family="style_and_tone", count=1)[0]
    prompt = render_dpo_chosen_prompt([spec])
    assert "preference_dimension" in prompt
    if "holdout_key" in spec:
        assert json.dumps(spec["holdout_key"]) not in prompt
    assert "repository code attaches IDs, metadata, and tools" in prompt
    assert "if and only if interaction_modes contains system_conditioned" in prompt
    assert "exactly one user turn for single_turn" in prompt
    assert "chosen branch may not rely on hidden input-spec fields" in prompt
    assert "target their midpoint rather than either boundary" in prompt
    assert "hard machine-checked requirements" in prompt



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


def test_validate_dpo_batch_response_does_not_rewrite_branch_roles():
    spec = build_specs(family="helpfulness_and_completeness", count=1)[0]
    response = _response(spec)
    response["items"][0]["chosen"][0]["role"] = "user"
    with pytest.raises(ValueError, match="branch contains unsupported role"):
        validate_dpo_batch_response(response)
