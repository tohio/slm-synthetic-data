import json

import pytest

from slm_synth.sft.batches import (
    SFT_BATCH_RESPONSE_SCHEMA,
    SFT_METADATA_SCHEMA,
    attach_sft_code_fields,
    build_sft_teacher_request_items,
    render_sft_batch_prompt,
    validate_sft_batch_response,
    validate_sft_rows_against_specs,
)
from slm_synth.sft.spec_builders import build_specs


def _response(spec):
    messages = [{"role": "user", "content": "Use the supplied passage."}, {"role": "assistant", "content": "Grounded answer."}]
    return {"items": [{"id": spec["id"], "messages": messages, "metadata": spec["metadata"]}]}


def test_sft_batch_schema_uses_new_metadata_only():
    required = set(SFT_METADATA_SCHEMA["required"])
    assert {"task_family", "interaction_modes", "output_mode", "context_mode"} <= required
    assert "eval_family" not in json.dumps(SFT_BATCH_RESPONSE_SCHEMA)
    assert "category" not in SFT_METADATA_SCHEMA["properties"]
    item_schema = SFT_BATCH_RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert set(item_schema["properties"]) == {"messages"}
    assert "metadata" not in item_schema["properties"]
    assert "tool_calls" in item_schema["properties"]["messages"]["items"]["properties"]
    assert "content" in item_schema["properties"]["messages"]["items"]["required"]
    assert item_schema["properties"]["messages"]["items"]["allOf"]


def test_sft_teacher_request_hides_holdout_key():
    spec = build_specs(family="applied_math_and_reasoning", count=1)[0]
    item = build_sft_teacher_request_items([spec])[0]
    assert "holdout_key" in spec
    assert "holdout_key" not in item


def test_sft_prompt_contains_generation_contract():
    prompt = render_sft_batch_prompt(build_specs(family="grounded_qa_and_reading", count=1))
    assert "high-quality generic SFT" in prompt
    assert "Repository code owns IDs, metadata, tools, taxonomy, and run fields" in prompt
    assert "Do not invent a tools array" in prompt
    assert "if and only if interaction_modes contains system_conditioned" in prompt
    assert "exactly one user turn for single_turn" in prompt
    assert "assistant must never answer from hidden input-spec fields" in prompt
    assert "structured_json is only a parseable JSON" in prompt
    assert "target their midpoint rather than either boundary" in prompt
    assert "hard machine-checked requirements" in prompt
    assert "copy every listed phrase exactly" in prompt
    assert "may not appear only in the answer" in prompt


def test_validate_sft_batch_and_spec_metadata_binding():
    spec = build_specs(family="grounded_qa_and_reading", count=1)[0]
    rows = validate_sft_batch_response(_response(spec), expected_ids=[spec["id"]], expected_count=1)
    validate_sft_rows_against_specs(rows, [spec])


def test_validate_sft_rows_rejects_metadata_drift():
    spec = build_specs(family="grounded_qa_and_reading", count=1)[0]
    response = _response(spec)
    response["items"][0]["metadata"] = dict(spec["metadata"], output_mode="concise")
    rows = validate_sft_batch_response(response)
    with pytest.raises(ValueError, match="metadata does not match"):
        validate_sft_rows_against_specs(rows, [spec])


def test_validate_sft_batch_rejects_id_mismatch():
    spec = build_specs(family="grounded_qa_and_reading", count=1)[0]
    with pytest.raises(ValueError, match="id mismatch"):
        validate_sft_batch_response({"items": []}, expected_ids=[spec["id"]])


def test_attach_sft_code_fields_builds_tools_without_model_owned_structure():
    spec = build_specs(family="grounded_qa_and_reading", count=1)[0]
    spec["metadata"] = {
        **spec["metadata"],
        "interaction_modes": ["single_turn", "tool_mediated"],
    }
    spec["variables"] = {**spec["variables"], "tool": "lookup(query, region?)"}
    response = {
        "items": [
            {
                "messages": [
                    {"role": "user", "content": "Look this up."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": {"query": "this"},
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "result"},
                    {"role": "assistant", "content": "Here is the result."},
                ]
            }
        ]
    }

    attached = attach_sft_code_fields(response, [spec])["items"][0]

    assert attached["id"] == spec["id"]
    assert attached["metadata"] == spec["metadata"]
    assert attached["tools"][0]["function"]["name"] == "lookup"
    assert attached["tools"][0]["function"]["parameters"]["required"] == ["query"]
