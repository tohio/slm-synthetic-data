"""LLM batch formatting and response validation for generic SFT generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.chat_schema import CHAT_MESSAGE_JSON_SCHEMA
from slm_synth.sft.schema import validate_sft_row
from slm_synth.sft.specs import teacher_visible_sft_spec
from slm_synth.render_contract import validate_rendered_output_mode
from slm_synth.tool_catalog import tools_from_spec

SFT_BATCH_RESPONSE_FIELDS = frozenset({"items"})

CHAT_MESSAGE_SCHEMA = CHAT_MESSAGE_JSON_SCHEMA

SFT_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": [
        "task_family", "interaction_modes", "output_mode", "context_mode",
        "difficulty", "template_family",
    ],
    "properties": {
        "task_family": {"type": "string", "minLength": 1},
        "interaction_modes": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "output_mode": {"type": "string", "minLength": 1},
        "context_mode": {"type": "string", "minLength": 1},
        "difficulty": {"type": "integer"},
        "template_family": {"type": "string", "minLength": 1},
    },
}

SFT_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["messages"],
        "properties": {
            "messages": {"type": "array", "minItems": 2, "items": CHAT_MESSAGE_SCHEMA},
        },
    }}},
}


def build_sft_teacher_request_items(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = [teacher_visible_sft_spec(spec) for spec in specs]
    ids = [item["id"] for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"SFT specs contain duplicate id(s): {duplicates}")
    return items


def build_sft_teacher_request_object(specs: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"items": build_sft_teacher_request_items(specs)}


def render_sft_batch_prompt(specs: Iterable[Mapping[str, Any]]) -> str:
    request_json = json.dumps(build_sft_teacher_request_object(specs), ensure_ascii=False, indent=2)
    return (
        "Generate one high-quality generic SFT row for each input spec. Return exactly one JSON object shaped as "
        '{"items":[{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}]}. '
        "Return items in input order and do not add fields outside messages. Repository code owns IDs, metadata, tools, taxonomy, and run fields. "
        "Materialize the requested interaction mode and do not expose "
        "variables, constraints, holdout_key, fingerprints, provider data, or run data. Do not copy known evaluation prompts. "
        "Put every source passage, document, code sample, question, option, constraint, or other fact needed to perform the "
        "task into the user-visible conversation. The assistant must never answer from hidden input-spec fields. "
        "When public_prompt_requirements is present, copy every listed phrase exactly into a system or user message before "
        "the final assistant response; these phrases are checked locally and may not appear only in the answer. "
        "Every message object must include content. System, user, tool, and ordinary assistant messages require non-empty "
        "string content; only an assistant message containing tool_calls may use null content. The final message must always "
        "be an assistant message. Include a leading system message if and only if interaction_modes contains "
        "system_conditioned. For single_turn, use exactly one user message followed by the final assistant response (with an "
        "optional leading system message only when system_conditioned is declared). For multi_turn, use at least two user "
        "messages, with assistant turns between them as needed, and end with the final assistant response. Never end on a user "
        "message. Do not add tools, tool_calls, or tool messages unless interaction_modes contains tool_mediated. For derived "
        "tasks whose instruction asks for a fresh task, the first user-visible task content must be the newly instantiated "
        "concrete task, not the capability-anchor/meta instruction. "
        "Do not invent a tools array. Tool definitions and structural fields are attached by repository code. Apply output_mode to "
        "the final assistant content: structured_json is only a parseable JSON object or array with no surrounding prose; "
        "table is a Markdown table with a header and separator row; code contains the requested implementation in a fenced "
        "code block; concise remains brief; exact_constraints obeys every explicit count, length, heading, and forbidden-term "
        "rule; free_text uses the form required by the instruction. Treat output_constraints as hard machine-checked "
        "requirements. When both min_words and max_words are present, target their midpoint rather than either boundary and "
        "count the final assistant words before returning. Verify every declared line count, item count, required term, "
        "forbidden term, heading, and JSON key before returning.\n\n"
        f"Input specs:\n{request_json}"
    )


def validate_sft_batch_response(
    response_object: Mapping[str, Any], *, expected_ids: Iterable[str] | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(response_object, Mapping):
        raise TypeError("SFT batch response must be a mapping")
    missing = SFT_BATCH_RESPONSE_FIELDS - set(response_object)
    extra = set(response_object) - SFT_BATCH_RESPONSE_FIELDS
    if missing:
        raise ValueError(f"SFT batch response missing required field(s): {sorted(missing)}")
    if extra:
        raise ValueError(f"SFT batch response contains unexpected field(s): {sorted(extra)}")
    items = response_object["items"]
    if not isinstance(items, list):
        raise ValueError("SFT batch response field 'items' must be a list")
    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"SFT batch response expected {expected_count} item(s), got {len(items)}")
    rows = [validate_sft_row(item) for item in items]
    _validate_response_ids([row["id"] for row in rows], expected_ids)
    return rows


def validate_sft_rows_against_specs(rows: Iterable[Mapping[str, Any]], specs: Iterable[Mapping[str, Any]]) -> None:
    spec_by_id = {str(spec["id"]): spec for spec in specs}
    for row in rows:
        spec = spec_by_id.get(str(row["id"]))
        if spec is None:
            raise ValueError(f"SFT row {row['id']} has no matching spec")
        if dict(row["metadata"]) != dict(spec["metadata"]):
            raise ValueError(f"SFT row {row['id']} metadata does not match spec metadata")
        validate_rendered_output_mode(
            row["messages"], output_mode=spec["metadata"]["output_mode"], row_id=row["id"]
        )


def attach_sft_code_fields(
    response_object: Mapping[str, Any], specs: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Attach public structure that must never be delegated to a model."""
    if not isinstance(response_object, Mapping) or set(response_object) != {"items"}:
        raise ValueError("SFT generator response must contain only an items field")
    if not isinstance(response_object["items"], list):
        raise TypeError("SFT generator items must be a list")
    validated_specs = list(specs)
    if len(response_object["items"]) != len(validated_specs):
        raise ValueError("SFT generator item count must match the input spec count")
    items: list[dict[str, Any]] = []
    for raw, spec in zip(response_object["items"], validated_specs, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {"messages"}:
            raise ValueError("SFT generator item must contain only messages")
        item = {
            "id": str(spec["id"]),
            "messages": raw["messages"],
            "metadata": dict(spec["metadata"]),
        }
        tools = tools_from_spec(spec)
        if tools is not None:
            item["tools"] = tools
        items.append(item)
    return {"items": items}


def _validate_response_ids(row_ids: list[str], expected_ids: Iterable[str] | None) -> None:
    duplicates = sorted({row_id for row_id in row_ids if row_ids.count(row_id) > 1})
    if duplicates:
        raise ValueError(f"SFT batch response contains duplicate id(s): {duplicates}")
    if expected_ids is None:
        return
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_ids contains duplicate id(s)")
    missing = sorted(set(expected) - set(row_ids))
    unexpected = sorted(set(row_ids) - set(expected))
    if missing or unexpected:
        raise ValueError(f"SFT batch response id mismatch: missing={missing}, unexpected={unexpected}")
