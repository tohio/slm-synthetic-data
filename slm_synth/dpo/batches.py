"""LLM batch formatting and response validation for generic DPO generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.chat_schema import CHAT_MESSAGE_JSON_SCHEMA
from slm_synth.dpo.schema import validate_dpo_chosen_candidate, validate_dpo_row
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec
from slm_synth.render_contract import validate_rendered_output_mode
from slm_synth.tool_catalog import tools_from_spec

DPO_BATCH_RESPONSE_FIELDS = frozenset({"items"})

_MESSAGE = CHAT_MESSAGE_JSON_SCHEMA

DPO_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": [
        "task_family", "interaction_modes", "output_mode", "context_mode",
        "difficulty", "template_family", "preference_dimension", "failure_mode",
    ],
    "properties": {
        "task_family": {"type": "string", "minLength": 1},
        "interaction_modes": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
        "output_mode": {"type": "string", "minLength": 1},
        "context_mode": {"type": "string", "minLength": 1},
        "difficulty": {"type": "integer"},
        "template_family": {"type": "string", "minLength": 1},
        "preference_dimension": {"type": "string", "minLength": 1},
        "failure_mode": {"type": "string", "minLength": 1},
    },
}

def build_dpo_teacher_request_items(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = [teacher_visible_dpo_spec(spec) for spec in specs]
    ids = [item["id"] for item in items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"DPO specs contain duplicate id(s): {duplicates}")
    return items


def build_dpo_teacher_request_object(specs: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"items": build_dpo_teacher_request_items(specs)}


DPO_CHOSEN_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["prompt", "chosen"],
        "properties": {
            "prompt": {"type": "array", "minItems": 1, "items": _MESSAGE},
            "chosen": {"type": "array", "minItems": 1, "items": _MESSAGE},
        },
    }}},
}

DPO_REJECTED_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["rejected"],
        "properties": {
            "rejected": {"type": "array", "minItems": 1, "items": _MESSAGE},
        },
    }}},
}


def render_dpo_chosen_prompt(specs: Iterable[Mapping[str, Any]]) -> str:
    request_json = json.dumps(build_dpo_teacher_request_object(specs), ensure_ascii=False, indent=2)
    return (
        "Generate the shared prompt and one highest-quality chosen response for each grounded DPO brief. "
        "Return exactly one JSON object with an items array; each item contains only prompt and chosen, in input order. "
        "Do not generate a rejected response; repository code attaches IDs, metadata, and tools. "
        "The chosen branch must be correct, complete, grounded in supplied material, and satisfy every source constraint. "
        "Put every source passage, document, code sample, question, option, constraint, or other fact needed to perform the "
        "task into the shared user-visible prompt; the chosen branch may not rely on hidden input-spec fields. "
        "Every message object must include content. System, user, tool, and ordinary assistant messages require non-empty "
        "string content; only an assistant message containing tool_calls may use null content. The shared prompt must start "
        "with a system message if and only if interaction_modes contains system_conditioned, contain exactly one user turn "
        "for single_turn or at least two user turns for multi_turn, and end with a user message. Do not add tools or structured "
        "tool activity unless interaction_modes contains tool_mediated. "
        "For tool-mediated tasks, use valid structured tool activity in chosen but do not return a tools array. Apply output_mode to the "
        "chosen branch's final content: structured_json is only parseable JSON; table is a Markdown table with a separator "
        "row; code uses a fenced code block; exact_constraints obeys every explicit surface constraint; concise remains brief. "
        "Treat output_constraints as hard machine-checked requirements. When both min_words and max_words are present, "
        "target their midpoint rather than either boundary and count the final chosen words before returning. Verify every "
        "declared line count, item count, required term, forbidden term, heading, and JSON key before returning.\n\n"
        f"Input specs:\n{request_json}"
    )


def render_dpo_rejected_prompt(
    specs: Iterable[Mapping[str, Any]], chosen_items: Iterable[Mapping[str, Any]]
) -> str:
    payload = {
        "items": [
            {"spec": teacher_visible_dpo_spec(spec), "accepted_chosen_candidate": dict(chosen)}
            for spec, chosen in zip(specs, chosen_items, strict=True)
        ]
    }
    return (
        "Generate only the rejected branch for each DPO item. Return exactly one JSON object with an items array; each item "
        "contains only rejected, in input order. Start from the supplied "
        "chosen candidate and introduce exactly one plausible controlled weakness: the requested failure_mode on the named "
        "preference_dimension. Preserve all unrelated strengths. Do not alter or repeat the shared prompt, tools, metadata, "
        "or chosen branch. Do not use a fabricated wrong-number shortcut unless the grounded brief explicitly requests a "
        "numeric error. Every branch message must include content; only an assistant message containing tool_calls may use "
        "null content. Tool branches may use only the supplied shared tools. Preserve the chosen branch's machine-checkable "
        "output constraints unless violating one is exactly the requested failure_mode.\n\n"
        f"Input specs and chosen candidates:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def merge_dpo_generation_stages(
    *, specs: Iterable[Mapping[str, Any]], chosen_response: Mapping[str, Any],
    rejected_response: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    expected_ids = [spec["id"] for spec in validated_specs]
    chosen_items = _validate_stage_items(chosen_response, expected_ids=expected_ids, stage="chosen")
    rejected_items = _validate_stage_items(rejected_response, expected_ids=expected_ids, stage="rejected")
    merged: list[dict[str, Any]] = []
    specs_by_id = {spec["id"]: spec for spec in validated_specs}
    for chosen, rejected in zip(chosen_items, rejected_items, strict=True):
        spec = specs_by_id[chosen["id"]]
        item = {
            **chosen,
            "rejected": rejected["rejected"],
            "metadata": dict(spec["metadata"]),
        }
        tools = tools_from_spec(spec)
        if tools is not None:
            item["tools"] = tools
        row = validate_dpo_row(item)
        if row["metadata"] != spec["metadata"]:
            raise ValueError(f"DPO row {row['id']} metadata does not match its input spec")
        validate_rendered_output_mode(
            row["chosen"], output_mode=spec["metadata"]["output_mode"], row_id=row["id"]
        )
        merged.append(row)
    return {"items": merged}


def validate_dpo_chosen_stage(
    response: Mapping[str, Any], *, specs: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    expected_ids = [spec["id"] for spec in validated_specs]
    specs_by_id = {spec["id"]: spec for spec in validated_specs}
    items = _validate_stage_items(response, expected_ids=expected_ids, stage="chosen")
    normalized: list[dict[str, Any]] = []
    for item in items:
        spec = specs_by_id[item["id"]]
        value = {**item, "metadata": dict(spec["metadata"])}
        tools = tools_from_spec(spec)
        if tools is not None:
            value["tools"] = tools
        normalized.append(value)
    rows = [validate_dpo_chosen_candidate(item) for item in normalized]
    for row in rows:
        spec = specs_by_id[row["id"]]
        if row["metadata"] != spec["metadata"]:
            raise ValueError(f"DPO row {row['id']} metadata does not match its input spec")
        validate_rendered_output_mode(
            row["chosen"], output_mode=spec["metadata"]["output_mode"], row_id=row["id"]
        )
    return rows


def _validate_stage_items(
    response: Mapping[str, Any], *, expected_ids: list[str], stage: str
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != {"items"} or not isinstance(response["items"], list):
        raise ValueError(f"DPO {stage} stage must contain only an items list")
    items = [dict(item) for item in response["items"] if isinstance(item, Mapping)]
    if len(items) != len(response["items"]):
        raise TypeError(f"DPO {stage} stage item must be an object")
    if len(items) != len(expected_ids):
        raise ValueError(f"DPO {stage} stage item count must match the input spec count")
    required = {"rejected"} if stage == "rejected" else {"prompt", "chosen"}
    optional = set()
    for item in items:
        if not required <= set(item) or set(item) - required - optional:
            raise ValueError(f"DPO {stage} stage item fields do not match the contract")
    return [{"id": item_id, **item} for item_id, item in zip(expected_ids, items, strict=True)]


def validate_dpo_batch_response(
    response_object: Mapping[str, Any], *, expected_ids: Iterable[str] | None = None,
    expected_count: int | None = None, expected_specs: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(response_object, Mapping):
        raise TypeError("DPO batch response must be a mapping")
    missing = DPO_BATCH_RESPONSE_FIELDS - set(response_object)
    extra = set(response_object) - DPO_BATCH_RESPONSE_FIELDS
    if missing:
        raise ValueError(f"DPO batch response missing required field(s): {sorted(missing)}")
    if extra:
        raise ValueError(f"DPO batch response contains unexpected field(s): {sorted(extra)}")
    items = response_object["items"]
    if not isinstance(items, list):
        raise ValueError("DPO batch response field 'items' must be a list")
    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"DPO batch response expected {expected_count} item(s), got {len(items)}")
    rows = [validate_dpo_row(item) for item in items]
    _validate_ids([row["id"] for row in rows], expected_ids)
    if expected_specs is not None:
        specs_by_id = {spec["id"]: validate_dpo_spec(spec) for spec in expected_specs}
        for row in rows:
            spec = specs_by_id.get(row["id"])
            if spec is not None and row["metadata"] != spec["metadata"]:
                raise ValueError(f"DPO row {row['id']} metadata does not match its input spec")
            if spec is not None:
                validate_rendered_output_mode(
                    row["chosen"], output_mode=spec["metadata"]["output_mode"], row_id=row["id"]
                )
    return rows


def _validate_ids(row_ids: list[str], expected_ids: Iterable[str] | None) -> None:
    duplicates = sorted({row_id for row_id in row_ids if row_ids.count(row_id) > 1})
    if duplicates:
        raise ValueError(f"DPO batch response contains duplicate id(s): {duplicates}")
    if expected_ids is None:
        return
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_ids contains duplicate id(s)")
    missing = sorted(set(expected) - set(row_ids))
    unexpected = sorted(set(row_ids) - set(expected))
    if missing or unexpected:
        raise ValueError(f"DPO batch response id mismatch: missing={missing}, unexpected={unexpected}")
