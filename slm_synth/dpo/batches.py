"""LLM batch formatting and response validation for generic DPO generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec

DPO_BATCH_RESPONSE_FIELDS = frozenset({"items"})

_MESSAGE = {
    "type": "object", "additionalProperties": False, "required": ["role", "content"],
    "properties": {"role": {"type": "string"}, "content": {"type": "string", "minLength": 1}},
}

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

DPO_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "prompt", "chosen", "rejected", "metadata"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "prompt": {"type": "array", "minItems": 1, "items": _MESSAGE},
            "chosen": {"type": "array", "minItems": 1, "maxItems": 1, "items": _MESSAGE},
            "rejected": {"type": "array", "minItems": 1, "maxItems": 1, "items": _MESSAGE},
            "metadata": DPO_METADATA_SCHEMA,
        },
    }}},
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


def render_dpo_batch_prompt(specs: Iterable[Mapping[str, Any]]) -> str:
    request_json = json.dumps(build_dpo_teacher_request_object(specs), ensure_ascii=False, indent=2)
    return (
        "Generate one high-quality generic DPO preference row for each input spec. Return only JSON matching the schema.\n"
        "Preserve every id and metadata value exactly. The chosen response must be materially better on the named "
        "preference_dimension; the rejected response must be plausible and exhibit failure_mode. Do not expose variables, "
        "constraints, holdout_key, fingerprints, provider data, or run data. Do not copy known evaluation prompts.\n\n"
        f"Input specs:\n{request_json}"
    )


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
    rows = [validate_dpo_row(_normalize_assistant_roles(item)) for item in items]
    _validate_ids([row["id"] for row in rows], expected_ids)
    if expected_specs is not None:
        specs_by_id = {spec["id"]: validate_dpo_spec(spec) for spec in expected_specs}
        for row in rows:
            spec = specs_by_id.get(row["id"])
            if spec is not None and row["metadata"] != spec["metadata"]:
                raise ValueError(f"DPO row {row['id']} metadata does not match its input spec")
    return rows


def _normalize_assistant_roles(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    normalized = dict(item)
    for field in ("chosen", "rejected"):
        messages = normalized.get(field)
        if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], Mapping):
            message = dict(messages[0])
            if message.get("role") in {"user", "system"}:
                message["role"] = "assistant"
                normalized[field] = [message]
    return normalized


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
