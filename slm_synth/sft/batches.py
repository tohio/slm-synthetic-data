"""LLM batch formatting and response validation for generic SFT generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.chat_schema import CHAT_MESSAGE_JSON_SCHEMA, TOOLS_JSON_SCHEMA
from slm_synth.sft.schema import validate_sft_row
from slm_synth.sft.specs import teacher_visible_sft_spec

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
        "required": ["id", "messages", "metadata"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "messages": {"type": "array", "minItems": 2, "items": CHAT_MESSAGE_SCHEMA},
            "tools": TOOLS_JSON_SCHEMA,
            "metadata": SFT_METADATA_SCHEMA,
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
        "Generate one high-quality generic SFT row for each input spec. Return only JSON matching the supplied schema.\n"
        "Preserve every id and metadata value exactly. Materialize the requested interaction mode and do not expose "
        "variables, constraints, holdout_key, fingerprints, provider data, or run data. Do not copy known evaluation prompts. "
        "For tool-mediated tasks, emit one shared tools array, structured assistant tool_calls with object arguments, matching "
        "tool responses, and a final assistant answer. Never serialize tool arguments as JSON strings.\n\n"
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
