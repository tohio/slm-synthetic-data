"""Teacher batch formatting for response distillation.

This module prepares local prompt records for teacher calls and validates the
teacher's batch-shaped response. It does not perform provider requests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.distillation.prompts import validate_prompt_record
from slm_synth.distillation.signals import validate_signal
from slm_synth.distillation.validate import validate_teacher_output

TEACHER_REQUEST_FIELDS = frozenset({"id", "prompt"})
TEACHER_RESPONSE_FIELDS = frozenset({"items"})

TEACHER_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "reasoning", "response"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "reasoning": {"type": "null"},
                    "response": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


def build_teacher_request_items(
    prompt_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return the prompt payload that is safe to send to the teacher.

    Signal names, metadata, teacher model, provider, run ids, cost, and retry
    details stay local. The teacher sees only ids and prompts.
    """
    items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for record in prompt_records:
        validated = validate_prompt_record(record)
        record_id = validated["id"]
        if record_id in seen_ids:
            duplicate_ids.add(record_id)
        seen_ids.add(record_id)
        items.append({"id": record_id, "prompt": validated["prompt"]})

    if duplicate_ids:
        raise ValueError(f"prompt records contain duplicate id(s): {sorted(duplicate_ids)}")
    return items


def build_teacher_request_object(
    prompt_records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Return the JSON object payload used inside the rendered teacher prompt."""
    return {"items": build_teacher_request_items(prompt_records)}


def render_teacher_batch_prompt(
    *,
    signal: str,
    prompt_records: Iterable[Mapping[str, Any]],
) -> str:
    """Render a JSON-only batch prompt for teacher response generation."""
    normalized_signal = validate_signal(signal)
    request_object = build_teacher_request_object(prompt_records)
    request_json = json.dumps(request_object, ensure_ascii=False, indent=2)

    return (
        "You are generating response-distillation data.\n"
        f"Signal: {normalized_signal}\n\n"
        "For each input item, return exactly one output item with the same id.\n"
        "Return only a valid JSON object with this shape:\n"
        '{"items":[{"id":"string","reasoning":null,"response":"string"}]}\n\n'
        "Rules:\n"
        "- Do not add, remove, rename, or reorder ids.\n"
        "- Always set reasoning to null.\n"
        "- Keep the final response in response.\n"
        "- Do not include prompt, signal, metadata, teacher_model, teacher_provider, generation_run, or difficulty.\n\n"
        "Input items:\n"
        f"{request_json}"
    )


def validate_teacher_batch_response(
    response_object: Mapping[str, Any],
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate the teacher's batch response object and return teacher outputs."""
    if not isinstance(response_object, Mapping):
        raise TypeError("teacher batch response must be a mapping")

    keys = set(response_object)
    missing = TEACHER_RESPONSE_FIELDS - keys
    if missing:
        raise ValueError(f"teacher batch response missing required field(s): {sorted(missing)}")

    unexpected = keys - TEACHER_RESPONSE_FIELDS
    if unexpected:
        raise ValueError(f"teacher batch response contains unexpected field(s): {sorted(unexpected)}")

    items = response_object["items"]
    if not isinstance(items, list):
        raise ValueError("teacher batch response field 'items' must be a list")

    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"teacher batch response expected {expected_count} item(s), got {len(items)}")

    return [validate_teacher_output(item) for item in items]
