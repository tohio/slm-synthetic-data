"""LLM batch formatting and response validation for distillation-DPO generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row

DISTILLATION_DPO_BATCH_RESPONSE_FIELDS = frozenset({"items"})

DISTILLATION_DPO_PROMPT_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": ["system", "user"]},
        "content": {"type": "string", "minLength": 1},
    },
}

DISTILLATION_DPO_ASSISTANT_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": ["assistant"]},
        "content": {"type": "string", "minLength": 1},
    },
}

DISTILLATION_DPO_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "category",
        "difficulty",
        "template_family",
        "eval_family",
        "failure_mode",
    ],
    "properties": {
        "category": {"type": "string", "minLength": 1},
        "difficulty": {"type": "integer"},
        "template_family": {"type": "string", "minLength": 1},
        "eval_family": {"type": ["string", "null"]},
        "failure_mode": {"type": "string", "minLength": 1},
    },
}

DISTILLATION_DPO_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "prompt", "chosen", "rejected", "metadata"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "prompt": {"type": "array", "items": DISTILLATION_DPO_PROMPT_MESSAGE_SCHEMA},
                    "chosen": {"type": "array", "items": DISTILLATION_DPO_ASSISTANT_MESSAGE_SCHEMA},
                    "rejected": {"type": "array", "items": DISTILLATION_DPO_ASSISTANT_MESSAGE_SCHEMA},
                    "metadata": DISTILLATION_DPO_METADATA_SCHEMA,
                },
            },
        }
    },
}


def build_teacher_request_items(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return distillation-DPO row specs that are safe to send to the LLM."""
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for row in rows:
        validated = validate_distillation_dpo_row(row)
        row_id = validated["id"]
        if row_id in seen_ids:
            duplicate_ids.add(row_id)
        seen_ids.add(row_id)
        items.append(
            {
                "id": row_id,
                "prompt": validated["prompt"],
                "metadata": validated["metadata"],
                "chosen_contract": "Write the preferred teacher-quality response. It must be correct, safe, and satisfy the prompt.",
                "rejected_contract": "Write a realistic lower-quality response that demonstrates metadata.failure_mode.",
                "reference_chosen": validated["chosen"],
                "reference_rejected": validated["rejected"],
            }
        )

    if duplicate_ids:
        raise ValueError(f"distillation-DPO specs contain duplicate id(s): {sorted(duplicate_ids)}")
    return items


def render_distillation_dpo_batch_prompt(rows: Iterable[Mapping[str, Any]]) -> str:
    """Render a JSON-only prompt for batched LLM distillation-DPO generation."""
    request_object = {"items": build_teacher_request_items(rows)}
    request_json = json.dumps(request_object, ensure_ascii=False, indent=2)
    return (
        "You are generating synthetic preference data for distillation-DPO training.\n\n"
        "For each input spec, return exactly one public distillation-DPO row with the same id.\n"
        "Return only a valid JSON object with this shape:\n"
        '{"items":[{"id":"string","prompt":[{"role":"user","content":"string"}],'
        '"chosen":[{"role":"assistant","content":"string"}],'
        '"rejected":[{"role":"assistant","content":"string"}],'
        '"metadata":{"category":"string","difficulty":1,"template_family":"string",'
        '"eval_family":"string|null","failure_mode":"string"}}]}\n\n'
        "Rules:\n"
        "- Do not add, remove, rename, or reorder ids.\n"
        "- Preserve prompt and metadata values from each input spec exactly.\n"
        "- prompt must contain only system/user messages and must end with a user message.\n"
        "- Do not put assistant messages in prompt.\n"
        "- chosen must contain exactly one message whose role is assistant.\n"
        "- rejected must contain exactly one message whose role is assistant.\n"
        "- Never use role=user or role=system inside chosen or rejected.\n"
        "- The chosen response is the teacher-quality preferred answer.\n"
        "- The rejected response is a controlled weak answer and must reflect metadata.failure_mode.\n"
        "- The chosen and rejected responses must differ.\n"
        "- Use reference_chosen and reference_rejected as semantic anchors, but do not include those fields in output.\n"
        "- Do not include chosen_contract, rejected_contract, reference_chosen, reference_rejected, teacher_model, teacher_provider, or generation_run.\n\n"
        "Input specs:\n"
        f"{request_json}"
    )


def validate_distillation_dpo_batch_response(
    response_object: Mapping[str, Any],
    *,
    expected_ids: Iterable[str] | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate a batched LLM distillation-DPO response and return normalized rows."""
    if not isinstance(response_object, Mapping):
        raise TypeError("distillation-DPO batch response must be a mapping")

    keys = set(response_object)
    missing = DISTILLATION_DPO_BATCH_RESPONSE_FIELDS - keys
    if missing:
        raise ValueError(f"distillation-DPO batch response missing required field(s): {sorted(missing)}")

    unexpected = keys - DISTILLATION_DPO_BATCH_RESPONSE_FIELDS
    if unexpected:
        raise ValueError(f"distillation-DPO batch response contains unexpected field(s): {sorted(unexpected)}")

    items = response_object["items"]
    if not isinstance(items, list):
        raise ValueError("distillation-DPO batch response field 'items' must be a list")
    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"distillation-DPO batch response expected {expected_count} item(s), got {len(items)}")

    normalized_items = [_normalize_response_role_fields(item) for item in items]
    rows = [validate_distillation_dpo_row(item) for item in normalized_items]
    _validate_response_ids([row["id"] for row in rows], expected_ids=expected_ids)
    return rows



def _normalize_response_role_fields(item: Any) -> Any:
    if not isinstance(item, Mapping):
        return item
    normalized = dict(item)
    for field_name in ("chosen", "rejected"):
        normalized[field_name] = _coerce_single_response_message_to_assistant(normalized.get(field_name))
    return normalized



def _coerce_single_response_message_to_assistant(messages: Any) -> Any:
    """Repair the common teacher error of tagging answer content as user/system."""
    if not isinstance(messages, list) or len(messages) != 1:
        return messages
    message = messages[0]
    if not isinstance(message, Mapping):
        return messages
    role = message.get("role")
    if role == "assistant":
        return messages
    content = message.get("content")
    if role in {"user", "system"} and isinstance(content, str) and content.strip():
        repaired = dict(message)
        repaired["role"] = "assistant"
        return [repaired]
    return messages



def _validate_response_ids(row_ids: list[str], *, expected_ids: Iterable[str] | None) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row_id in row_ids:
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
    if duplicates:
        raise ValueError(f"distillation-DPO batch response contains duplicate id(s): {sorted(duplicates)}")

    if expected_ids is None:
        return

    expected = list(expected_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("expected_ids contains duplicate id(s)")

    actual_set = set(row_ids)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing and unexpected:
        raise ValueError(
            f"distillation-DPO batch response missing expected id(s): {missing}; "
            f"contains unexpected id(s): {unexpected}"
        )
    if missing:
        raise ValueError(f"distillation-DPO batch response missing expected id(s): {missing}")
    if unexpected:
        raise ValueError(f"distillation-DPO batch response contains unexpected id(s): {unexpected}")
