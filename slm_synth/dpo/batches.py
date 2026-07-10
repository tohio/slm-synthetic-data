"""LLM batch formatting and response validation for DPO generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row
from slm_synth.dpo.specs import teacher_visible_dpo_spec, validate_dpo_spec

DPO_BATCH_RESPONSE_FIELDS = frozenset({"items"})

CHAT_MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": ["user", "assistant"]},
        "content": {"type": "string", "minLength": 1},
    },
}

DPO_METADATA_SCHEMA: dict[str, Any] = {
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

DPO_BATCH_RESPONSE_SCHEMA: dict[str, Any] = {
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
                    "prompt": {
                        "type": "array",
                        "items": CHAT_MESSAGE_SCHEMA,
                    },
                    "chosen": {
                        "type": "array",
                        "items": CHAT_MESSAGE_SCHEMA,
                    },
                    "rejected": {
                        "type": "array",
                        "items": CHAT_MESSAGE_SCHEMA,
                    },
                    "metadata": DPO_METADATA_SCHEMA,
                },
            },
        }
    },
}


def build_dpo_teacher_request_items(specs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return batchable DPO specs that are safe to send to the LLM."""
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for spec in specs:
        item = teacher_visible_dpo_spec(spec)
        item_id = item["id"]
        if item_id in seen_ids:
            duplicate_ids.add(item_id)
        seen_ids.add(item_id)
        items.append(item)

    if duplicate_ids:
        raise ValueError(f"DPO specs contain duplicate id(s): {sorted(duplicate_ids)}")
    return items


def build_dpo_teacher_request_object(specs: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the DPO teacher request object."""
    return {"items": build_dpo_teacher_request_items(specs)}


def render_dpo_batch_prompt(specs: Iterable[Mapping[str, Any]]) -> str:
    """Render a JSON-only prompt for batched LLM DPO pair generation."""
    request_object = build_dpo_teacher_request_object(specs)
    request_json = json.dumps(request_object, ensure_ascii=False, indent=2)

    return (
        "You are generating synthetic preference data for DPO training.\n\n"
        "For each input spec, return exactly one DPO preference row with the same id.\n"
        "Return only a valid JSON object with this shape:\n"
        '{"items":[{"id":"string","prompt":[{"role":"user","content":"string"}],'
        '"chosen":[{"role":"assistant","content":"string"}],'
        '"rejected":[{"role":"assistant","content":"string"}],'
        '"metadata":{"category":"string","difficulty":1,"template_family":"string",'
        '"eval_family":"string|null","failure_mode":"string"}}]}\n\n'
        "Rules:\n"
        "- Do not add, remove, rename, or reorder ids.\n"
        "- Preserve metadata values from each input spec exactly.\n"
        "- The chosen response must be correct and preferred.\n"
        "- The rejected response must be realistic and reflect metadata.failure_mode.\n"
        "- The chosen and rejected responses must differ.\n"
        "- If variables.chosen_answer is present, use it exactly for the chosen assistant content.\n"
        "- If variables.rejected_answer is present, use it exactly for the rejected assistant content.\n"
        "- Do not copy known eval prompts exactly.\n"
        "- Do not include variables, constraints, holdout_key, teacher_model, teacher_provider, or generation_run.\n\n"
        "Input specs:\n"
        f"{request_json}"
    )


def validate_dpo_batch_response(
    response_object: Mapping[str, Any],
    *,
    expected_ids: Iterable[str] | None = None,
    expected_count: int | None = None,
    expected_specs: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Validate a batched LLM DPO response and return normalized rows."""
    if not isinstance(response_object, Mapping):
        raise TypeError("DPO batch response must be a mapping")

    keys = set(response_object)
    missing = DPO_BATCH_RESPONSE_FIELDS - keys
    if missing:
        raise ValueError(f"DPO batch response missing required field(s): {sorted(missing)}")

    unexpected = keys - DPO_BATCH_RESPONSE_FIELDS
    if unexpected:
        raise ValueError(f"DPO batch response contains unexpected field(s): {sorted(unexpected)}")

    items = response_object["items"]
    if not isinstance(items, list):
        raise ValueError("DPO batch response field 'items' must be a list")
    if expected_count is not None and len(items) != expected_count:
        raise ValueError(f"DPO batch response expected {expected_count} item(s), got {len(items)}")

    rows = [validate_dpo_row(item) for item in items]
    _validate_response_ids([row["id"] for row in rows], expected_ids=expected_ids)
    _validate_rows_against_specs(rows, expected_specs=expected_specs)
    return rows



def _validate_rows_against_specs(
    rows: list[dict[str, Any]],
    *,
    expected_specs: Iterable[Mapping[str, Any]] | None,
) -> None:
    if expected_specs is None:
        return
    specs_by_id = {spec["id"]: validate_dpo_spec(spec) for spec in expected_specs}
    for row in rows:
        spec = specs_by_id.get(row["id"])
        if spec is None:
            continue
        _validate_exact_targeted_row(row, spec=spec)
        _validate_prompt_does_not_leak_answer(row, spec=spec)


def _validate_exact_targeted_row(row: dict[str, Any], *, spec: Mapping[str, Any]) -> None:
    variables = spec.get("variables")
    if not isinstance(variables, Mapping):
        return
    chosen_answer = _optional_string(variables.get("chosen_answer"))
    if chosen_answer is not None:
        chosen_content = _single_assistant_content(row["chosen"], field_name="chosen")
        if chosen_content.strip() != chosen_answer.strip():
            raise ValueError(f"DPO row {row['id']} chosen content does not match variables.chosen_answer")

    rejected_answer = _optional_string(variables.get("rejected_answer"))
    if rejected_answer is not None:
        rejected_content = _single_assistant_content(row["rejected"], field_name="rejected")
        if rejected_content.strip() != rejected_answer.strip():
            raise ValueError(f"DPO row {row['id']} rejected content does not match variables.rejected_answer")


def _validate_prompt_does_not_leak_answer(row: dict[str, Any], *, spec: Mapping[str, Any]) -> None:
    metadata = spec.get("metadata")
    variables = spec.get("variables")
    if not isinstance(metadata, Mapping) or metadata.get("eval_family") != "list_exact_n_items":
        return
    if not isinstance(variables, Mapping):
        return
    prompt_text = "\n".join(message["content"] for message in row["prompt"] if message["role"] == "user").lower()
    answer = _optional_string(variables.get("answer"))
    if answer is not None and answer.lower() in prompt_text:
        raise ValueError(f"DPO row {row['id']} prompt leaks variables.answer")
    items = variables.get("items")
    if isinstance(items, list) and items and all(str(item).lower() in prompt_text for item in items):
        raise ValueError(f"DPO row {row['id']} prompt leaks variables.items")


def _single_assistant_content(messages: Any, *, field_name: str) -> str:
    if not isinstance(messages, list):
        raise ValueError(f"DPO row {field_name} must be a message list")
    assistant_messages = [message for message in messages if isinstance(message, Mapping) and message.get("role") == "assistant"]
    if len(assistant_messages) != 1:
        raise ValueError(f"DPO row {field_name} must contain exactly one assistant message")
    content = assistant_messages[0].get("content")
    if not isinstance(content, str):
        raise ValueError(f"DPO row {field_name} assistant content must be a string")
    return content


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None

def _validate_response_ids(row_ids: list[str], *, expected_ids: Iterable[str] | None) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row_id in row_ids:
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
    if duplicates:
        raise ValueError(f"DPO batch response contains duplicate id(s): {sorted(duplicates)}")

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
            f"DPO batch response missing expected id(s): {missing}; "
            f"contains unexpected id(s): {unexpected}"
        )
    if missing:
        raise ValueError(f"DPO batch response missing expected id(s): {missing}")
    if unexpected:
        raise ValueError(f"DPO batch response contains unexpected id(s): {unexpected}")
