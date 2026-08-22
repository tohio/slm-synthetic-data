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
        "oneOf": [
            {
                "type": "object", "additionalProperties": False,
                "required": ["messages"],
                "properties": {
                    "messages": {"type": "array", "minItems": 2, "items": CHAT_MESSAGE_SCHEMA},
                },
            },
            {
                "type": "object", "additionalProperties": False,
                "required": ["assistant_content"],
                "properties": {
                    "assistant_content": {"type": "string", "minLength": 1},
                },
            },
        ]
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



def _public_task_role_skeleton(interaction_modes: Iterable[str]) -> list[str]:
    """Return the code-owned role sequence for a materialized public task prefix."""
    modes = set(interaction_modes)
    roles: list[str] = []
    if "system_conditioned" in modes:
        roles.append("system")
    if "multi_turn" in modes:
        roles.extend(["user", "assistant", "user"])
    else:
        roles.append("user")
    return roles


def render_sft_task_materialization_prompt(
    specs: Iterable[Mapping[str, Any]],
) -> str:
    """Render derived plans into language for code-owned public task slots."""
    items: list[dict[str, Any]] = []
    for spec in specs:
        visible = teacher_visible_sft_spec(spec)
        visible["materialization_roles"] = _public_task_role_skeleton(
            visible["metadata"]["interaction_modes"]
        )
        items.append(visible)
    request_json = json.dumps({"items": items}, ensure_ascii=False, indent=2)
    return (
        "Materialize one concrete public SFT task for each input plan. Do not answer the "
        "final task. Repository code owns the conversation role sequence. Return exactly one "
        "JSON object shaped as "
        '{"items":[{"contents":["text for slot 1","text for slot 2"]}]}. '
        "Return items in input order and do not add fields outside contents. For each input, "
        "return exactly one non-empty content string for every role listed in "
        "materialization_roles, in that same order. Do not return role names yourself. Replace "
        "the capability-anchor/meta planning brief with a genuinely new concrete task: "
        "instantiate fresh actors, facts, source passages, code, quantities, dates, examples, "
        "or other task material as appropriate. Do not merely rename entities or swap numbers. "
        "Every fact, source, rubric, label set, and task-specific requirement needed for the "
        "final answer must appear in the task language. Never expose variables, "
        "derivation_profile, capability-anchor language, repository metadata, or planning "
        "commentary. Preserve the declared task family, interaction mode, output mode, safety "
        "posture, and capability. Repository code will append code-owned response constraints "
        "to the final user turn after materialization, so do not try to reproduce "
        "public_prompt_requirements literally. The final materialization slot is always a user "
        "request ready for the final assistant answer. Any assistant slot before it is prior "
        "conversation context, not the final answer.\n\n"
        f"Input plans:\n{request_json}"
    )


def attach_and_validate_sft_task_materializations(
    response_object: Mapping[str, Any], specs: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Attach materialized language to code-owned public task role skeletons."""
    validated_specs = [dict(spec) for spec in specs]
    if not isinstance(response_object, Mapping) or set(response_object) != {"items"}:
        raise ValueError("SFT task materializer response must contain only an items field")
    items = response_object["items"]
    if not isinstance(items, list):
        raise TypeError("SFT task materializer items must be a list")
    if len(items) != len(validated_specs):
        raise ValueError("SFT task materializer item count must match the input spec count")

    materialized: list[dict[str, Any]] = []
    for raw, spec in zip(items, validated_specs, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {"contents"}:
            raise ValueError("SFT task materializer item must contain only contents")
        contents = raw["contents"]
        roles = _public_task_role_skeleton(spec["metadata"]["interaction_modes"])
        if not isinstance(contents, list):
            raise TypeError("SFT task materializer contents must be a list")
        if len(contents) != len(roles):
            raise ValueError(
                "SFT task materializer content count must match the code-owned role skeleton"
            )
        messages: list[dict[str, str]] = []
        for index, (role, content) in enumerate(zip(roles, contents, strict=True)):
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"SFT task materializer content {index} must be a non-empty string"
                )
            messages.append({"role": role, "content": content.strip()})
        _append_public_prompt_requirements(
            messages, spec.get("public_prompt_requirements", [])
        )
        updated = dict(spec)
        updated["public_task_messages"] = messages
        materialized.append(updated)
    return materialized


def _append_public_prompt_requirements(
    messages: list[dict[str, str]], requirements: Any
) -> None:
    """Append code-owned response requirements to the final public user turn."""
    if not requirements:
        return
    if not isinstance(requirements, list):
        raise TypeError("public_prompt_requirements must be a list")
    normalized: list[str] = []
    for requirement in requirements:
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("public_prompt_requirements entries must be non-empty strings")
        normalized.append(requirement.strip())
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("materialized public task must end with user before requirements")
    section = "Response requirements:\n" + "\n".join(
        f"- {requirement}" for requirement in normalized
    )
    messages[-1]["content"] = messages[-1]["content"].rstrip() + "\n\n" + section


def build_sft_answer_request_object(
    specs: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the answer request while keeping code-owned structure out of model control."""
    items: list[dict[str, Any]] = []
    for spec in specs:
        validated = teacher_visible_sft_spec(spec)
        if "public_task_messages" not in validated:
            items.append(validated)
            continue
        answer_item: dict[str, Any] = {
            "id": validated["id"],
            "public_task_messages": validated["public_task_messages"],
            "output_mode": validated["metadata"]["output_mode"],
        }
        if "output_constraints" in validated:
            answer_item["output_constraints"] = validated["output_constraints"]
        items.append(answer_item)
    return {"items": items}


def render_sft_batch_prompt(specs: Iterable[Mapping[str, Any]]) -> str:
    request_json = json.dumps(
        build_sft_answer_request_object(specs), ensure_ascii=False, indent=2
    )
    return (
        "Generate one high-quality generic SFT result for each input item. Return exactly one "
        "JSON object with an items array and preserve input order. Repository code owns IDs, "
        "metadata, task prefixes, roles, tools, taxonomy, and run fields. For an item containing "
        "public_task_messages, the concrete public task is already complete: return exactly "
        '{"assistant_content":"..."} for that item and nothing else. Do not copy, rewrite, or '
        "return the existing task messages. Repository code will append assistant_content as the "
        "single final assistant turn. For an item without public_task_messages, return the legacy "
        '{"messages":[...]} object for that item and follow the declared interaction mode. '
        "Never expose variables, constraints, holdout keys, fingerprints, provider data, or run "
        "data. For legacy items, put every source passage, document, code sample, question, "
        "option, constraint, or other fact needed to perform the task into the user-visible "
        "conversation. The assistant must never answer from hidden input-spec fields. Include a "
        "leading system message if and only if interaction_modes contains system_conditioned. "
        "For single_turn legacy items, use exactly one user message followed by the final "
        "assistant response. For multi_turn legacy items, use at least two user messages with "
        "assistant turns between them and end with the final assistant response. Do not add tools, "
        "tool_calls, or tool messages unless interaction_modes contains tool_mediated. Apply "
        "output_mode to the final assistant content: structured_json is only a parseable JSON "
        "object or array with no surrounding prose; table is a Markdown table with a header and "
        "separator row; code contains the requested implementation in a fenced code block; concise "
        "remains brief; exact_constraints obeys every explicit count, length, heading, and "
        "forbidden-term rule; free_text uses the form required by the task. Treat "
        "output_constraints as hard machine-checked requirements. When both min_words and "
        "max_words are present, target their midpoint rather than either boundary and count the "
        "final assistant words before returning. Verify every declared line count, item count, "
        "required term, forbidden term, heading, and JSON key before returning.\n\n"
        f"Input items:\n{request_json}"
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
    """Attach structure owned by repository code to model-generated language."""
    if not isinstance(response_object, Mapping) or set(response_object) != {"items"}:
        raise ValueError("SFT generator response must contain only an items field")
    if not isinstance(response_object["items"], list):
        raise TypeError("SFT generator items must be a list")
    validated_specs = list(specs)
    if len(response_object["items"]) != len(validated_specs):
        raise ValueError("SFT generator item count must match the input spec count")
    items: list[dict[str, Any]] = []
    for raw, spec in zip(response_object["items"], validated_specs, strict=True):
        prefix = spec.get("public_task_messages")
        if prefix is not None:
            if not isinstance(raw, Mapping) or set(raw) != {"assistant_content"}:
                raise ValueError(
                    "derived SFT generator item must contain only assistant_content"
                )
            assistant_content = raw["assistant_content"]
            if not isinstance(assistant_content, str) or not assistant_content.strip():
                raise ValueError(
                    "derived SFT generator assistant_content must be a non-empty string"
                )
            messages = [dict(message) for message in prefix]
            messages.append(
                {"role": "assistant", "content": assistant_content.strip()}
            )
        else:
            if not isinstance(raw, Mapping) or set(raw) != {"messages"}:
                raise ValueError("SFT generator item must contain only messages")
            messages = raw["messages"]
        item = {
            "id": str(spec["id"]),
            "messages": messages,
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
