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



def render_sft_task_materialization_prompt(
    specs: Iterable[Mapping[str, Any]],
) -> str:
    """Render derived spec plans into concrete public conversation prefixes."""
    request_json = json.dumps(
        build_sft_teacher_request_object(specs), ensure_ascii=False, indent=2
    )
    return (
        "Materialize one concrete public SFT task for each input plan. Do not answer the "
        "final task. Return exactly one JSON object shaped as "
        '{"items":[{"messages":[{"role":"user","content":"..."}]}]}. '
        "Return items in input order and do not add fields outside messages. The returned "
        "messages are the exact public conversation prefix that will later be given to the "
        "answer generator. Replace the capability-anchor/meta planning brief with a genuinely "
        "new concrete task: instantiate fresh actors, facts, source passages, code, quantities, "
        "dates, examples, or other task material as appropriate. Do not merely rename entities "
        "or swap numbers. Every fact, source, rubric, label set, and response requirement needed "
        "for the final answer must appear explicitly in these public messages. Never expose "
        "variables, derivation_profile, capability-anchor language, repository metadata, or "
        "planning commentary. Preserve the declared task family, interaction mode, output mode, "
        "safety posture, and structural response requirements. When public_prompt_requirements is "
        "present, copy every listed phrase exactly into a system or user message in the materialized "
        "prefix; repository validation checks those phrases literally. If the instruction says the "
        "newly instantiated user task must state particular response requirements, state them "
        "explicitly in a user message. For single_turn, return exactly one user message, plus a "
        "leading system "
        "message if and only if system_conditioned is declared. For multi_turn, return at least two "
        "user messages with assistant context turns between them as needed, plus a leading system "
        "message if and only if system_conditioned is declared. In every case, end the prefix with "
        "a user message whose request is ready for the final assistant answer. Do not include that "
        "final assistant answer.\n\n"
        f"Input plans:\n{request_json}"
    )


def attach_and_validate_sft_task_materializations(
    response_object: Mapping[str, Any], specs: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Attach concrete public task prefixes to derived specs and validate their roles."""
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
        if not isinstance(raw, Mapping) or set(raw) != {"messages"}:
            raise ValueError("SFT task materializer item must contain only messages")
        messages = raw["messages"]
        _validate_public_task_prefix(
            messages, interaction_modes=list(spec["metadata"]["interaction_modes"])
        )
        _validate_materialized_prompt_requirements(
            messages, spec.get("public_prompt_requirements", [])
        )
        updated = dict(spec)
        updated["public_task_messages"] = [dict(message) for message in messages]
        materialized.append(updated)
    return materialized


def validate_sft_generated_task_prefixes(
    response_object: Mapping[str, Any], specs: Iterable[Mapping[str, Any]]
) -> None:
    """Require the answer generator to preserve each materialized task exactly."""
    items = response_object.get("items") if isinstance(response_object, Mapping) else None
    if not isinstance(items, list):
        raise ValueError("SFT generator items must be a list")
    validated_specs = list(specs)
    if len(items) != len(validated_specs):
        raise ValueError("SFT generator item count must match the input spec count")
    for raw, spec in zip(items, validated_specs, strict=True):
        prefix = spec.get("public_task_messages")
        if prefix is None:
            continue
        if not isinstance(raw, Mapping) or set(raw) != {"messages"}:
            raise ValueError("SFT generator item must contain only messages")
        messages = raw["messages"]
        if not isinstance(messages, list):
            raise TypeError("SFT generator messages must be a list")
        if len(messages) != len(prefix) + 1:
            raise ValueError(
                "SFT generator must preserve the materialized public task prefix and append "
                "exactly one final assistant response"
            )
        if messages[: len(prefix)] != prefix:
            raise ValueError(
                "SFT generator changed the materialized public task prefix"
            )
        final = messages[-1]
        if not isinstance(final, Mapping) or final.get("role") != "assistant":
            raise ValueError("SFT generator final message must be from assistant")



def _validate_materialized_prompt_requirements(
    messages: list[Mapping[str, Any]], requirements: Any
) -> None:
    if not requirements:
        return
    if not isinstance(requirements, list):
        raise TypeError("public_prompt_requirements must be a list")
    public_text = "\n".join(
        str(message.get("content", "")) for message in messages
    ).casefold()
    missing = [
        requirement
        for requirement in requirements
        if not isinstance(requirement, str)
        or requirement.casefold() not in public_text
    ]
    if missing:
        raise ValueError(
            "SFT task materializer omitted public prompt requirement(s): "
            + repr(missing)
        )


def _validate_public_task_prefix(
    messages: Any, *, interaction_modes: list[str]
) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT task materializer messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(
                f"SFT task materializer message {index} must contain only role and content"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                "SFT task materializer messages may contain only system, user, and assistant"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("SFT task materializer message content must be non-empty")
        normalized.append({"role": role, "content": content})

    roles = [message["role"] for message in normalized]
    system_conditioned = "system_conditioned" in interaction_modes
    if (roles[0] == "system") != system_conditioned:
        raise ValueError(
            "SFT task materializer system message must match system_conditioned"
        )
    conversational_roles = roles[1:] if system_conditioned else roles
    if not conversational_roles or conversational_roles[0] != "user":
        raise ValueError("SFT task materializer conversation must start with user")
    if conversational_roles[-1] != "user":
        raise ValueError("SFT task materializer prefix must end with user")
    if any(
        left == right
        for left, right in zip(
            conversational_roles, conversational_roles[1:], strict=False
        )
    ):
        raise ValueError("SFT task materializer user/assistant roles must alternate")
    user_turns = conversational_roles.count("user")
    if "multi_turn" in interaction_modes:
        if user_turns < 2:
            raise ValueError(
                "multi_turn SFT task materializer prefix requires at least two user messages"
            )
    elif user_turns != 1 or len(conversational_roles) != 1:
        raise ValueError(
            "single_turn SFT task materializer prefix requires exactly one user message"
        )

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
        "system_conditioned. When public_task_messages is present, copy that list exactly as the conversation prefix, "
        "without rewriting, reordering, adding, or deleting prefix messages, and append exactly one final assistant "
        "response. The concrete task has already been materialized; do not invent a replacement task or expose the "
        "planning brief. For single_turn, use exactly one user message followed by the final assistant response (with an "
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
