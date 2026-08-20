"""Strict schema validation for generic DPO preference records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.chat_schema import (
    conversation_has_tool_activity,
    validate_conversation,
    validate_message,
    validate_tools,
)
from slm_synth.taxonomy import validate_alignment_metadata

DPO_REQUIRED_FIELDS = frozenset({"id", "prompt", "chosen", "rejected", "metadata"})
DPO_OPTIONAL_FIELDS = frozenset({"tools"})


def validate_dpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one shared prompt/tools prefix and two complete branches."""
    if not isinstance(row, Mapping):
        raise TypeError("DPO row must be an object")
    missing = sorted(DPO_REQUIRED_FIELDS - set(row))
    extra = sorted(set(row) - DPO_REQUIRED_FIELDS - DPO_OPTIONAL_FIELDS)
    if missing:
        raise ValueError(f"DPO row missing required field(s): {missing}")
    if extra:
        raise ValueError(f"DPO row contains unsupported field(s): {extra}")

    tools = validate_tools(row["tools"]) if "tools" in row else None
    prompt = validate_conversation(
        row["prompt"],
        tools=tools,
        field_name="DPO prompt",
        require_final_assistant=False,
        require_final_user=True,
    )
    chosen = _validate_branch(row["chosen"], prompt=prompt, tools=tools, field_name="chosen")
    rejected = _validate_branch(row["rejected"], prompt=prompt, tools=tools, field_name="rejected")
    if chosen == rejected:
        raise ValueError("chosen and rejected branches must differ")

    metadata = validate_alignment_metadata(row["metadata"], preference=True)
    _validate_interaction_contract(prompt, chosen, rejected, metadata["interaction_modes"])
    validated: dict[str, Any] = {
        "id": _require_non_empty_string(row["id"], "id"),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": metadata,
    }
    if tools is not None:
        validated["tools"] = tools
    return validated


def _validate_branch(
    value: Any,
    *,
    prompt: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    field_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError(f"{field_name} must be a non-empty list")
    branch = [validate_message(message) for message in value]
    disallowed = sorted({message["role"] for message in branch} - {"assistant", "tool"})
    if disallowed:
        raise ValueError(f"{field_name} branch contains unsupported role(s): {disallowed}")
    validate_conversation(
        [*prompt, *branch],
        tools=tools,
        field_name=f"DPO {field_name} branch",
        require_final_assistant=True,
    )
    return branch


def _validate_interaction_contract(
    prompt: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    modes: list[str],
) -> None:
    user_turns = sum(message["role"] == "user" for message in prompt)
    if ("multi_turn" in modes) != (user_turns > 1):
        raise ValueError("interaction_modes single_turn/multi_turn must match the shared prompt")
    if ("system_conditioned" in modes) != (prompt[0]["role"] == "system"):
        raise ValueError("system_conditioned must match the shared prompt system message")
    tool_activity = any(
        conversation_has_tool_activity(messages)
        for messages in (prompt, chosen, rejected)
    )
    if ("tool_mediated" in modes) != tool_activity:
        raise ValueError("tool_mediated must match structured tool activity in the prompt or branches")


def validate_message_list(
    messages: Any,
    *,
    field_name: str,
    allowed_roles: set[str],
    required_roles: set[str],
    final_role: str | None = None,
    exact_length: int | None = None,
) -> list[dict[str, Any]]:
    """Compatibility-free list validation retained for direct callers."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise TypeError(f"{field_name} must be a non-empty list")
    if exact_length is not None and len(messages) != exact_length:
        raise ValueError(f"{field_name} must contain exactly {exact_length} message(s)")
    validated = [validate_message(message) for message in messages]
    disallowed = sorted({message["role"] for message in validated} - allowed_roles)
    if disallowed:
        raise ValueError(f"{field_name} contains unsupported role(s): {disallowed}")
    for role in sorted(required_roles):
        if not any(message["role"] == role for message in validated):
            raise ValueError(f"{field_name} must contain at least one {role} message")
    if final_role is not None and validated[-1]["role"] != final_role:
        raise ValueError(f"final {field_name} message must be from {final_role}")
    return validated


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
