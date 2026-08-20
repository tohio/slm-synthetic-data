"""Strict schema validation for generic SFT chat records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from slm_synth.chat_schema import (
    CHAT_ROLES,
    conversation_has_tool_activity,
    validate_conversation,
    validate_message,
    validate_tools,
)
from slm_synth.taxonomy import validate_alignment_metadata

SFT_ALLOWED_ROLES = CHAT_ROLES
SFT_REQUIRED_FIELDS = frozenset({"id", "messages", "metadata"})
SFT_OPTIONAL_FIELDS = frozenset({"tools"})


def validate_sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete SFT conversation with optional shared tools."""
    if not isinstance(row, Mapping):
        raise TypeError("SFT row must be an object")
    missing = sorted(SFT_REQUIRED_FIELDS - set(row))
    extra = sorted(set(row) - SFT_REQUIRED_FIELDS - SFT_OPTIONAL_FIELDS)
    if missing:
        raise ValueError(f"SFT row missing required field(s): {missing}")
    if extra:
        raise ValueError(f"SFT row contains unsupported field(s): {extra}")

    tools = validate_tools(row["tools"]) if "tools" in row else None
    messages = validate_conversation(
        row["messages"],
        tools=tools,
        field_name="SFT messages",
        require_final_assistant=True,
    )
    metadata = validate_alignment_metadata(row["metadata"])
    _validate_interaction_contract(messages, metadata["interaction_modes"])

    validated: dict[str, Any] = {
        "id": _require_non_empty_string(row["id"], "id"),
        "messages": messages,
        "metadata": metadata,
    }
    if tools is not None:
        validated["tools"] = tools
    return validated


def validate_messages(messages: Any, *, tools: Any = None) -> list[dict[str, Any]]:
    """Validate a complete SFT message sequence."""
    validated_tools = validate_tools(tools) if tools is not None else None
    return validate_conversation(
        messages,
        tools=validated_tools,
        field_name="SFT messages",
        require_final_assistant=True,
    )


def _validate_interaction_contract(messages: list[dict[str, Any]], modes: list[str]) -> None:
    roles = [message["role"] for message in messages]
    user_turns = roles.count("user")
    if ("multi_turn" in modes) != (user_turns > 1):
        raise ValueError("interaction_modes single_turn/multi_turn must match the message sequence")
    if ("system_conditioned" in modes) != (roles[0] == "system"):
        raise ValueError("system_conditioned must match the presence of a leading system message")
    if ("tool_mediated" in modes) != conversation_has_tool_activity(messages):
        raise ValueError("tool_mediated must match structured tool activity in messages")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


__all__ = [
    "SFT_ALLOWED_ROLES",
    "SFT_OPTIONAL_FIELDS",
    "SFT_REQUIRED_FIELDS",
    "validate_message",
    "validate_messages",
    "validate_sft_row",
]
