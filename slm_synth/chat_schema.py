"""Shared strict chat, tool-definition, and tool-call validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
CHAT_ROLES = frozenset({"system", "user", "assistant", "tool"})

TOOL_CALL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "type", "function"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "type": {"type": "string", "enum": ["function"]},
        "function": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "arguments"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "arguments": {"type": "object"},
            },
        },
    },
}

CHAT_MESSAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "content"],
    "properties": {
        "role": {"type": "string", "enum": sorted(CHAT_ROLES)},
        "content": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "tool_calls": {"type": "array", "minItems": 1, "items": TOOL_CALL_JSON_SCHEMA},
        "tool_call_id": {"type": "string", "minLength": 1},
    },
    "allOf": [
        {
            "if": {
                "properties": {
                    "role": {"enum": ["system", "user", "tool"]},
                },
                "required": ["role"],
            },
            "then": {
                "properties": {
                    "content": {"type": "string", "minLength": 1},
                },
            },
        },
        {
            "if": {
                "properties": {
                    "role": {"const": "assistant"},
                    "content": {"type": "null"},
                },
                "required": ["role", "content"],
            },
            "then": {"required": ["tool_calls"]},
        },
    ],
}

TOOL_DEFINITION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "function"],
    "properties": {
        "type": {"type": "string", "enum": ["function"]},
        "function": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "description", "parameters"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "parameters": {"type": "object"},
            },
        },
    },
}

TOOLS_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": TOOL_DEFINITION_JSON_SCHEMA,
}


def validate_tools(value: Any) -> list[dict[str, Any]]:
    """Validate a shared list of function-tool definitions."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError("tools must be a non-empty list")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"tools[{index}] must be an object")
        _require_fields(item, {"type", "function"}, label=f"tools[{index}]")
        if item["type"] != "function":
            raise ValueError(f"tools[{index}].type must be 'function'")
        function = item["function"]
        if not isinstance(function, Mapping):
            raise TypeError(f"tools[{index}].function must be an object")
        _require_fields(function, {"name", "description", "parameters"}, label=f"tools[{index}].function")
        name = _require_name(function["name"], f"tools[{index}].function.name")
        if name in names:
            raise ValueError(f"tools contains duplicate function name {name!r}")
        names.add(name)
        description = _require_non_empty_string(function["description"], f"tools[{index}].function.description")
        parameters = function["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError(f"tools[{index}].function.parameters must be an object")
        parameters = dict(parameters)
        if parameters.get("type") != "object":
            raise ValueError(f"tools[{index}].function.parameters.type must be 'object'")
        if "properties" in parameters and not isinstance(parameters["properties"], Mapping):
            raise TypeError(f"tools[{index}].function.parameters.properties must be an object")
        tools.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        })
    return tools


def validate_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one role-specific chat message without compatibility repair."""
    if not isinstance(message, Mapping):
        raise TypeError("message must be an object")
    role = _require_non_empty_string(message.get("role"), "role").lower()
    if role not in CHAT_ROLES:
        raise ValueError(f"unsupported message role {role!r}. Supported roles: {', '.join(sorted(CHAT_ROLES))}")

    if role in {"system", "user"}:
        _require_fields(message, {"role", "content"}, label=f"{role} message")
        return {"role": role, "content": _require_non_empty_string(message["content"], "content")}
    if role == "tool":
        _require_fields(message, {"role", "content", "tool_call_id"}, label="tool message")
        return {
            "role": "tool",
            "content": _require_non_empty_string(message["content"], "content"),
            "tool_call_id": _require_non_empty_string(message["tool_call_id"], "tool_call_id"),
        }

    allowed = {"role", "content", "tool_calls"}
    extra = sorted(set(message) - allowed)
    if extra:
        raise ValueError(f"assistant message contains unsupported field(s): {extra}")
    content = message.get("content")
    if content is not None:
        content = _require_non_empty_string(content, "content")
    tool_calls_value = message.get("tool_calls")
    if content is None and tool_calls_value is None:
        raise ValueError("assistant message requires non-empty content or tool_calls")
    validated: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls_value is not None:
        validated["tool_calls"] = validate_tool_calls(tool_calls_value)
    return validated


def validate_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise TypeError("tool_calls must be a non-empty list")
    calls: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"tool_calls[{index}] must be an object")
        _require_fields(item, {"id", "type", "function"}, label=f"tool_calls[{index}]")
        call_id = _require_non_empty_string(item["id"], f"tool_calls[{index}].id")
        if call_id in ids:
            raise ValueError(f"tool_calls contains duplicate id {call_id!r}")
        ids.add(call_id)
        if item["type"] != "function":
            raise ValueError(f"tool_calls[{index}].type must be 'function'")
        function = item["function"]
        if not isinstance(function, Mapping):
            raise TypeError(f"tool_calls[{index}].function must be an object")
        _require_fields(function, {"name", "arguments"}, label=f"tool_calls[{index}].function")
        arguments = function["arguments"]
        if not isinstance(arguments, Mapping):
            raise TypeError(f"tool_calls[{index}].function.arguments must be an object")
        calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": _require_name(function["name"], f"tool_calls[{index}].function.name"),
                "arguments": dict(arguments),
            },
        })
    return calls


def validate_conversation(
    messages: Any,
    *,
    tools: list[dict[str, Any]] | None,
    field_name: str,
    require_final_assistant: bool,
    require_final_user: bool = False,
    allowed_new_user_turns: bool = True,
) -> list[dict[str, Any]]:
    """Validate strict role order and complete tool-call/result cycles."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise TypeError(f"{field_name} must be a non-empty list")
    validated = [validate_message(message) for message in messages]
    declared_names = {tool["function"]["name"] for tool in tools or []}
    pending: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    expect = "system_or_user"
    user_count = 0
    tool_activity = False

    for index, message in enumerate(validated):
        role = message["role"]
        if role == "system":
            if index != 0:
                raise ValueError(f"{field_name} system message is permitted only at index 0")
            expect = "user"
            continue
        if role == "user":
            if expect not in {"system_or_user", "user", "next_user"}:
                raise ValueError(f"{field_name} has malformed role sequence at user message {index}")
            if not allowed_new_user_turns and user_count:
                raise ValueError(f"{field_name} branches cannot contain user messages")
            user_count += 1
            expect = "assistant"
            continue
        if role == "assistant":
            if expect == "tool":
                raise ValueError(f"{field_name} has unresolved tool_call id(s): {sorted(pending)}")
            if expect not in {"assistant", "assistant_after_tools"}:
                raise ValueError(f"{field_name} has malformed role sequence at assistant message {index}")
            calls = message.get("tool_calls", [])
            if calls:
                tool_activity = True
                if tools is None:
                    raise ValueError(f"{field_name} contains tool_calls but row has no shared tools")
                for call in calls:
                    name = call["function"]["name"]
                    if name not in declared_names:
                        raise ValueError(f"{field_name} calls undeclared tool {name!r}")
                    if call["id"] in seen_call_ids:
                        raise ValueError(f"{field_name} reuses tool_call id {call['id']!r}")
                    seen_call_ids.add(call["id"])
                    pending[call["id"]] = name
                expect = "tool"
            else:
                expect = "next_user"
            continue
        if role == "tool":
            tool_activity = True
            if expect != "tool":
                raise ValueError(f"{field_name} has tool response without a preceding tool call")
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError(f"{field_name} has unknown or duplicate tool_call_id {call_id!r}")
            del pending[call_id]
            if not pending:
                expect = "assistant_after_tools"

    if pending:
        raise ValueError(f"{field_name} has unresolved tool_call id(s): {sorted(pending)}")
    if user_count == 0:
        raise ValueError(f"{field_name} must contain at least one user message")
    if require_final_assistant and validated[-1]["role"] != "assistant":
        raise ValueError(f"final {field_name} message must be from assistant")
    if require_final_user and validated[-1]["role"] != "user":
        raise ValueError(f"final {field_name} message must be from user")
    if require_final_assistant and validated[-1].get("tool_calls"):
        raise ValueError(f"final {field_name} assistant message cannot contain tool_calls")
    return validated


def conversation_has_tool_activity(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(message["role"] == "tool" or message.get("tool_calls") for message in messages)


def _require_fields(value: Mapping[str, Any], exact: set[str], *, label: str) -> None:
    missing = sorted(exact - set(value))
    extra = sorted(set(value) - exact)
    if missing:
        raise ValueError(f"{label} missing required field(s): {missing}")
    if extra:
        raise ValueError(f"{label} contains unsupported field(s): {extra}")


def _require_name(value: Any, field_name: str) -> str:
    name = _require_non_empty_string(value, field_name)
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"{field_name} must be a valid tool name")
    return name


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
