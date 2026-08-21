"""Deterministic tool definitions derived from grounded source briefs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_SIGNATURE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*$")


def tools_from_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    modes = spec.get("metadata", {}).get("interaction_modes", [])
    if "tool_mediated" not in modes:
        return None
    variables = spec.get("variables", {})
    raw_values = variables.get("tools") or [variables.get("tool")]
    if not isinstance(raw_values, list):
        raw_values = [raw_values]
    tools = [_parse_tool(value) for value in raw_values if isinstance(value, str) and value.strip()]
    if not tools:
        raise ValueError(f"tool-mediated spec {spec.get('id')} has no deterministic tool signature")
    return tools


def _parse_tool(value: str) -> dict[str, Any]:
    match = _SIGNATURE.match(value.strip())
    if not match:
        raise ValueError(f"invalid grounded tool signature: {value!r}")
    name, raw_parameters = match.groups()
    properties: dict[str, Any] = {}
    required: list[str] = []
    for raw in (part.strip() for part in raw_parameters.split(",")):
        if not raw:
            continue
        token = raw.split(":", 1)[0].strip().rstrip("?")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            continue
        properties[token] = {"type": "string"}
        if "?" not in raw:
            required.append(token)
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Grounded {name} operation for this candidate.",
            "parameters": parameters,
        },
    }
