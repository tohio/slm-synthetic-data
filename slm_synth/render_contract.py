"""Deterministic checks for locally verifiable rendered-output contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def validate_rendered_output_mode(
    messages: Sequence[Mapping[str, Any]], *, output_mode: str, row_id: str
) -> None:
    """Reject rendered rows whose observable form contradicts the source brief."""
    assistant_content = [
        message.get("content")
        for message in messages
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    ]
    final = assistant_content[-1].strip() if assistant_content else ""
    if not final:
        raise ValueError(f"rendered row {row_id} has no final assistant content")
    if output_mode == "structured_json":
        candidate = _strip_json_fence(final)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"rendered row {row_id} does not satisfy structured_json output_mode") from exc
        if not isinstance(parsed, (dict, list)):
            raise ValueError(f"rendered row {row_id} structured_json output must be an object or array")
    elif output_mode == "table":
        lines = [line for line in final.splitlines() if line.strip()]
        if not any("|" in line for line in lines) or not any(_is_table_separator(line) for line in lines):
            raise ValueError(f"rendered row {row_id} does not satisfy table output_mode")
    elif output_mode == "code":
        if "```" not in final:
            raise ValueError(f"rendered row {row_id} does not satisfy code output_mode")


def _strip_json_fence(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return value


def _is_table_separator(line: str) -> bool:
    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return compact == "" and "-" in line
