"""Schema validation for synthetic SFT records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.taxonomy import validate_metadata

SFT_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
SFT_REQUIRED_FIELDS = frozenset({"id", "messages", "metadata"})


def validate_sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one SFT training row.

    Required row shape:
      - id: non-empty string
      - messages: chat message list ending with an assistant message
      - metadata: category/difficulty/template/eval labels
    """
    if not isinstance(row, Mapping):
        raise TypeError("SFT row must be an object")

    missing = sorted(field for field in SFT_REQUIRED_FIELDS if field not in row)
    if missing:
        raise ValueError(f"SFT row missing required field(s): {missing}")

    extra = sorted(field for field in row if field not in SFT_REQUIRED_FIELDS)
    if extra:
        raise ValueError(f"SFT row contains unsupported field(s): {extra}")

    row_id = _require_non_empty_string(row["id"], "id")
    messages = validate_messages(row["messages"])
    metadata = validate_metadata(row["metadata"], require_failure_mode=False)

    return {
        "id": row_id,
        "messages": messages,
        "metadata": metadata,
    }


def validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Validate the SFT chat message sequence."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a list")
    if not messages:
        raise ValueError("messages must contain at least one message")

    validated = [validate_message(message) for message in messages]

    if not any(message["role"] == "user" for message in validated):
        raise ValueError("messages must contain at least one user message")
    if validated[-1]["role"] != "assistant":
        raise ValueError("final SFT message must be from assistant")

    assistant_count = sum(1 for message in validated if message["role"] == "assistant")
    if assistant_count < 1:
        raise ValueError("messages must contain at least one assistant message")

    return validated


def validate_message(message: Mapping[str, Any]) -> dict[str, str]:
    """Validate one chat message."""
    if not isinstance(message, Mapping):
        raise TypeError("message must be an object")

    missing = sorted(field for field in ("role", "content") if field not in message)
    if missing:
        raise ValueError(f"message missing required field(s): {missing}")

    extra = sorted(field for field in message if field not in {"role", "content"})
    if extra:
        raise ValueError(f"message contains unsupported field(s): {extra}")

    role = _require_non_empty_string(message["role"], "role").lower()
    if role not in SFT_ALLOWED_ROLES:
        supported = ", ".join(sorted(SFT_ALLOWED_ROLES))
        raise ValueError(f"unsupported message role '{role}'. Supported roles: {supported}")

    content = _require_non_empty_string(message["content"], "content")
    return {"role": role, "content": content}


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
