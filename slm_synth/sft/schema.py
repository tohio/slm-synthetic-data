"""Schema validation for synthetic SFT records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.taxonomy import validate_metadata

SFT_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
SFT_ALLOWED_ROLE_CONTRACTS = (
    ("user", "assistant"),
    ("system", "user", "assistant"),
)
SFT_REQUIRED_FIELDS = frozenset({"id", "messages", "metadata"})


def validate_sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one SFT training row.

    Required row shape:
      - id: non-empty string
      - messages: one user turn and one assistant turn, optionally prefixed by one system turn
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
    roles = tuple(message["role"] for message in validated)
    if roles not in SFT_ALLOWED_ROLE_CONTRACTS:
        supported = " or ".join(str(contract) for contract in SFT_ALLOWED_ROLE_CONTRACTS)
        raise ValueError(f"SFT messages must follow role contract {supported}; got {roles}")

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

# BEGIN adjacent SFT role normalization
def _normalize_adjacent_sft_role_messages(row):
    """Merge adjacent same-role SFT messages before role-contract validation."""
    if not isinstance(row, dict):
        return row

    messages = row.get("messages")
    if not isinstance(messages, list):
        return row

    normalized_messages = []

    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue

        copied = dict(message)
        role = copied.get("role")
        content = copied.get("content")

        if isinstance(content, str):
            copied["content"] = content.strip()

        if (
            role in {"user", "assistant"}
            and normalized_messages
            and isinstance(normalized_messages[-1], dict)
            and normalized_messages[-1].get("role") == role
            and isinstance(normalized_messages[-1].get("content"), str)
            and isinstance(copied.get("content"), str)
        ):
            left = normalized_messages[-1]["content"].rstrip()
            right = copied["content"].strip()
            normalized_messages[-1]["content"] = (
                f"{left}\n{right}".strip() if left and right else left or right
            )
            continue

        normalized_messages.append(copied)

    normalized = dict(row)
    normalized["messages"] = normalized_messages
    return normalized


def _install_adjacent_sft_role_normalization():
    candidate_names = (
        "validate_sft_row",
        "validate_row",
        "validate_sft_record",
        "validate_record",
    )

    for name in candidate_names:
        original = globals().get(name)
        if not callable(original):
            continue

        if getattr(original, "_sft_adjacent_role_normalized", False):
            return

        def wrapped(row, *args, __original=original, **kwargs):
            return __original(
                _normalize_adjacent_sft_role_messages(row),
                *args,
                **kwargs,
            )

        wrapped.__name__ = getattr(original, "__name__", name)
        wrapped.__doc__ = getattr(original, "__doc__", None)
        wrapped._sft_adjacent_role_normalized = True
        globals()[name] = wrapped
        return

    raise RuntimeError("No SFT validation function found for adjacent-role normalization")


_install_adjacent_sft_role_normalization()
# END adjacent SFT role normalization
