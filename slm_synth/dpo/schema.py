"""Schema validation for synthetic DPO preference rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.sft.schema import validate_message
from slm_synth.taxonomy import validate_metadata

DPO_REQUIRED_FIELDS = frozenset({"id", "prompt", "chosen", "rejected", "metadata"})


def validate_dpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one DPO preference row.

    Required row shape:
      - id: non-empty string
      - prompt: chat messages ending with a user message
      - chosen: assistant message list
      - rejected: assistant message list
      - metadata: shared taxonomy labels plus DPO failure_mode
    """
    if not isinstance(row, Mapping):
        raise TypeError("DPO row must be an object")

    missing = sorted(field for field in DPO_REQUIRED_FIELDS if field not in row)
    if missing:
        raise ValueError(f"DPO row missing required field(s): {missing}")

    extra = sorted(field for field in row if field not in DPO_REQUIRED_FIELDS)
    if extra:
        raise ValueError(f"DPO row contains unsupported field(s): {extra}")

    prompt = validate_message_list(
        row["prompt"],
        field_name="prompt",
        allowed_roles={"system", "user"},
        required_roles={"user"},
        final_role="user",
    )
    chosen = validate_message_list(
        row["chosen"],
        field_name="chosen",
        allowed_roles={"assistant"},
        required_roles={"assistant"},
    )
    rejected = validate_message_list(
        row["rejected"],
        field_name="rejected",
        allowed_roles={"assistant"},
        required_roles={"assistant"},
    )

    if chosen == rejected:
        raise ValueError("chosen and rejected messages must differ")

    return {
        "id": _require_non_empty_string(row["id"], "id"),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": validate_metadata(row["metadata"], require_failure_mode=True),
    }


def validate_message_list(
    messages: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    allowed_roles: set[str],
    required_roles: set[str],
    final_role: str | None = None,
) -> list[dict[str, str]]:
    """Validate one DPO message list."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError(f"{field_name} must be a list")
    if not messages:
        raise ValueError(f"{field_name} must contain at least one message")

    validated = [validate_message(message) for message in messages]

    disallowed = sorted({message["role"] for message in validated if message["role"] not in allowed_roles})
    if disallowed:
        supported = ", ".join(sorted(allowed_roles))
        raise ValueError(f"{field_name} contains unsupported role(s): {disallowed}. Supported roles: {supported}")

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
