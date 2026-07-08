"""Public row schema for distillation-DPO preference datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.taxonomy import validate_metadata

DISTILLATION_DPO_REQUIRED_FIELDS = frozenset({"id", "prompt", "chosen", "rejected", "metadata"})
FORBIDDEN_ROW_FIELDS = frozenset(
    {
        "teacher_model",
        "teacher_provider",
        "generation_run",
        "chosen_source",
        "rejected_source",
        "target_consumer",
        "provider",
        "cost",
        "retry_count",
    }
)


def validate_distillation_dpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one public distillation-DPO row.

    Public rows keep only training-facing preference data. Teacher lineage,
    provider metadata, run ids, retry information, and cost details belong in
    manifests and dataset cards.
    """
    if not isinstance(row, Mapping):
        raise TypeError("distillation-DPO row must be an object")

    keys = set(row)
    missing = sorted(field for field in DISTILLATION_DPO_REQUIRED_FIELDS if field not in keys)
    if missing:
        raise ValueError(f"distillation-DPO row missing required field(s): {missing}")

    forbidden = sorted(keys & FORBIDDEN_ROW_FIELDS)
    if forbidden:
        raise ValueError(f"distillation-DPO row contains forbidden field(s): {forbidden}")

    extra = sorted(keys - DISTILLATION_DPO_REQUIRED_FIELDS)
    if extra:
        raise ValueError(f"distillation-DPO row contains unsupported field(s): {extra}")

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


def validate_message(message: Mapping[str, Any]) -> dict[str, str]:
    """Validate one public chat message."""
    if not isinstance(message, Mapping):
        raise TypeError("message must be an object")
    missing = sorted(field for field in ("role", "content") if field not in message)
    if missing:
        raise ValueError(f"message missing required field(s): {missing}")
    extra = sorted(field for field in message if field not in {"role", "content"})
    if extra:
        raise ValueError(f"message contains unsupported field(s): {extra}")
    return {
        "role": _require_non_empty_string(message["role"], "message role"),
        "content": _require_non_empty_string(message["content"], "message content"),
    }


def validate_message_list(
    messages: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    allowed_roles: set[str],
    required_roles: set[str],
    final_role: str | None = None,
) -> list[dict[str, str]]:
    """Validate a prompt/chosen/rejected message list."""
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
