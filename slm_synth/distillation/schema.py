"""Public row schema for signal-specific response-distillation datasets."""

from __future__ import annotations

from typing import Any, Mapping

PUBLIC_ROW_FIELDS = frozenset({"id", "prompt", "reasoning", "response"})
FORBIDDEN_PUBLIC_ROW_FIELDS = frozenset(
    {
        "signal",
        "metadata",
        "teacher_model",
        "teacher_provider",
        "generation_run",
        "difficulty",
    }
)


def _require_string(row: Mapping[str, Any], field: str) -> None:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"public distillation row field '{field}' must be a non-empty string")


def validate_public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized public distillation row.

    Public rows intentionally contain only the fields needed for training:
    id, prompt, optional reasoning, and response. Internal metadata and teacher
    provenance must stay in local manifests or dataset cards.
    """
    if not isinstance(row, Mapping):
        raise TypeError("public distillation row must be a mapping")

    keys = set(row)
    missing = PUBLIC_ROW_FIELDS - keys
    if missing:
        raise ValueError(f"public distillation row missing required field(s): {sorted(missing)}")

    forbidden = keys & FORBIDDEN_PUBLIC_ROW_FIELDS
    if forbidden:
        raise ValueError(f"public distillation row contains forbidden field(s): {sorted(forbidden)}")

    unexpected = keys - PUBLIC_ROW_FIELDS
    if unexpected:
        raise ValueError(f"public distillation row contains unexpected field(s): {sorted(unexpected)}")

    _require_string(row, "id")
    _require_string(row, "prompt")
    _require_string(row, "response")

    reasoning = row["reasoning"]
    if reasoning is not None:
        if not isinstance(reasoning, list):
            raise ValueError("public distillation row field 'reasoning' must be null or list[str]")
        if not all(isinstance(step, str) and step.strip() for step in reasoning):
            raise ValueError("public distillation row field 'reasoning' must be null or list[str]")

    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "reasoning": reasoning,
        "response": row["response"],
    }
