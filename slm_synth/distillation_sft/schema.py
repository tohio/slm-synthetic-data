"""Public row schema for signal-specific response-distillation datasets."""

from __future__ import annotations

from typing import Any, Mapping

from slm_synth.corruption import require_no_high_confidence_corruption
from slm_synth.distillation_sft.public_metadata import extract_public_metadata


PUBLIC_ROW_FIELDS = frozenset({"id", "prompt", "reasoning", "response", "metadata"})
FORBIDDEN_PUBLIC_ROW_FIELDS = frozenset(
    {
        "signal",
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

    Public rows contain the fields needed for training plus audit metadata:
    id, prompt, reasoning fixed to null, response, and public taxonomy metadata.
    Teacher provenance and generation-only metadata stay in manifests or cards.
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
        raise ValueError("public distillation row field 'reasoning' must be null")

    require_no_high_confidence_corruption(
        row["prompt"],
        row["response"],
        artifact_name="public distillation-SFT row",
    )

    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "reasoning": None,
        "response": row["response"],
        "metadata": extract_public_metadata(row["metadata"]),
    }
