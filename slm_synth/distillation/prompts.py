"""Local prompt records for response distillation.

Prompt records are internal inputs to teacher generation. They carry signal and
metadata locally, but those fields are not emitted in public training rows.
"""

from __future__ import annotations

from typing import Any, Mapping

from slm_synth.distillation.signals import validate_signal


def validate_prompt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a local prompt record used before teacher generation."""
    if not isinstance(record, Mapping):
        raise TypeError("prompt record must be a mapping")

    for field in ("id", "prompt", "signal"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompt record field '{field}' must be a non-empty string")

    metadata = record.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("prompt record field 'metadata' must be a mapping when provided")

    return {
        "id": record["id"],
        "prompt": record["prompt"],
        "signal": validate_signal(record["signal"]),
        "metadata": dict(metadata),
    }
