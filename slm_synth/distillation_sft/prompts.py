"""Local prompt records for response distillation.

Prompt records are internal inputs to teacher generation. They carry signal,
public audit metadata, and generation-only metadata locally. Only the validated
public metadata subset is emitted in training rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.distillation_sft.public_metadata import build_public_metadata
from slm_synth.distillation_sft.signals import validate_signal


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prompt record field '{field}' must be a non-empty string")
    return value


def format_prompt_id(signal: str, index: int, *, width: int = 6) -> str:
    """Return a stable local prompt id for a signal-specific distillation row."""
    normalized_signal = validate_signal(signal)
    if not isinstance(index, int) or index < 1:
        raise ValueError("prompt id index must be a positive integer")
    if not isinstance(width, int) or width < 1:
        raise ValueError("prompt id width must be a positive integer")
    return f"{normalized_signal}-{index:0{width}d}"


def build_prompt_record(
    *,
    signal: str,
    prompt: str,
    index: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one internal prompt record for teacher generation.

    Local code owns ids, prompts, signals, and metadata. Teacher calls should
    receive only the prompt batch and should return id, reasoning, and response.
    """
    record = {
        "id": format_prompt_id(signal, index),
        "prompt": _require_non_empty_string(prompt, "prompt"),
        "signal": validate_signal(signal),
        "metadata": dict(metadata or {}),
    }
    return validate_prompt_record(record)


def build_prompt_records(
    *,
    signal: str,
    prompts: Iterable[str],
    start_index: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build internal prompt records from a sequence of prompt strings."""
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    rows: list[dict[str, Any]] = []
    for offset, prompt in enumerate(prompts):
        rows.append(
            build_prompt_record(
                signal=signal,
                prompt=prompt,
                index=start_index + offset,
                metadata=metadata,
            )
        )
    return rows


def validate_prompt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a local prompt record used before teacher generation."""
    if not isinstance(record, Mapping):
        raise TypeError("prompt record must be a mapping")

    for field in ("id", "prompt", "signal"):
        _require_non_empty_string(record.get(field), field)

    metadata = record.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("prompt record field 'metadata' must be a mapping when provided")

    normalized_signal = validate_signal(record["signal"])
    normalized_metadata = dict(metadata)
    normalized_metadata.update(
        build_public_metadata(
            signal=normalized_signal,
            prompt=record["prompt"],
            template_family=normalized_metadata.get("template_family"),
            difficulty=normalized_metadata.get("difficulty"),
        )
    )

    return {
        "id": record["id"],
        "prompt": record["prompt"],
        "signal": normalized_signal,
        "metadata": normalized_metadata,
    }
