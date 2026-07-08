"""Non-network orchestration helpers for response-distillation batches.

This module connects the already-separated pieces of the distillation pipeline:
local prompt records, teacher batch response validation, public row merge, dataset
writing, and local manifest writing. It intentionally does not call providers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.distillation.batches import validate_teacher_batch_response
from slm_synth.distillation.io import write_manifest, write_signal_dataset
from slm_synth.distillation.prompts import validate_prompt_record
from slm_synth.distillation.response_quality import filter_public_rows_by_response_quality
from slm_synth.distillation.signals import validate_signal
from slm_synth.distillation.validate import merge_teacher_outputs


@dataclass(frozen=True)
class DistillationRunResult:
    """Result of materializing one signal-specific teacher batch."""

    signal: str
    dataset_path: Path
    manifest_path: Path
    row_count: int


def _validate_prompt_records_for_signal(
    *,
    signal: str,
    prompt_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized_signal = validate_signal(signal)
    records = [validate_prompt_record(record) for record in prompt_records]

    mismatched = sorted({record["id"] for record in records if record["signal"] != normalized_signal})
    if mismatched:
        raise ValueError(f"prompt record signal mismatch for id(s): {mismatched}")

    return records


def default_manifest_path(
    *,
    manifest_dir: str | Path,
    signal: str,
    generation_run: str,
) -> Path:
    """Return the default local manifest path for one signal/run pair."""
    normalized_signal = validate_signal(signal)
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(manifest_dir) / f"{normalized_signal}.{generation_run}.manifest.json"


def materialize_teacher_batch(
    *,
    signal: str,
    prompt_records: Sequence[Mapping[str, Any]],
    teacher_response: Mapping[str, Any],
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    token_target: str | int | None = None,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DistillationRunResult:
    """Validate, merge, and write one signal-specific distillation batch.

    The teacher response must be the strict batch object produced by the teacher
    contract: {"items": [{"id", "reasoning", "response"}]}. Public JSONL rows
    contain only id, prompt, reasoning, and response. Provider/model/run details
    are written only to the local manifest.
    """
    normalized_signal = validate_signal(signal)
    records = _validate_prompt_records_for_signal(signal=normalized_signal, prompt_records=prompt_records)
    teacher_outputs = validate_teacher_batch_response(teacher_response, expected_count=len(records))
    public_rows = merge_teacher_outputs(records, teacher_outputs)
    accepted_rows, response_quality = filter_public_rows_by_response_quality(
        signal=normalized_signal,
        rows=public_rows,
    )

    dataset_path = write_signal_dataset(
        signal=normalized_signal,
        rows=accepted_rows,
        output_dir=output_dir,
        filename=dataset_filename,
    )

    if manifest_filename is None:
        manifest_path = default_manifest_path(
            manifest_dir=manifest_dir,
            signal=normalized_signal,
            generation_run=generation_run,
        )
    else:
        manifest_path = Path(manifest_dir) / manifest_filename

    write_manifest(
        manifest_path=manifest_path,
        signal=normalized_signal,
        dataset_path=dataset_path,
        row_count=len(accepted_rows),
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=token_target,
        metadata={
            **dict(metadata or {}),
            "prompt_count": len(records),
            "planned_prompt_rows": len(records),
            "accepted_rows": len(accepted_rows),
            "rejected_rows": response_quality.rejected_rows,
            "rejection_reasons": response_quality.rejection_reasons,
            "response_quality": response_quality.to_dict(),
        },
    )

    return DistillationRunResult(
        signal=normalized_signal,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(accepted_rows),
    )
