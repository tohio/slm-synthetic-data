"""Small multi-signal orchestration helpers for response distillation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.distillation.batches import validate_teacher_batch_response
from slm_synth.distillation.generation import (
    StructuredTeacherBackend,
    build_openrouter_backend,
    generate_teacher_batch_response,
)
from slm_synth.distillation.io import write_manifest, write_run_manifest, write_signal_dataset
from slm_synth.distillation.runs import DistillationRunResult, default_manifest_path
from slm_synth.distillation.seeds import build_seed_prompt_records
from slm_synth.distillation.signals import DISTILLATION_SIGNALS, validate_signal
from slm_synth.distillation.validate import merge_teacher_outputs


@dataclass(frozen=True)
class MultiSignalRunResult:
    """Result of generating and materializing multiple signal-specific batches."""

    generation_run: str
    results: tuple[DistillationRunResult, ...]
    manifest_path: Path

    @property
    def row_count(self) -> int:
        """Return the total number of public rows written across all signals."""
        return sum(result.row_count for result in self.results)

    @property
    def signals(self) -> tuple[str, ...]:
        """Return generated signals in execution order."""
        return tuple(result.signal for result in self.results)


BackendFactory = Callable[[str], StructuredTeacherBackend]


def normalize_signal_sequence(signals: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return validated signals in deterministic order.

    When no signals are provided, all supported signals are returned sorted. Duplicate
    explicit signals are rejected so a run cannot accidentally overwrite outputs.
    """
    if signals is None:
        return tuple(sorted(DISTILLATION_SIGNALS))

    normalized = tuple(validate_signal(signal) for signal in signals)
    duplicates = sorted({signal for signal in normalized if normalized.count(signal) > 1})
    if duplicates:
        raise ValueError(f"duplicate signal(s) requested: {duplicates}")
    return normalized


def normalize_signal_counts(
    *,
    signals: Sequence[str] | None = None,
    count_per_signal: int | None = None,
    counts_by_signal: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Build a validated mapping of signal -> prompt count.

    Use count_per_signal for fixed-size runs. counts_by_signal is available for
    callers that already computed counts, but this module does not estimate token
    budgets yet.
    """
    normalized_signals = normalize_signal_sequence(signals)

    if count_per_signal is not None and counts_by_signal is not None:
        raise ValueError("provide either count_per_signal or counts_by_signal, not both")

    if counts_by_signal is not None:
        normalized_counts: dict[str, int] = {}
        for signal, count in counts_by_signal.items():
            normalized_signal = validate_signal(signal)
            if normalized_signal not in normalized_signals:
                raise ValueError(f"count provided for unrequested signal '{normalized_signal}'")
            normalized_counts[normalized_signal] = _validate_count(count)
        missing = [signal for signal in normalized_signals if signal not in normalized_counts]
        if missing:
            raise ValueError(f"missing count(s) for signal(s): {missing}")
        return {signal: normalized_counts[signal] for signal in normalized_signals}

    if count_per_signal is None:
        raise ValueError("count_per_signal is required when counts_by_signal is not provided")

    count = _validate_count(count_per_signal)
    return {signal: count for signal in normalized_signals}


def generate_seed_multi_signal_run(
    *,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    count_per_signal: int | None = None,
    counts_by_signal: Mapping[str, int] | None = None,
    signals: Sequence[str] | None = None,
    token_target: str | int | None = None,
    start_index: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 1,
    batch_size: int | None = None,
    run_manifest_filename: str | None = None,
    backend_factory: BackendFactory | None = None,
) -> MultiSignalRunResult:
    """Generate one seed batch for each requested signal and materialize outputs.

    This is intentionally a thin loop over the existing single-signal live path.
    It does not perform dataset-card generation or Makefile integration.
    """
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    signal_counts = normalize_signal_counts(
        signals=signals,
        count_per_signal=count_per_signal,
        counts_by_signal=counts_by_signal,
    )
    normalized_batch_size = _validate_batch_size(batch_size)

    results: list[DistillationRunResult] = []
    for signal, count in signal_counts.items():
        prompt_records = build_seed_prompt_records(signal=signal, count=count, start_index=start_index)
        backend = backend_factory(signal) if backend_factory is not None else None
        result = _generate_and_materialize_signal_batches(
            signal=signal,
            prompt_records=prompt_records,
            output_dir=output_dir,
            manifest_dir=manifest_dir,
            teacher_model=teacher_model,
            generation_run=generation_run,
            max_tokens=max_tokens,
            token_target=token_target,
            temperature=temperature,
            top_p=top_p,
            request_timeout=request_timeout,
            max_request_retries=max_request_retries,
            max_retryable_request_attempts=max_retryable_request_attempts,
            retry_max_elapsed_seconds=retry_max_elapsed_seconds,
            adaptive_maximum_in_flight=adaptive_maximum_in_flight,
            adaptive_initial_in_flight=adaptive_initial_in_flight,
            batch_size=normalized_batch_size,
            backend=backend,
        )
        results.append(result)

    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    write_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=generation_run,
        teacher_model=teacher_model,
        teacher_provider="openrouter",
        token_target=token_target,
        datasets=[
            {
                "signal": result.signal,
                "dataset_path": result.dataset_path,
                "manifest_path": result.manifest_path,
                "row_count": result.row_count,
            }
            for result in results
        ],
        metadata={
            "signal_count": len(results),
            "start_index": start_index,
        },
    )

    return MultiSignalRunResult(
        generation_run=generation_run,
        results=tuple(results),
        manifest_path=run_manifest_path,
    )


def _validate_count(count: Any) -> int:
    if not isinstance(count, int) or count < 1:
        raise ValueError("signal prompt counts must be positive integers")
    return count


def _validate_batch_size(batch_size: Any) -> int | None:
    if batch_size is None:
        return None
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return batch_size


def _chunks(records: Sequence[Mapping[str, Any]], batch_size: int | None) -> list[list[Mapping[str, Any]]]:
    if batch_size is None or batch_size >= len(records):
        return [list(records)]
    return [list(records[index : index + batch_size]) for index in range(0, len(records), batch_size)]


def _generate_and_materialize_signal_batches(
    *,
    signal: str,
    prompt_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    token_target: str | int | None = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 1,
    batch_size: int | None = None,
    backend: StructuredTeacherBackend | None = None,
) -> DistillationRunResult:
    normalized_signal = validate_signal(signal)
    batches = _chunks(prompt_records, batch_size)
    active_backend = backend or build_openrouter_backend(
        model=teacher_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        request_timeout=request_timeout,
        max_request_retries=max_request_retries,
        max_retryable_request_attempts=max_retryable_request_attempts,
        retry_max_elapsed_seconds=retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=adaptive_maximum_in_flight,
        adaptive_initial_in_flight=adaptive_initial_in_flight,
    )

    public_rows: list[dict[str, Any]] = []
    for batch in batches:
        teacher_response = generate_teacher_batch_response(
            signal=normalized_signal,
            prompt_records=batch,
            backend=active_backend,
        )
        teacher_outputs = validate_teacher_batch_response(teacher_response, expected_count=len(batch))
        public_rows.extend(merge_teacher_outputs(batch, teacher_outputs))

    dataset_path = write_signal_dataset(
        signal=normalized_signal,
        rows=public_rows,
        output_dir=output_dir,
    )
    manifest_path = default_manifest_path(
        manifest_dir=manifest_dir,
        signal=normalized_signal,
        generation_run=generation_run,
    )
    write_manifest(
        manifest_path=manifest_path,
        signal=normalized_signal,
        dataset_path=dataset_path,
        row_count=len(public_rows),
        teacher_model=teacher_model,
        teacher_provider="openrouter",
        generation_run=generation_run,
        token_target=token_target,
        metadata={
            "prompt_count": len(prompt_records),
            "batch_count": len(batches),
            "batch_size": batch_size,
        },
    )

    return DistillationRunResult(
        signal=normalized_signal,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(public_rows),
    )
