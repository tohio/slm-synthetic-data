"""Small multi-signal orchestration helpers for response distillation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.accepted_target import accepted_target_metadata
from slm_synth.adaptive_batch import AdaptiveBatchSizeController
from slm_synth.distillation_sft.generation import (
    StructuredTeacherBackend,
    build_openrouter_backend,
    generate_and_materialize_signal_batch,
)
from slm_synth.distillation_sft.io import write_jsonl, write_manifest, write_run_manifest
from slm_synth.distillation_sft.prompt_quality import validate_prompt_preflight
from slm_synth.distillation_sft.response_quality import (
    RESPONSE_QUALITY_CHECKS,
    aggregate_rejection_reasons,
)
from slm_synth.distillation_sft.runs import DistillationRunResult
from slm_synth.distillation_sft.seeds import build_seed_prompt_records
from slm_synth.distillation_sft.spec_builders import build_prompt_spec_records
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS, validate_signal
from slm_synth.telemetry import aggregate_llm_telemetry_from_manifests
from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    MAX_OPENROUTER_BATCH_SIZE,
    MAX_OPENROUTER_CONCURRENCY,
)
from slm_synth.run_summary import print_batch_failure, print_batch_progress


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
    target_rows: int | None = None,
) -> dict[str, int]:
    """Build a validated mapping of signal -> planned prompt count.

    Production runs should use target_rows as the dataset-size planning knob.
    count_per_signal remains available for explicit smoke/bakeoff-sized runs, and
    counts_by_signal remains available for callers that already computed a fixed
    per-signal plan. Exactly one planning strategy must be provided.
    """
    normalized_signals = normalize_signal_sequence(signals)

    provided_strategies = sum(
        value is not None for value in (count_per_signal, counts_by_signal, target_rows)
    )
    if provided_strategies > 1:
        raise ValueError("provide only one of count_per_signal, counts_by_signal, or target_rows")

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

    if target_rows is not None:
        target = _validate_target_rows(target_rows)
        if target < len(normalized_signals):
            raise ValueError("target_rows must be at least the number of requested signals")
        base_count, remainder = divmod(target, len(normalized_signals))
        return {
            signal: base_count + (1 if index < remainder else 0)
            for index, signal in enumerate(normalized_signals)
        }

    if count_per_signal is None:
        raise ValueError("one of count_per_signal, counts_by_signal, or target_rows is required")

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
    adaptive_maximum_in_flight: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    adaptive_initial_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
    adaptive_initial_batch_size: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    adaptive_batch_increase_successes: int = DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    batch_size: int | None = None,
    concurrency: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    run_manifest_filename: str | None = None,
    backend_factory: BackendFactory | None = None,
    max_backfill_rounds: int = 2,
) -> MultiSignalRunResult:
    """Generate smoke seed prompts across signals and materialize public datasets."""
    return _generate_multi_signal_run(
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        generation_run=generation_run,
        max_tokens=max_tokens,
        count_per_signal=count_per_signal,
        counts_by_signal=counts_by_signal,
        target_rows=None,
        signals=signals,
        token_target=token_target,
        start_index=start_index,
        temperature=temperature,
        top_p=top_p,
        request_timeout=request_timeout,
        max_request_retries=max_request_retries,
        max_retryable_request_attempts=max_retryable_request_attempts,
        retry_max_elapsed_seconds=retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=adaptive_maximum_in_flight,
        adaptive_initial_in_flight=adaptive_initial_in_flight,
        openrouter_routing_mode=openrouter_routing_mode,
        openrouter_provider=openrouter_provider,
        adaptive_initial_batch_size=adaptive_initial_batch_size,
        adaptive_batch_increase_successes=adaptive_batch_increase_successes,
        batch_size=batch_size,
        concurrency=concurrency,
        run_manifest_filename=run_manifest_filename,
        backend_factory=backend_factory,
        max_backfill_rounds=max_backfill_rounds,
        prompt_record_builder=build_seed_prompt_records,
        prompt_source="builtin_seed",
        require_unique_prompt_text=False,
    )


def generate_prompt_spec_multi_signal_run(
    *,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    count_per_signal: int | None = None,
    counts_by_signal: Mapping[str, int] | None = None,
    target_rows: int | None = None,
    signals: Sequence[str] | None = None,
    token_target: str | int | None = None,
    start_index: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    adaptive_initial_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
    adaptive_initial_batch_size: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    adaptive_batch_increase_successes: int = DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    batch_size: int | None = None,
    concurrency: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    run_manifest_filename: str | None = None,
    backend_factory: BackendFactory | None = None,
    max_backfill_rounds: int = 2,
) -> MultiSignalRunResult:
    """Generate production prompt specs across signals and materialize public datasets."""
    return _generate_multi_signal_run(
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        generation_run=generation_run,
        max_tokens=max_tokens,
        count_per_signal=count_per_signal,
        counts_by_signal=counts_by_signal,
        target_rows=target_rows,
        signals=signals,
        token_target=token_target,
        start_index=start_index,
        temperature=temperature,
        top_p=top_p,
        request_timeout=request_timeout,
        max_request_retries=max_request_retries,
        max_retryable_request_attempts=max_retryable_request_attempts,
        retry_max_elapsed_seconds=retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=adaptive_maximum_in_flight,
        adaptive_initial_in_flight=adaptive_initial_in_flight,
        openrouter_routing_mode=openrouter_routing_mode,
        openrouter_provider=openrouter_provider,
        adaptive_initial_batch_size=adaptive_initial_batch_size,
        adaptive_batch_increase_successes=adaptive_batch_increase_successes,
        batch_size=batch_size,
        concurrency=concurrency,
        run_manifest_filename=run_manifest_filename,
        backend_factory=backend_factory,
        max_backfill_rounds=max_backfill_rounds,
        prompt_record_builder=build_prompt_spec_records,
        prompt_source="production_spec",
        require_unique_prompt_text=True,
    )


def _generate_multi_signal_run(
    *,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    count_per_signal: int | None,
    counts_by_signal: Mapping[str, int] | None,
    target_rows: int | None,
    signals: Sequence[str] | None,
    token_target: str | int | None,
    start_index: int,
    temperature: float,
    top_p: float,
    request_timeout: float | None,
    max_request_retries: int,
    max_retryable_request_attempts: int,
    retry_max_elapsed_seconds: float,
    adaptive_maximum_in_flight: int,
    adaptive_initial_in_flight: int,
    openrouter_routing_mode: str | None,
    openrouter_provider: str | None,
    adaptive_initial_batch_size: int,
    adaptive_batch_increase_successes: int,
    batch_size: int | None,
    concurrency: int,
    run_manifest_filename: str | None,
    backend_factory: BackendFactory | None,
    max_backfill_rounds: int,
    prompt_record_builder: Callable[..., list[dict[str, Any]]],
    prompt_source: str,
    require_unique_prompt_text: bool,
) -> MultiSignalRunResult:
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    signal_counts = normalize_signal_counts(
        signals=signals,
        count_per_signal=count_per_signal,
        counts_by_signal=counts_by_signal,
        target_rows=target_rows,
    )
    normalized_batch_size = _validate_batch_size(batch_size)
    _validate_openrouter_concurrency(concurrency)
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")

    signal_items = list(signal_counts.items())
    prompt_records_by_signal = {
        signal: prompt_record_builder(signal=signal, count=count, start_index=start_index)
        for signal, count in signal_items
    }
    prompt_preflight = validate_prompt_preflight(
        [record for records in prompt_records_by_signal.values() for record in records],
        require_unique_prompt_text=require_unique_prompt_text,
    )

    def run_signal(item: tuple[str, int]) -> DistillationRunResult:
        signal, _count = item
        prompt_records = prompt_records_by_signal[signal]
        backend = backend_factory(signal) if backend_factory is not None else None
        return _generate_and_materialize_signal_batches(
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
            openrouter_routing_mode=openrouter_routing_mode,
            openrouter_provider=openrouter_provider,
            adaptive_initial_batch_size=adaptive_initial_batch_size,
            adaptive_batch_increase_successes=adaptive_batch_increase_successes,
            batch_size=normalized_batch_size,
            concurrency=concurrency,
            prompt_source=prompt_source,
            require_unique_prompt_text=require_unique_prompt_text,
            prompt_record_builder=prompt_record_builder,
            target_rows=_count,
            start_index=start_index,
            max_backfill_rounds=max_backfill_rounds,
            backend=backend,
        )

    results = [run_signal(item) for item in signal_items]
    target_prompt_rows = sum(signal_counts.values())
    signal_manifest_paths = [result.manifest_path for result in results]
    signal_metadata = _metadata_from_manifests(signal_manifest_paths)
    planned_prompt_rows = sum(_metadata_int(metadata, "planned_prompt_rows") for metadata in signal_metadata)
    accepted_rows = sum(result.row_count for result in results)
    rejected_rows = sum(_metadata_int(metadata, "rejected_rows") for metadata in signal_metadata)
    remaining_rows = max(target_prompt_rows - accepted_rows, 0)
    backfill_rounds = max((_metadata_int(metadata, "backfill_rounds") for metadata in signal_metadata), default=0)
    rejection_reasons = _aggregate_rejection_reasons_from_metadata(signal_metadata)
    llm_telemetry = aggregate_llm_telemetry_from_manifests(signal_manifest_paths)

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
            "signals": [signal for signal, _count in signal_items],
            "rows_per_signal": dict(signal_counts),
            "target_rows": target_rows,
            "target_prompt_rows": target_prompt_rows,
            "planned_prompt_rows": planned_prompt_rows,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
            "rejection_reasons": rejection_reasons,
            "response_quality": {
                "checked_rows": planned_prompt_rows,
                "accepted_rows": accepted_rows,
                "rejected_rows": rejected_rows,
                "rejection_reasons": rejection_reasons,
                "checks": list(RESPONSE_QUALITY_CHECKS),
            },
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": backfill_rounds,
            **accepted_target_metadata(
                unit="rows",
                target_count=target_prompt_rows,
                accepted_count=accepted_rows,
                attempted_count=planned_prompt_rows,
                max_backfill_rounds=max_backfill_rounds,
                backfill_rounds=backfill_rounds,
            ),
            "start_index": start_index,
            "batch_size": normalized_batch_size,
            "concurrency": concurrency,
            "adaptive_maximum_in_flight": concurrency,
            "adaptive_initial_in_flight": adaptive_initial_in_flight,
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            "prompt_source": prompt_source,
            "prompt_preflight": prompt_preflight.to_dict(),
            "llm_telemetry": llm_telemetry,
        },
    )

    return MultiSignalRunResult(
        generation_run=generation_run,
        results=tuple(results),
        manifest_path=run_manifest_path,
    )


def _aggregate_rejection_reasons_from_manifests(manifest_paths: Iterable[Path]) -> dict[str, int]:
    return _aggregate_rejection_reasons_from_metadata(_metadata_from_manifests(manifest_paths))


def _aggregate_rejection_reasons_from_metadata(metadata_items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return aggregate_rejection_reasons(metadata_items)


def _metadata_from_manifests(manifest_paths: Iterable[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})
        if isinstance(metadata, Mapping):
            summaries.append(dict(metadata))
    return summaries


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key, 0)
    return value if isinstance(value, int) and value > 0 else 0


def _validate_count(count: Any) -> int:
    if not isinstance(count, int) or count < 1:
        raise ValueError("signal prompt counts must be positive integers")
    return count


def _validate_batch_size(batch_size: Any) -> int | None:
    if batch_size is None:
        return None
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if batch_size > MAX_OPENROUTER_BATCH_SIZE:
        raise ValueError(f"batch_size must be at most {MAX_OPENROUTER_BATCH_SIZE}")
    return batch_size


def _validate_positive_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_openrouter_concurrency(value: Any) -> None:
    _validate_positive_int(value, "concurrency")
    if value > MAX_OPENROUTER_CONCURRENCY:
        raise ValueError(f"concurrency must be at most {MAX_OPENROUTER_CONCURRENCY}")


def _validate_target_rows(value: Any) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError("target_rows must be a positive integer")
    return value


def _chunks(records: Sequence[Mapping[str, Any]], batch_size: int | None) -> list[list[Mapping[str, Any]]]:
    if batch_size is None or batch_size >= len(records):
        return [list(records)]
    return [list(records[index : index + batch_size]) for index in range(0, len(records), batch_size)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def default_batch_output_dir(output_dir: str | Path) -> Path:
    """Return the sibling internal batch directory for a public dataset directory."""
    public_dir = Path(output_dir)
    return public_dir.parent / "batches"


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
    adaptive_maximum_in_flight: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    adaptive_initial_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
    adaptive_initial_batch_size: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    adaptive_batch_increase_successes: int = DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    batch_size: int | None = None,
    concurrency: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    prompt_source: str = "builtin_seed",
    require_unique_prompt_text: bool = False,
    prompt_record_builder: Callable[..., list[dict[str, Any]]] | None = None,
    target_rows: int | None = None,
    start_index: int = 1,
    max_backfill_rounds: int = 2,
    backend: StructuredTeacherBackend | None = None,
) -> DistillationRunResult:
    initial_prompt_records = list(prompt_records)
    if not initial_prompt_records:
        raise ValueError("at least one distillation prompt record is required")
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")
    target_row_count = len(initial_prompt_records) if target_rows is None else _validate_target_rows(target_rows)
    all_prompt_records = list(initial_prompt_records)
    prompt_preflight = validate_prompt_preflight(
        all_prompt_records,
        require_unique_prompt_text=require_unique_prompt_text,
    )
    maximum_batch_size = batch_size or len(initial_prompt_records)
    batch_controller = AdaptiveBatchSizeController(
        maximum=maximum_batch_size,
        minimum=1,
        initial=adaptive_initial_batch_size,
        increase_successes=adaptive_batch_increase_successes,
    )
    print(
        "[generate] Starting distillation signal: "
        f"{signal} (target_rows={target_row_count}, batch_size={maximum_batch_size}, "
        f"min_batch_size=1, parallel_requests={concurrency}, model={teacher_model})",
        flush=True,
    )
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
        openrouter_routing_mode=openrouter_routing_mode,
        openrouter_provider=openrouter_provider,
    )

    def run_batch(job: dict[str, Any]) -> DistillationRunResult:
        batch_number = job["batch_number"]
        return generate_and_materialize_signal_batch(
            signal=signal,
            prompt_records=job["prompt_records"],
            output_dir=default_batch_output_dir(output_dir),
            manifest_dir=manifest_dir,
            teacher_model=teacher_model,
            generation_run=generation_run,
            max_tokens=max_tokens,
            token_target=token_target,
            dataset_filename=f"{signal}.batch{batch_number:06d}.jsonl",
            manifest_filename=f"{signal}.batch{batch_number:06d}.{generation_run}.manifest.json",
            temperature=temperature,
            top_p=top_p,
            request_timeout=request_timeout,
            max_request_retries=max_request_retries,
            max_retryable_request_attempts=max_retryable_request_attempts,
            retry_max_elapsed_seconds=retry_max_elapsed_seconds,
            adaptive_maximum_in_flight=adaptive_maximum_in_flight,
            adaptive_initial_in_flight=adaptive_initial_in_flight,
            backend=active_backend,
        )

    jobs: list[dict[str, Any]] = []
    next_batch_number = 1
    signal_rows_done = 0
    backfill_rounds_used = 0

    def active_job_limit() -> int:
        return min(concurrency, max(1, adaptive_initial_in_flight, batch_controller.current))

    def run_prompt_records_round(round_records: list[Mapping[str, Any]], *, round_number: int, round_start_offset: int) -> None:
        nonlocal next_batch_number, signal_rows_done
        pending_ranges: deque[tuple[int, int]] = deque([(0, len(round_records))])
        active: dict[Any, dict[str, Any]] = {}

        def submit_available(executor: ThreadPoolExecutor) -> None:
            nonlocal next_batch_number
            while pending_ranges and len(active) < active_job_limit():
                offset, remaining = pending_ranges.popleft()
                size = min(batch_controller.current, remaining)
                if remaining > size:
                    pending_ranges.appendleft((offset + size, remaining - size))
                batch_start_offset = round_start_offset + offset
                job = {
                    "batch_number": next_batch_number,
                    "batch_start_offset": batch_start_offset,
                    "backfill_round": round_number,
                    "prompt_records": list(round_records[offset : offset + size]),
                }
                next_batch_number += 1
                active[executor.submit(run_batch, job)] = job

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            submit_available(executor)
            while active:
                done, _ = wait(set(active), return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        batch_controller.record_failure()
                        print_batch_failure(
                            workflow="distillation",
                            group_key="signal",
                            group_value=signal,
                            batch_number=job["batch_number"],
                            batch_start=job["batch_start_offset"],
                            batch_size=len(job["prompt_records"]),
                            adaptive_batch_size=batch_controller.snapshot(),
                            error=exc,
                        )
                        if len(job["prompt_records"]) <= batch_controller.minimum:
                            raise
                        local_offset = job["batch_start_offset"] - round_start_offset
                        pending_ranges.appendleft((local_offset, len(job["prompt_records"])))
                        submit_available(executor)
                        continue
                    batch_controller.record_success()
                    job["result"] = result
                    job["adaptive_batch_size"] = batch_controller.snapshot()
                    jobs.append(job)
                    signal_rows_done += result.row_count
                    print_batch_progress(
                        workflow="distillation",
                        group_key="signal",
                        group_value=signal,
                        batch_number=job["batch_number"],
                        batch_start=job["batch_start_offset"],
                        batch_size=len(job["prompt_records"]),
                        rows_done=signal_rows_done,
                        rows_total=target_row_count,
                        manifest_path=result.manifest_path,
                        adaptive_batch_size=job["adaptive_batch_size"],
                    )
                    submit_available(executor)

    round_records = list(initial_prompt_records)
    round_start_offset = 0
    round_number = 0
    while round_records and signal_rows_done < target_row_count:
        run_prompt_records_round(round_records, round_number=round_number, round_start_offset=round_start_offset)
        if signal_rows_done >= target_row_count:
            break
        if prompt_record_builder is None or backfill_rounds_used >= max_backfill_rounds:
            break
        remaining = target_row_count - signal_rows_done
        backfill_rounds_used += 1
        next_start_index = start_index + len(all_prompt_records)
        round_records = prompt_record_builder(signal=signal, count=remaining, start_index=next_start_index)
        round_start_offset = len(all_prompt_records)
        all_prompt_records.extend(round_records)
        prompt_preflight = validate_prompt_preflight(
            all_prompt_records,
            require_unique_prompt_text=require_unique_prompt_text,
        )
        print(
            "[generate] Backfilling distillation signal: "
            f"{signal} round={backfill_rounds_used}/{max_backfill_rounds} "
            f"remaining_rows={remaining}",
            flush=True,
        )
        round_number = backfill_rounds_used

    jobs.sort(key=lambda item: (item["batch_start_offset"], item["batch_number"]))
    batch_results = [job["result"] for job in jobs]

    public_rows: list[dict[str, Any]] = []
    for batch_result in batch_results:
        public_rows.extend(_read_jsonl(batch_result.dataset_path))
    if len(public_rows) > target_row_count:
        public_rows = public_rows[:target_row_count]
    rejection_reasons = _aggregate_rejection_reasons_from_manifests(
        result.manifest_path for result in batch_results
    )

    attempted_rows = len(all_prompt_records)
    dataset_path = Path(output_dir) / f"{signal}.jsonl"
    row_count = write_jsonl(public_rows, dataset_path)
    manifest_path = Path(manifest_dir) / f"{signal}.{generation_run}.manifest.json"
    write_manifest(
        manifest_path=manifest_path,
        signal=signal,
        dataset_path=dataset_path,
        row_count=row_count,
        teacher_model=teacher_model,
        teacher_provider="openrouter",
        generation_run=generation_run,
        token_target=token_target,
        metadata={
            "prompt_count": attempted_rows,
            "initial_prompt_rows": len(initial_prompt_records),
            "target_prompt_rows": target_row_count,
            "planned_prompt_rows": attempted_rows,
            "accepted_rows": row_count,
            "rejected_rows": max(attempted_rows - row_count, 0),
            "rejection_reasons": rejection_reasons,
            "response_quality": {
                "checked_rows": attempted_rows,
                "accepted_rows": row_count,
                "rejected_rows": max(attempted_rows - row_count, 0),
                "rejection_reasons": rejection_reasons,
                "checks": list(RESPONSE_QUALITY_CHECKS),
            },
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": backfill_rounds_used,
            **accepted_target_metadata(
                unit="rows",
                target_count=target_row_count,
                accepted_count=row_count,
                attempted_count=attempted_rows,
                max_backfill_rounds=max_backfill_rounds,
                backfill_rounds=backfill_rounds_used,
            ),
            "batch_count": len(batch_results),
            "batch_size": maximum_batch_size,
            "concurrency": concurrency,
            "prompt_source": prompt_source,
            "prompt_preflight": prompt_preflight.to_dict(),
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            **batch_controller.snapshot(),
            "llm_telemetry": aggregate_llm_telemetry_from_manifests(
                result.manifest_path for result in batch_results
            ),
            "batch_manifests": [str(result.manifest_path) for result in batch_results],
        },
    )
    print(
        "[generate] Completed distillation signal: "
        f"{signal} rows={row_count}, target_rows={target_row_count}, "
        f"remaining_rows={max(target_row_count - row_count, 0)}, backfill_rounds={backfill_rounds_used}, "
        f"batches={len(batch_results)}, batch_size={maximum_batch_size}, min_batch_size=1, "
        f"parallel_requests={concurrency}, adaptive_batch_size_observed_minimum={batch_controller.observed_minimum}, "
        f"adaptive_batch_size_observed_peak={batch_controller.observed_peak}, "
        f"adaptive_batch_size_increases={batch_controller.increases}, "
        f"adaptive_batch_size_decreases={batch_controller.decreases}, "
        f"adaptive_batch_size_failures={batch_controller.failures}",
        flush=True,
    )
    return DistillationRunResult(
        signal=signal,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=row_count,
    )
