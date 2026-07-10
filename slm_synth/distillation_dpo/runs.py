"""Run helpers for isolated distillation-DPO artifacts."""

from __future__ import annotations

from collections import Counter
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.accepted_target import accepted_target_metadata, raise_for_underfilled_manifest
from slm_synth.adaptive_batch import AdaptiveBatchSizeController
from slm_synth.distillation_dpo.batches import (
    DISTILLATION_DPO_BATCH_RESPONSE_SCHEMA,
    render_distillation_dpo_batch_prompt,
    validate_distillation_dpo_batch_response,
)
from slm_synth.distillation_dpo.io import write_family_dataset, write_jsonl, write_manifest, write_run_manifest
from slm_synth.distillation_dpo.pair_quality import (
    PairQualitySummary,
    aggregate_rejection_reasons,
    filter_pairs_by_quality,
)
from slm_synth.distillation_dpo.seeds import DISTILLATION_DPO_FAMILIES, build_seed_rows, validate_family
from slm_synth.distillation_dpo.spec_builders import build_production_rows
from slm_synth.dpo.generation import StructuredTeacherBackend, build_openrouter_backend
from slm_synth.planning import build_count_plan
from slm_synth.run_summary import print_batch_failure, print_batch_progress
from slm_synth.telemetry import aggregate_llm_telemetry_from_manifests
from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    MAX_OPENROUTER_BATCH_SIZE,
    MAX_OPENROUTER_CONCURRENCY,
    MIN_OPENROUTER_BATCH_SIZE,
    MIN_OPENROUTER_CONCURRENCY,
)


@dataclass(frozen=True)
class DistillationDPOFamilyResult:
    family: str
    dataset_path: Path
    manifest_path: Path
    row_count: int
    planned_pairs: int
    accepted_pairs: int
    rejected_pairs: int
    rejection_reasons: dict[str, int]
    max_backfill_rounds: int = 0
    backfill_rounds: int = 0
    batch_manifest_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class DistillationDPORunResult:
    generation_run: str
    results: tuple[DistillationDPOFamilyResult, ...]
    manifest_path: Path
    target_pairs: int | None = None
    planned_pairs: int = 0
    accepted_pairs: int = 0
    rejected_pairs: int = 0
    rejection_reasons: dict[str, int] | None = None

    @property
    def row_count(self) -> int:
        return sum(result.row_count for result in self.results)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(result.family for result in self.results)


def default_manifest_path(*, manifest_dir: str | Path, family: str, generation_run: str) -> Path:
    normalized_family = validate_family(family)
    return Path(manifest_dir) / f"{normalized_family}.{generation_run}.manifest.json"


def default_run_manifest_path(*, manifest_dir: str | Path, generation_run: str) -> Path:
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(manifest_dir) / f"{generation_run}.manifest.json"


def resolve_families(families: Sequence[str] | None = None) -> tuple[str, ...]:
    if families is None or not families or "all" in families:
        if families and len(families) > 1:
            raise ValueError("'all' cannot be combined with explicit distillation-DPO families")
        return tuple(sorted(DISTILLATION_DPO_FAMILIES))

    normalized = tuple(validate_family(family) for family in families)
    duplicates = sorted({family for family in normalized if normalized.count(family) > 1})
    if duplicates:
        raise ValueError(f"duplicate distillation-DPO family/families requested: {duplicates}")
    return normalized


def normalize_family_pair_counts(
    *,
    families: Sequence[str] | None,
    target_pairs: int,
) -> dict[str, int]:
    """Split a run-level target pair count deterministically across families."""
    normalized_families = resolve_families(families)
    _validate_positive_int(target_pairs, "target_pairs")
    if target_pairs < len(normalized_families):
        raise ValueError("target_pairs must be at least the number of requested distillation-DPO families")

    base = target_pairs // len(normalized_families)
    remainder = target_pairs % len(normalized_families)
    counts: dict[str, int] = {}
    for index, family in enumerate(normalized_families):
        counts[family] = base + (1 if index < remainder else 0)
    return counts


def materialize_seed_dataset(
    *,
    family: str,
    count: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    token_target: str | int | None = None,
    start_index: int = 1,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    max_backfill_rounds: int = 2,
    raise_on_underfill: bool = True,
) -> DistillationDPOFamilyResult:
    """Materialize one smoke/control distillation-DPO family dataset."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    rows, accepted_rows, quality, backfill_rounds = _build_rows_until_target(
        family=normalized_family,
        target_pairs=count,
        start_index=start_index,
        max_backfill_rounds=max_backfill_rounds,
        builder=lambda *, count, start_index: build_seed_rows(
            family=normalized_family, count=count, start_index=start_index
        ),
    )
    return _materialize_family_dataset(
        family=normalized_family,
        accepted_rows=accepted_rows,
        quality=quality,
        target_pairs=count,
        planned_pairs=len(rows),
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=token_target,
        dataset_filename=dataset_filename,
        manifest_filename=manifest_filename,
        metadata={"generation_mode": "seed_controlled_weak"},
        raise_on_underfill=raise_on_underfill,
    )


def materialize_production_dataset(
    *,
    family: str,
    count: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    start_index: int = 1,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    max_backfill_rounds: int = 2,
    raise_on_underfill: bool = True,
) -> DistillationDPOFamilyResult:
    """Materialize one production distillation-DPO family dataset."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    rows, accepted_rows, quality, backfill_rounds = _build_rows_until_target(
        family=normalized_family,
        target_pairs=count,
        start_index=start_index,
        max_backfill_rounds=max_backfill_rounds,
        builder=lambda *, count, start_index: build_production_rows(
            family=normalized_family, count=count, start_index=start_index
        ),
    )
    return _materialize_family_dataset(
        family=normalized_family,
        accepted_rows=accepted_rows,
        quality=quality,
        target_pairs=count,
        planned_pairs=len(rows),
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=None,
        dataset_filename=dataset_filename,
        manifest_filename=manifest_filename,
        metadata={"generation_mode": "production_controlled_weak"},
        raise_on_underfill=raise_on_underfill,
    )


def materialize_seed_run(
    *,
    families: Sequence[str] | None,
    count_per_family: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    token_target: str | int | None = None,
    start_index: int = 1,
    run_manifest_filename: str | None = None,
    max_backfill_rounds: int = 2,
) -> DistillationDPORunResult:
    """Materialize a deterministic distillation-DPO smoke/control run."""
    _validate_positive_int(count_per_family, "count_per_family")
    normalized_families = resolve_families(families)
    results: list[DistillationDPOFamilyResult] = []
    for family in normalized_families:
        results.append(
            materialize_seed_dataset(
                family=family,
                count=count_per_family,
                output_dir=output_dir,
                manifest_dir=manifest_dir,
                teacher_model=teacher_model,
                teacher_provider=teacher_provider,
                generation_run=generation_run,
                token_target=token_target,
                start_index=start_index,
                max_backfill_rounds=max_backfill_rounds,
                raise_on_underfill=False,
            )
        )

    return _write_run_result(
        generation_run=generation_run,
        results=results,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=token_target,
        target_pairs=len(normalized_families) * count_per_family,
        run_manifest_filename=run_manifest_filename,
        metadata={
            "generation_mode": "seed_controlled_weak_run",
            "count_per_family": count_per_family,
            "max_backfill_rounds": max_backfill_rounds,
        },
    )


def materialize_production_run(
    *,
    families: Sequence[str] | None,
    target_pairs: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    start_index: int = 1,
    run_manifest_filename: str | None = None,
    max_backfill_rounds: int = 2,
) -> DistillationDPORunResult:
    """Materialize a deterministic production distillation-DPO run."""
    pair_counts = normalize_family_pair_counts(families=families, target_pairs=target_pairs)
    results: list[DistillationDPOFamilyResult] = []
    for family, count in pair_counts.items():
        results.append(
            materialize_production_dataset(
                family=family,
                count=count,
                output_dir=output_dir,
                manifest_dir=manifest_dir,
                teacher_model=teacher_model,
                teacher_provider=teacher_provider,
                generation_run=generation_run,
                start_index=start_index,
                max_backfill_rounds=max_backfill_rounds,
                raise_on_underfill=False,
            )
        )

    return _write_run_result(
        generation_run=generation_run,
        results=results,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=None,
        target_pairs=target_pairs,
        run_manifest_filename=run_manifest_filename,
        metadata={
            "generation_mode": "production_controlled_weak_run",
            "target_pairs": target_pairs,
            "pairs_per_family": pair_counts,
            "max_backfill_rounds": max_backfill_rounds,
        },
    )


def generate_llm_run(
    *,
    families: Sequence[str] | None,
    count_per_family: int | None = None,
    target_pairs: int | None = None,
    batch_size: int = 1,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    teacher_provider: str = "openrouter",
    start_index: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_initial_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    adaptive_initial_batch_size: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_BATCH_SIZE,
    adaptive_batch_increase_successes: int = DEFAULT_OPENROUTER_ADAPTIVE_BATCH_INCREASE_SUCCESSES,
    concurrency: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    max_backfill_rounds: int = 2,
    run_manifest_filename: str | None = None,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    backend: StructuredTeacherBackend | None = None,
) -> DistillationDPORunResult:
    """Generate distillation-DPO preference pairs with an LLM teacher."""
    normalized_families = resolve_families(families)
    count_plan = build_count_plan(
        keys=normalized_families,
        count_per_key=count_per_family,
        target_count=target_pairs,
        key_name="family",
        count_per_key_name="count_per_family",
        target_count_name="target_pairs",
        target_mode="target_pairs",
    )
    _validate_openrouter_batch_size(batch_size)
    _validate_openrouter_concurrency(concurrency)
    _validate_positive_int(max_tokens, "max_tokens")
    _validate_positive_int(start_index, "start_index")
    _validate_positive_int(adaptive_initial_in_flight, "adaptive_initial_in_flight")
    _validate_positive_int(adaptive_initial_batch_size, "adaptive_initial_batch_size")
    _validate_positive_int(adaptive_batch_increase_successes, "adaptive_batch_increase_successes")
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")

    active_backend = backend or build_openrouter_backend(
        model=teacher_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        request_timeout=request_timeout,
        max_request_retries=max_request_retries,
        max_retryable_request_attempts=max_retryable_request_attempts,
        retry_max_elapsed_seconds=retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=concurrency,
        adaptive_initial_in_flight=adaptive_initial_in_flight,
        openrouter_routing_mode=openrouter_routing_mode,
        openrouter_provider=openrouter_provider,
    )

    results: list[DistillationDPOFamilyResult] = []
    for family in normalized_families:
        results.append(
            _generate_llm_family(
                family=family,
                target_pairs=count_plan.counts_by_key[family],
                output_dir=output_dir,
                manifest_dir=manifest_dir,
                teacher_model=teacher_model,
                teacher_provider=teacher_provider,
                generation_run=generation_run,
                max_tokens=max_tokens,
                start_index=start_index,
                batch_size=batch_size,
                concurrency=concurrency,
                adaptive_initial_in_flight=adaptive_initial_in_flight,
                adaptive_initial_batch_size=adaptive_initial_batch_size,
                adaptive_batch_increase_successes=adaptive_batch_increase_successes,
                max_backfill_rounds=max_backfill_rounds,
                backend=active_backend,
                metadata=dict(metadata or {}),
            )
        )

    llm_manifest_paths = [path for result in results for path in result.batch_manifest_paths]
    return _write_run_result(
        generation_run=generation_run,
        results=results,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=None,
        target_pairs=count_plan.planned_count,
        run_manifest_filename=run_manifest_filename,
        metadata={
            "generation_mode": "live_llm_run",
            "planning_mode": count_plan.planning_mode,
            "target_pairs": target_pairs,
            "pairs_per_family": dict(count_plan.counts_by_key),
            "count_per_family": count_per_family,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "adaptive_maximum_in_flight": concurrency,
            "adaptive_initial_in_flight": adaptive_initial_in_flight,
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            "llm_telemetry": aggregate_llm_telemetry_from_manifests(llm_manifest_paths),
            **dict(metadata or {}),
        },
    )


def _generate_llm_family(
    *,
    family: str,
    target_pairs: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    max_tokens: int,
    start_index: int,
    batch_size: int,
    concurrency: int,
    adaptive_initial_in_flight: int,
    adaptive_initial_batch_size: int,
    adaptive_batch_increase_successes: int,
    max_backfill_rounds: int,
    backend: StructuredTeacherBackend,
    metadata: Mapping[str, Any],
) -> DistillationDPOFamilyResult:
    normalized_family = validate_family(family)
    print(
        "[generate] Starting distillation-DPO family: "
        f"{normalized_family} (target_pairs={target_pairs}, batch_size={batch_size}, "
        f"min_batch_size=1, parallel_requests={concurrency}, model={teacher_model})",
        flush=True,
    )

    accepted_rows: list[dict[str, Any]] = []
    attempted_rows: list[dict[str, Any]] = []
    batch_manifest_paths: list[Path] = []
    quality = PairQualitySummary(checked_pairs=0, accepted_pairs=0, rejected_pairs=0, rejection_reasons={})
    backfill_rounds = 0
    next_start_index = start_index
    planned_pairs = 0
    batch_controller = AdaptiveBatchSizeController(
        maximum=batch_size,
        minimum=1,
        initial=adaptive_initial_batch_size,
        increase_successes=adaptive_batch_increase_successes,
    )

    while len(accepted_rows) < target_pairs:
        if planned_pairs and backfill_rounds >= max_backfill_rounds:
            break
        remaining = target_pairs - len(accepted_rows)
        if planned_pairs:
            backfill_rounds += 1
        source_rows = build_production_rows(
            family=normalized_family,
            count=remaining,
            start_index=next_start_index,
        )
        next_start_index += remaining
        planned_pairs += len(source_rows)
        round_rows, round_manifests = _generate_llm_rows_for_source_rows(
            family=normalized_family,
            source_rows=source_rows,
            output_dir=output_dir,
            manifest_dir=manifest_dir,
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
            generation_run=generation_run,
            max_tokens=max_tokens,
            batch_controller=batch_controller,
            concurrency=concurrency,
            adaptive_initial_in_flight=adaptive_initial_in_flight,
            backend=backend,
            metadata={
                "generation_mode": "live_llm_batch",
                "family": normalized_family,
                "max_tokens": max_tokens,
                **dict(metadata),
            },
        )
        attempted_rows.extend(round_rows)
        batch_manifest_paths.extend(round_manifests)
        accepted_rows, quality = filter_pairs_by_quality(family=normalized_family, rows=attempted_rows)
        if len(accepted_rows) > target_pairs:
            accepted_rows = accepted_rows[:target_pairs]
            quality = PairQualitySummary(
                checked_pairs=quality.checked_pairs,
                accepted_pairs=len(accepted_rows),
                rejected_pairs=max(quality.checked_pairs - len(accepted_rows), 0),
                rejection_reasons=quality.rejection_reasons,
            )

    dataset_path = write_family_dataset(
        family=normalized_family,
        rows=accepted_rows,
        output_dir=output_dir,
    )
    manifest_path = default_manifest_path(
        manifest_dir=manifest_dir,
        family=normalized_family,
        generation_run=generation_run,
    )
    manifest_metadata = _planning_metadata(
        base_metadata={
            "generation_mode": "live_llm_family",
            "batch_size": batch_size,
            "concurrency": concurrency,
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            **batch_controller.snapshot(),
            "llm_telemetry": aggregate_llm_telemetry_from_manifests(batch_manifest_paths),
            **dict(metadata),
        },
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        quality=quality,
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
    )
    write_manifest(
        manifest_path=manifest_path,
        family=normalized_family,
        dataset_path=dataset_path,
        row_count=len(accepted_rows),
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=None,
        metadata=manifest_metadata,
    )
    print(
        "[generate] Completed distillation-DPO family: "
        f"{normalized_family} rows={len(accepted_rows)}, target_pairs={target_pairs}, "
        f"planned_pairs={planned_pairs}, rejected_pairs={quality.rejected_pairs}, "
        f"adaptive_batch_size_observed_minimum={batch_controller.observed_minimum}, "
        f"adaptive_batch_size_observed_peak={batch_controller.observed_peak}, "
        f"adaptive_batch_size_increases={batch_controller.increases}, "
        f"adaptive_batch_size_decreases={batch_controller.decreases}, "
        f"adaptive_batch_size_failures={batch_controller.failures}",
        flush=True,
    )
    return DistillationDPOFamilyResult(
        family=normalized_family,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(accepted_rows),
        planned_pairs=planned_pairs,
        accepted_pairs=quality.accepted_pairs,
        rejected_pairs=quality.rejected_pairs,
        rejection_reasons=quality.rejection_reasons,
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
        batch_manifest_paths=tuple(batch_manifest_paths),
    )


def _generate_llm_rows_for_source_rows(
    *,
    family: str,
    source_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    max_tokens: int,
    batch_controller: AdaptiveBatchSizeController,
    concurrency: int,
    adaptive_initial_in_flight: int,
    backend: StructuredTeacherBackend,
    metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    pending_ranges: deque[tuple[int, int]] = deque([(0, len(source_rows))])
    active: dict[Any, dict[str, Any]] = {}
    completed_jobs: list[dict[str, Any]] = []
    rows_done = 0

    def make_job(offset: int, size: int) -> dict[str, Any]:
        batch_rows = list(source_rows[offset : offset + size])
        batch_start_index = _row_index(batch_rows[0]) if batch_rows else offset + 1
        return {
            "family": family,
            "batch_number": batch_start_index,
            "batch_start_index": batch_start_index,
            "rows": batch_rows,
            "dataset_path": default_batch_output_dir(output_dir) / f"{family}.batch{batch_start_index:06d}.jsonl",
            "manifest_path": Path(manifest_dir) / f"{family}.batch{batch_start_index:06d}.{generation_run}.manifest.json",
        }

    def active_job_limit() -> int:
        return min(concurrency, max(1, adaptive_initial_in_flight, batch_controller.current))

    def submit_available(executor: ThreadPoolExecutor) -> None:
        while pending_ranges and len(active) < active_job_limit():
            offset, remaining = pending_ranges.popleft()
            size = min(batch_controller.current, remaining)
            if remaining > size:
                pending_ranges.appendleft((offset + size, remaining - size))
            job = make_job(offset, size)
            active[executor.submit(_run_llm_batch_job, job, teacher_model, teacher_provider, generation_run, max_tokens, backend, metadata)] = job

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
                        workflow="distillation-DPO",
                        group_key="family",
                        group_value=family,
                        batch_number=job["batch_number"],
                        batch_start=job["batch_start_index"],
                        batch_size=len(job["rows"]),
                        adaptive_batch_size=batch_controller.snapshot(),
                        error=exc,
                    )
                    if len(job["rows"]) <= batch_controller.minimum:
                        raise
                    offset = source_rows.index(job["rows"][0])
                    pending_ranges.appendleft((offset, len(job["rows"])))
                    submit_available(executor)
                    continue
                batch_controller.record_success()
                job["result"] = result
                job["adaptive_batch_size"] = batch_controller.snapshot()
                completed_jobs.append(job)
                rows_done += len(result["rows"])
                print_batch_progress(
                    workflow="distillation-DPO",
                    group_key="family",
                    group_value=family,
                    batch_number=job["batch_number"],
                    batch_start=job["batch_start_index"],
                    batch_size=len(job["rows"]),
                    rows_done=rows_done,
                    rows_total=len(source_rows),
                    manifest_path=result["manifest_path"],
                    adaptive_batch_size=job["adaptive_batch_size"],
                )
                submit_available(executor)

    completed_jobs.sort(key=lambda item: (item["batch_start_index"], item["batch_number"]))
    generated_rows: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    for job in completed_jobs:
        generated_rows.extend(job["result"]["rows"])
        manifest_paths.append(job["result"]["manifest_path"])
    return generated_rows, manifest_paths


def _run_llm_batch_job(
    job: Mapping[str, Any],
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    max_tokens: int,
    backend: StructuredTeacherBackend,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    source_rows = list(job["rows"])
    prompt = render_distillation_dpo_batch_prompt(source_rows)
    result = backend.generate_structured_object_with_metadata(
        prompt=prompt,
        schema=DISTILLATION_DPO_BATCH_RESPONSE_SCHEMA,
        schema_name="distillation_dpo_batch",
    )
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("distillation-DPO teacher backend returned non-object data")
    telemetry = result.get("telemetry")
    expected_ids = [str(row["id"]) for row in source_rows]
    rows = validate_distillation_dpo_batch_response(
        data,
        expected_ids=expected_ids,
        expected_count=len(source_rows),
    )
    dataset_path = Path(job["dataset_path"])
    row_count = write_jsonl(rows, dataset_path)
    manifest_path = write_manifest(
        manifest_path=job["manifest_path"],
        family=str(job["family"]),
        dataset_path=dataset_path,
        row_count=row_count,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=None,
        metadata={
            "generation_mode": "live_llm_batch",
            "batch_number": job["batch_number"],
            "batch_start_index": job["batch_start_index"],
            "batch_size": len(source_rows),
            "max_tokens": max_tokens,
            "llm_telemetry": dict(telemetry) if isinstance(telemetry, Mapping) else {},
            **dict(metadata),
        },
    )
    return {"rows": rows, "dataset_path": dataset_path, "manifest_path": manifest_path}


def default_batch_output_dir(output_dir: str | Path) -> Path:
    """Return the sibling internal batch directory for a public dataset directory."""
    public_dir = Path(output_dir)
    return public_dir.parent / "batches"


def _row_index(row: Mapping[str, Any]) -> int:
    row_id = str(row.get("id", ""))
    suffix = row_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else 1


def _build_rows_until_target(
    *,
    family: str,
    target_pairs: int,
    start_index: int,
    max_backfill_rounds: int,
    builder: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PairQualitySummary, int]:
    _validate_positive_int(target_pairs, "target_pairs")
    _validate_positive_int(start_index, "start_index")
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")

    attempted_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    quality = PairQualitySummary(checked_pairs=0, accepted_pairs=0, rejected_pairs=0, rejection_reasons={})
    next_start_index = start_index
    backfill_rounds = 0

    while len(accepted_rows) < target_pairs:
        if attempted_rows and backfill_rounds >= max_backfill_rounds:
            break
        remaining = target_pairs - len(accepted_rows)
        if attempted_rows:
            backfill_rounds += 1
        new_rows = builder(count=remaining, start_index=next_start_index)
        attempted_rows.extend(new_rows)
        next_start_index += remaining
        accepted_rows, quality = filter_pairs_by_quality(family=family, rows=attempted_rows)

    if len(accepted_rows) > target_pairs:
        accepted_rows = accepted_rows[:target_pairs]
        quality = PairQualitySummary(
            checked_pairs=quality.checked_pairs,
            accepted_pairs=len(accepted_rows),
            rejected_pairs=max(quality.checked_pairs - len(accepted_rows), 0),
            rejection_reasons=quality.rejection_reasons,
        )
    return attempted_rows, accepted_rows, quality, backfill_rounds


def _materialize_family_dataset(
    *,
    family: str,
    accepted_rows: Sequence[Mapping[str, Any]],
    quality: PairQualitySummary,
    target_pairs: int,
    planned_pairs: int,
    max_backfill_rounds: int,
    backfill_rounds: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    teacher_provider: str,
    generation_run: str,
    token_target: str | int | None,
    metadata: Mapping[str, Any],
    dataset_filename: str | None,
    manifest_filename: str | None,
    raise_on_underfill: bool = True,
) -> DistillationDPOFamilyResult:
    dataset_path = write_family_dataset(
        family=family,
        rows=accepted_rows,
        output_dir=output_dir,
        filename=dataset_filename,
    )
    manifest_path = (
        Path(manifest_dir) / manifest_filename
        if manifest_filename is not None
        else default_manifest_path(manifest_dir=manifest_dir, family=family, generation_run=generation_run)
    )
    manifest_metadata = _planning_metadata(
        base_metadata=metadata,
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        quality=quality,
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
    )
    write_manifest(
        manifest_path=manifest_path,
        family=family,
        dataset_path=dataset_path,
        row_count=len(accepted_rows),
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        token_target=token_target,
        metadata=manifest_metadata,
    )
    if raise_on_underfill:
        raise_for_underfilled_manifest(manifest_path, artifact_name=f"distillation-dpo family {family}")
    return DistillationDPOFamilyResult(
        family=family,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=len(accepted_rows),
        planned_pairs=planned_pairs,
        accepted_pairs=quality.accepted_pairs,
        rejected_pairs=quality.rejected_pairs,
        rejection_reasons=quality.rejection_reasons,
        max_backfill_rounds=max_backfill_rounds,
        backfill_rounds=backfill_rounds,
    )


def _write_run_result(
    *,
    generation_run: str,
    results: Sequence[DistillationDPOFamilyResult],
    manifest_dir: str | Path,
    teacher_model: str,
    teacher_provider: str,
    token_target: str | int | None,
    target_pairs: int,
    run_manifest_filename: str | None,
    metadata: Mapping[str, Any],
) -> DistillationDPORunResult:
    planned_pairs = sum(result.planned_pairs for result in results)
    accepted_pairs = sum(result.accepted_pairs for result in results)
    rejected_pairs = sum(result.rejected_pairs for result in results)
    rejection_reasons = _merge_rejection_reasons(result.rejection_reasons for result in results)
    quality_summary = PairQualitySummary(
        checked_pairs=planned_pairs,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        rejection_reasons=rejection_reasons,
    )
    manifest_metadata = _planning_metadata(
        base_metadata=metadata,
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        quality=quality_summary,
        max_backfill_rounds=max(_metadata_max_backfill_rounds(results), int(metadata.get("max_backfill_rounds", 0)) if isinstance(metadata.get("max_backfill_rounds"), int) else 0),
        backfill_rounds=max(_metadata_backfill_rounds(results), 0),
    )
    manifest_path = (
        Path(manifest_dir) / run_manifest_filename
        if run_manifest_filename is not None
        else default_run_manifest_path(manifest_dir=manifest_dir, generation_run=generation_run)
    )
    write_run_manifest(
        manifest_path=manifest_path,
        generation_run=generation_run,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        token_target=token_target,
        datasets=(
            {
                "family": result.family,
                "dataset_path": result.dataset_path,
                "manifest_path": result.manifest_path,
                "row_count": result.row_count,
            }
            for result in results
        ),
        metadata=manifest_metadata,
    )
    raise_for_underfilled_manifest(manifest_path, artifact_name="distillation-dpo")
    return DistillationDPORunResult(
        generation_run=generation_run,
        results=tuple(results),
        manifest_path=manifest_path,
        target_pairs=target_pairs,
        planned_pairs=planned_pairs,
        accepted_pairs=accepted_pairs,
        rejected_pairs=rejected_pairs,
        rejection_reasons=rejection_reasons,
    )


def _planning_metadata(
    *,
    base_metadata: Mapping[str, Any],
    target_pairs: int,
    planned_pairs: int,
    quality: PairQualitySummary,
    max_backfill_rounds: int,
    backfill_rounds: int,
) -> dict[str, Any]:
    metadata = dict(base_metadata)
    metadata.update(
        {
            "source_contract": {
                "chosen_source": "teacher",
                "rejected_source": "controlled_weak",
                "target_consumer": "slm-distillation",
            },
            "target_pairs": target_pairs,
            "planned_pairs": planned_pairs,
            "accepted_pairs": quality.accepted_pairs,
            "rejected_pairs": quality.rejected_pairs,
            "rejection_reasons": dict(sorted(quality.rejection_reasons.items())),
            "pair_quality": quality.to_dict(),
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": backfill_rounds,
            **accepted_target_metadata(
                unit="pairs",
                target_count=target_pairs,
                accepted_count=quality.accepted_pairs,
                attempted_count=planned_pairs,
                max_backfill_rounds=max_backfill_rounds,
                backfill_rounds=backfill_rounds,
            ),
        }
    )
    return metadata


def _metadata_max_backfill_rounds(results: Sequence[DistillationDPOFamilyResult]) -> int:
    return max((result.max_backfill_rounds for result in results), default=0)


def _metadata_backfill_rounds(results: Sequence[DistillationDPOFamilyResult]) -> int:
    return max((result.backfill_rounds for result in results), default=0)


def _merge_rejection_reasons(summaries: Sequence[Mapping[str, int]] | Any) -> dict[str, int]:
    if not isinstance(summaries, Sequence):
        return aggregate_rejection_reasons([])
    counter: Counter[str] = Counter()
    for summary in summaries:
        if isinstance(summary, Mapping):
            for reason, count in summary.items():
                if isinstance(reason, str) and isinstance(count, int) and count > 0:
                    counter[reason] += count
    return dict(sorted(counter.items()))


def _validate_openrouter_batch_size(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("batch_size must be an integer")
    if not MIN_OPENROUTER_BATCH_SIZE <= value <= MAX_OPENROUTER_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be between {MIN_OPENROUTER_BATCH_SIZE} and {MAX_OPENROUTER_BATCH_SIZE}"
        )


def _validate_openrouter_concurrency(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("concurrency must be an integer")
    if not MIN_OPENROUTER_CONCURRENCY <= value <= MAX_OPENROUTER_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between {MIN_OPENROUTER_CONCURRENCY} and {MAX_OPENROUTER_CONCURRENCY}"
        )


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
