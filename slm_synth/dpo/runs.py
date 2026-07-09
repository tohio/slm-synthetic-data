"""DPO dataset materialization helpers."""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.accepted_target import accepted_target_metadata, raise_for_underfilled_manifest
from slm_synth.adaptive_batch import AdaptiveBatchSizeController
from slm_synth.planning import build_count_plan
from slm_synth.dpo.generation import StructuredTeacherBackend, build_openrouter_backend, generate_llm_batch
from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.manifest import write_manifest, write_run_manifest
from slm_synth.dpo.seeds import DPO_SEED_FAMILIES, build_seed_rows
from slm_synth.dpo.spec_builders import DPO_SPEC_FAMILIES, build_specs
from slm_synth.taxonomy.holdouts import HoldoutRegistry
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
class DPOSeedRunResult:
    """Result of materializing one DPO seed dataset."""

    dataset_path: Path
    manifest_path: Path
    row_count: int
    family: str
    generation_run: str


@dataclass(frozen=True)
class DPOSeedFamilyRunResult:
    """Result of materializing multiple DPO seed families."""

    results: tuple[DPOSeedRunResult, ...]
    row_count: int
    families: tuple[str, ...]
    generation_run: str
    manifest_path: Path


@dataclass(frozen=True)
class DPOLLMRunResult:
    """Result of running one multi-batch LLM-generated DPO job."""

    results: tuple[Any, ...]
    row_count: int
    families: tuple[str, ...]
    generation_run: str
    manifest_path: Path


def default_dataset_path(*, output_dir: str | Path, family: str) -> Path:
    """Return the default DPO JSONL path for a seed family."""
    return Path(output_dir) / f"{family}.jsonl"


def default_manifest_path(*, manifest_dir: str | Path, family: str, generation_run: str) -> Path:
    """Return the default local DPO manifest path for a seed run."""
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(manifest_dir) / f"{family}.{generation_run}.manifest.json"


def default_batch_output_dir(output_dir: str | Path) -> Path:
    """Return the sibling internal batch directory for a public dataset directory."""
    public_dir = Path(output_dir)
    return public_dir.parent / "batches"


def materialize_seed_dataset(
    *,
    family: str,
    count: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    generation_run: str,
    start_index: int = 1,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
) -> DPOSeedRunResult:
    """Build DPO seed rows and write JSONL plus a local manifest."""
    rows = build_seed_rows(
        family=family,
        count=count,
        start_index=start_index,
        holdout_registry=holdout_registry,
    )

    dataset_path = Path(output_dir) / dataset_filename if dataset_filename else default_dataset_path(
        output_dir=output_dir,
        family=family,
    )
    row_count = write_jsonl(rows, dataset_path)

    manifest_path = Path(manifest_dir) / manifest_filename if manifest_filename else default_manifest_path(
        manifest_dir=manifest_dir,
        family=family,
        generation_run=generation_run,
    )
    write_manifest(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        rows=rows,
        generation_run=generation_run,
        metadata={
            "family": family,
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )

    return DPOSeedRunResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        row_count=row_count,
        family=family,
        generation_run=generation_run,
    )


def materialize_seed_run(
    *,
    families: list[str] | tuple[str, ...] | None,
    count_per_family: int,
    output_dir: str | Path,
    manifest_dir: str | Path,
    generation_run: str,
    start_index: int = 1,
    run_manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
) -> DPOSeedFamilyRunResult:
    """Materialize one DPO seed dataset per requested family."""
    resolved_families = resolve_seed_families(families)
    if not isinstance(count_per_family, int) or count_per_family < 1:
        raise ValueError("count_per_family must be a positive integer")

    results = tuple(
        materialize_seed_dataset(
            family=family,
            count=count_per_family,
            output_dir=output_dir,
            manifest_dir=manifest_dir,
            generation_run=generation_run,
            start_index=start_index,
            metadata={
                "seed_run_family_count": len(resolved_families),
                **dict(metadata or {}),
            },
            holdout_registry=holdout_registry,
        )
        for family in resolved_families
    )
    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    write_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=generation_run,
        datasets=[
            {
                "family": result.family,
                "dataset_path": result.dataset_path,
                "manifest_path": result.manifest_path,
                "row_count": result.row_count,
            }
            for result in results
        ],
        metadata={
            "family_count": len(resolved_families),
            "count_per_family": count_per_family,
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )

    return DPOSeedFamilyRunResult(
        results=results,
        row_count=sum(result.row_count for result in results),
        families=resolved_families,
        generation_run=generation_run,
        manifest_path=run_manifest_path,
    )


def resolve_seed_families(families: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve requested DPO seed families, where None or ['all'] means all families."""
    if families is None or tuple(families) == ("all",):
        return tuple(sorted(DPO_SEED_FAMILIES))
    if "all" in families:
        raise ValueError("'all' cannot be combined with explicit DPO seed families")

    resolved: list[str] = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("DPO seed family must be a non-empty string")
        normalized = family.strip().lower()
        if normalized not in DPO_SEED_FAMILIES:
            supported = ", ".join(sorted(DPO_SEED_FAMILIES))
            raise ValueError(f"Unsupported DPO seed family '{family}'. Supported families: {supported}")
        if normalized in seen:
            raise ValueError(f"Duplicate DPO seed family: {normalized}")
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        raise ValueError("at least one DPO seed family is required")
    return tuple(resolved)


def generate_llm_run(
    *,
    families: list[str] | tuple[str, ...] | None,
    count_per_family: int | None = None,
    target_pairs: int | None = None,
    batch_size: int = 1,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    start_index: int = 1,
    teacher_provider: str = "openrouter",
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
    concurrency: int = DEFAULT_OPENROUTER_SMOKE_CONCURRENCY,
    max_backfill_rounds: int = 2,
    run_manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
    backend: StructuredTeacherBackend | None = None,
) -> DPOLLMRunResult:
    """Build specs and generate DPO datasets across families and batches."""
    resolved_families = resolve_spec_families(families)
    count_plan = build_count_plan(
        keys=resolved_families,
        count_per_key=count_per_family,
        target_count=target_pairs,
        key_name="family",
        count_per_key_name="count_per_family",
        target_count_name="target_pairs",
        target_mode="target_pairs",
    )
    _validate_openrouter_batch_size(batch_size)
    _validate_positive_int(start_index, "start_index")
    _validate_openrouter_concurrency(concurrency)
    _validate_non_negative_int(max_backfill_rounds, "max_backfill_rounds")
    _validate_positive_int(adaptive_initial_batch_size, "adaptive_initial_batch_size")
    _validate_positive_int(adaptive_batch_increase_successes, "adaptive_batch_increase_successes")
    adaptive_maximum_in_flight = concurrency

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

    def run_job(job: dict[str, Any]) -> Any:
        return generate_llm_batch(
            specs=job["specs"],
            output_path=job["dataset_path"],
            manifest_path=job["manifest_path"],
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
            generation_run=generation_run,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_timeout=request_timeout,
            max_request_retries=max_request_retries,
            max_retryable_request_attempts=max_retryable_request_attempts,
            retry_max_elapsed_seconds=retry_max_elapsed_seconds,
            adaptive_maximum_in_flight=adaptive_maximum_in_flight,
            adaptive_initial_in_flight=adaptive_initial_in_flight,
            metadata={
                "family": job["family"],
                "batch_number": job["batch_number"],
                "batch_start_index": job["batch_start_index"],
                "batch_size": len(job["specs"]),
                **job.get("adaptive_batch_size", {}),
                **dict(metadata or {}),
            },
            holdout_registry=holdout_registry,
            backend=active_backend,
        )

    jobs: list[dict[str, Any]] = []
    batch_controller = AdaptiveBatchSizeController(
        maximum=batch_size,
        minimum=1,
        initial=adaptive_initial_batch_size,
        increase_successes=adaptive_batch_increase_successes,
    )
    for family in resolved_families:
        family_target_pairs = count_plan.counts_by_key[family]
        specs = build_specs(family=family, count=family_target_pairs, start_index=start_index)
        print(
            "[generate] Starting DPO family: "
            f"{family} (target_rows={len(specs)}, batch_size={batch_size}, "
            f"min_batch_size=1, parallel_requests={concurrency}, model={teacher_model})",
            flush=True,
        )
        pending_ranges: deque[tuple[int, int]] = deque([(0, len(specs))])
        active: dict[Any, dict[str, Any]] = {}
        next_batch_number = 1
        family_rows_done = 0

        def make_job(batch_specs: list[dict[str, Any]], batch_number: int, offset: int) -> dict[str, Any]:
            batch_start_index = start_index + offset
            return {
                "family": family,
                "batch_number": batch_number,
                "batch_start_index": batch_start_index,
                "specs": batch_specs,
                "dataset_path": default_batch_output_dir(output_dir) / f"{family}.batch{batch_number:06d}.jsonl",
                "manifest_path": Path(manifest_dir) / f"{family}.batch{batch_number:06d}.{generation_run}.manifest.json",
            }

        def active_job_limit() -> int:
            return min(concurrency, max(1, adaptive_initial_in_flight, batch_controller.current))

        def submit_available(executor: ThreadPoolExecutor) -> None:
            nonlocal next_batch_number
            while pending_ranges and len(active) < active_job_limit():
                offset, remaining = pending_ranges.popleft()
                size = min(batch_controller.current, remaining)
                if remaining > size:
                    pending_ranges.appendleft((offset + size, remaining - size))
                job = make_job(specs[offset : offset + size], next_batch_number, offset)
                next_batch_number += 1
                active[executor.submit(run_job, job)] = job

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
                            workflow="DPO",
                            group_key="family",
                            group_value=family,
                            batch_number=job["batch_number"],
                            batch_start=job["batch_start_index"],
                            batch_size=len(job["specs"]),
                            adaptive_batch_size=batch_controller.snapshot(),
                            error=exc,
                        )
                        if len(job["specs"]) <= batch_controller.minimum:
                            raise
                        offset = job["batch_start_index"] - start_index
                        pending_ranges.appendleft((offset, len(job["specs"])))
                        submit_available(executor)
                        continue
                    batch_controller.record_success()
                    job["result"] = result
                    job["adaptive_batch_size"] = batch_controller.snapshot()
                    jobs.append(job)
                    family_rows_done += result.row_count
                    print_batch_progress(
                        workflow="DPO",
                        group_key="family",
                        group_value=family,
                        batch_number=job["batch_number"],
                        batch_start=job["batch_start_index"],
                        batch_size=len(job["specs"]),
                        rows_done=family_rows_done,
                        rows_total=len(specs),
                        manifest_path=result.manifest_path,
                        adaptive_batch_size=job["adaptive_batch_size"],
                    )
                    submit_available(executor)
        print(
            "[generate] Completed DPO family: "
            f"{family} rows={family_rows_done}, target_rows={len(specs)}, "
            f"batch_size={batch_size}, min_batch_size=1, parallel_requests={concurrency}, "
            f"adaptive_batch_size_observed_minimum={batch_controller.observed_minimum}, "
            f"adaptive_batch_size_observed_peak={batch_controller.observed_peak}, "
            f"adaptive_batch_size_increases={batch_controller.increases}, "
            f"adaptive_batch_size_decreases={batch_controller.decreases}, "
            f"adaptive_batch_size_failures={batch_controller.failures}",
            flush=True,
        )

    jobs.sort(key=lambda item: (item["family"], item["batch_start_index"], item["batch_number"]))
    results = [job["result"] for job in jobs]
    datasets = _write_public_family_files(jobs=jobs, output_dir=output_dir)
    planned_pairs = count_plan.planned_count
    accepted_pairs = sum(dataset["row_count"] for dataset in datasets)
    rejected_pairs = max(planned_pairs - accepted_pairs, 0)

    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    _write_llm_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=generation_run,
        families=resolved_families,
        datasets=datasets,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        metadata={
            "generation_mode": "live_llm_run",
            "planning_mode": count_plan.planning_mode,
            "target_pairs": target_pairs,
            "planned_pairs": planned_pairs,
            "accepted_pairs": accepted_pairs,
            "rejected_pairs": rejected_pairs,
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": 0,
            **accepted_target_metadata(
                unit="pairs",
                target_count=planned_pairs,
                accepted_count=accepted_pairs,
                attempted_count=planned_pairs,
                max_backfill_rounds=max_backfill_rounds,
                backfill_rounds=0,
            ),
            "pairs_per_family": dict(count_plan.counts_by_key),
            "count_per_family": count_per_family,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "adaptive_maximum_in_flight": adaptive_maximum_in_flight,
            "adaptive_initial_in_flight": adaptive_initial_in_flight,
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            **batch_controller.snapshot(),
            "llm_telemetry": aggregate_llm_telemetry_from_manifests(result.manifest_path for result in results),
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )
    raise_for_underfilled_manifest(run_manifest_path, artifact_name="DPO")

    return DPOLLMRunResult(
        results=tuple(results),
        row_count=accepted_pairs,
        families=resolved_families,
        generation_run=generation_run,
        manifest_path=run_manifest_path,
    )


def resolve_spec_families(families: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve requested DPO spec families, where None or ['all'] means all families."""
    if families is None or tuple(families) == ("all",):
        return tuple(sorted(DPO_SPEC_FAMILIES))
    if "all" in families:
        raise ValueError("'all' cannot be combined with explicit DPO spec families")

    resolved: list[str] = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("DPO spec family must be a non-empty string")
        normalized = family.strip().lower()
        if normalized not in DPO_SPEC_FAMILIES:
            supported = ", ".join(sorted(DPO_SPEC_FAMILIES))
            raise ValueError(f"Unsupported DPO spec family '{family}'. Supported families: {supported}")
        if normalized in seen:
            raise ValueError(f"Duplicate DPO spec family: {normalized}")
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        raise ValueError("at least one DPO spec family is required")
    return tuple(resolved)


def _write_llm_run_manifest(
    *,
    manifest_path: str | Path,
    generation_run: str,
    families: tuple[str, ...],
    datasets: list[dict[str, Any]],
    teacher_model: str,
    teacher_provider: str,
    metadata: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "dpo",
        "generation_run": generation_run,
        "generation_mode": "live_llm_run",
        "teacher_model": teacher_model,
        "teacher_provider": teacher_provider,
        "families": list(families),
        "datasets": [
            {
                "family": item["family"],
                "dataset_path": str(Path(item["dataset_path"])),
                "row_count": item["row_count"],
                "batch_count": item["batch_count"],
                "batch_manifests": [str(Path(path)) for path in item["batch_manifests"]],
            }
            for item in datasets
        ],
        "total_rows": sum(item["row_count"] for item in datasets),
        "metadata": metadata,
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_public_family_files(*, jobs: list[dict[str, Any]], output_dir: str | Path) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for family in sorted({job["family"] for job in jobs}):
        family_jobs = [job for job in jobs if job["family"] == family]
        rows: list[dict[str, Any]] = []
        batch_manifests: list[Path] = []
        for job in family_jobs:
            result = job["result"]
            rows.extend(read_jsonl(result.dataset_path))
            batch_manifests.append(result.manifest_path)

        dataset_path = Path(output_dir) / f"{family}.jsonl"
        row_count = write_jsonl(rows, dataset_path)
        datasets.append(
            {
                "family": family,
                "dataset_path": dataset_path,
                "row_count": row_count,
                "batch_count": len(family_jobs),
                "batch_manifests": batch_manifests,
            }
        )
    return datasets


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_openrouter_batch_size(value: int) -> None:
    _validate_positive_int(value, "batch_size")
    if value > MAX_OPENROUTER_BATCH_SIZE:
        raise ValueError(f"batch_size must be at most {MAX_OPENROUTER_BATCH_SIZE}")


def _validate_openrouter_concurrency(value: int) -> None:
    _validate_positive_int(value, "concurrency")
    if value > MAX_OPENROUTER_CONCURRENCY:
        raise ValueError(f"concurrency must be at most {MAX_OPENROUTER_CONCURRENCY}")
