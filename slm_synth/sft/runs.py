"""SFT dataset materialization helpers."""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.adaptive_batch import AdaptiveBatchSizeController
from slm_synth.sft.generation import StructuredTeacherBackend, build_openrouter_backend, generate_llm_batch
from slm_synth.sft.io import write_jsonl
from slm_synth.sft.manifest import write_manifest, write_run_manifest
from slm_synth.sft.seeds import SFT_SEED_FAMILIES, build_seed_rows
from slm_synth.sft.spec_builders import SFT_SPEC_FAMILIES, build_specs
from slm_synth.taxonomy.holdouts import HoldoutRegistry
from slm_synth.telemetry import aggregate_llm_telemetry_from_manifests
from slm_synth.run_summary import print_batch_failure, print_batch_progress


@dataclass(frozen=True)
class SFTSeedRunResult:
    """Result of materializing one SFT seed dataset."""

    dataset_path: Path
    manifest_path: Path
    row_count: int
    family: str
    generation_run: str


@dataclass(frozen=True)
class SFTSeedFamilyRunResult:
    """Result of materializing multiple SFT seed families."""

    results: tuple[SFTSeedRunResult, ...]
    row_count: int
    families: tuple[str, ...]
    generation_run: str
    manifest_path: Path


@dataclass(frozen=True)
class SFTLLMRunResult:
    """Result of running one multi-batch LLM-generated SFT job."""

    results: tuple[Any, ...]
    row_count: int
    families: tuple[str, ...]
    generation_run: str
    manifest_path: Path


def default_dataset_path(*, output_dir: str | Path, family: str) -> Path:
    """Return the default SFT JSONL path for a seed family."""
    return Path(output_dir) / f"{family}.jsonl"


def default_manifest_path(*, manifest_dir: str | Path, family: str, generation_run: str) -> Path:
    """Return the default local SFT manifest path for a seed run."""
    if not isinstance(generation_run, str) or not generation_run.strip():
        raise ValueError("generation_run must be a non-empty string")
    return Path(manifest_dir) / f"{family}.{generation_run}.manifest.json"


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
) -> SFTSeedRunResult:
    """Build SFT seed rows and write JSONL plus a local manifest."""
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

    return SFTSeedRunResult(
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
) -> SFTSeedFamilyRunResult:
    """Materialize one SFT seed dataset per requested family."""
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

    return SFTSeedFamilyRunResult(
        results=results,
        row_count=sum(result.row_count for result in results),
        families=resolved_families,
        generation_run=generation_run,
        manifest_path=run_manifest_path,
    )


def resolve_seed_families(families: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve requested SFT seed families, where None or ['all'] means all families."""
    if families is None or tuple(families) == ("all",):
        return tuple(sorted(SFT_SEED_FAMILIES))
    if "all" in families:
        raise ValueError("'all' cannot be combined with explicit SFT seed families")

    resolved: list[str] = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("SFT seed family must be a non-empty string")
        normalized = family.strip().lower()
        if normalized not in SFT_SEED_FAMILIES:
            supported = ", ".join(sorted(SFT_SEED_FAMILIES))
            raise ValueError(f"Unsupported SFT seed family '{family}'. Supported families: {supported}")
        if normalized in seen:
            raise ValueError(f"Duplicate SFT seed family: {normalized}")
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        raise ValueError("at least one SFT seed family is required")
    return tuple(resolved)


def generate_llm_run(
    *,
    families: list[str] | tuple[str, ...] | None,
    count_per_family: int,
    batch_size: int,
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
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 8,
    concurrency: int = 1,
    run_manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
    backend: StructuredTeacherBackend | None = None,
) -> SFTLLMRunResult:
    """Build specs and generate SFT datasets across families and batches."""
    resolved_families = resolve_spec_families(families)
    _validate_positive_int(count_per_family, "count_per_family")
    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(start_index, "start_index")
    _validate_positive_int(concurrency, "concurrency")
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
    batch_controller = AdaptiveBatchSizeController(maximum=batch_size, minimum=1)
    for family in resolved_families:
        specs = build_specs(family=family, count=count_per_family, start_index=start_index)
        print(
            "[generate] Starting SFT family: "
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
                "dataset_path": Path(output_dir) / f"{family}.batch{batch_number:06d}.jsonl",
                "manifest_path": Path(manifest_dir) / f"{family}.batch{batch_number:06d}.{generation_run}.manifest.json",
            }

        def submit_available(executor: ThreadPoolExecutor) -> None:
            nonlocal next_batch_number
            while pending_ranges and len(active) < concurrency:
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
                            workflow="SFT",
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
                        workflow="SFT",
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
            "[generate] Completed SFT family: "
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

    datasets: list[dict[str, Any]] = []
    for job, result in zip(jobs, results):
        datasets.append(
            {
                "family": job["family"],
                "batch_number": job["batch_number"],
                "batch_start_index": job["batch_start_index"],
                "dataset_path": result.dataset_path,
                "manifest_path": result.manifest_path,
                "row_count": result.row_count,
            }
        )

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
            "count_per_family": count_per_family,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "adaptive_maximum_in_flight": adaptive_maximum_in_flight,
            "adaptive_initial_in_flight": adaptive_initial_in_flight,
            **batch_controller.snapshot(),
            "llm_telemetry": aggregate_llm_telemetry_from_manifests(result.manifest_path for result in results),
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )

    return SFTLLMRunResult(
        results=tuple(results),
        row_count=sum(result.row_count for result in results),
        families=resolved_families,
        generation_run=generation_run,
        manifest_path=run_manifest_path,
    )


def resolve_spec_families(families: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve requested SFT spec families, where None or ['all'] means all families."""
    if families is None or tuple(families) == ("all",):
        return tuple(sorted(SFT_SPEC_FAMILIES))
    if "all" in families:
        raise ValueError("'all' cannot be combined with explicit SFT spec families")

    resolved: list[str] = []
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, str) or not family.strip():
            raise ValueError("SFT spec family must be a non-empty string")
        normalized = family.strip().lower()
        if normalized not in SFT_SPEC_FAMILIES:
            supported = ", ".join(sorted(SFT_SPEC_FAMILIES))
            raise ValueError(f"Unsupported SFT spec family '{family}'. Supported families: {supported}")
        if normalized in seen:
            raise ValueError(f"Duplicate SFT spec family: {normalized}")
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        raise ValueError("at least one SFT spec family is required")
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
        "dataset_type": "sft",
        "generation_run": generation_run,
        "generation_mode": "live_llm_run",
        "teacher_model": teacher_model,
        "teacher_provider": teacher_provider,
        "families": list(families),
        "datasets": [
            {
                "family": item["family"],
                "batch_number": item["batch_number"],
                "batch_start_index": item["batch_start_index"],
                "dataset_path": str(Path(item["dataset_path"])),
                "manifest_path": str(Path(item["manifest_path"])),
                "row_count": item["row_count"],
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


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
