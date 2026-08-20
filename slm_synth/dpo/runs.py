"""DPO LLM-backed run orchestration helpers."""

from __future__ import annotations

import json
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_synth.adaptive_batch import (
    AdaptiveBatchSizeController,
    aggregate_adaptive_batch_size_controllers,
)
from slm_synth.alignment_tokens import estimate_dpo_tokens
from slm_synth.planning import CountPlan
from slm_synth.dpo.generation import (
    DPOBatchAcceptanceError,
    StructuredTeacherBackend,
    build_openrouter_backend,
    generate_llm_batch,
)
from slm_synth.dpo.acceptance import partition_unique_dpo_rows
from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.manifest import write_manifest, write_run_manifest
from slm_synth.dpo.spec_builders import DPO_PREFERENCE_DIMENSIONS, build_specs, validate_spec_range
from slm_synth.taxonomy.holdouts import HoldoutRegistry
from slm_synth.telemetry import aggregate_llm_telemetry, aggregate_llm_telemetry_from_manifests
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
class DPOLLMRunResult:
    """Result of running one multi-batch LLM-generated DPO job."""

    results: tuple[Any, ...]
    row_count: int
    preference_dimensions: tuple[str, ...]
    generation_run: str
    manifest_path: Path


def default_batch_output_dir(output_dir: str | Path) -> Path:
    """Return the sibling internal batch directory for a public dataset directory."""
    public_dir = Path(output_dir)
    return public_dir.parent / "batches"


def generate_llm_run(
    *,
    preference_dimensions: list[str] | tuple[str, ...] | None,
    candidate_counts_by_dimension: dict[str, int],
    batch_size: int = 1,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    adjudicator_model: str | None = None,
    adjudicator_max_tokens: int | None = None,
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
    run_manifest_filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
    backend: StructuredTeacherBackend | None = None,
    adjudicator_backend: StructuredTeacherBackend | None = None,
) -> DPOLLMRunResult:
    """Build specs and generate DPO datasets across preference dimensions."""
    resolved_families = resolve_preference_dimensions(preference_dimensions)
    normalized_counts = {
        str(dimension).strip().lower(): count
        for dimension, count in candidate_counts_by_dimension.items()
    }
    if set(normalized_counts) != set(resolved_families):
        raise ValueError(
            "candidate_counts_by_dimension must contain exactly the requested preference dimensions"
        )
    for dimension, count in normalized_counts.items():
        _validate_positive_int(count, f"candidate count for {dimension}")
    count_plan = CountPlan(
        planning_mode="candidate_counts_by_dimension",
        counts_by_key={dimension: normalized_counts[dimension] for dimension in resolved_families},
    )
    _validate_openrouter_batch_size(batch_size)
    _validate_positive_int(start_index, "start_index")
    _validate_openrouter_concurrency(concurrency)
    _validate_positive_int(adaptive_initial_batch_size, "adaptive_initial_batch_size")
    _validate_positive_int(adaptive_batch_increase_successes, "adaptive_batch_increase_successes")
    adaptive_maximum_in_flight = concurrency

    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    for family in resolved_families:
        validate_spec_range(
            family=family,
            count=count_plan.counts_by_key[family],
            start_index=start_index,
        )

    # Preflight every DPO source and its separation from SFT before credentials
    # are read or a provider backend can be constructed.
    from slm_synth.alignment_preflight import preflight_dpo_inventory

    preflight_dpo_inventory()

    active_backend = backend
    active_adjudicator_backend = adjudicator_backend

    def get_backend() -> StructuredTeacherBackend:
        nonlocal active_backend
        if active_backend is None:
            active_backend = build_openrouter_backend(
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
        return active_backend

    def get_adjudicator_backend() -> StructuredTeacherBackend:
        nonlocal active_adjudicator_backend
        if active_adjudicator_backend is None:
            active_adjudicator_backend = build_openrouter_backend(
                model=adjudicator_model if adjudicator_model is not None else teacher_model,
                max_tokens=adjudicator_max_tokens if adjudicator_max_tokens is not None else max_tokens,
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
        return active_adjudicator_backend

    def run_job(job: dict[str, Any]) -> Any:
        return generate_llm_batch(
            specs=job["specs"],
            output_path=job["dataset_path"],
            manifest_path=job["manifest_path"],
            teacher_model=teacher_model,
            teacher_provider=teacher_provider,
            generation_run=generation_run,
            max_tokens=max_tokens,
            adjudicator_model=adjudicator_model,
            adjudicator_max_tokens=adjudicator_max_tokens,
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
            backend=get_backend(),
            adjudicator_backend=get_adjudicator_backend(),
        )

    results: list[Any] = []
    rejected_llm_telemetry: list[dict[str, Any]] = []
    batch_controllers: list[AdaptiveBatchSizeController] = []
    rejection_diagnostics: list[dict[str, Any]] = []
    initial_state = _empty_run_state(families=resolved_families, start_index=start_index)
    next_batch_numbers = dict(initial_state["next_batch_numbers"])
    next_source_indexes = dict(initial_state["next_source_indexes"])
    accepted_rows_by_family = dict(initial_state["accepted_rows_by_family"])
    datasets = list(initial_state["datasets"])
    output_acceptance = dict(initial_state["acceptance"])

    def run_generation(
        request_counts: dict[str, int],
    ) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
        round_jobs: list[dict[str, Any]] = []
        rejected_pairs_per_family = {family: 0 for family in resolved_families}
        rejection_reason_counts: Counter[str] = Counter()
        for family in resolved_families:
            requested_pairs = request_counts.get(family, 0)
            if requested_pairs == 0:
                continue
            round_start_index = next_source_indexes[family]
            validate_spec_range(family=family, count=requested_pairs, start_index=round_start_index)
            specs = build_specs(family=family, count=requested_pairs, start_index=round_start_index)
            next_source_indexes[family] += requested_pairs
            batch_controller = AdaptiveBatchSizeController(
                maximum=batch_size,
                minimum=1,
                initial=adaptive_initial_batch_size,
                increase_successes=adaptive_batch_increase_successes,
            )
            batch_controllers.append(batch_controller)
            print(
                "[generate] Starting DPO family: "
                f"{family} (candidate_pairs={len(specs)}, batch_size={batch_size}, "
                f"min_batch_size=1, parallel_requests={concurrency}, model={teacher_model})",
                flush=True,
            )
            pending_ranges: deque[tuple[int, int]] = deque([(0, len(specs))])
            active: dict[Any, dict[str, Any]] = {}
            family_pairs_done = 0

            def make_job(batch_specs: list[dict[str, Any]], batch_number: int, offset: int) -> dict[str, Any]:
                batch_start_index = round_start_index + offset
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
                while pending_ranges and len(active) < active_job_limit():
                    offset, remaining = pending_ranges.popleft()
                    size = min(batch_controller.current, remaining)
                    if remaining > size:
                        pending_ranges.appendleft((offset + size, remaining - size))
                    batch_number = next_batch_numbers[family]
                    next_batch_numbers[family] += 1
                    job = make_job(specs[offset : offset + size], batch_number, offset)
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
                            if isinstance(exc, DPOBatchAcceptanceError):
                                rejected_llm_telemetry.append(exc.telemetry)
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
                                if isinstance(exc, DPOBatchAcceptanceError):
                                    rejected_pairs_per_family[family] += 1
                                    rejection_reason_counts[exc.failure_type] += 1
                                    rejection_diagnostics.append(
                                        {
                                            "id": job["specs"][0]["id"],
                                            "preference_dimension": family,
                                            "failure_type": exc.failure_type,
                                            "reason": str(exc),
                                        }
                                    )
                                    submit_available(executor)
                                    continue
                                raise
                            offset = job["batch_start_index"] - round_start_index
                            pending_ranges.appendleft((offset, len(job["specs"])))
                            submit_available(executor)
                            continue
                        batch_controller.record_success()
                        job["result"] = result
                        job["adaptive_batch_size"] = batch_controller.snapshot()
                        round_jobs.append(job)
                        results.append(result)
                        family_pairs_done += result.row_count
                        print_batch_progress(
                            workflow="DPO",
                            group_key="family",
                            group_value=family,
                            batch_number=job["batch_number"],
                            batch_start=job["batch_start_index"],
                            batch_size=len(job["specs"]),
                            rows_done=family_pairs_done,
                            rows_total=len(specs),
                            manifest_path=result.manifest_path,
                            adaptive_batch_size=job["adaptive_batch_size"],
                        )
                        submit_available(executor)
            print(
                "[generate] Completed DPO family: "
                f"{family} rendered_pairs={family_pairs_done}, candidate_pairs={len(specs)}, "
                f"batch_size={batch_size}, min_batch_size=1, parallel_requests={concurrency}, "
                f"adaptive_batch_size_observed_minimum={batch_controller.observed_minimum}, "
                f"adaptive_batch_size_observed_peak={batch_controller.observed_peak}, "
                f"adaptive_batch_size_increases={batch_controller.increases}, "
                f"adaptive_batch_size_decreases={batch_controller.decreases}, "
                f"adaptive_batch_size_failures={batch_controller.failures}",
                flush=True,
            )
        round_jobs.sort(key=lambda item: (item["family"], item["batch_start_index"], item["batch_number"]))
        return round_jobs, rejected_pairs_per_family, dict(rejection_reason_counts)

    initial_counts = dict(count_plan.counts_by_key)
    if any(initial_counts.values()):
        initial_jobs, initial_rejected, initial_rejection_reasons = run_generation(initial_counts)
        datasets, output_acceptance, accepted_rows_by_family = _write_public_family_files(
            jobs=initial_jobs,
            output_dir=output_dir,
            families=resolved_families,
            accepted_rows_by_family=accepted_rows_by_family,
            prior_datasets=datasets,
            prior_acceptance=output_acceptance,
            new_rejected_pairs_per_family=initial_rejected,
            new_rejection_reason_counts=initial_rejection_reasons,
        )

    candidate_pairs = count_plan.planned_count
    accepted_pairs = sum(dataset["row_count"] for dataset in datasets)
    attempted_pairs = output_acceptance["attempted_pairs"]
    duplicate_pairs = output_acceptance["duplicate_pairs"]
    rejected_pairs = max(attempted_pairs - accepted_pairs - duplicate_pairs, 0)
    estimated_tokens_per_dimension = {
        family: sum(estimate_dpo_tokens(row) for row in accepted_rows_by_family[family])
        for family in resolved_families
    }
    empty_preference_dimensions = sorted(
        family for family in resolved_families if not accepted_rows_by_family[family]
    )
    _write_llm_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=generation_run,
        preference_dimensions=resolved_families,
        datasets=datasets,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        metadata={
            "generation_mode": "live_llm_run",
            "planning_mode": count_plan.planning_mode,
            "adjudicator_model": adjudicator_model if adjudicator_model is not None else teacher_model,
            "adjudicator_max_tokens": adjudicator_max_tokens if adjudicator_max_tokens is not None else max_tokens,
            "candidate_pairs": candidate_pairs,
            "accepted_pairs": accepted_pairs,
            "estimated_tokens": sum(estimated_tokens_per_dimension.values()),
            "rejected_pairs": rejected_pairs,
            "rejection_reason_counts": output_acceptance["rejection_reason_counts"],
            "rejection_diagnostics": rejection_diagnostics,
            "attempted_pairs": attempted_pairs,
            "duplicate_pairs": duplicate_pairs,
            "duplicate_reason_counts": output_acceptance["duplicate_reason_counts"],
            "attempted_pairs_per_dimension": output_acceptance["attempted_pairs_per_family"],
            "accepted_pairs_per_dimension": output_acceptance["accepted_pairs_per_family"],
            "estimated_tokens_per_dimension": estimated_tokens_per_dimension,
            "rejected_pairs_per_dimension": output_acceptance["rejected_pairs_per_family"],
            "duplicate_pairs_per_dimension": output_acceptance["duplicate_pairs_per_family"],
            "next_start_index_per_family": next_source_indexes,
            "candidate_pairs_per_dimension": dict(count_plan.counts_by_key),
            "generation_status": "complete",
            "publish_ready": not empty_preference_dimensions,
            "empty_preference_dimensions": empty_preference_dimensions,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "adaptive_maximum_in_flight": adaptive_maximum_in_flight,
            "adaptive_initial_in_flight": adaptive_initial_in_flight,
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            **aggregate_adaptive_batch_size_controllers(batch_controllers),
            "llm_telemetry": aggregate_llm_telemetry(
                [
                    telemetry
                    for telemetry in (
                        aggregate_llm_telemetry_from_manifests(
                            result.manifest_path for result in results
                        ),
                        *rejected_llm_telemetry,
                    )
                    if telemetry
                ]
            ),
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )
    return DPOLLMRunResult(
        results=tuple(results),
        row_count=accepted_pairs,
        preference_dimensions=resolved_families,
        generation_run=generation_run,
        manifest_path=run_manifest_path,
    )


def resolve_preference_dimensions(
    preference_dimensions: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve requested DPO preference dimensions; ``all`` selects the catalog."""
    if preference_dimensions is None or tuple(preference_dimensions) == ("all",):
        return tuple(sorted(DPO_PREFERENCE_DIMENSIONS))
    if "all" in preference_dimensions:
        raise ValueError("'all' cannot be combined with explicit DPO preference dimensions")

    resolved: list[str] = []
    seen: set[str] = set()
    for dimension in preference_dimensions:
        if not isinstance(dimension, str) or not dimension.strip():
            raise ValueError("DPO preference dimension must be a non-empty string")
        normalized = dimension.strip().lower()
        if normalized not in DPO_PREFERENCE_DIMENSIONS:
            supported = ", ".join(sorted(DPO_PREFERENCE_DIMENSIONS))
            raise ValueError(
                f"Unsupported DPO preference dimension '{dimension}'. "
                f"Supported dimensions: {supported}"
            )
        if normalized in seen:
            raise ValueError(f"Duplicate DPO preference dimension: {normalized}")
        seen.add(normalized)
        resolved.append(normalized)
    if not resolved:
        raise ValueError("at least one DPO preference dimension is required")
    return tuple(resolved)


def _write_llm_run_manifest(
    *,
    manifest_path: str | Path,
    generation_run: str,
    preference_dimensions: tuple[str, ...],
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
        "preference_dimensions": list(preference_dimensions),
        "datasets": [
            {
                "preference_dimension": item["family"],
                "dataset_path": str(Path(item["dataset_path"])),
                "row_count": item["row_count"],
                "batch_count": item["batch_count"],
                "batch_manifests": [str(Path(path)) for path in item["batch_manifests"]],
            }
            for item in datasets
        ],
        "total_rows": sum(item["row_count"] for item in datasets),
        "total_pairs": sum(item["row_count"] for item in datasets),
        "metadata": metadata,
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_public_family_files(
    *,
    jobs: list[dict[str, Any]],
    output_dir: str | Path,
    families: tuple[str, ...],
    accepted_rows_by_family: dict[str, list[dict[str, Any]]],
    prior_datasets: list[dict[str, Any]],
    prior_acceptance: dict[str, Any],
    new_rejected_pairs_per_family: dict[str, int],
    new_rejection_reason_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    candidate_rows = [
        row for family in families for row in accepted_rows_by_family.get(family, [])
    ]
    new_candidate_pairs_per_family = {family: 0 for family in families}
    for family in families:
        for job in [item for item in jobs if item["family"] == family]:
            rows = read_jsonl(job["result"].dataset_path)
            candidate_rows.extend(rows)
            new_candidate_pairs_per_family[family] += len(rows)
    new_attempted_pairs_per_family = {
        family: new_candidate_pairs_per_family[family] + new_rejected_pairs_per_family[family]
        for family in families
    }

    accepted_rows, round_acceptance = partition_unique_dpo_rows(candidate_rows)
    current_rows_by_family = {
        family: [row for row in accepted_rows if row["metadata"]["preference_dimension"] == family]
        for family in families
    }
    prior_datasets_by_family = {dataset["family"]: dataset for dataset in prior_datasets}
    datasets: list[dict[str, Any]] = []
    for family in families:
        family_jobs = [job for job in jobs if job["family"] == family]
        prior_dataset = prior_datasets_by_family.get(family, {})
        batch_manifests = [Path(path) for path in prior_dataset.get("batch_manifests", [])]
        batch_manifests.extend(job["result"].manifest_path for job in family_jobs)
        rows = current_rows_by_family[family]
        dataset_path = Path(output_dir) / f"{family}.jsonl"
        row_count = write_jsonl(rows, dataset_path)
        datasets.append(
            {
                "family": family,
                "dataset_path": dataset_path,
                "row_count": row_count,
                "batch_count": len(batch_manifests),
                "batch_manifests": batch_manifests,
            }
        )

    prior_attempted = prior_acceptance.get("attempted_pairs_per_family", {})
    prior_duplicates = prior_acceptance.get("duplicate_pairs_per_family", {})
    prior_rejected = prior_acceptance.get("rejected_pairs_per_family", {})
    attempted_pairs_per_family = {
        family: prior_attempted.get(family, 0) + new_attempted_pairs_per_family[family]
        for family in families
    }
    accepted_pairs_per_family = {family: len(current_rows_by_family[family]) for family in families}
    new_duplicate_pairs_per_family = {
        family: (
            len(accepted_rows_by_family.get(family, []))
            + new_candidate_pairs_per_family[family]
            - accepted_pairs_per_family[family]
        )
        for family in families
    }
    duplicate_pairs_per_family = {
        family: prior_duplicates.get(family, 0) + new_duplicate_pairs_per_family[family]
        for family in families
    }
    rejected_pairs_per_family = {
        family: prior_rejected.get(family, 0) + new_rejected_pairs_per_family[family]
        for family in families
    }
    duplicate_reasons = Counter(prior_acceptance.get("duplicate_reason_counts", {}))
    duplicate_reasons.update(round_acceptance["duplicate_reason_counts"])
    rejection_reasons = Counter(prior_acceptance.get("rejection_reason_counts", {}))
    rejection_reasons.update(new_rejection_reason_counts)
    acceptance = {
        "attempted_pairs": sum(attempted_pairs_per_family.values()),
        "accepted_pairs": sum(accepted_pairs_per_family.values()),
        "duplicate_pairs": sum(duplicate_pairs_per_family.values()),
        "duplicate_reason_counts": {
            reason: duplicate_reasons[reason] for reason in sorted(duplicate_reasons)
        },
        "rejection_reason_counts": {
            reason: rejection_reasons[reason] for reason in sorted(rejection_reasons)
        },
        "attempted_pairs_per_family": attempted_pairs_per_family,
        "accepted_pairs_per_family": accepted_pairs_per_family,
        "rejected_pairs_per_family": rejected_pairs_per_family,
        "duplicate_pairs_per_family": duplicate_pairs_per_family,
    }
    return datasets, acceptance, current_rows_by_family


def _empty_run_state(*, families: tuple[str, ...], start_index: int) -> dict[str, Any]:
    empty_counts = {family: 0 for family in families}
    return {
        "accepted_rows_by_family": {family: [] for family in families},
        "datasets": [],
        "acceptance": {
            "attempted_pairs": 0,
            "accepted_pairs": 0,
            "duplicate_pairs": 0,
            "duplicate_reason_counts": {},
            "rejection_reason_counts": {},
            "attempted_pairs_per_family": dict(empty_counts),
            "accepted_pairs_per_family": dict(empty_counts),
            "duplicate_pairs_per_family": dict(empty_counts),
            "rejected_pairs_per_family": dict(empty_counts),
        },
        "next_source_indexes": {family: start_index for family in families},
        "next_batch_numbers": {family: 1 for family in families},
    }


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_openrouter_batch_size(value: int) -> None:
    _validate_positive_int(value, "batch_size")
    if value > MAX_OPENROUTER_BATCH_SIZE:
        raise ValueError(f"batch_size must be at most {MAX_OPENROUTER_BATCH_SIZE}")


def _validate_openrouter_concurrency(value: int) -> None:
    _validate_positive_int(value, "concurrency")
    if value > MAX_OPENROUTER_CONCURRENCY:
        raise ValueError(f"concurrency must be at most {MAX_OPENROUTER_CONCURRENCY}")
