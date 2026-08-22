"""SFT LLM-backed run orchestration helpers."""

from __future__ import annotations

import json
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from slm_synth.adaptive_batch import (
    AdaptiveBatchSizeController,
    aggregate_adaptive_batch_size_controllers,
)
from slm_synth.alignment_tokens import estimate_sft_tokens
from slm_synth.planning import CountPlan
from slm_synth.sft.acceptance import partition_unique_sft_rows
from slm_synth.sft.generation import (
    SFTBatchAcceptanceError,
    StructuredTeacherBackend,
    build_openrouter_backend,
    generate_llm_batch,
)
from slm_synth.sft.io import read_jsonl, write_jsonl
from slm_synth.sft.manifest import write_manifest, write_run_manifest
from slm_synth.sft.planning import DEFAULT_SFT_SPEC_PLANNER
from slm_synth.sft.spec_builders import SFT_SPEC_FAMILIES, build_specs, validate_spec_range
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
from slm_synth.quality_policy import summarize_judge_rejections


@dataclass(frozen=True)
class SFTLLMRunResult:
    """Result of running one multi-batch LLM-generated SFT job."""

    results: tuple[Any, ...]
    row_count: int
    families: tuple[str, ...]
    generation_run: str
    manifest_path: Path


def default_batch_output_dir(output_dir: str | Path) -> Path:
    """Return the sibling internal batch directory for a public dataset directory."""
    public_dir = Path(output_dir)
    return public_dir.parent / "batches"


def generate_llm_run(
    *,
    families: list[str] | tuple[str, ...] | None,
    candidate_counts_by_family: dict[str, int] | None = None,
    accepted_targets_by_family: dict[str, int] | None = None,
    candidate_wave_size: int = 1000,
    resume: bool = True,
    batch_size: int = 1,
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    adjudicator_model: str | None = None,
    adjudicator_max_tokens: int | None = None,
    reviewer_model: str | None = None,
    reviewer_max_tokens: int | None = None,
    start_index: int = 1,
    teacher_provider: str = "openrouter",
    temperature: float | None = None,
    top_p: float | None = None,
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
    reviewer_backend: StructuredTeacherBackend | None = None,
) -> SFTLLMRunResult:
    """Build specs and generate SFT datasets across families and batches."""
    resolved_families = resolve_spec_families(families)
    if (candidate_counts_by_family is None) == (accepted_targets_by_family is None):
        raise ValueError(
            "provide exactly one of candidate_counts_by_family or accepted_targets_by_family"
        )

    count_plan: CountPlan | None = None
    accepted_targets: dict[str, int] | None = None
    if candidate_counts_by_family is not None:
        normalized_counts = {
            str(family).strip().lower(): count
            for family, count in candidate_counts_by_family.items()
        }
        if set(normalized_counts) != set(resolved_families):
            raise ValueError(
                "candidate_counts_by_family must contain exactly the requested families"
            )
        for family, count in normalized_counts.items():
            _validate_positive_int(count, f"candidate count for {family}")
        count_plan = CountPlan(
            planning_mode="candidate_counts_by_family",
            counts_by_key={
                family: normalized_counts[family] for family in resolved_families
            },
        )
    else:
        accepted_targets = {
            str(family).strip().lower(): count
            for family, count in (accepted_targets_by_family or {}).items()
        }
        if set(accepted_targets) != set(resolved_families):
            raise ValueError(
                "accepted_targets_by_family must contain exactly the requested families"
            )
        for family, count in accepted_targets.items():
            _validate_positive_int(count, f"accepted target for {family}")
        _validate_positive_int(candidate_wave_size, "candidate_wave_size")

    _validate_openrouter_batch_size(batch_size)
    _validate_positive_int(start_index, "start_index")
    _validate_openrouter_concurrency(concurrency)
    _validate_positive_int(adaptive_initial_batch_size, "adaptive_initial_batch_size")
    _validate_positive_int(adaptive_batch_increase_successes, "adaptive_batch_increase_successes")
    adaptive_maximum_in_flight = concurrency

    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    checkpoint_path = Path(manifest_dir) / f"{generation_run}.checkpoint.json"
    checkpoint_signature = _run_checkpoint_signature(
        generation_run=generation_run,
        families=resolved_families,
        candidate_counts_by_family=(dict(count_plan.counts_by_key) if count_plan is not None else None),
        accepted_targets_by_family=accepted_targets,
        candidate_wave_size=candidate_wave_size if accepted_targets is not None else None,
        start_index=start_index,
        output_dir=output_dir,
        teacher_model=teacher_model,
        adjudicator_model=adjudicator_model if adjudicator_model is not None else teacher_model,
        reviewer_model=reviewer_model or adjudicator_model or teacher_model,
        max_tokens=max_tokens,
        adjudicator_max_tokens=adjudicator_max_tokens if adjudicator_max_tokens is not None else max_tokens,
        reviewer_max_tokens=reviewer_max_tokens or adjudicator_max_tokens or max_tokens,
        batch_size=batch_size,
        adaptive_initial_batch_size=adaptive_initial_batch_size,
        adaptive_batch_increase_successes=adaptive_batch_increase_successes,
        temperature=temperature,
        top_p=top_p,
    )
    resumed_from_checkpoint = checkpoint_path.exists()
    if resumed_from_checkpoint and not resume:
        raise RuntimeError(
            f"SFT checkpoint already exists at {checkpoint_path}; "
            "use a new generation_run or enable resume"
        )
    if resumed_from_checkpoint:
        initial_state = _load_run_checkpoint(
            checkpoint_path=checkpoint_path,
            expected_signature=checkpoint_signature,
            families=resolved_families,
        )
        if initial_state["complete"]:
            raise RuntimeError(
                f"SFT generation run {generation_run!r} is already complete according to "
                f"{checkpoint_path}; use a new generation_run for a new run"
            )
        print(
            "[generate] Resuming SFT run from checkpoint: "
            f"{checkpoint_path} planning_rounds={initial_state['planning_rounds']} "
            f"next_start_index_per_family={initial_state['next_source_indexes']}",
            flush=True,
        )
    else:
        initial_state = _empty_run_state(families=resolved_families, start_index=start_index)
    if count_plan is not None:
        for family in resolved_families:
            validate_spec_range(
                family=family,
                count=count_plan.counts_by_key[family],
                start_index=start_index,
            )
    else:
        # Accepted-target mode validates only each bounded wave before it is
        # consumed. This lets normal rejection/duplication draw fresh unique
        # candidate opportunities without reserving or regenerating rejected
        # specs up front.
        assert accepted_targets is not None
        for family in resolved_families:
            available_capacity = DEFAULT_SFT_SPEC_PLANNER.capacity(family) - start_index + 1
            if accepted_targets[family] > available_capacity:
                raise ValueError(
                    f"accepted target for {family} is {accepted_targets[family]}, but only "
                    f"{available_capacity} candidate opportunities are available from "
                    f"start_index={start_index}"
                )
            validate_spec_range(
                family=family,
                count=min(candidate_wave_size, accepted_targets[family]),
                start_index=start_index,
            )

    # Validate the full finite inventory before credentials are read or a
    # provider backend can be constructed. A clean requested slice cannot hide
    # a defect elsewhere in the catalog.
    from slm_synth.alignment_preflight import preflight_sft_inventory

    preflight_sft_inventory()

    active_backend = backend
    active_adjudicator_backend = adjudicator_backend
    active_reviewer_backend = reviewer_backend

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

    def get_reviewer_backend() -> StructuredTeacherBackend:
        nonlocal active_reviewer_backend
        if active_reviewer_backend is None and reviewer_model is None:
            active_reviewer_backend = get_adjudicator_backend()
        if active_reviewer_backend is None:
            active_reviewer_backend = build_openrouter_backend(
                model=reviewer_model or adjudicator_model or teacher_model,
                max_tokens=reviewer_max_tokens or adjudicator_max_tokens or max_tokens,
                temperature=temperature, top_p=top_p, request_timeout=request_timeout,
                max_request_retries=max_request_retries,
                max_retryable_request_attempts=max_retryable_request_attempts,
                retry_max_elapsed_seconds=retry_max_elapsed_seconds,
                adaptive_maximum_in_flight=adaptive_maximum_in_flight,
                adaptive_initial_in_flight=adaptive_initial_in_flight,
                openrouter_routing_mode=openrouter_routing_mode,
                openrouter_provider=openrouter_provider,
            )
        return active_reviewer_backend

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
            reviewer_model=reviewer_model,
            reviewer_max_tokens=reviewer_max_tokens,
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
            reviewer_backend=get_reviewer_backend(),
        )

    results: list[Any] = []
    rejected_llm_telemetry: list[dict[str, Any]] = list(
        initial_state.get("rejected_llm_telemetry", [])
    )
    batch_controllers: list[AdaptiveBatchSizeController] = []
    rejection_diagnostics: list[dict[str, Any]] = list(
        initial_state.get("rejection_diagnostics", [])
    )
    next_batch_numbers = dict(initial_state["next_batch_numbers"])
    next_source_indexes = dict(initial_state["next_source_indexes"])
    accepted_rows_by_family = dict(initial_state["accepted_rows_by_family"])
    datasets = list(initial_state["datasets"])
    output_acceptance = dict(initial_state["acceptance"])
    planning_rounds = int(initial_state.get("planning_rounds", 0))
    requested_candidate_rows_per_family = dict(
        initial_state.get(
            "requested_candidate_rows_per_family",
            {family: 0 for family in resolved_families},
        )
    )
    checkpoint_llm_telemetry = dict(initial_state.get("llm_telemetry", {}))
    checkpointed_rejected_telemetry_count = int(
        initial_state.get("checkpointed_rejected_telemetry_count", len(rejected_llm_telemetry))
    )
    checkpointed_batch_manifests = {
        str(Path(path))
        for dataset in datasets
        for path in dataset.get("batch_manifests", [])
    }
    accepted_content_fingerprints = dict(
        initial_state.get(
            "accepted_content_fingerprints_per_family",
            {
                family: _rows_fingerprint(accepted_rows_by_family.get(family, []))
                for family in resolved_families
            },
        )
    )

    def write_checkpoint(
        *,
        complete: bool = False,
        changed_families: tuple[str, ...] = (),
    ) -> None:
        nonlocal checkpoint_llm_telemetry, checkpointed_rejected_telemetry_count
        for family in changed_families:
            accepted_content_fingerprints[family] = _rows_fingerprint(
                accepted_rows_by_family.get(family, [])
            )
        current_batch_manifests = [
            str(Path(path))
            for dataset in datasets
            for path in dataset.get("batch_manifests", [])
        ]
        new_batch_manifests = [
            Path(path)
            for path in current_batch_manifests
            if path not in checkpointed_batch_manifests
        ]
        new_rejected_telemetry = rejected_llm_telemetry[
            checkpointed_rejected_telemetry_count:
        ]
        new_telemetry = aggregate_llm_telemetry(
            [
                item
                for item in (
                    aggregate_llm_telemetry_from_manifests(new_batch_manifests),
                    *new_rejected_telemetry,
                )
                if item
            ]
        )
        if new_telemetry:
            checkpoint_llm_telemetry = aggregate_llm_telemetry(
                [item for item in (checkpoint_llm_telemetry, new_telemetry) if item]
            )
        checkpointed_batch_manifests.update(current_batch_manifests)
        checkpointed_rejected_telemetry_count = len(rejected_llm_telemetry)
        _write_run_checkpoint(
            checkpoint_path=checkpoint_path,
            signature=checkpoint_signature,
            complete=complete,
            next_source_indexes=next_source_indexes,
            next_batch_numbers=next_batch_numbers,
            datasets=datasets,
            acceptance=output_acceptance,
            requested_candidate_rows_per_family=requested_candidate_rows_per_family,
            planning_rounds=planning_rounds,
            rejection_diagnostics=rejection_diagnostics,
            rejected_llm_telemetry=rejected_llm_telemetry,
            llm_telemetry=checkpoint_llm_telemetry,
            checkpointed_rejected_telemetry_count=checkpointed_rejected_telemetry_count,
            accepted_content_fingerprints_per_family=accepted_content_fingerprints,
        )

    def run_generation(request_counts: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
        round_jobs: list[dict[str, Any]] = []
        rejected_rows_per_family = {family: 0 for family in resolved_families}
        rejection_reason_counts: Counter[str] = Counter()
        for family in resolved_families:
            requested_rows = request_counts.get(family, 0)
            if requested_rows == 0:
                continue
            round_start_index = next_source_indexes[family]
            validate_spec_range(family=family, count=requested_rows, start_index=round_start_index)
            specs = build_specs(family=family, count=requested_rows, start_index=round_start_index)
            next_source_indexes[family] += requested_rows
            batch_controller = AdaptiveBatchSizeController(
                maximum=batch_size,
                minimum=1,
                initial=adaptive_initial_batch_size,
                increase_successes=adaptive_batch_increase_successes,
            )
            batch_controllers.append(batch_controller)
            print(
                "[generate] Starting SFT family: "
                f"{family} (candidate_rows={len(specs)}, batch_size={batch_size}, "
                f"min_batch_size=1, parallel_requests={concurrency}, model={teacher_model})",
                flush=True,
            )
            pending_ranges: deque[tuple[int, int]] = deque([(0, len(specs))])
            active: dict[Any, dict[str, Any]] = {}
            family_rows_done = 0

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
                            if isinstance(exc, SFTBatchAcceptanceError):
                                rejected_llm_telemetry.append(exc.telemetry)
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
                                if isinstance(exc, SFTBatchAcceptanceError):
                                    rejected_rows_per_family[family] += 1
                                    rejection_reason_counts[exc.failure_type] += 1
                                    rejection_diagnostics.append(
                                        {
                                            "id": job["specs"][0]["id"],
                                            "family": family,
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
                        if result.semantic_rejected_count:
                            rejected_rows_per_family[family] += result.semantic_rejected_count
                            rejection_reason_counts["semantic_quality_rejected"] += result.semantic_rejected_count
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
                f"{family} rendered_rows={family_rows_done}, candidate_rows={len(specs)}, "
                f"batch_size={batch_size}, min_batch_size=1, parallel_requests={concurrency}, "
                f"adaptive_batch_size_observed_minimum={batch_controller.observed_minimum}, "
                f"adaptive_batch_size_observed_peak={batch_controller.observed_peak}, "
                f"adaptive_batch_size_increases={batch_controller.increases}, "
                f"adaptive_batch_size_decreases={batch_controller.decreases}, "
                f"adaptive_batch_size_failures={batch_controller.failures}",
                flush=True,
            )
        round_jobs.sort(key=lambda item: (item["family"], item["batch_start_index"], item["batch_number"]))
        return round_jobs, rejected_rows_per_family, dict(rejection_reason_counts)

    def execute_round(request_counts: dict[str, int]) -> None:
        nonlocal datasets, output_acceptance, accepted_rows_by_family, planning_rounds
        if not any(request_counts.values()):
            return
        planning_rounds += 1
        # Commit one family-wave at a time. If the process stops, all public
        # data and planner cursors up to the last completed family-wave are
        # restartable without consuming or skipping uncommitted candidates.
        for family in resolved_families:
            requested_rows = request_counts.get(family, 0)
            if requested_rows == 0:
                continue
            round_jobs, round_rejected, round_rejection_reasons = run_generation(
                {family: requested_rows}
            )
            datasets, output_acceptance, accepted_rows_by_family = _write_public_family_files(
                jobs=round_jobs,
                output_dir=output_dir,
                families=resolved_families,
                accepted_rows_by_family=accepted_rows_by_family,
                prior_datasets=datasets,
                prior_acceptance=output_acceptance,
                new_rejected_rows_per_family=round_rejected,
                new_rejection_reason_counts=round_rejection_reasons,
            )
            requested_candidate_rows_per_family[family] += requested_rows
            write_checkpoint(changed_families=(family,))

    if count_plan is not None:
        remaining_counts = {
            family: max(
                count_plan.counts_by_key[family]
                - requested_candidate_rows_per_family.get(family, 0),
                0,
            )
            for family in resolved_families
        }
        execute_round(remaining_counts)
        planning_mode = count_plan.planning_mode
    else:
        assert accepted_targets is not None
        planning_mode = "accepted_targets_by_family"
        while True:
            accepted_now = output_acceptance["accepted_rows_per_family"]
            remaining = {
                family: max(accepted_targets[family] - accepted_now.get(family, 0), 0)
                for family in resolved_families
            }
            if not any(remaining.values()):
                break

            request_counts: dict[str, int] = {}
            for family in resolved_families:
                if remaining[family] == 0:
                    request_counts[family] = 0
                    continue
                available_capacity = (
                    DEFAULT_SFT_SPEC_PLANNER.capacity(family)
                    - next_source_indexes[family]
                    + 1
                )
                if available_capacity <= 0:
                    raise RuntimeError(
                        "SFT accepted-target planning exhausted candidate capacity before "
                        f"reaching the target for {family}: "
                        f"accepted={accepted_now.get(family, 0)}, "
                        f"target={accepted_targets[family]}, "
                        f"next_start_index={next_source_indexes[family]}"
                    )
                request_counts[family] = min(
                    candidate_wave_size, remaining[family], available_capacity
                )
            execute_round(request_counts)

    candidate_rows = sum(requested_candidate_rows_per_family.values())
    accepted_rows = sum(dataset["row_count"] for dataset in datasets)
    attempted_rows = output_acceptance["attempted_rows"]
    duplicate_rows = output_acceptance["duplicate_rows"]
    rejected_rows = max(attempted_rows - accepted_rows - duplicate_rows, 0)
    estimated_tokens_per_family = {
        family: sum(estimate_sft_tokens(row) for row in accepted_rows_by_family[family])
        for family in resolved_families
    }
    empty_families = sorted(
        family for family in resolved_families if not accepted_rows_by_family[family]
    )
    all_batch_manifests = [
        Path(path)
        for dataset in datasets
        for path in dataset.get("batch_manifests", [])
    ]
    judge_rejection_policy = summarize_judge_rejections(all_batch_manifests)

    _write_llm_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=generation_run,
        families=resolved_families,
        datasets=datasets,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        metadata={
            "generation_mode": "live_llm_run",
            "adjudicator_model": adjudicator_model if adjudicator_model is not None else teacher_model,
            "reviewer_model": reviewer_model or adjudicator_model or teacher_model,
            "adjudicator_max_tokens": adjudicator_max_tokens if adjudicator_max_tokens is not None else max_tokens,
            "planning_mode": planning_mode,
            "candidate_rows": candidate_rows,
            "attempted_rows": attempted_rows,
            "accepted_rows": accepted_rows,
            "estimated_tokens": sum(estimated_tokens_per_family.values()),
            "rejected_rows": rejected_rows,
            "rejection_reason_counts": output_acceptance["rejection_reason_counts"],
            "rejection_diagnostics": rejection_diagnostics,
            **judge_rejection_policy,
            "duplicate_rows": duplicate_rows,
            "duplicate_reason_counts": output_acceptance["duplicate_reason_counts"],
            "attempted_rows_per_family": output_acceptance["attempted_rows_per_family"],
            "accepted_rows_per_family": output_acceptance["accepted_rows_per_family"],
            "estimated_tokens_per_family": estimated_tokens_per_family,
            "rejected_rows_per_family": output_acceptance["rejected_rows_per_family"],
            "duplicate_rows_per_family": output_acceptance["duplicate_rows_per_family"],
            "next_start_index_per_family": next_source_indexes,
            "accepted_content_fingerprints_per_family": {
                family: _rows_fingerprint(accepted_rows_by_family[family])
                for family in resolved_families
            },
            "generation_status": "complete",
            "publish_ready": not empty_families,
            "empty_families": empty_families,
            "candidate_rows_per_family": dict(requested_candidate_rows_per_family),
            "accepted_targets_per_family": dict(accepted_targets or {}),
            "candidate_wave_size": candidate_wave_size if accepted_targets is not None else None,
            "planning_rounds": planning_rounds,
            "checkpoint_path": str(checkpoint_path),
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "adaptive_maximum_in_flight": adaptive_maximum_in_flight,
            "adaptive_initial_in_flight": adaptive_initial_in_flight,
            "adaptive_initial_batch_size": adaptive_initial_batch_size,
            "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
            **aggregate_adaptive_batch_size_controllers(batch_controllers),
            "llm_telemetry": checkpoint_llm_telemetry,
            "start_index": start_index,
            **dict(metadata or {}),
        },
    )
    write_checkpoint(complete=True)
    return SFTLLMRunResult(
        results=tuple(results),
        row_count=accepted_rows,
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


def _write_public_family_files(
    *,
    jobs: list[dict[str, Any]],
    output_dir: str | Path,
    families: tuple[str, ...],
    accepted_rows_by_family: dict[str, list[dict[str, Any]]],
    prior_datasets: list[dict[str, Any]],
    prior_acceptance: dict[str, Any],
    new_rejected_rows_per_family: dict[str, int],
    new_rejection_reason_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    candidate_rows = [
        row
        for family in families
        for row in accepted_rows_by_family.get(family, [])
    ]
    new_candidate_rows_per_family = {family: 0 for family in families}
    for family in families:
        for job in [item for item in jobs if item["family"] == family]:
            rows = read_jsonl(job["result"].dataset_path)
            candidate_rows.extend(rows)
            new_candidate_rows_per_family[family] += len(rows)
    new_attempted_rows_per_family = {
        family: new_candidate_rows_per_family[family] + new_rejected_rows_per_family[family]
        for family in families
    }

    accepted_rows, round_acceptance = partition_unique_sft_rows(candidate_rows)
    current_rows_by_family = {
        family: [row for row in accepted_rows if row["metadata"]["task_family"] == family]
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

    prior_attempted = prior_acceptance.get("attempted_rows_per_family", {})
    prior_duplicates = prior_acceptance.get("duplicate_rows_per_family", {})
    prior_rejected = prior_acceptance.get("rejected_rows_per_family", {})
    attempted_rows_per_family = {
        family: prior_attempted.get(family, 0) + new_attempted_rows_per_family[family]
        for family in families
    }
    accepted_rows_per_family = {family: len(current_rows_by_family[family]) for family in families}
    new_duplicate_rows_per_family = {
        family: (
            len(accepted_rows_by_family.get(family, []))
            + new_candidate_rows_per_family[family]
            - accepted_rows_per_family[family]
        )
        for family in families
    }
    duplicate_rows_per_family = {
        family: prior_duplicates.get(family, 0) + new_duplicate_rows_per_family[family]
        for family in families
    }
    rejected_rows_per_family = {
        family: prior_rejected.get(family, 0) + new_rejected_rows_per_family[family]
        for family in families
    }
    reason_counts = Counter(prior_acceptance.get("duplicate_reason_counts", {}))
    reason_counts.update(round_acceptance["duplicate_reason_counts"])
    rejection_reasons = Counter(prior_acceptance.get("rejection_reason_counts", {}))
    rejection_reasons.update(new_rejection_reason_counts)
    acceptance = {
        "attempted_rows": sum(attempted_rows_per_family.values()),
        "accepted_rows": sum(accepted_rows_per_family.values()),
        "duplicate_rows": sum(duplicate_rows_per_family.values()),
        "duplicate_reason_counts": {
            reason: reason_counts[reason]
            for reason in sorted(reason_counts)
        },
        "rejection_reason_counts": {
            reason: rejection_reasons[reason]
            for reason in sorted(rejection_reasons)
        },
        "attempted_rows_per_family": attempted_rows_per_family,
        "accepted_rows_per_family": accepted_rows_per_family,
        "duplicate_rows_per_family": duplicate_rows_per_family,
        "rejected_rows_per_family": rejected_rows_per_family,
    }
    return datasets, acceptance, current_rows_by_family


def _empty_run_state(*, families: tuple[str, ...], start_index: int) -> dict[str, Any]:
    empty_counts = {family: 0 for family in families}
    return {
        "complete": False,
        "accepted_rows": 0,
        "accepted_rows_by_family": {family: [] for family in families},
        "datasets": [],
        "acceptance": {
            "attempted_rows": 0,
            "accepted_rows": 0,
            "duplicate_rows": 0,
            "duplicate_reason_counts": {},
            "rejection_reason_counts": {},
            "attempted_rows_per_family": dict(empty_counts),
            "accepted_rows_per_family": dict(empty_counts),
            "duplicate_rows_per_family": dict(empty_counts),
            "rejected_rows_per_family": dict(empty_counts),
        },
        "next_source_indexes": {family: start_index for family in families},
        "next_batch_numbers": {family: 1 for family in families},
        "requested_candidate_rows_per_family": dict(empty_counts),
        "planning_rounds": 0,
        "rejection_diagnostics": [],
        "rejected_llm_telemetry": [],
    }


def _run_checkpoint_signature(
    *,
    generation_run: str,
    families: tuple[str, ...],
    candidate_counts_by_family: dict[str, int] | None,
    accepted_targets_by_family: dict[str, int] | None,
    candidate_wave_size: int | None,
    start_index: int,
    output_dir: str | Path,
    teacher_model: str,
    adjudicator_model: str,
    reviewer_model: str,
    max_tokens: int,
    adjudicator_max_tokens: int,
    reviewer_max_tokens: int,
    batch_size: int,
    adaptive_initial_batch_size: int,
    adaptive_batch_increase_successes: int,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    planning_mode = (
        "candidate_counts_by_family"
        if candidate_counts_by_family is not None
        else "accepted_targets_by_family"
    )
    return {
        "generation_run": generation_run,
        "families": list(families),
        "planning_mode": planning_mode,
        "candidate_counts_by_family": dict(candidate_counts_by_family or {}),
        "accepted_targets_by_family": dict(accepted_targets_by_family or {}),
        "candidate_wave_size": candidate_wave_size,
        "start_index": start_index,
        "output_dir": str(Path(output_dir)),
        "teacher_model": teacher_model,
        "adjudicator_model": adjudicator_model,
        "reviewer_model": reviewer_model,
        "max_tokens": max_tokens,
        "adjudicator_max_tokens": adjudicator_max_tokens,
        "reviewer_max_tokens": reviewer_max_tokens,
        "batch_size": batch_size,
        "adaptive_initial_batch_size": adaptive_initial_batch_size,
        "adaptive_batch_increase_successes": adaptive_batch_increase_successes,
        "temperature": temperature,
        "top_p": top_p,
    }


def _serialize_checkpoint_datasets(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "family": item["family"],
            "dataset_path": str(Path(item["dataset_path"])),
            "row_count": int(item["row_count"]),
            "batch_count": int(item["batch_count"]),
            "batch_manifests": [str(Path(path)) for path in item["batch_manifests"]],
        }
        for item in datasets
    ]


def _write_run_checkpoint(
    *,
    checkpoint_path: str | Path,
    signature: dict[str, Any],
    complete: bool,
    next_source_indexes: dict[str, int],
    next_batch_numbers: dict[str, int],
    datasets: list[dict[str, Any]],
    acceptance: dict[str, Any],
    requested_candidate_rows_per_family: dict[str, int],
    planning_rounds: int,
    rejection_diagnostics: list[dict[str, Any]],
    rejected_llm_telemetry: list[dict[str, Any]],
    llm_telemetry: dict[str, Any],
    checkpointed_rejected_telemetry_count: int,
    accepted_content_fingerprints_per_family: dict[str, str],
) -> Path:
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_type": "sft",
        "generation_run": signature["generation_run"],
        "complete": bool(complete),
        "signature": signature,
        "state": {
            "next_source_indexes": dict(next_source_indexes),
            "next_batch_numbers": dict(next_batch_numbers),
            "datasets": _serialize_checkpoint_datasets(datasets),
            "acceptance": acceptance,
            "requested_candidate_rows_per_family": dict(
                requested_candidate_rows_per_family
            ),
            "planning_rounds": int(planning_rounds),
            "rejection_diagnostics": list(rejection_diagnostics),
            "rejected_llm_telemetry": list(rejected_llm_telemetry),
            "llm_telemetry": dict(llm_telemetry),
            "checkpointed_rejected_telemetry_count": int(
                checkpointed_rejected_telemetry_count
            ),
            "accepted_content_fingerprints_per_family": dict(
                accepted_content_fingerprints_per_family
            ),
        },
    }
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def _load_run_checkpoint(
    *,
    checkpoint_path: str | Path,
    expected_signature: dict[str, Any],
    families: tuple[str, ...],
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read SFT checkpoint {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("dataset_type") != "sft":
        raise RuntimeError(f"unsupported or invalid SFT checkpoint: {path}")
    actual_signature = payload.get("signature")
    if actual_signature != expected_signature:
        raise RuntimeError(
            "SFT checkpoint configuration does not match this invocation; "
            f"checkpoint={path}"
        )
    state = payload.get("state")
    if not isinstance(state, dict):
        raise RuntimeError(f"SFT checkpoint is missing state: {path}")

    datasets = list(state.get("datasets", []))
    datasets_by_family = {item["family"]: item for item in datasets}
    accepted_rows_by_family: dict[str, list[dict[str, Any]]] = {}
    expected_fingerprints = state.get("accepted_content_fingerprints_per_family", {})
    for family in families:
        dataset = datasets_by_family.get(family)
        if dataset is None:
            rows: list[dict[str, Any]] = []
        else:
            dataset_path = Path(dataset["dataset_path"])
            if not dataset_path.exists():
                raise RuntimeError(
                    f"SFT checkpoint expects dataset file for {family}, but it is missing: "
                    f"{dataset_path}"
                )
            rows = read_jsonl(dataset_path)
            if len(rows) != int(dataset["row_count"]):
                raise RuntimeError(
                    f"SFT checkpoint row count mismatch for {family}: "
                    f"checkpoint={dataset['row_count']} file={len(rows)}"
                )
        expected_fingerprint = expected_fingerprints.get(family)
        if expected_fingerprint is not None and _rows_fingerprint(rows) != expected_fingerprint:
            raise RuntimeError(
                f"SFT checkpoint content fingerprint mismatch for {family}; "
                "refusing to resume from modified public output"
            )
        accepted_rows_by_family[family] = rows

    empty_state = _empty_run_state(
        families=families,
        start_index=int(expected_signature["start_index"]),
    )
    return {
        **empty_state,
        "complete": bool(payload.get("complete", False)),
        "accepted_rows_by_family": accepted_rows_by_family,
        "datasets": datasets,
        "acceptance": dict(state.get("acceptance", empty_state["acceptance"])),
        "next_source_indexes": dict(
            state.get("next_source_indexes", empty_state["next_source_indexes"])
        ),
        "next_batch_numbers": dict(
            state.get("next_batch_numbers", empty_state["next_batch_numbers"])
        ),
        "requested_candidate_rows_per_family": dict(
            state.get(
                "requested_candidate_rows_per_family",
                empty_state["requested_candidate_rows_per_family"],
            )
        ),
        "planning_rounds": int(state.get("planning_rounds", 0)),
        "rejection_diagnostics": list(state.get("rejection_diagnostics", [])),
        "rejected_llm_telemetry": list(state.get("rejected_llm_telemetry", [])),
        "llm_telemetry": dict(state.get("llm_telemetry", {})),
        "checkpointed_rejected_telemetry_count": int(
            state.get(
                "checkpointed_rejected_telemetry_count",
                len(state.get("rejected_llm_telemetry", [])),
            )
        ),
        "accepted_content_fingerprints_per_family": dict(expected_fingerprints),
    }


def _rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


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
