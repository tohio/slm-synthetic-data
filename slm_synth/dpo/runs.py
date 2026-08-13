"""DPO LLM-backed run orchestration helpers."""

from __future__ import annotations

import json
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from slm_synth.accepted_target import accepted_target_metadata, raise_for_underfilled_manifest
from slm_synth.adaptive_batch import AdaptiveBatchSizeController
from slm_synth.planning import build_count_plan
from slm_synth.dpo.generation import (
    DPOBatchAcceptanceError,
    StructuredTeacherBackend,
    build_openrouter_backend,
    generate_llm_batch,
)
from slm_synth.dpo.acceptance import partition_unique_dpo_rows
from slm_synth.dpo.io import read_jsonl, write_jsonl
from slm_synth.dpo.manifest import write_manifest, write_run_manifest
from slm_synth.dpo.batches import is_exact_target_dpo_spec
from slm_synth.dpo.spec_builders import DPO_SPEC_FAMILIES, build_specs, validate_spec_range
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
    resume: bool = False,
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

    run_manifest_path = Path(manifest_dir) / (run_manifest_filename or f"{generation_run}.manifest.json")
    resume_state = _load_resume_state(
        resume=resume,
        manifest_path=run_manifest_path,
        output_dir=output_dir,
        generation_run=generation_run,
        families=resolved_families,
        target_pairs_per_family=dict(count_plan.counts_by_key),
        start_index=start_index,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
    )
    if resume_state["complete"]:
        return DPOLLMRunResult(
            results=(),
            row_count=resume_state["accepted_pairs"],
            families=resolved_families,
            generation_run=generation_run,
            manifest_path=run_manifest_path,
        )
    if resume and max_backfill_rounds < resume_state["backfill_rounds"]:
        raise ValueError(
            "max_backfill_rounds cannot be lower than rounds already recorded by the resumed DPO run"
        )

    # Validate the complete worst-case source range before credentials are read
    # or a provider backend is constructed.
    if resume:
        remaining_for_resume = _remaining_pairs_by_family(
            targets=dict(count_plan.counts_by_key),
            accepted_rows_by_family=resume_state["accepted_rows_by_family"],
        )
        remaining_rounds = max(max_backfill_rounds - resume_state["backfill_rounds"], 0)
        for family in resolved_families:
            possible_attempts = remaining_for_resume[family] * remaining_rounds
            if possible_attempts:
                validate_spec_range(
                    family=family,
                    count=possible_attempts,
                    start_index=resume_state["next_source_indexes"][family],
                )
    else:
        for family in resolved_families:
            validate_spec_range(
                family=family,
                count=count_plan.counts_by_key[family] * (max_backfill_rounds + 1),
                start_index=start_index,
            )

    active_backend = backend

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
                "backfill_round": job["backfill_round"],
                "is_backfill": job["backfill_round"] > 0,
                **job.get("adaptive_batch_size", {}),
                **dict(metadata or {}),
            },
            holdout_registry=holdout_registry,
            backend=(
                active_backend
                if all(is_exact_target_dpo_spec(spec) for spec in job["specs"])
                else get_backend()
            ),
        )

    results: list[Any] = []
    rejected_llm_telemetry: list[dict[str, Any]] = []
    batch_controller = AdaptiveBatchSizeController(
        maximum=batch_size,
        minimum=1,
        initial=adaptive_initial_batch_size,
        increase_successes=adaptive_batch_increase_successes,
    )
    next_batch_numbers = dict(resume_state["next_batch_numbers"])
    next_source_indexes = dict(resume_state["next_source_indexes"])
    accepted_rows_by_family = dict(resume_state["accepted_rows_by_family"])
    datasets = list(resume_state["datasets"])
    output_acceptance = dict(resume_state["acceptance"])
    backfill_rounds = resume_state["backfill_rounds"]

    def run_generation_round(
        request_counts: dict[str, int], *, round_number: int
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
            print(
                "[generate] Starting DPO family: "
                f"{family} (round={round_number}, requested_pairs={len(specs)}, batch_size={batch_size}, "
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
                    "backfill_round": round_number,
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
                                    rejection_reason_counts["batch_acceptance_error"] += 1
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
                f"{family} round={round_number} pairs={family_pairs_done}, requested_pairs={len(specs)}, "
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

    initial_counts = {
        family: 0 if resume else count_plan.counts_by_key[family]
        for family in resolved_families
    }
    if any(initial_counts.values()):
        initial_jobs, initial_rejected, initial_rejection_reasons = run_generation_round(
            initial_counts, round_number=0
        )
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

    remaining_by_family = _remaining_pairs_by_family(
        targets=dict(count_plan.counts_by_key),
        accepted_rows_by_family=accepted_rows_by_family,
    )
    while any(remaining_by_family.values()) and backfill_rounds < max_backfill_rounds:
        backfill_rounds += 1
        backfill_jobs, backfill_rejected, backfill_rejection_reasons = run_generation_round(
            remaining_by_family, round_number=backfill_rounds
        )
        datasets, output_acceptance, accepted_rows_by_family = _write_public_family_files(
            jobs=backfill_jobs,
            output_dir=output_dir,
            families=resolved_families,
            accepted_rows_by_family=accepted_rows_by_family,
            prior_datasets=datasets,
            prior_acceptance=output_acceptance,
            new_rejected_pairs_per_family=backfill_rejected,
            new_rejection_reason_counts=backfill_rejection_reasons,
        )
        remaining_by_family = _remaining_pairs_by_family(
            targets=dict(count_plan.counts_by_key),
            accepted_rows_by_family=accepted_rows_by_family,
        )

    planned_pairs = count_plan.planned_count
    accepted_pairs = sum(dataset["row_count"] for dataset in datasets)
    attempted_pairs = output_acceptance["attempted_pairs"]
    duplicate_pairs = output_acceptance["duplicate_pairs"]
    rejected_pairs = max(attempted_pairs - accepted_pairs - duplicate_pairs, 0)
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
            "rejection_reason_counts": output_acceptance["rejection_reason_counts"],
            "attempted_pairs": attempted_pairs,
            "duplicate_pairs": duplicate_pairs,
            "duplicate_reason_counts": output_acceptance["duplicate_reason_counts"],
            "attempted_pairs_per_family": output_acceptance["attempted_pairs_per_family"],
            "accepted_pairs_per_family": output_acceptance["accepted_pairs_per_family"],
            "rejected_pairs_per_family": output_acceptance["rejected_pairs_per_family"],
            "duplicate_pairs_per_family": output_acceptance["duplicate_pairs_per_family"],
            "max_backfill_rounds": max_backfill_rounds,
            "backfill_rounds": backfill_rounds,
            "next_start_index_per_family": next_source_indexes,
            "accepted_content_fingerprints_per_family": {
                family: _rows_fingerprint(accepted_rows_by_family[family])
                for family in resolved_families
            },
            **accepted_target_metadata(
                unit="pairs",
                target_count=planned_pairs,
                accepted_count=accepted_pairs,
                attempted_count=attempted_pairs,
                max_backfill_rounds=max_backfill_rounds,
                backfill_rounds=backfill_rounds,
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
            "llm_telemetry": aggregate_llm_telemetry(
                [
                    telemetry
                    for telemetry in (
                        resume_state["llm_telemetry"],
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
        family: [row for row in accepted_rows if row["metadata"]["eval_family"] == family]
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


def _remaining_pairs_by_family(
    *, targets: dict[str, int], accepted_rows_by_family: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    return {
        family: max(target - len(accepted_rows_by_family.get(family, [])), 0)
        for family, target in targets.items()
    }


def _load_resume_state(
    *,
    resume: bool,
    manifest_path: Path,
    output_dir: str | Path,
    generation_run: str,
    families: tuple[str, ...],
    target_pairs_per_family: dict[str, int],
    start_index: int,
    teacher_model: str,
    teacher_provider: str,
) -> dict[str, Any]:
    if not resume:
        return _empty_resume_state(families=families, start_index=start_index)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cannot resume DPO run without run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"DPO resume manifest must contain a JSON object: {manifest_path}")
    if manifest.get("dataset_type") != "dpo" or manifest.get("generation_run") != generation_run:
        raise ValueError("DPO resume manifest does not match dataset type and generation run")
    if tuple(manifest.get("families", [])) != families:
        raise ValueError("DPO resume families do not match the existing run manifest")
    if manifest.get("teacher_model") != teacher_model or manifest.get("teacher_provider") != teacher_provider:
        raise ValueError("DPO resume teacher configuration does not match the existing run manifest")

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("DPO resume manifest is missing metadata")
    if metadata.get("pairs_per_family") != target_pairs_per_family or metadata.get("start_index") != start_index:
        raise ValueError("DPO resume source plan does not match the existing run manifest")
    next_source_indexes = metadata.get("next_start_index_per_family")
    if not isinstance(next_source_indexes, dict) or set(next_source_indexes) != set(families):
        raise ValueError("DPO resume manifest is missing next source indexes")
    for family, value in next_source_indexes.items():
        _validate_positive_int(value, f"next_start_index_per_family[{family}]")

    manifest_datasets = manifest.get("datasets")
    if not isinstance(manifest_datasets, list):
        raise ValueError("DPO resume manifest is missing datasets")
    datasets_by_family = {
        item.get("family"): item
        for item in manifest_datasets
        if isinstance(item, dict) and isinstance(item.get("family"), str)
    }
    content_fingerprints = metadata.get("accepted_content_fingerprints_per_family")
    if not isinstance(content_fingerprints, dict) or set(content_fingerprints) != set(families):
        raise ValueError("DPO resume manifest is missing accepted content fingerprints")

    accepted_rows_by_family: dict[str, list[dict[str, Any]]] = {}
    datasets: list[dict[str, Any]] = []
    next_batch_numbers: dict[str, int] = {}
    for family in families:
        item = datasets_by_family.get(family)
        if not isinstance(item, dict):
            raise ValueError(f"DPO resume manifest is missing dataset entry for {family}")
        dataset_path = Path(output_dir) / f"{family}.jsonl"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"DPO resume dataset does not exist: {dataset_path}")
        rows = read_jsonl(dataset_path)
        if item.get("row_count") != len(rows):
            raise ValueError(f"DPO resume dataset row count does not match manifest for {family}")
        if any(row["metadata"]["eval_family"] != family for row in rows):
            raise ValueError(f"DPO resume dataset contains the wrong family metadata for {family}")
        if content_fingerprints.get(family) != _rows_fingerprint(rows):
            raise ValueError(f"DPO resume dataset content fingerprint does not match manifest for {family}")
        accepted_rows_by_family[family] = rows
        raw_batch_manifests = item.get("batch_manifests")
        if not isinstance(raw_batch_manifests, list):
            raise ValueError(f"DPO resume batch manifests must be a list for {family}")
        batch_manifests = [Path(path) for path in raw_batch_manifests]
        if item.get("batch_count") != len(batch_manifests):
            raise ValueError(f"DPO resume batch count does not match manifest list for {family}")
        missing = [path for path in batch_manifests if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"DPO resume batch manifest does not exist: {missing[0]}")
        datasets.append(
            {
                "family": family,
                "dataset_path": dataset_path,
                "row_count": len(rows),
                "batch_count": len(batch_manifests),
                "batch_manifests": batch_manifests,
            }
        )
        next_batch_numbers[family] = _next_batch_number(batch_manifests)

    accepted_rows, unique_summary = partition_unique_dpo_rows(
        row for family in families for row in accepted_rows_by_family[family]
    )
    if unique_summary["duplicate_pairs"]:
        raise ValueError("DPO resume public datasets contain duplicate accepted content")
    accepted_count = len(accepted_rows)
    accepted_target = metadata.get("accepted_target")
    expected_target = sum(target_pairs_per_family.values())
    expected_remaining = expected_target - accepted_count
    if (
        not isinstance(accepted_target, dict)
        or accepted_target.get("target") != expected_target
        or accepted_target.get("accepted") != accepted_count
        or accepted_target.get("remaining") != expected_remaining
    ):
        raise ValueError("DPO resume accepted count does not match public datasets")

    acceptance = {
        "attempted_pairs": metadata.get("attempted_pairs"),
        "accepted_pairs": metadata.get("accepted_pairs"),
        "duplicate_pairs": metadata.get("duplicate_pairs"),
        "duplicate_reason_counts": metadata.get("duplicate_reason_counts", {}),
        "rejection_reason_counts": metadata.get("rejection_reason_counts", {}),
        "attempted_pairs_per_family": metadata.get("attempted_pairs_per_family"),
        "accepted_pairs_per_family": metadata.get("accepted_pairs_per_family"),
        "duplicate_pairs_per_family": metadata.get("duplicate_pairs_per_family"),
        "rejected_pairs_per_family": metadata.get("rejected_pairs_per_family"),
    }
    _validate_resume_acceptance(acceptance, families=families, accepted_pairs=accepted_count)
    llm_telemetry = metadata.get("llm_telemetry", {})
    if not isinstance(llm_telemetry, dict):
        raise ValueError("DPO resume manifest has invalid llm_telemetry")
    backfill_rounds = metadata.get("backfill_rounds", 0)
    _validate_non_negative_int(backfill_rounds, "backfill_rounds")
    remaining = _remaining_pairs_by_family(
        targets=target_pairs_per_family,
        accepted_rows_by_family=accepted_rows_by_family,
    )
    complete = (
        not any(remaining.values())
        and metadata.get("publish_ready") is True
        and accepted_target.get("publish_ready") is True
    )
    return {
        "complete": complete,
        "accepted_pairs": accepted_count,
        "accepted_rows_by_family": accepted_rows_by_family,
        "datasets": datasets,
        "acceptance": acceptance,
        "backfill_rounds": backfill_rounds,
        "next_source_indexes": dict(next_source_indexes),
        "next_batch_numbers": next_batch_numbers,
        "llm_telemetry": dict(llm_telemetry),
    }


def _empty_resume_state(*, families: tuple[str, ...], start_index: int) -> dict[str, Any]:
    empty_counts = {family: 0 for family in families}
    return {
        "complete": False,
        "accepted_pairs": 0,
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
        "backfill_rounds": 0,
        "next_source_indexes": {family: start_index for family in families},
        "next_batch_numbers": {family: 1 for family in families},
        "llm_telemetry": {},
    }


def _validate_resume_acceptance(
    acceptance: dict[str, Any], *, families: tuple[str, ...], accepted_pairs: int
) -> None:
    for field in ("attempted_pairs", "accepted_pairs", "duplicate_pairs"):
        _validate_non_negative_int(acceptance.get(field), field)
    if acceptance["accepted_pairs"] != accepted_pairs:
        raise ValueError("DPO resume accepted-pair accounting does not match public datasets")
    for field in (
        "attempted_pairs_per_family",
        "accepted_pairs_per_family",
        "duplicate_pairs_per_family",
        "rejected_pairs_per_family",
    ):
        values = acceptance.get(field)
        if not isinstance(values, dict) or set(values) != set(families):
            raise ValueError(f"DPO resume manifest has invalid {field}")
        for family, value in values.items():
            _validate_non_negative_int(value, f"{field}[{family}]")
    if sum(acceptance["accepted_pairs_per_family"].values()) != acceptance["accepted_pairs"]:
        raise ValueError("DPO resume per-family accepted counts do not match aggregate accounting")
    if sum(acceptance["attempted_pairs_per_family"].values()) != acceptance["attempted_pairs"]:
        raise ValueError("DPO resume per-family attempted counts do not match aggregate accounting")
    if sum(acceptance["duplicate_pairs_per_family"].values()) != acceptance["duplicate_pairs"]:
        raise ValueError("DPO resume per-family duplicate counts do not match aggregate accounting")
    for family in families:
        if (
            acceptance["accepted_pairs_per_family"][family]
            + acceptance["duplicate_pairs_per_family"][family]
            + acceptance["rejected_pairs_per_family"][family]
            != acceptance["attempted_pairs_per_family"][family]
        ):
            raise ValueError(f"DPO resume accounting does not balance for {family}")
    for field in ("duplicate_reason_counts", "rejection_reason_counts"):
        reasons = acceptance.get(field)
        if not isinstance(reasons, dict):
            raise ValueError(f"DPO resume manifest has invalid {field}")
        for reason, value in reasons.items():
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"DPO resume manifest has invalid {field} key")
            _validate_non_negative_int(value, f"{field}[{reason}]")


def _next_batch_number(batch_manifests: list[Path]) -> int:
    numbers: list[int] = []
    for path in batch_manifests:
        name = path.name
        if ".batch" not in name:
            continue
        suffix = name.split(".batch", 1)[1].split(".", 1)[0]
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def _rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


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
