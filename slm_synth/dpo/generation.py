"""Non-network materialization for LLM-generated DPO batches."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
)
from slm_synth.dpo.batches import (
    merge_dpo_generation_stages,
    render_dpo_chosen_prompt,
    render_dpo_rejected_prompt,
    validate_dpo_chosen_stage,
    validate_dpo_batch_response,
)
from slm_synth.dpo.adjudication import adjudicate_dpo_rows
from slm_synth.quality_telemetry import combine_telemetry
from slm_synth.output_constraints import (
    OutputConstraintError,
    evaluate_dpo_output_constraints,
)
from slm_synth.dpo.io import write_jsonl
from slm_synth.dpo.manifest import write_manifest
from slm_synth.dpo.specs import validate_dpo_spec
from slm_synth.taxonomy.holdouts import HoldoutRegistry
from slm_synth.model_contract import PlainTextBackend, call_plain_parsed, parse_json_object

if TYPE_CHECKING:
    from slm_synth.llm import LLMBackend

SUPPORTED_TEACHER_PROVIDERS = frozenset({"openrouter"})


StructuredTeacherBackend = PlainTextBackend


class DPOBatchAcceptanceError(ValueError):
    """Raised when a completed teacher response fails local DPO acceptance."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str = "batch_acceptance_error",
        telemetry: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.failure_type = failure_type
        self.telemetry = dict(telemetry or {})


@dataclass(frozen=True)
class DPOLLMBatchResult:
    """Result of materializing one saved LLM DPO batch."""

    dataset_path: Path
    manifest_path: Path
    row_count: int
    generation_run: str
    teacher_model: str
    teacher_provider: str
    semantic_rejected_count: int = 0


def build_openrouter_backend(
    *,
    model: str,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    adaptive_initial_in_flight: int = 8,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
) -> "LLMBackend":
    """Create the supported production DPO teacher backend."""
    from slm_synth.llm import LLMBackend

    return LLMBackend(
        provider="openrouter",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        json_mode=False,
        request_timeout=request_timeout,
        max_request_retries=max_request_retries,
        max_retryable_request_attempts=max_retryable_request_attempts,
        retry_max_elapsed_seconds=retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=adaptive_maximum_in_flight,
        adaptive_initial_in_flight=adaptive_initial_in_flight,
        openrouter_routing_mode=openrouter_routing_mode,
        openrouter_provider=openrouter_provider,
    )


def generate_llm_batch(
    *,
    specs: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    manifest_path: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    adjudicator_model: str | None = None,
    adjudicator_max_tokens: int | None = None,
    reviewer_model: str | None = None,
    reviewer_max_tokens: int | None = None,
    teacher_provider: str = "openrouter",
    temperature: float | None = None,
    top_p: float | None = None,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    adaptive_initial_in_flight: int = 8,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
    backend: StructuredTeacherBackend | None = None,
    adjudicator_backend: StructuredTeacherBackend | None = None,
    reviewer_backend: StructuredTeacherBackend | None = None,
) -> DPOLLMBatchResult:
    """Generate one DPO batch with OpenRouter and write dataset + manifest."""
    provider = _validate_teacher_provider(teacher_provider)
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    if not validated_specs:
        raise ValueError("at least one DPO spec is required")

    active_backend = backend
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
    try:
        chosen_response, chosen_telemetry = _generate_dpo_stage(
            backend=active_backend,
            prompt=render_dpo_chosen_prompt(validated_specs),
        )
        chosen_items = validate_dpo_chosen_stage(chosen_response, specs=validated_specs)
    except (TypeError, ValueError) as exc:
        raise DPOBatchAcceptanceError(
            str(exc),
            failure_type="chosen_render_error",
            telemetry=locals().get("chosen_telemetry", {}),
        ) from exc

    try:
        rejected_response, rejected_telemetry = _generate_dpo_stage(
            backend=active_backend,
            prompt=render_dpo_rejected_prompt(validated_specs, chosen_items),
        )
        teacher_response = merge_dpo_generation_stages(
            specs=validated_specs,
            chosen_response=chosen_response,
            rejected_response=rejected_response,
        )
        rows = _validate_candidate_rows(
            specs=validated_specs,
            teacher_response=teacher_response,
            holdout_registry=holdout_registry,
        )
    except (TypeError, ValueError) as exc:
        raise DPOBatchAcceptanceError(
            str(exc),
            failure_type="rejected_render_error",
            telemetry=combine_telemetry(
                chosen_telemetry, locals().get("rejected_telemetry", {})
            ),
        ) from exc

    try:
        deterministic_validation = evaluate_dpo_output_constraints(specs=validated_specs, rows=rows)
    except OutputConstraintError as exc:
        raise DPOBatchAcceptanceError(
            str(exc),
            failure_type="deterministic_constraint_error",
            telemetry=combine_telemetry(chosen_telemetry, rejected_telemetry),
        ) from exc

    active_adjudicator = adjudicator_backend or build_openrouter_backend(
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
    active_reviewer = reviewer_backend or (active_adjudicator if reviewer_model is None else None)
    if active_reviewer is None:
        active_reviewer = build_openrouter_backend(
            model=reviewer_model or adjudicator_model or teacher_model,
            max_tokens=reviewer_max_tokens or adjudicator_max_tokens or max_tokens,
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
    try:
        decisions, adjudication_telemetry = adjudicate_dpo_rows(
            specs=validated_specs, rows=rows, backend=active_adjudicator,
            reviewer_backend=active_reviewer,
        )
    except (TypeError, ValueError) as exc:
        raise DPOBatchAcceptanceError(
            str(exc),
            failure_type="semantic_adjudication_error",
            telemetry=combine_telemetry(chosen_telemetry, rejected_telemetry),
        ) from exc

    rejected_audit_path = _write_rejected_candidate_audit(
        specs=validated_specs,
        rows=rows,
        decisions=decisions,
        deterministic_validation=deterministic_validation,
        manifest_path=manifest_path,
    )
    accepted_ids = {item_id for item_id, decision in decisions.items() if decision["accepted"]}
    accepted_specs = [spec for spec in validated_specs if spec["id"] in accepted_ids]
    accepted_rows = [row for row in rows if row["id"] in accepted_ids]
    rejected_count = len(validated_specs) - len(accepted_specs)
    telemetry = combine_telemetry(chosen_telemetry, rejected_telemetry, adjudication_telemetry)
    if not accepted_specs:
        return _write_accepted_dpo_rows(
            rows=[], output_path=output_path, manifest_path=manifest_path,
            teacher_model=teacher_model, teacher_provider=provider, generation_run=generation_run,
            metadata={
                "generation_mode": "live_llm_batch", "spec_count": len(validated_specs),
                "accepted_count": 0, "semantic_rejected_count": rejected_count,
                "rejected_audit_path": str(rejected_audit_path) if rejected_audit_path else None,
                "llm_telemetry": telemetry,
                "llm_stage_telemetry": {"chosen_renderer": chosen_telemetry, "rejected_renderer": rejected_telemetry, "quality_pipeline": adjudication_telemetry},
                "adjudicator_model": adjudicator_model or teacher_model,
                "reviewer_model": reviewer_model or adjudicator_model or teacher_model,
                "quality_adjudication": decisions, **dict(metadata or {}),
            }, semantic_rejected_count=rejected_count,
        )
    accepted_response = {"items": accepted_rows}
    try:
        result = materialize_llm_batch(
            specs=accepted_specs,
            teacher_response=accepted_response,
            output_path=output_path,
            manifest_path=manifest_path,
            teacher_model=teacher_model,
            teacher_provider=provider,
            generation_run=generation_run,
            metadata={
                "generation_mode": "live_llm_batch",
                "spec_count": len(validated_specs),
                "accepted_count": len(accepted_specs),
                "semantic_rejected_count": rejected_count,
                "rejected_audit_path": str(rejected_audit_path) if rejected_audit_path else None,
                "llm_telemetry": telemetry,
                "llm_stage_telemetry": {
                    "chosen_renderer": chosen_telemetry,
                    "rejected_renderer": rejected_telemetry,
                    "quality_pipeline": adjudication_telemetry,
                },
                "adjudicator_model": adjudicator_model if adjudicator_model is not None else teacher_model,
                "reviewer_model": reviewer_model or adjudicator_model or teacher_model,
                "quality_adjudication": decisions,
                **dict(metadata or {}),
            },
            holdout_registry=holdout_registry,
        )
        return replace(result, semantic_rejected_count=rejected_count)
    except DPOBatchAcceptanceError as exc:
        raise DPOBatchAcceptanceError(
            str(exc),
            failure_type="materialization_validation_error",
            telemetry=telemetry,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise DPOBatchAcceptanceError(
            str(exc),
            failure_type="materialization_validation_error",
            telemetry=telemetry,
        ) from exc


def _write_rejected_candidate_audit(
    *,
    specs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    deterministic_validation: Any,
    manifest_path: str | Path,
) -> Path | None:
    """Persist rejected candidate evidence outside the publishable dataset."""
    rejected_ids = [
        item_id for item_id, decision in decisions.items() if not decision.get("accepted", False)
    ]
    if not rejected_ids:
        return None

    specs_by_id = {spec["id"]: spec for spec in specs}
    rows_by_id = {row["id"]: row for row in rows}
    validation_by_id: dict[str, Any] = {}
    if isinstance(deterministic_validation, Mapping):
        validation_by_id = dict(deterministic_validation)

    manifest = Path(manifest_path)
    run_dir = manifest.parent.parent
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest.name
    if stem.endswith(".manifest.json"):
        stem = stem[: -len(".manifest.json")]
    else:
        stem = manifest.stem
    audit_path = audit_dir / f"{stem}.rejected.jsonl"

    with audit_path.open("w", encoding="utf-8") as handle:
        for item_id in rejected_ids:
            decision = dict(decisions[item_id])
            record = {
                "id": item_id,
                "spec": specs_by_id.get(item_id),
                "candidate": rows_by_id.get(item_id),
                "deterministic_validation": validation_by_id.get(item_id, deterministic_validation),
                "judge": {
                    "assessable": decision.get("assessable"),
                    "accepted": decision.get("judge_accepted"),
                    "reason": decision.get("judge_reason"),
                },
                "reviewer": {
                    "reviewed": decision.get("reviewed", False),
                    "agreed": decision.get("reviewer_agreed"),
                    "reason": decision.get("reviewer_reason"),
                },
                "final_rejection_stage": (
                    "judge" if not decision.get("judge_accepted", False) else "reviewer"
                ),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return audit_path


def materialize_llm_batch(
    *,
    specs: Iterable[Mapping[str, Any]],
    teacher_response: Mapping[str, Any],
    output_path: str | Path,
    manifest_path: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    metadata: Mapping[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
) -> DPOLLMBatchResult:
    """Validate a saved LLM DPO response and write JSONL plus local manifest."""
    provider = _validate_teacher_provider(teacher_provider)
    model = _require_non_empty_string(teacher_model, "teacher_model")
    run = _require_non_empty_string(generation_run, "generation_run")
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    if not validated_specs:
        raise ValueError("at least one DPO spec is required")

    expected_ids = [spec["id"] for spec in validated_specs]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("DPO specs contain duplicate id(s)")

    try:
        rows = _validate_candidate_rows(
            specs=validated_specs,
            teacher_response=teacher_response,
            holdout_registry=holdout_registry,
        )
    except (TypeError, ValueError) as exc:
        raise DPOBatchAcceptanceError(str(exc)) from exc
    try:
        deterministic_validation = evaluate_dpo_output_constraints(
            specs=validated_specs, rows=rows
        )
    except OutputConstraintError as exc:
        raise DPOBatchAcceptanceError(str(exc)) from exc

    dataset_path = Path(output_path)
    row_count = write_jsonl(rows, dataset_path)
    local_manifest_path = write_manifest(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        rows=rows,
        generation_run=run,
        metadata={
            "generation_mode": "llm_batch",
            "teacher_model": model,
            "teacher_provider": provider,
            "spec_count": len(validated_specs),
            **dict(metadata or {}),
            "deterministic_output_validation": deterministic_validation,
        },
    )

    return DPOLLMBatchResult(
        dataset_path=dataset_path,
        manifest_path=local_manifest_path,
        row_count=row_count,
        generation_run=run,
        teacher_model=model,
        teacher_provider=provider,
    )


def _write_accepted_dpo_rows(
    *, rows: list[dict[str, Any]], output_path: str | Path, manifest_path: str | Path,
    teacher_model: str, teacher_provider: str, generation_run: str,
    metadata: Mapping[str, Any], semantic_rejected_count: int,
) -> DPOLLMBatchResult:
    dataset_path = Path(output_path)
    row_count = write_jsonl(rows, dataset_path)
    local_manifest_path = write_manifest(
        manifest_path=manifest_path, dataset_path=dataset_path, rows=rows,
        generation_run=generation_run, metadata=metadata,
    )
    return DPOLLMBatchResult(
        dataset_path=dataset_path, manifest_path=local_manifest_path,
        row_count=row_count, generation_run=generation_run,
        teacher_model=teacher_model, teacher_provider=teacher_provider,
        semantic_rejected_count=semantic_rejected_count,
    )


def _generate_dpo_stage(
    *, backend: StructuredTeacherBackend, prompt: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    data, telemetry = call_plain_parsed(
        backend, prompt=prompt,
        system_prompt="Generate dataset content. Return one valid JSON object and no commentary.",
        parser=parse_json_object,
    )
    return data, telemetry


def _validate_candidate_rows(
    *, specs: list[dict[str, Any]], teacher_response: Mapping[str, Any],
    holdout_registry: HoldoutRegistry | None
) -> list[dict[str, Any]]:
    rows = validate_dpo_batch_response(
        teacher_response,
        expected_ids=[spec["id"] for spec in specs],
        expected_count=len(specs),
        expected_specs=specs,
    )
    _reject_holdout_matches(rows=rows, specs=specs, holdout_registry=holdout_registry)
    return rows


def _reject_holdout_matches(
    *,
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    holdout_registry: HoldoutRegistry | None,
) -> None:
    if holdout_registry is None:
        return
    specs_by_id = {spec["id"]: spec for spec in specs}
    for row in rows:
        spec = specs_by_id[row["id"]]
        holdout_key = spec.get("holdout_key")
        for message in row["prompt"]:
            if message["role"] == "user":
                holdout_registry.reject_if_holdout(
                    prompt=message["content"],
                    holdout_key=holdout_key,
                )


def _validate_teacher_provider(value: Any) -> str:
    provider = _require_non_empty_string(value, "teacher_provider").lower()
    if provider not in SUPPORTED_TEACHER_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_TEACHER_PROVIDERS))
        raise ValueError(f"Unsupported teacher_provider '{value}'. Supported providers: {supported}")
    return provider


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
