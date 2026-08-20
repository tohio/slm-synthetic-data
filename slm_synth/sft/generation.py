"""Non-network materialization for LLM-generated SFT batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from slm_synth.throughput_defaults import (
    DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
)
from slm_synth.sft.batches import (
    SFT_BATCH_RESPONSE_SCHEMA,
    render_sft_batch_prompt,
    validate_sft_batch_response,
    validate_sft_rows_against_specs,
)
from slm_synth.sft.io import write_jsonl
from slm_synth.sft.manifest import write_manifest
from slm_synth.sft.specs import validate_sft_spec
from slm_synth.sft.adjudication import adjudicate_sft_rows
from slm_synth.quality_adjudication import combine_telemetry
from slm_synth.taxonomy.holdouts import HoldoutRegistry

if TYPE_CHECKING:
    from slm_synth.llm import LLMBackend

SUPPORTED_TEACHER_PROVIDERS = frozenset({"openrouter"})


class StructuredTeacherBackend(Protocol):
    """Small protocol used by tests and live LLMBackend instances."""

    def generate_structured_object_with_metadata(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        ...


class SFTBatchAcceptanceError(ValueError):
    """Raised when a completed teacher response fails local SFT acceptance."""

    def __init__(self, message: str, *, telemetry: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.telemetry = dict(telemetry or {})


@dataclass(frozen=True)
class SFTLLMBatchResult:
    """Result of materializing one saved LLM SFT batch."""

    dataset_path: Path
    manifest_path: Path
    row_count: int
    generation_run: str
    teacher_model: str
    teacher_provider: str


def build_openrouter_backend(
    *,
    model: str,
    max_tokens: int,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = DEFAULT_OPENROUTER_ADAPTIVE_INITIAL_IN_FLIGHT,
    adaptive_initial_in_flight: int = 8,
    openrouter_routing_mode: str | None = None,
    openrouter_provider: str | None = None,
) -> "LLMBackend":
    """Create the supported production SFT teacher backend."""
    from slm_synth.llm import LLMBackend

    return LLMBackend(
        provider="openrouter",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        json_mode=True,
        request_timeout=request_timeout,
        max_request_retries=max_request_retries,
        max_retryable_request_attempts=max_retryable_request_attempts,
        retry_max_elapsed_seconds=retry_max_elapsed_seconds,
        adaptive_maximum_in_flight=adaptive_maximum_in_flight,
        adaptive_initial_in_flight=adaptive_initial_in_flight,
        openrouter_routing_mode=openrouter_routing_mode,
        openrouter_provider=openrouter_provider,
    )


def generate_teacher_batch_response(
    *,
    specs: Iterable[Mapping[str, Any]],
    backend: StructuredTeacherBackend,
) -> dict[str, Any]:
    """Call a teacher backend and return the strict SFT batch response object."""
    data, _telemetry = generate_teacher_batch_response_with_metadata(specs=specs, backend=backend)
    return data


def generate_teacher_batch_response_with_metadata(
    *,
    specs: Iterable[Mapping[str, Any]],
    backend: StructuredTeacherBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call a teacher backend and return the response object plus operational telemetry."""
    validated_specs = [validate_sft_spec(spec) for spec in specs]
    rendered_prompt = render_sft_batch_prompt(validated_specs)
    result = backend.generate_structured_object_with_metadata(
        prompt=rendered_prompt,
        schema=SFT_BATCH_RESPONSE_SCHEMA,
        schema_name="sft_batch",
    )
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("SFT teacher backend returned non-object data")
    telemetry = result.get("telemetry")
    return dict(data), dict(telemetry) if isinstance(telemetry, Mapping) else {}


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
    teacher_provider: str = "openrouter",
    temperature: float = 0.2,
    top_p: float = 0.95,
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
) -> SFTLLMBatchResult:
    """Generate one SFT batch with OpenRouter and write dataset + manifest."""
    provider = _validate_teacher_provider(teacher_provider)
    validated_specs = [validate_sft_spec(spec) for spec in specs]
    if not validated_specs:
        raise ValueError("at least one SFT spec is required")

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
    teacher_response, telemetry = generate_teacher_batch_response_with_metadata(
        specs=validated_specs,
        backend=active_backend,
    )
    try:
        rows = _validate_candidate_rows(
            specs=validated_specs,
            teacher_response=teacher_response,
            holdout_registry=holdout_registry,
        )
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
        decisions, adjudication_telemetry = adjudicate_sft_rows(
            specs=validated_specs, rows=rows, backend=active_adjudicator
        )
        combined_telemetry = combine_telemetry(telemetry, adjudication_telemetry)
        return materialize_llm_batch(
            specs=validated_specs,
            teacher_response=teacher_response,
            output_path=output_path,
            manifest_path=manifest_path,
            teacher_model=teacher_model,
            teacher_provider=provider,
            generation_run=generation_run,
            metadata={
                "generation_mode": "live_llm_batch",
                "spec_count": len(validated_specs),
                "llm_telemetry": combined_telemetry,
                "llm_stage_telemetry": {
                    "renderer": telemetry,
                    "adjudicator": adjudication_telemetry,
                },
                "adjudicator_model": adjudicator_model if adjudicator_model is not None else teacher_model,
                "quality_adjudication": decisions,
                **dict(metadata or {}),
            },
            holdout_registry=holdout_registry,
        )
    except SFTBatchAcceptanceError as exc:
        raise SFTBatchAcceptanceError(str(exc), telemetry=telemetry) from exc
    except (TypeError, ValueError) as exc:
        raise SFTBatchAcceptanceError(str(exc), telemetry=telemetry) from exc


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
) -> SFTLLMBatchResult:
    """Validate a saved LLM SFT response and write JSONL plus local manifest."""
    provider = _validate_teacher_provider(teacher_provider)
    model = _require_non_empty_string(teacher_model, "teacher_model")
    run = _require_non_empty_string(generation_run, "generation_run")
    validated_specs = [validate_sft_spec(spec) for spec in specs]
    if not validated_specs:
        raise ValueError("at least one SFT spec is required")

    expected_ids = [spec["id"] for spec in validated_specs]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("SFT specs contain duplicate id(s)")

    try:
        rows = _validate_candidate_rows(
            specs=validated_specs,
            teacher_response=teacher_response,
            holdout_registry=holdout_registry,
        )
    except (TypeError, ValueError) as exc:
        raise SFTBatchAcceptanceError(str(exc)) from exc

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
        },
    )

    return SFTLLMBatchResult(
        dataset_path=dataset_path,
        manifest_path=local_manifest_path,
        row_count=row_count,
        generation_run=run,
        teacher_model=model,
        teacher_provider=provider,
    )


def _validate_candidate_rows(
    *, specs: list[dict[str, Any]], teacher_response: Mapping[str, Any],
    holdout_registry: HoldoutRegistry | None
) -> list[dict[str, Any]]:
    expected_ids = [spec["id"] for spec in specs]
    rows = validate_sft_batch_response(
        teacher_response, expected_ids=expected_ids, expected_count=len(specs)
    )
    validate_sft_rows_against_specs(rows, specs)
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
        for message in row["messages"]:
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
