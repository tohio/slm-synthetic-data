"""Live teacher generation for response-distillation batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from slm_synth.distillation.batches import TEACHER_BATCH_RESPONSE_SCHEMA, render_teacher_batch_prompt
from slm_synth.distillation.prompt_quality import validate_prompt_preflight
from slm_synth.distillation.runs import DistillationRunResult, materialize_teacher_batch
from slm_synth.distillation.signals import validate_signal

if TYPE_CHECKING:
    from slm_synth.llm import LLMBackend


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
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 8,
) -> "LLMBackend":
    """Create the supported production teacher backend.

    OpenRouter is intentionally the only provider accepted by this repo.
    """
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
    )


def generate_teacher_batch_response(
    *,
    signal: str,
    prompt_records: Sequence[Mapping[str, Any]],
    backend: StructuredTeacherBackend,
) -> dict[str, Any]:
    """Call a teacher backend and return the strict teacher batch response object.

    The rendered prompt sends only id + prompt fields. Local signal/metadata remain
    local and are reattached only during materialization.
    """
    data, _telemetry = generate_teacher_batch_response_with_metadata(
        signal=signal,
        prompt_records=prompt_records,
        backend=backend,
    )
    return data


def generate_teacher_batch_response_with_metadata(
    *,
    signal: str,
    prompt_records: Sequence[Mapping[str, Any]],
    backend: StructuredTeacherBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call a teacher backend and return the response object plus operational telemetry."""
    normalized_signal = validate_signal(signal)
    rendered_prompt = render_teacher_batch_prompt(signal=normalized_signal, prompt_records=prompt_records)
    result = backend.generate_structured_object_with_metadata(
        prompt=rendered_prompt,
        schema=TEACHER_BATCH_RESPONSE_SCHEMA,
        schema_name=f"{normalized_signal}_distillation_batch",
    )
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("teacher backend returned non-object data")
    telemetry = result.get("telemetry")
    return dict(data), dict(telemetry) if isinstance(telemetry, Mapping) else {}


def generate_and_materialize_signal_batch(
    *,
    signal: str,
    prompt_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    manifest_dir: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    token_target: str | int | None = None,
    dataset_filename: str | None = None,
    manifest_filename: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 8,
    backend: StructuredTeacherBackend | None = None,
) -> DistillationRunResult:
    """Generate one signal batch with OpenRouter and write dataset + manifest."""
    normalized_signal = validate_signal(signal)
    prompt_records = list(prompt_records)
    prompt_preflight = validate_prompt_preflight(
        prompt_records,
        require_unique_prompt_text=False,
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
    )
    teacher_response, telemetry = generate_teacher_batch_response_with_metadata(
        signal=normalized_signal,
        prompt_records=prompt_records,
        backend=active_backend,
    )
    return materialize_teacher_batch(
        signal=normalized_signal,
        prompt_records=prompt_records,
        teacher_response=teacher_response,
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        teacher_model=teacher_model,
        teacher_provider="openrouter",
        generation_run=generation_run,
        token_target=token_target,
        dataset_filename=dataset_filename,
        manifest_filename=manifest_filename,
        metadata={
            "prompt_count": len(prompt_records),
            "prompt_preflight": prompt_preflight.to_dict(),
            "llm_telemetry": telemetry,
        },
    )
