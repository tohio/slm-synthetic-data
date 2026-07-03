"""Non-network materialization for LLM-generated DPO batches."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from slm_synth.dpo.batches import (
    DPO_BATCH_RESPONSE_SCHEMA,
    render_dpo_batch_prompt,
    validate_dpo_batch_response,
)
from slm_synth.dpo.io import write_jsonl
from slm_synth.dpo.manifest import write_manifest
from slm_synth.dpo.specs import validate_dpo_spec
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


@dataclass(frozen=True)
class DPOLLMBatchResult:
    """Result of materializing one saved LLM DPO batch."""

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
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 8,
) -> "LLMBackend":
    """Create the supported production DPO teacher backend."""
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


def read_specs_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate DPO task specs from JSONL."""
    input_path = Path(path)
    specs: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid DPO spec JSONL in {input_path} at line {line_number}: {exc}") from exc
        specs.append(validate_dpo_spec(value))
    return specs


def read_teacher_response_json(path: str | Path) -> dict[str, Any]:
    """Read a saved teacher batch response JSON object."""
    input_path = Path(path)
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid DPO teacher response JSON in {input_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("DPO teacher response JSON must be an object")
    return value


def generate_teacher_batch_response(
    *,
    specs: Iterable[Mapping[str, Any]],
    backend: StructuredTeacherBackend,
) -> dict[str, Any]:
    """Call a teacher backend and return the strict DPO batch response object."""
    data, _telemetry = generate_teacher_batch_response_with_metadata(specs=specs, backend=backend)
    return data


def generate_teacher_batch_response_with_metadata(
    *,
    specs: Iterable[Mapping[str, Any]],
    backend: StructuredTeacherBackend,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call a teacher backend and return the response object plus operational telemetry."""
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    rendered_prompt = render_dpo_batch_prompt(validated_specs)
    result = backend.generate_structured_object_with_metadata(
        prompt=rendered_prompt,
        schema=DPO_BATCH_RESPONSE_SCHEMA,
        schema_name="dpo_batch",
    )
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("DPO teacher backend returned non-object data")
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
    teacher_provider: str = "openrouter",
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 8,
    metadata: Mapping[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
    backend: StructuredTeacherBackend | None = None,
) -> DPOLLMBatchResult:
    """Generate one DPO batch with OpenRouter and write dataset + manifest."""
    provider = _validate_teacher_provider(teacher_provider)
    validated_specs = [validate_dpo_spec(spec) for spec in specs]
    if not validated_specs:
        raise ValueError("at least one DPO spec is required")

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
        specs=validated_specs,
        backend=active_backend,
    )
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
            "llm_telemetry": telemetry,
            **dict(metadata or {}),
        },
        holdout_registry=holdout_registry,
    )


def generate_llm_batch_from_files(
    *,
    specs_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    teacher_model: str,
    generation_run: str,
    max_tokens: int,
    teacher_provider: str = "openrouter",
    temperature: float = 0.2,
    top_p: float = 0.95,
    request_timeout: float | None = None,
    max_request_retries: int = 3,
    max_retryable_request_attempts: int = 20,
    retry_max_elapsed_seconds: float = 1800.0,
    adaptive_maximum_in_flight: int = 1,
    adaptive_initial_in_flight: int = 8,
    metadata: Mapping[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
    backend: StructuredTeacherBackend | None = None,
) -> DPOLLMBatchResult:
    """Read DPO specs, call OpenRouter, then materialize the generated batch."""
    return generate_llm_batch(
        specs=read_specs_jsonl(specs_path),
        output_path=output_path,
        manifest_path=manifest_path,
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
        metadata=metadata,
        holdout_registry=holdout_registry,
        backend=backend,
    )


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

    rows = validate_dpo_batch_response(
        teacher_response,
        expected_ids=expected_ids,
        expected_count=len(validated_specs),
    )
    _reject_holdout_matches(rows=rows, specs=validated_specs, holdout_registry=holdout_registry)

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

    return DPOLLMBatchResult(
        dataset_path=dataset_path,
        manifest_path=local_manifest_path,
        row_count=row_count,
        generation_run=run,
        teacher_model=model,
        teacher_provider=provider,
    )


def materialize_llm_batch_from_files(
    *,
    specs_path: str | Path,
    teacher_response_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    teacher_model: str,
    generation_run: str,
    teacher_provider: str = "openrouter",
    metadata: Mapping[str, Any] | None = None,
    holdout_registry: HoldoutRegistry | None = None,
) -> DPOLLMBatchResult:
    """Read DPO specs and saved teacher JSON, then materialize the batch."""
    return materialize_llm_batch(
        specs=read_specs_jsonl(specs_path),
        teacher_response=read_teacher_response_json(teacher_response_path),
        output_path=output_path,
        manifest_path=manifest_path,
        teacher_model=teacher_model,
        teacher_provider=teacher_provider,
        generation_run=generation_run,
        metadata=metadata,
        holdout_registry=holdout_registry,
    )


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
