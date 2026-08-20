"""Independent semantic quality gate for live SFT candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from slm_synth.quality_adjudication import (
    QUALITY_ADJUDICATION_SCHEMA,
    render_quality_adjudication_prompt,
    validate_quality_adjudication,
)
from slm_synth.sft.specs import teacher_visible_sft_spec, validate_sft_spec


class StructuredAdjudicatorBackend(Protocol):
    def generate_structured_object_with_metadata(
        self, *, prompt: str, schema: dict[str, Any], schema_name: str
    ) -> dict[str, Any]: ...


def adjudicate_sft_rows(
    *, specs: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]],
    backend: StructuredAdjudicatorBackend
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validated_specs = [validate_sft_spec(spec) for spec in specs]
    rendered_rows = [dict(row) for row in rows]
    result = backend.generate_structured_object_with_metadata(
        prompt=render_quality_adjudication_prompt(
            dataset_type="SFT", specs=[teacher_visible_sft_spec(spec) for spec in validated_specs], rows=rendered_rows
        ),
        schema=QUALITY_ADJUDICATION_SCHEMA,
        schema_name="sft_quality_adjudication",
    )
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("SFT adjudicator returned non-object data")
    decisions = validate_quality_adjudication(data, specs=validated_specs)
    telemetry = result.get("telemetry")
    return decisions, dict(telemetry) if isinstance(telemetry, Mapping) else {}
