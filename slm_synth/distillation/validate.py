"""Validation and merge helpers for response-distillation teacher outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.distillation.prompts import validate_prompt_record
from slm_synth.distillation.schema import validate_public_row

TEACHER_OUTPUT_FIELDS = frozenset({"id", "reasoning", "response"})


def validate_teacher_output(output: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one teacher output record.

    The teacher only returns an id for matching, reasoning fixed to null, and
    the final response. Local code owns prompts, signal names, and metadata.
    """
    if not isinstance(output, Mapping):
        raise TypeError("teacher output must be a mapping")

    keys = set(output)
    missing = TEACHER_OUTPUT_FIELDS - keys
    if missing:
        raise ValueError(f"teacher output missing required field(s): {sorted(missing)}")

    unexpected = keys - TEACHER_OUTPUT_FIELDS
    if unexpected:
        raise ValueError(f"teacher output contains unexpected field(s): {sorted(unexpected)}")

    record_id = output["id"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("teacher output field 'id' must be a non-empty string")

    response = output["response"]
    if not isinstance(response, str) or not response.strip():
        raise ValueError("teacher output field 'response' must be a non-empty string")

    reasoning = output["reasoning"]
    if reasoning is not None:
        raise ValueError("teacher output field 'reasoning' must be null")

    return {"id": record_id, "reasoning": reasoning, "response": response}


def _index_teacher_outputs(outputs: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()

    for output in outputs:
        validated = validate_teacher_output(output)
        record_id = validated["id"]
        if record_id in indexed:
            duplicate_ids.add(record_id)
        indexed[record_id] = validated

    if duplicate_ids:
        raise ValueError(f"teacher output contains duplicate id(s): {sorted(duplicate_ids)}")
    return indexed


def merge_teacher_outputs(
    prompt_records: Iterable[Mapping[str, Any]],
    teacher_outputs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge local prompt records with validated teacher outputs.

    The returned rows are public training rows only. Signal, metadata, teacher
    model, provider, retry, cost, and run details stay outside this function.
    """
    prompts = [validate_prompt_record(record) for record in prompt_records]
    prompt_ids = [record["id"] for record in prompts]
    duplicate_prompt_ids = sorted({record_id for record_id in prompt_ids if prompt_ids.count(record_id) > 1})
    if duplicate_prompt_ids:
        raise ValueError(f"prompt records contain duplicate id(s): {duplicate_prompt_ids}")

    indexed_outputs = _index_teacher_outputs(teacher_outputs)
    expected_ids = set(prompt_ids)
    output_ids = set(indexed_outputs)

    missing = expected_ids - output_ids
    if missing:
        raise ValueError(f"teacher output missing id(s): {sorted(missing)}")

    unexpected = output_ids - expected_ids
    if unexpected:
        raise ValueError(f"teacher output contains unexpected id(s): {sorted(unexpected)}")

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        output = indexed_outputs[prompt["id"]]
        rows.append(
            validate_public_row(
                {
                    "id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "reasoning": None,
                    "response": output["response"],
                }
            )
        )
    return rows
