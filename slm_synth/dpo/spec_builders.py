"""Finite source-spec catalog for generic DPO preference generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_synth.dpo.specs import require_unique_dpo_sources, validate_dpo_spec
from slm_synth.sft.spec_builders import build_specs as build_sft_specs
from slm_synth.sft.spec_builders import unique_capacity as sft_unique_capacity
from slm_synth.taxonomy import PREFERENCE_DIMENSIONS, validate_preference_dimension

DPO_PREFERENCE_DIMENSIONS = PREFERENCE_DIMENSIONS

_DIMENSION_PLAN: dict[str, tuple[tuple[str, ...], str]] = {
    "helpfulness_and_completeness": (("planning_brainstorming_recommendations", "everyday_conversation", "grounded_qa_and_reading"), "incomplete_response"),
    "factual_accuracy": (("applied_math_and_reasoning", "grounded_qa_and_reading", "programming"), "unsupported_claim"),
    "instruction_adherence": (("classification_and_extraction", "rewriting_and_editing", "creative_writing"), "instruction_violation"),
    "appropriate_detail": (("summarization", "everyday_conversation", "planning_brainstorming_recommendations"), "excessive_detail"),
    "organization": (("planning_brainstorming_recommendations", "summarization", "programming"), "poor_organization"),
    "style_and_tone": (("rewriting_and_editing", "everyday_conversation", "creative_writing"), "tone_mismatch"),
    "tool_call_correctness": (("programming", "classification_and_extraction", "planning_brainstorming_recommendations"), "incorrect_tool_call"),
    "groundedness": (("grounded_qa_and_reading", "summarization", "classification_and_extraction"), "ungrounded_response"),
    "safe_refusal_calibration": (("safety_uncertainty_and_refusal", "everyday_conversation", "grounded_qa_and_reading"), "over_refusal"),
    "code_correctness": (("programming", "applied_math_and_reasoning", "classification_and_extraction"), "code_logic_error"),
}

DPO_SPEC_CAPACITIES = {
    dimension: sum(sft_unique_capacity(task_family) for task_family in task_families)
    for dimension, (task_families, _) in _DIMENSION_PLAN.items()
}


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    dimension = validate_preference_dimension(family)
    validate_spec_range(family=dimension, count=count, start_index=start_index)
    task_families, failure_mode = _DIMENSION_PLAN[dimension]
    source_specs = [_source_spec(task_families, index) for index in range(start_index, start_index + count)]
    specs = [
        validate_dpo_spec(_from_sft_spec(spec, dimension=dimension, failure_mode=failure_mode))
        for spec in source_specs
    ]
    require_unique_dpo_sources(specs)
    return specs


def unique_capacity(family: str) -> int:
    return DPO_SPEC_CAPACITIES[validate_preference_dimension(family)]


def validate_spec_range(*, family: str, count: int, start_index: int = 1) -> None:
    dimension = validate_preference_dimension(family)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 1:
        raise ValueError("start_index must be a positive integer")
    end = start_index + count - 1
    capacity = DPO_SPEC_CAPACITIES[dimension]
    if end > capacity:
        raise ValueError(f"DPO preference dimension {dimension!r} requested {start_index}..{end}; finite source capacity is {capacity}")


def write_specs_jsonl(specs: list[dict[str, Any]], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(validate_dpo_spec(spec), ensure_ascii=False) + "\n")
    return len(specs)


def _source_spec(task_families: tuple[str, ...], index: int) -> dict[str, Any]:
    offset = index - 1
    for task_family in task_families:
        capacity = sft_unique_capacity(task_family)
        if offset < capacity:
            return build_sft_specs(family=task_family, count=1, start_index=offset + 1)[0]
        offset -= capacity
    raise IndexError(index)


def build_and_write_specs(*, family: str, count: int, output_path: str | Path, start_index: int = 1) -> int:
    return write_specs_jsonl(build_specs(family=family, count=count, start_index=start_index), output_path)


def _from_sft_spec(spec: dict[str, Any], *, dimension: str, failure_mode: str) -> dict[str, Any]:
    metadata = dict(spec["metadata"])
    metadata.update({"preference_dimension": dimension, "failure_mode": failure_mode})
    result = {
        "id": spec["id"].replace("sft_", f"dpo_{dimension}_", 1),
        "instruction": (
            spec["instruction"]
            + f" Produce a clearly preferred response and a plausible rejected response that fails on {dimension}."
        ),
        "metadata": metadata,
        "variables": dict(spec.get("variables", {})),
        "constraints": [
            *list(spec.get("constraints", [])),
            "The chosen response must be correct and materially better, not merely differently worded.",
            "The rejected response must be plausible and demonstrate metadata.failure_mode.",
        ],
    }
    if "holdout_key" in spec:
        result["holdout_key"] = dict(spec["holdout_key"])
    return result
