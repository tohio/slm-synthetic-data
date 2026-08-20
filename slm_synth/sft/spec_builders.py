"""Finite source-spec catalog for generic SFT generation."""

from __future__ import annotations

from typing import Any

from slm_synth.sft.source_catalog import SFT_SOURCE_CATALOG
from slm_synth.sft.specs import require_unique_sft_sources, validate_sft_spec
from slm_synth.taxonomy import TASK_FAMILIES, validate_task_family

SFT_SPEC_FAMILIES = TASK_FAMILIES
SFT_SPEC_CAPACITIES = {family: len(sources) for family, sources in SFT_SOURCE_CATALOG.items()}


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    family = validate_task_family(family)
    validate_spec_range(family=family, count=count, start_index=start_index)
    specs = [validate_sft_spec(_build_spec(family, index)) for index in range(start_index, start_index + count)]
    require_unique_sft_sources(specs)
    return specs


def build_complete_inventory() -> list[dict[str, Any]]:
    """Build every declared SFT candidate in stable taxonomy order."""
    return [spec for family in sorted(SFT_SPEC_FAMILIES) for spec in build_specs(family=family, count=unique_capacity(family))]


def unique_capacity(family: str) -> int:
    return SFT_SPEC_CAPACITIES[validate_task_family(family)]


def validate_spec_range(*, family: str, count: int, start_index: int = 1) -> None:
    family = validate_task_family(family)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 1:
        raise ValueError("start_index must be a positive integer")
    end = start_index + count - 1
    capacity = SFT_SPEC_CAPACITIES[family]
    if end > capacity:
        raise ValueError(f"SFT task family {family!r} requested {start_index}..{end}; finite source capacity is {capacity}")


def _build_spec(family: str, index: int) -> dict[str, Any]:
    source = SFT_SOURCE_CATALOG[family][index - 1]
    context_mode = source["metadata"]["context_mode"]
    constraints = [
        "The public conversation must include every piece of source material needed to understand and answer the task; the assistant may not rely on hidden spec variables.",
    ]
    if context_mode == "self_contained":
        constraints.append(
            "Honor every supplied fact and explicit requirement. When the task requests creative, conversational, planning, or brainstorming content, appropriate invented details are allowed unless the brief prohibits them."
        )
    else:
        constraints.append(
            "Treat the supplied passage, document, transcript, code, or other context as the factual source of truth. Ordinary direct inference is allowed, but unsupported factual claims are not."
        )
    result: dict[str, Any] = {
        "id": f"sft_{family}_{index:06d}",
        "instruction": source["instruction"],
        "metadata": {"task_family": family, **dict(source["metadata"])},
        "variables": dict(source["variables"]),
        "constraints": constraints,
    }
    if "holdout_key" in source:
        result["holdout_key"] = dict(source["holdout_key"])
    if "output_constraints" in source:
        result["output_constraints"] = dict(source["output_constraints"])
    return result
