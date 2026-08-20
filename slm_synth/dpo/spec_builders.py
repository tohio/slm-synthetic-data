"""Finite, independently authored source specs for generic DPO generation."""

from __future__ import annotations

from typing import Any

from slm_synth.dpo.source_catalog import DPO_SOURCE_CATALOG
from slm_synth.dpo.specs import require_unique_dpo_sources, validate_dpo_spec
from slm_synth.taxonomy import PREFERENCE_DIMENSIONS, validate_preference_dimension

DPO_PREFERENCE_DIMENSIONS = PREFERENCE_DIMENSIONS
DPO_SPEC_CAPACITIES = {dimension: len(sources) for dimension, sources in DPO_SOURCE_CATALOG.items()}


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    dimension = validate_preference_dimension(family)
    validate_spec_range(family=dimension, count=count, start_index=start_index)
    specs = [validate_dpo_spec(_build_spec(dimension, index)) for index in range(start_index, start_index + count)]
    require_unique_dpo_sources(specs)
    return specs


def build_complete_inventory() -> list[dict[str, Any]]:
    """Build every declared DPO candidate in stable taxonomy order."""
    return [spec for dimension in sorted(DPO_PREFERENCE_DIMENSIONS) for spec in build_specs(family=dimension, count=unique_capacity(dimension))]


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


def _build_spec(dimension: str, index: int) -> dict[str, Any]:
    source = DPO_SOURCE_CATALOG[dimension][index - 1]
    task_family = source["metadata"]["task_family"]
    result: dict[str, Any] = {
        "id": f"dpo_{dimension}_{task_family}_{index:06d}",
        "instruction": source["instruction"],
        "metadata": {**dict(source["metadata"]), "preference_dimension": dimension, "failure_mode": source["failure_mode"]},
        "variables": dict(source["variables"]),
        "constraints": [
            "The chosen response must be correct and materially better, not merely differently worded.",
            "The rejected response must remain plausible while clearly demonstrating metadata.failure_mode.",
        ],
    }
    if "holdout_key" in source:
        result["holdout_key"] = dict(source["holdout_key"])
    return result
