"""SFT spec materialization from candidate plans."""

from __future__ import annotations

from typing import Any

from slm_synth.sft.planning import (
    DEFAULT_SFT_SPEC_PLANNER,
    SFTCandidatePlan,
    SFTSpecPlanner,
)
from slm_synth.sft.specs import require_unique_sft_sources, validate_sft_spec
from slm_synth.taxonomy import TASK_FAMILIES, validate_task_family

SFT_SPEC_FAMILIES = TASK_FAMILIES
# Compatibility snapshot for callers/tests that inspect the transition planner's
# finite capacity. Production-scale planners should query ``unique_capacity``.
SFT_SPEC_CAPACITIES = {
    family: DEFAULT_SFT_SPEC_PLANNER.capacity(family)
    for family in SFT_SPEC_FAMILIES
}


def build_specs(
    *,
    family: str,
    count: int,
    start_index: int = 1,
    planner: SFTSpecPlanner = DEFAULT_SFT_SPEC_PLANNER,
) -> list[dict[str, Any]]:
    family = validate_task_family(family)
    plans = planner.plan(
        family=family, count=count, start_index=start_index
    )
    specs = [validate_sft_spec(_build_spec(plan)) for plan in plans]
    require_unique_sft_sources(specs)
    return specs


def build_complete_inventory(
    *, planner: SFTSpecPlanner = DEFAULT_SFT_SPEC_PLANNER
) -> list[dict[str, Any]]:
    """Build every candidate exposed by the selected planner."""
    return [
        spec
        for family in sorted(SFT_SPEC_FAMILIES)
        for spec in build_specs(
            family=family,
            count=unique_capacity(family, planner=planner),
            planner=planner,
        )
    ]


def unique_capacity(
    family: str, *, planner: SFTSpecPlanner = DEFAULT_SFT_SPEC_PLANNER
) -> int:
    return planner.capacity(validate_task_family(family))


def validate_spec_range(
    *,
    family: str,
    count: int,
    start_index: int = 1,
    planner: SFTSpecPlanner = DEFAULT_SFT_SPEC_PLANNER,
) -> None:
    """Validate a range through the selected planner without materializing specs."""
    planner.plan(
        family=validate_task_family(family),
        count=count,
        start_index=start_index,
    )


def _build_spec(plan: SFTCandidatePlan) -> dict[str, Any]:
    family = plan.family
    index = plan.candidate_index
    source = plan.archetype
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
    constraints.extend(source.get("quality_requirements", ()))
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
    if "public_prompt_requirements" in source:
        result["public_prompt_requirements"] = list(
            source["public_prompt_requirements"]
        )
    return result
