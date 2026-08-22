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

# Only constraints that describe stable response structure are inherited by
# derived candidates. Seed-specific names, dates, passages, rubric text, and
# other literal content must be instantiated anew in the public task.
_DERIVED_STRUCTURAL_OUTPUT_CONSTRAINT_KEYS = frozenset(
    {
        "min_words",
        "max_words",
        "exact_list_items",
        "required_headings",
        "exact_json_keys",
        "exact_nonempty_lines",
        "forbidden_terms",
    }
)


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
    variables: dict[str, Any]
    instruction = source["instruction"]
    if plan.is_derived:
        profile = dict(plan.derivation_profile or {})
        instruction = _materialize_derived_instruction(
            family=family,
            archetype_instruction=source["instruction"],
            profile=profile,
        )
        # Derived candidates must not carry seed facts as hidden obligations.
        # The capability anchor remains visible in the instruction, while the
        # new task's concrete facts/rubric/source material are created afresh
        # and surfaced in the public conversation.
        variables = {"derivation_profile": profile}
        constraints.extend(
            [
                "Create a genuinely new concrete task instance that exercises the same capability as the capability anchor; do not merely rename entities or swap numbers.",
                "Use the derivation profile as substantive planning guidance while preserving the archetype's task family, interaction mode, output mode, safety posture, and structural response constraints.",
                "Do not copy literal proper nouns, source passages, code, quantities, dates, or scenario details from the capability anchor unless they are essential structural invariants.",
                "Any rubric, label set, source facts, required terms, or other content-specific requirements needed to answer the new task must appear explicitly in the public user conversation.",
                "Do not expose the derivation profile as metadata or commentary in the public conversation.",
            ]
        )
    else:
        variables = dict(source["variables"])
        constraints.extend(source.get("quality_requirements", ()))

    result: dict[str, Any] = {
        "id": f"sft_{family}_{index:06d}",
        "instruction": instruction,
        "metadata": {"task_family": family, **dict(source["metadata"])},
        "variables": variables,
        "constraints": constraints,
    }
    # Structured holdout keys describe the literal seed instance. Derived
    # candidates create new facts/content, so exact prompt-fingerprint holdout
    # protection remains active but the seed's structured key is not reused.
    if not plan.is_derived and "holdout_key" in source:
        result["holdout_key"] = dict(source["holdout_key"])
    if plan.is_derived:
        derived_output_constraints = _derived_structural_output_constraints(
            source.get("output_constraints")
        )
        if derived_output_constraints:
            result["output_constraints"] = derived_output_constraints
    else:
        if "output_constraints" in source:
            result["output_constraints"] = dict(source["output_constraints"])
        if "public_prompt_requirements" in source:
            result["public_prompt_requirements"] = list(
                source["public_prompt_requirements"]
            )
    return result


def _derived_structural_output_constraints(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in _DERIVED_STRUCTURAL_OUTPUT_CONSTRAINT_KEYS
    }


def _materialize_derived_instruction(
    *,
    family: str,
    archetype_instruction: str,
    profile: dict[str, str],
) -> str:
    """Return a teacher-visible brief that is distinct from the seed task.

    The archetype remains a capability anchor only. Derived candidates must
    expose their variation profile in the top-level instruction so planning
    capacity corresponds to distinct teacher-visible generation briefs rather
    than new IDs attached to the same seed instruction.
    """
    family_label = family.replace("_", " ")
    return (
        f"Create a fresh, concrete {family_label} training task and produce the "
        "assistant response for that new task. Do not answer or reproduce the "
        "archetype task literally; use it only as a capability anchor. "
        f"Capability anchor: {archetype_instruction} "
        f"New task context: {profile['context_lens']}. "
        f"Required variation: {profile['variation_lens']}. "
        f"Evidence requirement: {profile['evidence_lens']}. "
        "The public conversation must contain the complete newly instantiated "
        "task, including every fact or source needed to answer it. Preserve the "
        "archetype's capability, interaction mode, output mode, safety posture, "
        "and applicable output constraints, while changing the concrete task "
        "content substantially enough that it is not a renamed or number-swapped "
        "version of the archetype."
    )
