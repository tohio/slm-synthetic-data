"""Candidate-spec planning primitives for generic SFT generation.

Task archetypes define coverage; planners turn those archetypes into stable
candidate opportunities. The scalable planner keeps the first one-for-one
archetype candidates unchanged, then derives additional teacher-visible plans
from deterministic, semantically meaningful variation profiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from slm_synth.sft.source_catalog import SFT_ARCHETYPE_CATALOG
from slm_synth.taxonomy import TASK_FAMILIES, validate_task_family


# Fifteen materially different context lenses per task family. These are
# planning guidance, not literal templates: the generator must instantiate a
# fresh concrete task while preserving the selected archetype's capability.
_FAMILY_CONTEXT_LENSES: dict[str, tuple[str, ...]] = {
    "everyday_conversation": (
        "household coordination", "friendship expectations", "neighbor relations",
        "workplace peer communication", "family logistics", "community volunteering",
        "school or study coordination", "travel planning", "shared expenses",
        "event attendance", "communication repair", "time-bound favors",
        "social boundaries", "uncertain scheduling", "low-stakes conflict resolution",
    ),
    "rewriting_and_editing": (
        "customer communication", "technical status update", "research summary",
        "policy or procedure text", "event announcement", "executive briefing",
        "project handoff", "support response", "grant or proposal prose",
        "release notes", "accessibility information", "incident update",
        "instructional material", "internal memo", "public-facing notice",
    ),
    "summarization": (
        "operations report", "research note", "meeting record", "policy notice",
        "incident timeline", "contract excerpt", "product update", "experiment report",
        "multi-source briefing", "support case history", "technical document",
        "project status", "financial narrative", "educational passage", "service announcement",
    ),
    "classification_and_extraction": (
        "customer support", "billing and invoices", "calendar and scheduling",
        "operations alerts", "shipping and logistics", "service tickets", "procurement",
        "application intake", "inventory records", "incident triage", "contact records",
        "document metadata", "quality reports", "compliance forms", "work-order requests",
    ),
    "grounded_qa_and_reading": (
        "product notices", "policy excerpts", "service plans", "research passages",
        "meeting notes", "technical documentation", "travel information", "event notices",
        "procedural guidance", "contract snippets", "support documentation",
        "educational text", "operations bulletins", "comparison tables", "multi-document evidence",
    ),
    "planning_brainstorming_recommendations": (
        "study planning", "project sequencing", "household logistics", "event planning",
        "travel planning", "team coordination", "resource allocation", "maintenance scheduling",
        "content planning", "learning roadmap", "meeting preparation", "volunteer coordination",
        "personal organization", "work backlog", "small-business operations",
    ),
    "programming": (
        "data validation", "API pagination", "state management", "asynchronous workflows",
        "file processing", "database querying", "log processing", "scheduling systems",
        "inventory software", "report generation", "configuration parsing", "batch processing",
        "event handling", "data transformation", "command-line automation",
    ),
    "applied_math_and_reasoning": (
        "inventory and quantities", "rates and work", "pricing and break-even",
        "measurement and units", "probability and screening", "scheduling constraints",
        "mixtures and proportions", "capacity allocation", "piecewise costs",
        "uncertainty bounds", "time and distance", "resource planning", "weighted averages",
        "combinatorial ordering", "percentage change",
    ),
    "creative_writing": (
        "domestic interior", "workshop or studio", "public transit", "coastal setting",
        "rural setting", "urban nighttime", "museum or archive", "school or library",
        "market or shop", "industrial setting", "garden or park", "temporary shelter",
        "community venue", "travel stop", "weather-exposed setting",
    ),
    "safety_uncertainty_and_refusal": (
        "medical uncertainty", "privacy and confidential information", "account security",
        "physical access", "legal uncertainty", "financial uncertainty", "location privacy",
        "identity ambiguity", "benign household safety", "workplace authorization",
        "credential recovery", "high-stakes factual uncertainty", "personal data requests",
        "professional-boundary advice", "unsafe operational requests",
    ),
}

_VARIATION_LENSES: tuple[str, ...] = (
    "use different concrete actors, entities, and facts from the archetype seed",
    "introduce one meaningful edge case that changes how the task must be handled",
    "include two interacting constraints that both matter to a correct response",
    "include one uncertainty that must be acknowledged without blocking a useful answer",
    "include one explicit negative constraint that prevents a tempting but wrong response",
    "require the response to reconcile two pieces of evidence or requirements",
    "include a boundary condition that must be handled correctly",
    "require a concise verification or self-check appropriate to the task",
    "include one plausible distractor detail that must not change the correct result",
    "require a clear distinction between confirmed information and an unresolved detail",
)

_EVIDENCE_LENSES: tuple[str, ...] = (
    "ground the task in a fresh concrete scenario with all necessary facts visible",
    "use a different source shape or factual arrangement while preserving the capability",
    "make the decisive requirement explicit in the public user message",
    "include a concrete time, quantity, or named role when naturally applicable",
    "include one explicit exception or special case when naturally applicable",
    "make the expected output structure unambiguous in the public task",
    "ensure the public conversation contains enough evidence to reject unsupported guessing",
    "use concrete source material rather than generic placeholder text",
    "make one important relationship between facts require direct inference",
    "ensure the answer can be checked against facts present in the public conversation",
)


@dataclass(frozen=True)
class SFTCandidatePlan:
    """One planned candidate before its concrete SFT spec is materialized."""

    family: str
    candidate_index: int
    archetype_index: int
    archetype_key: str
    archetype: Mapping[str, Any]
    derivation_profile: Mapping[str, str] | None = None

    @property
    def is_derived(self) -> bool:
        return self.derivation_profile is not None


class SFTSpecPlanner(Protocol):
    """Planning contract consumed by the SFT spec builder."""

    def capacity(self, family: str) -> int:
        """Return the number of currently plannable candidates for ``family``."""

    def plan(
        self, *, family: str, count: int, start_index: int = 1
    ) -> list[SFTCandidatePlan]:
        """Return stable candidate plans for the requested range."""


class FiniteArchetypePlanner:
    """One-archetype-per-candidate planner retained for compatibility/debugging."""

    def __init__(
        self, archetypes_by_family: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> None:
        self._archetypes_by_family = _validate_archetype_catalog(
            archetypes_by_family
        )

    def capacity(self, family: str) -> int:
        family = validate_task_family(family)
        return len(self._archetypes_by_family[family])

    def plan(
        self, *, family: str, count: int, start_index: int = 1
    ) -> list[SFTCandidatePlan]:
        family = validate_task_family(family)
        _validate_requested_range(
            family=family,
            count=count,
            start_index=start_index,
            capacity=self.capacity(family),
        )
        archetypes = self._archetypes_by_family[family]
        return [
            SFTCandidatePlan(
                family=family,
                candidate_index=index,
                archetype_index=index,
                archetype_key=str(archetypes[index - 1]["source_key"]),
                archetype=archetypes[index - 1],
            )
            for index in range(start_index, start_index + count)
        ]


class ScalableArchetypePlanner:
    """Derive large numbers of stable candidate opportunities from archetypes.

    The first ``len(archetypes)`` plans are exactly the finite archetype plans,
    preserving smoke-test behavior. Later plans pair an archetype with one of
    1,500 semantically meaningful variation profiles (15 family contexts × 10
    variation lenses × 10 evidence lenses). With ten archetypes this exposes
    15,010 stable candidate opportunities per family, or 150,100 repo-wide.

    Capacity is planning headroom, not a promise that every generated row will
    survive quality, holdout, or final deduplication. Those remain downstream
    acceptance gates.
    """

    def __init__(
        self, archetypes_by_family: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> None:
        self._archetypes_by_family = _validate_archetype_catalog(
            archetypes_by_family
        )
        missing_lenses = set(TASK_FAMILIES) - set(_FAMILY_CONTEXT_LENSES)
        if missing_lenses:
            raise ValueError(
                f"SFT scalable planner missing family context lenses: {sorted(missing_lenses)}"
            )

    def capacity(self, family: str) -> int:
        family = validate_task_family(family)
        archetype_count = len(self._archetypes_by_family[family])
        profile_count = (
            len(_FAMILY_CONTEXT_LENSES[family])
            * len(_VARIATION_LENSES)
            * len(_EVIDENCE_LENSES)
        )
        return archetype_count + archetype_count * profile_count

    def plan(
        self, *, family: str, count: int, start_index: int = 1
    ) -> list[SFTCandidatePlan]:
        family = validate_task_family(family)
        _validate_requested_range(
            family=family,
            count=count,
            start_index=start_index,
            capacity=self.capacity(family),
        )
        return [
            self._plan_one(family=family, candidate_index=index)
            for index in range(start_index, start_index + count)
        ]

    def _plan_one(self, *, family: str, candidate_index: int) -> SFTCandidatePlan:
        archetypes = self._archetypes_by_family[family]
        archetype_count = len(archetypes)
        if candidate_index <= archetype_count:
            archetype = archetypes[candidate_index - 1]
            return SFTCandidatePlan(
                family=family,
                candidate_index=candidate_index,
                archetype_index=candidate_index,
                archetype_key=str(archetype["source_key"]),
                archetype=archetype,
            )

        derived_offset = candidate_index - archetype_count - 1
        archetype_position = derived_offset % archetype_count
        profile_index = derived_offset // archetype_count
        contexts = _FAMILY_CONTEXT_LENSES[family]

        context_index = profile_index % len(contexts)
        variation_index = (profile_index // len(contexts)) % len(_VARIATION_LENSES)
        evidence_index = (
            profile_index // (len(contexts) * len(_VARIATION_LENSES))
        ) % len(_EVIDENCE_LENSES)

        archetype = archetypes[archetype_position]
        return SFTCandidatePlan(
            family=family,
            candidate_index=candidate_index,
            archetype_index=archetype_position + 1,
            archetype_key=str(archetype["source_key"]),
            archetype=archetype,
            derivation_profile={
                "context_lens": contexts[context_index],
                "variation_lens": _VARIATION_LENSES[variation_index],
                "evidence_lens": _EVIDENCE_LENSES[evidence_index],
            },
        )


def _validate_requested_range(
    *, family: str, count: int, start_index: int, capacity: int
) -> None:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if (
        not isinstance(start_index, int)
        or isinstance(start_index, bool)
        or start_index < 1
    ):
        raise ValueError("start_index must be a positive integer")
    end = start_index + count - 1
    if end > capacity:
        raise ValueError(
            f"SFT task family {family!r} requested {start_index}..{end}; "
            f"planner capacity is {capacity}"
        )


def _validate_archetype_catalog(
    archetypes_by_family: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    expected = set(TASK_FAMILIES)
    observed = set(archetypes_by_family)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "SFT archetype catalog must cover the task taxonomy exactly; "
            f"missing={missing}, extra={extra}"
        )

    validated: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for family in TASK_FAMILIES:
        entries = tuple(archetypes_by_family[family])
        if not entries:
            raise ValueError(f"SFT task family {family!r} has no archetypes")
        seen_keys: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"SFT archetype for family {family!r} must be an object"
                )
            source_key = entry.get("source_key")
            if not isinstance(source_key, str) or not source_key.strip():
                raise ValueError(
                    f"SFT archetype for family {family!r} requires source_key"
                )
            if source_key in seen_keys:
                raise ValueError(
                    f"SFT task family {family!r} repeats source_key "
                    f"{source_key!r}"
                )
            seen_keys.add(source_key)
        validated[family] = entries
    return validated


FINITE_SFT_SPEC_PLANNER: SFTSpecPlanner = FiniteArchetypePlanner(
    SFT_ARCHETYPE_CATALOG
)
DEFAULT_SFT_SPEC_PLANNER: SFTSpecPlanner = ScalableArchetypePlanner(
    SFT_ARCHETYPE_CATALOG
)
