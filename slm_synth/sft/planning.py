"""Candidate-spec planning primitives for generic SFT generation.

This module separates task archetypes from candidate-spec planning.  The
default planner is deliberately one-archetype-per-candidate so the first
planner refactor is behavior-preserving.  Production-scale planners can later
implement the same interface while deriving many novel candidate specs from
each archetype.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from slm_synth.sft.source_catalog import SFT_ARCHETYPE_CATALOG
from slm_synth.taxonomy import TASK_FAMILIES, validate_task_family


@dataclass(frozen=True)
class SFTCandidatePlan:
    """One planned candidate before its concrete SFT spec is materialized."""

    family: str
    candidate_index: int
    archetype_index: int
    archetype_key: str
    archetype: Mapping[str, Any]


class SFTSpecPlanner(Protocol):
    """Planning contract consumed by the SFT spec builder."""

    def capacity(self, family: str) -> int:
        """Return the number of currently plannable candidates for ``family``."""

    def plan(
        self, *, family: str, count: int, start_index: int = 1
    ) -> list[SFTCandidatePlan]:
        """Return stable candidate plans for the requested range."""


class FiniteArchetypePlanner:
    """Behavior-preserving planner backed one-for-one by task archetypes.

    This is the transition implementation.  It keeps today's finite-capacity
    guardrail while moving that policy out of ``spec_builders`` and behind a
    planner interface that can later be replaced by a scalable derivation
    planner without changing generation callers.
    """

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


DEFAULT_SFT_SPEC_PLANNER: SFTSpecPlanner = FiniteArchetypePlanner(
    SFT_ARCHETYPE_CATALOG
)
