"""Deterministic candidate planning for grounded pretraining artifacts.

Finite factories remain the source of the initial quality-smoke candidates.
The scalable planner reuses each finite artifact only as a capability anchor and
adds a deterministic semantic derivation profile for later candidates. The
renderer must materialize fresh public content for those derived candidates;
all final records still pass deterministic validation, semantic review, and
final deduplication.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from slm_synth.pretrain.artifacts.arithmetic import ArithmeticArtifactFactory
from slm_synth.pretrain.artifacts.base import GroundedArtifact
from slm_synth.pretrain.artifacts.educational_qa_mcq_general import (
    EducationalQAMCQGeneralArtifactFactory,
)
from slm_synth.pretrain.artifacts.educational_qa_mcq_math import (
    EducationalQAMCQMathArtifactFactory,
)
from slm_synth.pretrain.artifacts.factual_restraint import FactualRestraintArtifactFactory
from slm_synth.pretrain.artifacts.task_code import TaskCodeArtifactFactory


BASE_FACTORY_MAP: dict[str, type[Any]] = {
    "arithmetic": ArithmeticArtifactFactory,
    "task_code": TaskCodeArtifactFactory,
    "educational_qa_mcq_math": EducationalQAMCQMathArtifactFactory,
    "educational_qa_mcq_general": EducationalQAMCQGeneralArtifactFactory,
    "factual_restraint": FactualRestraintArtifactFactory,
}

# These are semantic planning dimensions, not sentence templates. Their
# Cartesian product exposes 192,000 derivation profiles for every finite
# capability anchor. This provides production-scale replacement headroom while
# retaining the original finite catalogs as the local, validated capability
# basis for every derived request.
_DOMAIN_LENSES: tuple[str, ...] = (
    "archive and records operations",
    "community health administration",
    "environmental field research",
    "manufacturing quality assurance",
    "public transit operations",
    "shipping and warehouse logistics",
    "school and training administration",
    "library and information services",
    "agricultural planning",
    "energy-system operations",
    "water-utility operations",
    "construction project coordination",
    "equipment maintenance",
    "cybersecurity incident response",
    "scientific laboratory work",
    "retail inventory operations",
    "hospitality operations",
    "municipal public services",
    "nonprofit program delivery",
    "publishing and editorial work",
    "sports-event operations",
    "museum and collection management",
    "accessibility services",
    "emergency preparedness",
)

_VARIATION_LENSES: tuple[str, ...] = (
    "include a meaningful boundary condition",
    "make two constraints interact",
    "include one explicit exception",
    "require a multi-step state transition",
    "include one irrelevant but plausible detail",
    "distinguish confirmed facts from an unresolved detail",
    "compare two plausible alternatives",
    "require a check against a threshold",
    "include a resource-allocation constraint",
    "require careful ordering of operations",
    "include an input-validation edge case",
    "require explicit error handling",
    "use a time-dependent condition",
    "include a unit or representation conversion",
    "require reconciling two source facts",
    "include a special-case branch",
    "require a concise verification step",
    "make a tempting shortcut produce the wrong result",
    "require preserving an invariant",
    "include a realistic incomplete-information constraint",
)

_EVIDENCE_LENSES: tuple[str, ...] = (
    "ground the task in a short operations memo",
    "ground the task in a structured table",
    "ground the task in a chronological event log",
    "ground the task in a service ticket",
    "ground the task in a checklist",
    "ground the task in a schedule",
    "ground the task in an inventory snapshot",
    "ground the task in a procedure excerpt",
    "ground the task in a research note",
    "ground the task in a status report",
    "ground the task in a brief transcript",
    "ground the task in a configuration excerpt",
    "ground the task in a measurement record",
    "ground the task in a comparison summary",
    "ground the task in an incident timeline",
    "ground the task in a request form",
    "ground the task in a compact data sample",
    "ground the task in an audit observation",
    "ground the task in a policy excerpt",
    "ground the task in a handoff note",
)

_REASONING_LENSES: tuple[str, ...] = (
    "require causal diagnosis",
    "require explicit constraint satisfaction",
    "require sequential planning",
    "require tradeoff analysis",
    "require exception handling",
    "require data reconciliation",
    "require classification from supplied evidence",
    "require a threshold decision",
    "require state tracking",
    "require testing a plausible counterexample",
    "require preserving a stated invariant",
    "require calibrated uncertainty",
    "require source attribution",
    "require unit normalization",
    "require boundary-case analysis",
    "require prioritization",
    "require error localization",
    "require eligibility determination",
    "require comparative evaluation",
    "require a concise verification",
)

# The multiplier is coprime to the complete profile count, so this affine map
# is a deterministic permutation. The anchor-specific offset spreads adjacent
# candidate indices across the full semantic space instead of making an entire
# anchor cycle share the same early profile.
_PROFILE_PERMUTATION_MULTIPLIER = 7919
_PROFILE_ANCHOR_OFFSET = 104729


class ScalableGroundedArtifactFactory:
    """Expose stable derived candidate plans while preserving finite anchors.

    Candidate indices below the finite factory capacity are byte-for-byte the
    existing artifacts. Later indices combine one finite capability anchor with
    one semantic derivation profile. The derived payload deliberately retains
    the validated anchor fields so local preflight can still verify the
    capability seed before a provider request.
    """

    def __init__(self, signal: str, base_factory: Any) -> None:
        if signal not in BASE_FACTORY_MAP:
            raise ValueError(f"Unsupported grounded signal: {signal}")
        self.signal = signal
        self.base_factory = base_factory
        self.base_capacity = int(base_factory.UNIQUE_CANDIDATE_CAPACITY)
        self.profile_count = (
            len(_DOMAIN_LENSES)
            * len(_VARIATION_LENSES)
            * len(_EVIDENCE_LENSES)
            * len(_REASONING_LENSES)
        )
        self.UNIQUE_CANDIDATE_CAPACITY = self.base_capacity * (1 + self.profile_count)

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(int(batch_size))]

    def build(self, index: int) -> GroundedArtifact:
        index = int(index)
        if not 0 <= index < self.UNIQUE_CANDIDATE_CAPACITY:
            raise ValueError(
                f"{self.signal} index {index} exceeds scalable candidate capacity "
                f"{self.UNIQUE_CANDIDATE_CAPACITY}"
            )
        if index < self.base_capacity:
            return self.base_factory.build(index)

        derived_offset = index - self.base_capacity
        anchor_index = derived_offset % self.base_capacity
        profile_round = derived_offset // self.base_capacity
        anchor = self.base_factory.build(anchor_index)

        profile_index = (
            profile_round * _PROFILE_PERMUTATION_MULTIPLIER
            + anchor_index * _PROFILE_ANCHOR_OFFSET
        ) % self.profile_count
        reasoning_index = profile_index % len(_REASONING_LENSES)
        remaining = profile_index // len(_REASONING_LENSES)
        domain_index = remaining % len(_DOMAIN_LENSES)
        remaining //= len(_DOMAIN_LENSES)
        variation_index = remaining % len(_VARIATION_LENSES)
        evidence_index = (remaining // len(_VARIATION_LENSES)) % len(_EVIDENCE_LENSES)

        payload = copy.deepcopy(anchor.payload)
        payload.update(
            {
                "generation_mode": "derived",
                "capability_anchor_id": anchor.artifact_id,
                "derivation_profile": {
                    "domain_lens": _DOMAIN_LENSES[domain_index],
                    "variation_lens": _VARIATION_LENSES[variation_index],
                    "evidence_lens": _EVIDENCE_LENSES[evidence_index],
                    "reasoning_lens": _REASONING_LENSES[reasoning_index],
                },
            }
        )
        return GroundedArtifact(
            signal=anchor.signal,
            family=anchor.family,
            artifact_id=f"{anchor.signal}_{anchor.family}_derived_{index + 1:09d}",
            payload=payload,
        )


def build_artifact_factory(signal: str, mix_cfg: Mapping[str, Any] | None = None) -> Any:
    """Build the configured finite or scalable artifact factory for one signal."""

    if signal not in BASE_FACTORY_MAP:
        raise ValueError(f"Unsupported grounded signal: {signal}")
    cfg = mix_cfg or {}
    planner = str(cfg.get("candidate_planner", "finite")).strip().lower()
    base_factory = BASE_FACTORY_MAP[signal]()
    if planner == "finite":
        return base_factory
    if planner == "scalable":
        return ScalableGroundedArtifactFactory(signal, base_factory)
    raise ValueError(
        f"unsupported candidate_planner {planner!r} for {signal}; expected 'finite' or 'scalable'"
    )


def configured_candidate_capacity(
    signal: str,
    mix_cfg: Mapping[str, Any] | None = None,
    *,
    factory: Any | None = None,
) -> int:
    """Return the effective candidate capacity, including an optional lower cap."""

    cfg = mix_cfg or {}
    resolved_factory = factory or build_artifact_factory(signal, cfg)
    capacity = int(resolved_factory.UNIQUE_CANDIDATE_CAPACITY)
    explicit = cfg.get("max_unique_candidates")
    if explicit is None:
        return capacity
    explicit_capacity = int(explicit)
    if explicit_capacity <= 0:
        raise ValueError("max_unique_candidates must be positive")
    if explicit_capacity > capacity:
        raise ValueError(
            f"max_unique_candidates={explicit_capacity} exceeds {signal} "
            f"{str(cfg.get('candidate_planner', 'finite')).strip().lower()} planner capacity={capacity}"
        )
    return explicit_capacity


def is_derived_artifact(artifact: GroundedArtifact) -> bool:
    return artifact.payload.get("generation_mode") == "derived"
