"""Scalable candidate planning over finite grounded seed archetypes.

Seed artifacts remain deterministic, locally inspectable source material.  Once
production planning moves beyond the seed inventory, candidates reuse a seed as
an archetype plus a deterministic derivation profile.  The renderer must create
a materially distinct record for derived candidates; deterministic record
validation and the semantic judge/reviewer remain downstream acceptance gates.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from slm_synth.pretrain.artifacts.base import GroundedArtifact


_CONTEXT_LENSES = (
    "household or everyday operations",
    "education or training",
    "small-business operations",
    "scientific or technical work",
    "public-service administration",
    "logistics or scheduling",
    "data or software operations",
    "maintenance or repair",
    "community or nonprofit work",
    "research or analysis",
    "creative or media production",
    "personal planning or decision support",
)

_VARIATION_LENSES = (
    "change the underlying entities and relationships, not just names",
    "use a different causal or dependency structure",
    "use a different input representation or evidence layout",
    "change which quantity, state, or conclusion must be inferred",
    "introduce a materially different edge case that remains self-contained",
    "change the operational setting and constraints together",
    "use a different ordering or transformation of the supplied facts",
    "change the decision boundary while keeping the same broad skill family",
    "use a different failure or uncertainty pattern",
    "change the data shape or interaction pattern in a substantive way",
)

_PRESENTATION_LENSES = (
    "concise direct task",
    "short realistic scenario",
    "compact record or memo",
    "small structured evidence block",
    "brief troubleshooting context",
    "learner-facing exercise",
    "workplace request",
    "analysis note",
)


class ScalableArtifactFactory:
    """Expose stable production candidates from a finite seed factory.

    The first ``seed_capacity`` indexes are byte-for-byte the existing seed
    artifacts.  Later indexes carry a deterministic derivation profile so the
    renderer can create a new case rather than lightly paraphrasing the seed.
    """

    def __init__(self, seed_factory: Any):
        self.seed_factory = seed_factory
        self.seed_capacity = int(seed_factory.UNIQUE_CANDIDATE_CAPACITY)
        if self.seed_capacity <= 0:
            raise ValueError("seed artifact capacity must be positive")

    def build_batch(self, batch_id: int, batch_size: int) -> list[GroundedArtifact]:
        start = int(batch_id) * int(batch_size)
        return [self.build(start + offset) for offset in range(batch_size)]

    def build(self, index: int) -> GroundedArtifact:
        if index < 0:
            raise ValueError("candidate index must be nonnegative")
        if index < self.seed_capacity:
            return self.seed_factory.build(index)

        seed_index = index % self.seed_capacity
        derivation_ordinal = index // self.seed_capacity
        seed = self.seed_factory.build(seed_index)
        profile_index = derivation_ordinal - 1
        context = _CONTEXT_LENSES[profile_index % len(_CONTEXT_LENSES)]
        variation = _VARIATION_LENSES[
            (profile_index // len(_CONTEXT_LENSES)) % len(_VARIATION_LENSES)
        ]
        presentation = _PRESENTATION_LENSES[
            (profile_index // (len(_CONTEXT_LENSES) * len(_VARIATION_LENSES)))
            % len(_PRESENTATION_LENSES)
        ]
        payload = deepcopy(seed.payload)
        payload["_derivation"] = {
            "seed_artifact_id": seed.artifact_id,
            "derivation_ordinal": derivation_ordinal,
            "context_lens": context,
            "variation_lens": variation,
            "presentation_lens": presentation,
            "contract": (
                "Create a materially distinct case in the same broad skill family. "
                "The seed is an archetype, not authoritative facts for the new case. "
                "Do not make a renamed, renumbered, entity-swapped, or cosmetic paraphrase."
            ),
        }
        return GroundedArtifact(
            signal=seed.signal,
            family=seed.family,
            artifact_id=f"{seed.signal}_{seed.family}_derived_{index + 1:09d}",
            payload=payload,
        )
