"""Family definitions for distillation-DPO artifacts."""

from __future__ import annotations

from typing import Any

DISTILLATION_DPO_FAMILIES = frozenset({"teacher_response_preference"})


def validate_family(family: Any) -> str:
    """Return a normalized distillation-DPO family name or raise."""
    if not isinstance(family, str) or not family.strip():
        raise ValueError("distillation-DPO family must be a non-empty string")
    normalized = family.strip().lower()
    if normalized not in DISTILLATION_DPO_FAMILIES:
        supported = ", ".join(sorted(DISTILLATION_DPO_FAMILIES))
        raise ValueError(f"Unsupported distillation-DPO family '{family}'. Supported families: {supported}")
    return normalized
