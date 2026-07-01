"""Shared metadata validation for SFT and DPO rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from slm_synth.taxonomy.categories import validate_category
from slm_synth.taxonomy.difficulties import validate_difficulty
from slm_synth.taxonomy.eval_families import validate_eval_family
from slm_synth.taxonomy.failure_modes import validate_failure_mode
from slm_synth.taxonomy.template_families import validate_template_family


def validate_metadata(
    metadata: Mapping[str, Any],
    *,
    require_failure_mode: bool = False,
) -> dict[str, Any]:
    """Validate shared SFT/DPO metadata.

    SFT rows use category, difficulty, template_family, and eval_family. DPO
    rows use the same fields plus failure_mode.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be an object")

    required = {"category", "difficulty", "template_family", "eval_family"}
    if require_failure_mode:
        required.add("failure_mode")

    missing = sorted(field for field in required if field not in metadata)
    if missing:
        raise ValueError(f"metadata missing required field(s): {missing}")

    allowed = set(required)
    extra = sorted(field for field in metadata if field not in allowed)
    if extra:
        raise ValueError(f"metadata contains unsupported field(s): {extra}")

    validated: dict[str, Any] = {
        "category": validate_category(metadata["category"]),
        "difficulty": validate_difficulty(metadata["difficulty"]),
        "template_family": validate_template_family(metadata["template_family"]),
        "eval_family": validate_eval_family(metadata["eval_family"]),
    }
    if require_failure_mode:
        validated["failure_mode"] = validate_failure_mode(metadata["failure_mode"])
    return validated
