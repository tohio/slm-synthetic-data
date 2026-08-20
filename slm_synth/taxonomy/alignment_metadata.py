"""Public metadata validation for generic SFT and DPO rows."""

from collections.abc import Mapping
from typing import Any

from .context_modes import validate_context_mode
from .difficulties import validate_difficulty
from .failure_modes import validate_failure_mode
from .interaction_modes import validate_interaction_modes
from .output_modes import validate_output_mode
from .preference_dimensions import validate_preference_dimension
from .task_families import validate_task_family
from .template_families import validate_template_family


def validate_alignment_metadata(metadata: Mapping[str, Any], *, preference: bool = False) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be an object")
    required = {
        "task_family", "interaction_modes", "output_mode", "context_mode",
        "difficulty", "template_family",
    }
    if preference:
        required.update({"preference_dimension", "failure_mode"})
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"metadata missing required field(s): {missing}")
    extra = sorted(set(metadata) - required)
    if extra:
        raise ValueError(f"metadata contains unsupported field(s): {extra}")
    validated = {
        "task_family": validate_task_family(metadata["task_family"]),
        "interaction_modes": validate_interaction_modes(metadata["interaction_modes"]),
        "output_mode": validate_output_mode(metadata["output_mode"]),
        "context_mode": validate_context_mode(metadata["context_mode"]),
        "difficulty": validate_difficulty(metadata["difficulty"]),
        "template_family": validate_template_family(metadata["template_family"]),
    }
    if preference:
        validated["preference_dimension"] = validate_preference_dimension(metadata["preference_dimension"])
        validated["failure_mode"] = validate_failure_mode(metadata["failure_mode"])
    return validated
