"""Local task specs for LLM-generated DPO rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.taxonomy import validate_metadata

DPO_SPEC_REQUIRED_FIELDS = frozenset({"id", "instruction", "metadata"})
DPO_SPEC_OPTIONAL_FIELDS = frozenset({"variables", "constraints", "holdout_key"})
DPO_SPEC_FIELDS = DPO_SPEC_REQUIRED_FIELDS | DPO_SPEC_OPTIONAL_FIELDS


def validate_dpo_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one local DPO pair spec."""
    if not isinstance(spec, Mapping):
        raise TypeError("DPO spec must be an object")

    missing = sorted(field for field in DPO_SPEC_REQUIRED_FIELDS if field not in spec)
    if missing:
        raise ValueError(f"DPO spec missing required field(s): {missing}")

    extra = sorted(field for field in spec if field not in DPO_SPEC_FIELDS)
    if extra:
        raise ValueError(f"DPO spec contains unsupported field(s): {extra}")

    validated: dict[str, Any] = {
        "id": _require_non_empty_string(spec["id"], "id"),
        "instruction": _require_non_empty_string(spec["instruction"], "instruction"),
        "metadata": validate_metadata(spec["metadata"], require_failure_mode=True),
    }

    if "variables" in spec:
        validated["variables"] = _validate_mapping(spec["variables"], "variables")
    if "constraints" in spec:
        validated["constraints"] = _validate_string_list(spec["constraints"], "constraints")
    if "holdout_key" in spec:
        validated["holdout_key"] = _validate_mapping(spec["holdout_key"], "holdout_key")
    return validated


def teacher_visible_dpo_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the DPO spec fields sent to the LLM teacher."""
    validated = validate_dpo_spec(spec)
    visible = {
        "id": validated["id"],
        "instruction": validated["instruction"],
        "metadata": validated["metadata"],
    }
    if "variables" in validated:
        visible["variables"] = validated["variables"]
    if "constraints" in validated:
        visible["constraints"] = validated["constraints"]
    return visible


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _validate_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a list")
    result: list[str] = []
    for item in value:
        result.append(_require_non_empty_string(item, field_name))
    return result
