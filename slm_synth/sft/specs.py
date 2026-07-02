"""Local task specs for LLM-generated SFT rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from slm_synth.taxonomy import validate_metadata

SFT_SPEC_REQUIRED_FIELDS = frozenset({"id", "instruction", "metadata"})
SFT_SPEC_OPTIONAL_FIELDS = frozenset({"variables", "constraints", "holdout_key"})
SFT_SPEC_FIELDS = SFT_SPEC_REQUIRED_FIELDS | SFT_SPEC_OPTIONAL_FIELDS


def validate_sft_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one local SFT task spec.

    Specs are the scalable control plane for SFT generation. Local code owns
    ids, taxonomy labels, variables, constraints, and holdout keys. The LLM
    uses the spec to produce the final training row.
    """
    if not isinstance(spec, Mapping):
        raise TypeError("SFT spec must be an object")

    missing = sorted(field for field in SFT_SPEC_REQUIRED_FIELDS if field not in spec)
    if missing:
        raise ValueError(f"SFT spec missing required field(s): {missing}")

    extra = sorted(field for field in spec if field not in SFT_SPEC_FIELDS)
    if extra:
        raise ValueError(f"SFT spec contains unsupported field(s): {extra}")

    validated: dict[str, Any] = {
        "id": _require_non_empty_string(spec["id"], "id"),
        "instruction": _require_non_empty_string(spec["instruction"], "instruction"),
        "metadata": validate_metadata(spec["metadata"], require_failure_mode=False),
    }

    if "variables" in spec:
        validated["variables"] = _validate_mapping(spec["variables"], "variables")
    if "constraints" in spec:
        validated["constraints"] = _validate_string_list(spec["constraints"], "constraints")
    if "holdout_key" in spec:
        validated["holdout_key"] = _validate_mapping(spec["holdout_key"], "holdout_key")
    return validated


def teacher_visible_sft_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the SFT spec fields sent to the LLM teacher.

    Structured holdout keys stay local. They are for local rejection checks,
    not teacher prompting.
    """
    validated = validate_sft_spec(spec)
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
