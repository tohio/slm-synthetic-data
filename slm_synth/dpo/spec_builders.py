"""Scalable local task-spec builders for LLM-generated DPO data."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slm_synth.dpo.specs import validate_dpo_spec
from slm_synth.sft.spec_builders import SFT_SPEC_FAMILIES, build_specs as build_sft_specs

DPO_SPEC_FAMILIES = SFT_SPEC_FAMILIES

_FAILURE_MODE_BY_FAMILY = {
    "ai_concept_explanation": "wrong_factual_answer",
    "basic_arithmetic_qa": "wrong_numeric_answer",
    "capital_city_qa": "wrong_factual_answer",
    "clear_sky_color_qa": "wrong_factual_answer",
    "code_explanation_no_code": "code_includes_explanation",
    "code_expression_result": "wrong_numeric_answer",
    "code_generation_function": "code_includes_explanation",
    "direct_division": "wrong_numeric_answer",
    "direct_subtraction": "wrong_numeric_answer",
    "function_completion_body_only": "code_includes_explanation",
    "list_exact_n_items": "format_violation",
    "private_or_unverifiable_company_fact": "unknown_fact_fabrication",
    "repeat_exact_n_times": "format_violation",
    "short_factual_stop_behavior": "verbosity_mismatch",
}


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build validated DPO task specs for one eval family."""
    normalized = _validate_family(family)
    sft_specs = build_sft_specs(family=normalized, count=count, start_index=start_index)
    return [validate_dpo_spec(_dpo_from_sft_spec(spec, family=normalized)) for spec in sft_specs]


def write_specs_jsonl(specs: list[dict[str, Any]], path: str | Path) -> int:
    """Write validated DPO task specs to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(validate_dpo_spec(spec), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_and_write_specs(
    *,
    family: str,
    count: int,
    output_path: str | Path,
    start_index: int = 1,
) -> int:
    """Build and write one DPO spec JSONL file."""
    return write_specs_jsonl(build_specs(family=family, count=count, start_index=start_index), output_path)


def _dpo_from_sft_spec(spec: dict[str, Any], *, family: str) -> dict[str, Any]:
    metadata = dict(spec["metadata"])
    failure_mode = _FAILURE_MODE_BY_FAMILY[family]
    metadata["failure_mode"] = failure_mode

    return {
        "id": spec["id"].replace("sft_", "dpo_", 1),
        "instruction": _dpo_instruction_for_family(
            base_instruction=str(spec["instruction"]),
            family=family,
            failure_mode=failure_mode,
        ),
        "metadata": metadata,
        "variables": dict(spec.get("variables", {})),
        "constraints": _dpo_constraints_for_family(
            constraints=list(spec.get("constraints", [])),
            family=family,
            failure_mode=failure_mode,
        ),
        **({"holdout_key": spec["holdout_key"]} if "holdout_key" in spec else {}),
    }


def _dpo_instruction_for_family(*, base_instruction: str, family: str, failure_mode: str) -> str:
    if family == "code_explanation_no_code":
        return (
            base_instruction
            + " Create one preferred assistant response and one realistic rejected response. "
            "Both responses must be compact single-line assistant messages. "
            "The preferred response must explain the code in plain English without code. "
            "The rejected response must be a realistic lower-quality answer that reflects "
            f"the failure mode: {failure_mode}. "
            "Do not use Markdown, code fences, JSON, literal newlines, quotes, or backslashes "
            "inside either response content."
        )

    return (
        base_instruction
        + " Create a preferred assistant response and a realistic rejected response "
        f"that demonstrates the failure mode: {failure_mode}."
    )


def _dpo_constraints_for_family(
    *,
    constraints: list[Any],
    family: str,
    failure_mode: str,
) -> list[Any]:
    base_constraints = [
        *constraints,
        "Chosen response must be preferred over rejected response.",
        "Rejected response must be realistic, not random.",
    ]

    if family != "code_explanation_no_code":
        return base_constraints

    return [
        *base_constraints,
        "For code_explanation_no_code, chosen and rejected must each contain exactly one assistant message.",
        "For code_explanation_no_code, chosen.content and rejected.content must each be one line.",
        "For code_explanation_no_code, keep chosen.content and rejected.content under 220 characters each.",
        "For code_explanation_no_code, do not use Markdown fences, JSON, literal newlines, quotes, or backslashes in content.",
        "For code_explanation_no_code, chosen.content must be a plain-English explanation and must not include code.",
        f"For code_explanation_no_code, rejected.content must be concise and reflect failure_mode={failure_mode}.",
    ]


def _validate_family(family: str) -> str:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("DPO spec family must be a non-empty string")
    normalized = family.strip().lower()
    if normalized not in DPO_SPEC_FAMILIES:
        supported = ", ".join(sorted(DPO_SPEC_FAMILIES))
        raise ValueError(f"Unsupported DPO spec family '{family}'. Supported families: {supported}")
    return normalized
