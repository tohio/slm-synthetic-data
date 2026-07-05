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
    metadata["failure_mode"] = _FAILURE_MODE_BY_FAMILY[family]
    variables = dict(spec.get("variables", {}))
    rejected_answer = _build_rejected_answer(family=family, variables=variables)
    if rejected_answer is not None:
        variables["rejected_answer"] = rejected_answer
    return {
        "id": spec["id"].replace("sft_", "dpo_", 1),
        "instruction": (
            spec["instruction"]
            + " Create a preferred assistant response and a realistic rejected response "
            f"that demonstrates the failure mode: {metadata['failure_mode']}."
        ),
        "metadata": metadata,
        "variables": variables,
        "constraints": [
            *list(spec.get("constraints", [])),
            "Chosen response must be preferred over rejected response.",
            "Rejected response must be realistic, not random.",
            *(_rejected_answer_constraints(family) if rejected_answer is not None else []),
        ],
        **({"holdout_key": spec["holdout_key"]} if "holdout_key" in spec else {}),
    }


def _build_rejected_answer(*, family: str, variables: dict[str, Any]) -> str | None:
    if family in {"basic_arithmetic_qa", "direct_division", "direct_subtraction"}:
        return _wrong_number(variables.get("answer"))
    if family == "code_expression_result":
        return _wrong_expression_result(variables.get("answer"))
    if family == "capital_city_qa":
        return _alternate_text(variables.get("capital"), fallback="Toronto")
    if family == "clear_sky_color_qa":
        return _alternate_text(variables.get("answer"), fallback="purple")
    if family == "short_factual_stop_behavior":
        return _alternate_text(variables.get("capital"), fallback="Toronto")
    if family == "repeat_exact_n_times":
        answer = variables.get("answer")
        word = variables.get("word")
        if isinstance(answer, str) and isinstance(word, str) and answer:
            return f"{answer} {word}"
    if family == "list_exact_n_items":
        items = variables.get("items")
        if isinstance(items, list) and items:
            extra = "purple" if "purple" not in items else "silver"
            return ", ".join(str(item) for item in [*items, extra])
        answer = variables.get("answer")
        if isinstance(answer, str) and answer:
            return f"{answer}, purple"
    return None


def _rejected_answer_constraints(family: str) -> list[str]:
    if family in {
        "basic_arithmetic_qa",
        "capital_city_qa",
        "clear_sky_color_qa",
        "code_expression_result",
        "direct_division",
        "direct_subtraction",
        "list_exact_n_items",
        "repeat_exact_n_times",
        "short_factual_stop_behavior",
    }:
        return [
            "Chosen assistant content must exactly match variables.answer or variables.capital.",
            "Rejected assistant content must exactly match variables.rejected_answer.",
            "Rejected assistant content must not equal chosen assistant content.",
        ]
    return []


def _wrong_number(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value + 1)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return str(int(value) + 1)
    return None


def _wrong_expression_result(value: Any) -> str | None:
    wrong_number = _wrong_number(value)
    if wrong_number is not None:
        return wrong_number
    if value == "[1, 2, 3]":
        return "[3, 2, 1]"
    if isinstance(value, str) and value:
        return f"{value} extra"
    return None


def _alternate_text(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip().lower() == fallback.lower():
        return "Paris"
    return fallback


def _validate_family(family: str) -> str:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("DPO spec family must be a non-empty string")
    normalized = family.strip().lower()
    if normalized not in DPO_SPEC_FAMILIES:
        supported = ", ".join(sorted(DPO_SPEC_FAMILIES))
        raise ValueError(f"Unsupported DPO spec family '{family}'. Supported families: {supported}")
    return normalized
