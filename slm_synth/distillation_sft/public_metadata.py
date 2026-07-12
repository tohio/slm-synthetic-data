"""Public audit metadata for Distillation-SFT prompt/response rows."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from slm_synth.distillation_sft.signals import validate_signal
from slm_synth.taxonomy import validate_metadata


PUBLIC_METADATA_FIELDS = frozenset({"category", "difficulty", "template_family", "eval_family"})

_DEFAULT_TEMPLATE_FAMILY = {
    "code": "python_function_generation",
    "debugging": "python_debugging_explanation",
    "database": "sql_grouping_query",
    "cloud": "cloud_architecture_explanation",
    "data_transform": "data_transformation_plan",
    "educational_qa": "educational_explanation",
    "factual_restraint": "factual_restraint",
    "planning": "operational_planning_checklist",
    "instruction": "instruction_rewrite",
}

_DEFAULT_DIFFICULTY = {
    "arithmetic": 1,
    "educational_qa": 1,
    "instruction": 1,
}

_ARITHMETIC_EXPRESSION_RE = re.compile(r"(-?\d+)\s*([+\-*/x×÷])\s*(-?\d+)")


def build_public_metadata(
    *,
    signal: str,
    prompt: str,
    template_family: str | None = None,
    difficulty: int | None = None,
) -> dict[str, Any]:
    """Build validated row metadata from a locally owned prompt record."""
    normalized_signal = validate_signal(signal)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    resolved_template = template_family or _default_template_family(
        signal=normalized_signal,
        prompt=prompt,
    )
    category, eval_family = _category_and_eval_family(
        signal=normalized_signal,
        prompt=prompt,
        template_family=resolved_template,
    )
    resolved_difficulty = difficulty if difficulty is not None else _DEFAULT_DIFFICULTY.get(normalized_signal, 2)
    return validate_metadata(
        {
            "category": category,
            "difficulty": resolved_difficulty,
            "template_family": resolved_template,
            "eval_family": eval_family,
        }
    )


def extract_public_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the validated training-facing metadata fields."""
    if not isinstance(metadata, Mapping):
        raise TypeError("prompt metadata must be an object")
    return validate_metadata({field: metadata.get(field) for field in PUBLIC_METADATA_FIELDS})


def _default_template_family(*, signal: str, prompt: str) -> str:
    if signal != "arithmetic":
        return _DEFAULT_TEMPLATE_FAMILY[signal]

    match = _ARITHMETIC_EXPRESSION_RE.search(prompt)
    if match is None:
        return "arithmetic_word_problem"
    operator = match.group(2)
    return {
        "+": "integer_addition",
        "-": "integer_subtraction",
        "*": "integer_multiplication",
        "x": "integer_multiplication",
        "×": "integer_multiplication",
        "/": "integer_division",
        "÷": "integer_division",
    }[operator]


def _category_and_eval_family(
    *,
    signal: str,
    prompt: str,
    template_family: str,
) -> tuple[str, str | None]:
    if signal == "arithmetic":
        category = "direct_arithmetic" if template_family.startswith("integer_") else "word_problem_arithmetic"
        eval_family = {
            "integer_subtraction": "direct_subtraction",
            "integer_division": "direct_division",
        }.get(template_family, "basic_arithmetic_qa")
        return category, eval_family
    if signal == "code":
        return "code_generation", "code_generation_function"
    if signal == "factual_restraint":
        lowered = prompt.casefold()
        if any(term in lowered for term in ("next month", "next season", "future", "will definitely")):
            return "future_event_restraint", None
        if any(term in lowered for term in ("private", "password", "home address", "medical diagnosis")):
            return "private_info_restraint", None
        return "unknown_fact_restraint", None
    return "general_instruction_following", None
