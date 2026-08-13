"""Scalable local task-spec builders for LLM-generated DPO data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slm_synth.dpo.specs import require_unique_dpo_sources, validate_dpo_spec
from slm_synth.sft.spec_builders import (
    SFT_SPEC_CAPACITIES,
    SFT_SPEC_FAMILIES,
    build_specs as build_sft_specs,
)

DPO_SPEC_FAMILIES = SFT_SPEC_FAMILIES
DPO_SPEC_CAPACITIES = dict(SFT_SPEC_CAPACITIES)

_PROMPT_CONTEXTS = (
    "standalone question",
    "flashcard",
    "knowledge check",
    "short exercise",
    "practice item",
    "review question",
    "worksheet item",
    "oral quiz",
    "study prompt",
    "quick check",
)

_WRONG_CAPITALS = (
    "Toronto",
    "Milan",
    "Mombasa",
    "Rio de Janeiro",
    "Busan",
    "Sydney",
    "Zurich",
    "Barcelona",
)

_WRONG_COLORS = ("purple", "orange", "blue", "green", "red", "yellow", "black", "white")

def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build validated DPO task specs for one eval family."""
    normalized = _validate_family(family)
    validate_spec_range(family=normalized, count=count, start_index=start_index)
    sft_specs = build_sft_specs(family=normalized, count=count, start_index=start_index)
    specs = [validate_dpo_spec(_dpo_from_sft_spec(spec, family=normalized)) for spec in sft_specs]
    require_unique_dpo_sources(specs)
    return specs


def unique_capacity(family: str) -> int:
    """Return the declared unique source capacity for one DPO family."""
    return DPO_SPEC_CAPACITIES[_validate_family(family)]


def validate_spec_range(*, family: str, count: int, start_index: int = 1) -> None:
    """Fail locally when a requested index range exceeds DPO source capacity."""
    normalized = _validate_family(family)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 1:
        raise ValueError("start_index must be a positive integer")
    capacity = DPO_SPEC_CAPACITIES[normalized]
    end_index = start_index + count - 1
    if end_index > capacity:
        raise ValueError(
            f"DPO family '{normalized}' requested index range {start_index}..{end_index}, "
            f"which exceeds declared unique source capacity {capacity}"
        )


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
    index = _spec_index(spec["id"])
    metadata = dict(spec["metadata"])
    metadata["failure_mode"] = _failure_mode(family=family, index=index)
    variables = _dpo_variables(family=family, variables=dict(spec.get("variables", {})), index=index)
    rejected_answer = _build_rejected_answer(family=family, variables=variables, index=index)
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
            *_family_prompt_constraints(family),
            "Chosen response must be preferred over rejected response.",
            "Chosen response must be semantically correct, not merely better formatted.",
            "Rejected response must be realistic, not random.",
            *(_rejected_answer_constraints(family) if rejected_answer is not None else []),
        ],
        **({"holdout_key": spec["holdout_key"]} if "holdout_key" in spec else {}),
    }



def _dpo_variables(*, family: str, variables: dict[str, Any], index: int) -> dict[str, Any]:
    """Add DPO-specific exact targets that validators can enforce."""
    if family in {
        "basic_arithmetic_qa",
        "clear_sky_color_qa",
        "code_expression_result",
        "direct_division",
        "direct_subtraction",
        "list_exact_n_items",
        "repeat_exact_n_times",
    }:
        answer = variables.get("answer")
        if answer is not None:
            variables["chosen_answer"] = str(answer)
    elif family in {"capital_city_qa", "short_factual_stop_behavior"}:
        capital = variables.get("capital")
        if isinstance(capital, str) and capital.strip():
            variables["chosen_answer"] = capital.strip()
    elif family == "function_completion_body_only":
        signature, body = _function_signature_and_body(variables.get("function_name"))
        if signature and body:
            variables["function_signature"] = signature
            variables["chosen_answer"] = body
    elif family == "code_generation_function":
        signature, body = _function_signature_and_body(variables.get("function_name"))
        if signature and body:
            code = f"{signature}\n    {body}"
            variables["function_signature"] = signature
            variables["chosen_answer"] = code
    if family == "list_exact_n_items":
        variables["prompt_context"] = _PROMPT_CONTEXTS[((index - 1) // 10) % len(_PROMPT_CONTEXTS)]
    return variables


def _failure_mode(*, family: str, index: int) -> str:
    modes = {
        "ai_concept_explanation": ("wrong_factual_answer", "incomplete_instruction_guess"),
        "basic_arithmetic_qa": ("wrong_numeric_answer",),
        "capital_city_qa": ("wrong_factual_answer",),
        "clear_sky_color_qa": ("wrong_factual_answer",),
        "code_explanation_no_code": ("code_includes_explanation", "code_logic_error"),
        "code_expression_result": ("wrong_numeric_answer",),
        "code_generation_function": (
            "code_includes_explanation",
            "format_violation",
            "code_includes_explanation",
        ),
        "direct_division": ("wrong_numeric_answer",),
        "direct_subtraction": ("wrong_numeric_answer",),
        "function_completion_body_only": (
            "code_includes_explanation",
            "format_violation",
            "format_violation",
        ),
        "list_exact_n_items": ("format_violation",),
        "private_or_unverifiable_company_fact": (
            "unknown_fact_fabrication",
            "unsafe_private_info_guess",
            "persona_fabrication",
        ),
        "repeat_exact_n_times": ("format_violation",),
        "short_factual_stop_behavior": (
            "verbosity_mismatch",
            "extra_explanation",
            "extra_explanation",
        ),
    }[family]
    return modes[(index - 1) % len(modes)]


def _family_prompt_constraints(family: str) -> list[str]:
    if family == "function_completion_body_only":
        return [
            "Prompt must include variables.function_signature exactly so the function parameters are grounded.",
            "Chosen assistant content must exactly match variables.chosen_answer.",
            "Chosen assistant content must not invent different parameter names or a different operation.",
        ]
    if family == "code_generation_function":
        return [
            "Prompt must ask for a complete Python function, not only show a bare def line.",
            "Chosen assistant content must include variables.function_signature and the required return behavior.",
        ]
    if family == "list_exact_n_items":
        return [
            "Prompt must ask for the item type and exact count only; it must not reveal variables.items or variables.answer.",
        ]
    return []


def _function_signature_and_body(function_name: Any) -> tuple[str | None, str | None]:
    targets = {
        "add_numbers": ("def add_numbers(a, b):", "return a + b"),
        "is_even": ("def is_even(number):", "return number % 2 == 0"),
        "last_item": ("def last_item(items):", "return items[-1]"),
        "count_positive": ("def count_positive(numbers):", "return sum(1 for number in numbers if number > 0)"),
        "first_word": ("def first_word(text):", "return text.split()[0]"),
        "maximum_value": ("def maximum_value(numbers):", "return max(numbers)"),
        "minimum_value": ("def minimum_value(numbers):", "return min(numbers)"),
        "reverse_text": ("def reverse_text(text):", "return text[::-1]"),
        "sum_even": ("def sum_even(numbers):", "return sum(number for number in numbers if number % 2 == 0)"),
        "unique_items": ("def unique_items(items):", "return list(dict.fromkeys(items))"),
        "word_count": ("def word_count(text):", "return len(text.split())"),
        "clamp_value": ("def clamp_value(value, lower, upper):", "return max(lower, min(value, upper))"),
        "average_value": ("def average_value(numbers):", "return sum(numbers) / len(numbers)"),
        "starts_with_vowel": ("def starts_with_vowel(text):", "return text[0].lower() in \"aeiou\""),
        "filter_long_words": ("def filter_long_words(words, minimum):", "return [word for word in words if len(word) >= minimum]"),
        "merge_counts": ("def merge_counts(left, right):", "return {key: left.get(key, 0) + right.get(key, 0) for key in left.keys() | right.keys()}"),
        "index_of_smallest": ("def index_of_smallest(items):", "return min(range(len(items)), key=items.__getitem__)"),
        "all_non_negative": ("def all_non_negative(numbers):", "return all(number >= 0 for number in numbers)"),
        "difference": ("def difference(a, b):", "return a - b"),
        "normalize_spaces": ("def normalize_spaces(text):", "return \" \".join(text.split())"),
    }
    return targets.get(function_name, (None, None))

def _build_rejected_answer(*, family: str, variables: dict[str, Any], index: int) -> str | None:
    if family in {"basic_arithmetic_qa", "direct_division", "direct_subtraction"}:
        return _wrong_number(variables.get("answer"), index=index)
    if family == "code_expression_result":
        return _wrong_expression_result(variables.get("answer"), index=index)
    if family == "capital_city_qa":
        return _alternate_from_pool(variables.get("capital"), pool=_WRONG_CAPITALS, index=index)
    if family == "clear_sky_color_qa":
        return _alternate_from_pool(variables.get("answer"), pool=_WRONG_COLORS, index=index)
    if family == "short_factual_stop_behavior":
        country = variables.get("country")
        capital = variables.get("capital")
        if isinstance(country, str) and isinstance(capital, str) and country and capital:
            variants = (
                f"The capital of {country} is {capital}.",
                f"The answer is {capital} because it is the capital city of {country}.",
                f"{capital}. This is the official capital of {country}.",
            )
            return variants[(index - 1) % len(variants)]
        return _alternate_from_pool(variables.get("capital"), pool=_WRONG_CAPITALS, index=index)
    if family == "repeat_exact_n_times":
        answer = variables.get("answer")
        word = variables.get("word")
        if isinstance(answer, str) and isinstance(word, str) and answer:
            words = answer.split()
            variants = (
                f"{answer} {word}",
                " ".join(words[:-1]),
                f"{answer}.",
                f"Result: {answer}",
            )
            return variants[(index - 1) % len(variants)]
    if family == "list_exact_n_items":
        items = variables.get("items")
        if isinstance(items, list) and items:
            extra = "purple" if "purple" not in items else "silver"
            variants = (
                ", ".join(str(item) for item in [*items, extra]),
                ", ".join(str(item) for item in items[:-1]),
                "; ".join(str(item) for item in items),
                f"Here are the items: {', '.join(str(item) for item in items)}",
            )
            return variants[(index - 1) % len(variants)]
        answer = variables.get("answer")
        if isinstance(answer, str) and answer:
            return f"{answer}, purple"
    if family == "function_completion_body_only":
        chosen = variables.get("chosen_answer")
        if isinstance(chosen, str) and chosen.strip():
            variants = (
                f"# Explanation: implement the requested behavior.\n{chosen}",
                f"```python\n{chosen}\n```",
                f"{variables['function_signature']}\n    {chosen}",
            )
            return variants[(index - 1) % len(variants)]
    if family == "code_generation_function":
        chosen = variables.get("chosen_answer")
        if isinstance(chosen, str) and chosen.strip():
            variants = (
                f"Here is the function:\n{chosen}",
                f"```python\n{chosen}\n```",
                f"{chosen}\nThis function satisfies the request.",
            )
            return variants[(index - 1) % len(variants)]
    return None


def _rejected_answer_constraints(family: str) -> list[str]:
    if family in {
        "basic_arithmetic_qa",
        "capital_city_qa",
        "clear_sky_color_qa",
        "code_expression_result",
        "code_generation_function",
        "direct_division",
        "direct_subtraction",
        "function_completion_body_only",
        "list_exact_n_items",
        "repeat_exact_n_times",
        "short_factual_stop_behavior",
    }:
        return [
            "Chosen assistant content must exactly match variables.chosen_answer when present.",
            "Rejected assistant content must exactly match variables.rejected_answer.",
            "Rejected assistant content must not equal chosen assistant content.",
        ]
    return []


def _wrong_number(value: Any, *, index: int) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        variants = (value + 1, value - 1, -value)
        wrong = variants[(index - 1) % len(variants)]
        return str(wrong if wrong != value else value + 2)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        number = int(value)
        variants = (number + 1, number - 1, -number)
        wrong = variants[(index - 1) % len(variants)]
        return str(wrong if wrong != number else number + 2)
    return None


def _wrong_expression_result(value: Any, *, index: int) -> str | None:
    wrong_number = _wrong_number(value, index=index)
    if wrong_number is not None:
        return wrong_number
    if value == "[1, 2, 3]":
        return "[3, 2, 1]"
    if isinstance(value, str) and value:
        return f"{value} extra"
    return None


def _alternate_from_pool(value: Any, *, pool: tuple[str, ...], index: int) -> str:
    current = value.strip().lower() if isinstance(value, str) else ""
    for offset in range(len(pool)):
        candidate = pool[(index - 1 + offset) % len(pool)]
        if candidate.lower() != current:
            return candidate
    raise ValueError("alternate-answer pool does not contain a distinct value")


def _spec_index(spec_id: str) -> int:
    try:
        index = int(spec_id.rsplit("_", 1)[1])
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"DPO spec id does not end in a numeric index: {spec_id!r}") from exc
    if index < 1:
        raise ValueError(f"DPO spec id index must be positive: {spec_id!r}")
    return index


def _validate_family(family: str) -> str:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("DPO spec family must be a non-empty string")
    normalized = family.strip().lower()
    if normalized not in DPO_SPEC_FAMILIES:
        supported = ", ".join(sorted(DPO_SPEC_FAMILIES))
        raise ValueError(f"Unsupported DPO spec family '{family}'. Supported families: {supported}")
    return normalized
