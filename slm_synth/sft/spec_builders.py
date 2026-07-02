"""Scalable local task-spec builders for LLM-generated SFT data."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slm_synth.sft.specs import validate_sft_spec

SFT_SPEC_FAMILIES = frozenset(
    {
        "ai_concept_explanation",
        "basic_arithmetic_qa",
        "capital_city_qa",
        "clear_sky_color_qa",
        "code_explanation_no_code",
        "code_expression_result",
        "code_generation_function",
        "direct_division",
        "direct_subtraction",
        "function_completion_body_only",
        "list_exact_n_items",
        "private_or_unverifiable_company_fact",
        "repeat_exact_n_times",
        "short_factual_stop_behavior",
    }
)

_CITIES = [
    ("Canada", "Ottawa"),
    ("Italy", "Rome"),
    ("Kenya", "Nairobi"),
    ("Brazil", "Brasilia"),
    ("South Korea", "Seoul"),
]
_CONCEPTS = [
    ("embedding", "vectors that represent tokens or items"),
    ("attention mechanism", "a way to weight relevant context"),
    ("training loss", "a signal used to update model parameters"),
    ("tokenizer", "a component that converts text into tokens"),
]
_WORDS = ["dog", "blue", "sun", "code", "river"]
_COLORS = [["red", "green", "blue"], ["cyan", "magenta"], ["black", "white", "gray", "orange"]]
_PRIVATE_COMPANIES = ["Anthropic", "OpenAI", "Stripe", "Databricks", "Scale AI"]
_FUNCTIONS = [
    ("add_numbers", "Return the sum of two numbers."),
    ("is_even", "Return True when a number is even."),
    ("last_item", "Return the last item in a list."),
]
_EXPRESSIONS = [("2 + 3 * 4", "14"), ("len('cat')", "3"), ("10 // 3", "3"), ("sorted([3, 1, 2])", "[1, 2, 3]")]


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build validated SFT task specs for one eval family."""
    normalized = _validate_family(family)
    _validate_count(count)
    _validate_start_index(start_index)
    builder = _BUILDERS[normalized]
    return [validate_sft_spec(builder(index)) for index in range(start_index, start_index + count)]


def write_specs_jsonl(specs: list[dict[str, Any]], path: str | Path) -> int:
    """Write validated SFT task specs to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(validate_sft_spec(spec), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_and_write_specs(
    *,
    family: str,
    count: int,
    output_path: str | Path,
    start_index: int = 1,
) -> int:
    """Build and write one SFT spec JSONL file."""
    return write_specs_jsonl(build_specs(family=family, count=count, start_index=start_index), output_path)


def _spec(
    *,
    family: str,
    index: int,
    category: str,
    template_family: str,
    difficulty: int,
    instruction: str,
    variables: dict[str, Any],
    constraints: list[str] | None = None,
    holdout_key: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": f"sft_{family}_{index:06d}",
        "instruction": instruction,
        "metadata": {
            "category": category,
            "difficulty": difficulty,
            "template_family": template_family,
            "eval_family": family,
        },
        "variables": variables,
        "constraints": constraints or ["Generate exactly one user message and one correct assistant response."],
    }
    if holdout_key is not None:
        spec["holdout_key"] = holdout_key
    return spec


def _basic_arithmetic(index: int) -> dict[str, Any]:
    a = 11 + index
    b = 7 + (index * 3) % 29
    return _spec(
        family="basic_arithmetic_qa",
        index=index,
        category="direct_arithmetic",
        template_family="direct_addition",
        difficulty=1,
        instruction="Create a concise addition question and answer it with the correct number.",
        variables={"operation": "addition", "a": a, "b": b, "answer": a + b},
        constraints=["Do not use the exact eval prompt 'What is 2 + 2?'."],
        holdout_key={"operation": "addition", "a": a, "b": b},
    )


def _direct_subtraction(index: int) -> dict[str, Any]:
    a = 30 + index
    b = 3 + index % 11
    return _spec(
        family="direct_subtraction",
        index=index,
        category="direct_arithmetic",
        template_family="direct_subtraction",
        difficulty=1,
        instruction="Create a direct subtraction question and answer with the correct number.",
        variables={"operation": "subtraction", "a": a, "b": b, "answer": a - b},
        holdout_key={"operation": "subtraction", "a": a, "b": b},
    )


def _direct_division(index: int) -> dict[str, Any]:
    divisor = 2 + index % 9
    answer = 3 + index % 12
    dividend = divisor * answer
    return _spec(
        family="direct_division",
        index=index,
        category="direct_arithmetic",
        template_family="direct_division",
        difficulty=1,
        instruction="Create a direct integer division question and answer with the correct number.",
        variables={"operation": "division", "dividend": dividend, "divisor": divisor, "answer": answer},
        holdout_key={"operation": "division", "dividend": dividend, "divisor": divisor},
    )


def _capital_city(index: int) -> dict[str, Any]:
    country, capital = _CITIES[(index - 1) % len(_CITIES)]
    return _spec(
        family="capital_city_qa",
        index=index,
        category="concise_factual_qa",
        template_family="capital_city_direct",
        difficulty=1,
        instruction="Create a concise capital-city question and answer with only the correct capital city.",
        variables={"country": country, "capital": capital},
        holdout_key={"fact_type": "capital_city", "country": country},
    )


def _clear_sky(index: int) -> dict[str, Any]:
    return _spec(
        family="clear_sky_color_qa",
        index=index,
        category="concise_factual_qa",
        template_family="common_fact_color",
        difficulty=1,
        instruction="Create a simple factual color question about a common observable object or condition.",
        variables={"topic": ["clear sky", "ripe banana", "grass leaf", "stop sign"][(index - 1) % 4]},
        constraints=["Answer concisely and avoid unrelated details."],
    )


def _ai_concept(index: int) -> dict[str, Any]:
    concept, expected = _CONCEPTS[(index - 1) % len(_CONCEPTS)]
    return _spec(
        family="ai_concept_explanation",
        index=index,
        category="general_instruction_following",
        template_family="short_ai_definition",
        difficulty=2,
        instruction="Create a concise machine-learning concept explanation question and answer clearly.",
        variables={"concept": concept, "expected_content": expected},
        constraints=["Do not explain unrelated electrical or mechanical concepts."],
    )


def _private_company(index: int) -> dict[str, Any]:
    company = _PRIVATE_COMPANIES[(index - 1) % len(_PRIVATE_COMPANIES)]
    return _spec(
        family="private_or_unverifiable_company_fact",
        index=index,
        category="private_info_restraint",
        template_family="private_company_metric",
        difficulty=2,
        instruction="Create a question asking for a private or unverifiable current company metric and answer with restraint.",
        variables={"company": company, "metric": "private revenue last month"},
        constraints=["The assistant should not invent a dollar amount."],
    )


def _code_generation(index: int) -> dict[str, Any]:
    name, doc = _FUNCTIONS[(index - 1) % len(_FUNCTIONS)]
    return _spec(
        family="code_generation_function",
        index=index,
        category="code_generation",
        template_family="python_function_code_only",
        difficulty=2,
        instruction="Create a Python function generation request and answer with code only.",
        variables={"function_name": name, "requirement": doc},
        constraints=["Assistant response must not include prose outside code."],
    )


def _function_completion(index: int) -> dict[str, Any]:
    name, doc = _FUNCTIONS[(index - 1) % len(_FUNCTIONS)]
    return _spec(
        family="function_completion_body_only",
        index=index,
        category="code_generation",
        template_family="python_function_body_only",
        difficulty=2,
        instruction="Create a Python function-completion prompt and answer with only the function body.",
        variables={"function_name": name, "docstring": doc},
        constraints=["Do not repeat the function signature in the assistant response."],
    )


def _code_explanation(index: int) -> dict[str, Any]:
    expression, answer = _EXPRESSIONS[(index - 1) % len(_EXPRESSIONS)]
    return _spec(
        family="code_explanation_no_code",
        index=index,
        category="general_instruction_following",
        template_family="code_explanation_plain_text",
        difficulty=2,
        instruction="Create a prompt asking to explain a small code snippet and answer without code fences.",
        variables={"snippet": f"result = {expression}", "expected_result": answer},
        constraints=["The assistant response should explain behavior, not reproduce the full code."],
    )


def _code_expression(index: int) -> dict[str, Any]:
    expression, answer = _EXPRESSIONS[(index - 1) % len(_EXPRESSIONS)]
    return _spec(
        family="code_expression_result",
        index=index,
        category="code_expression_evaluation",
        template_family="python_expression_result",
        difficulty=2,
        instruction="Create a Python expression evaluation prompt and answer with the resulting value.",
        variables={"expression": expression, "answer": answer},
    )


def _repeat_exact(index: int) -> dict[str, Any]:
    word = _WORDS[(index - 1) % len(_WORDS)]
    count = 2 + index % 4
    return _spec(
        family="repeat_exact_n_times",
        index=index,
        category="exact_output_format_control",
        template_family="repeat_word_count",
        difficulty=1,
        instruction="Create an exact repeat instruction and answer with only the repeated text.",
        variables={"word": word, "count": count, "answer": " ".join([word] * count)},
        holdout_key={"task": "repeat", "word": word, "count": count},
    )


def _list_exact(index: int) -> dict[str, Any]:
    items = _COLORS[(index - 1) % len(_COLORS)]
    return _spec(
        family="list_exact_n_items",
        index=index,
        category="exact_output_format_control",
        template_family="list_exact_count",
        difficulty=1,
        instruction="Create an instruction to list an exact number of simple items and answer with exactly that many items.",
        variables={"item_type": "colors", "count": len(items), "items": items},
    )


def _short_stop(index: int) -> dict[str, Any]:
    country, capital = _CITIES[index % len(_CITIES)]
    return _spec(
        family="short_factual_stop_behavior",
        index=index,
        category="controlled_verbosity",
        template_family="short_factual_answer",
        difficulty=1,
        instruction="Create a short factual question and answer briefly, stopping when complete.",
        variables={"country": country, "capital": capital},
        constraints=["Answer in fewer than 12 words."],
    )


_BUILDERS: dict[str, Callable[[int], dict[str, Any]]] = {
    "ai_concept_explanation": _ai_concept,
    "basic_arithmetic_qa": _basic_arithmetic,
    "capital_city_qa": _capital_city,
    "clear_sky_color_qa": _clear_sky,
    "code_explanation_no_code": _code_explanation,
    "code_expression_result": _code_expression,
    "code_generation_function": _code_generation,
    "direct_division": _direct_division,
    "direct_subtraction": _direct_subtraction,
    "function_completion_body_only": _function_completion,
    "list_exact_n_items": _list_exact,
    "private_or_unverifiable_company_fact": _private_company,
    "repeat_exact_n_times": _repeat_exact,
    "short_factual_stop_behavior": _short_stop,
}


def _validate_family(family: str) -> str:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("SFT spec family must be a non-empty string")
    normalized = family.strip().lower()
    if normalized not in SFT_SPEC_FAMILIES:
        supported = ", ".join(sorted(SFT_SPEC_FAMILIES))
        raise ValueError(f"Unsupported SFT spec family '{family}'. Supported families: {supported}")
    return normalized


def _validate_count(count: int) -> None:
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")


def _validate_start_index(start_index: int) -> None:
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")
