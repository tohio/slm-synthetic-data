"""Scalable local task-spec builders for LLM-generated SFT data."""

from __future__ import annotations

import json
from collections.abc import Callable
from math import prod
from pathlib import Path
from typing import Any

from slm_synth.sft.specs import require_unique_sft_sources, validate_sft_spec

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

_CAPITALS = [
    ("Canada", "Ottawa"),
    ("Italy", "Rome"),
    ("Kenya", "Nairobi"),
    ("Brazil", "Brasilia"),
    ("South Korea", "Seoul"),
    ("Argentina", "Buenos Aires"),
    ("Australia", "Canberra"),
    ("Austria", "Vienna"),
    ("Belgium", "Brussels"),
    ("Chile", "Santiago"),
    ("China", "Beijing"),
    ("Colombia", "Bogota"),
    ("Croatia", "Zagreb"),
    ("Czech Republic", "Prague"),
    ("Denmark", "Copenhagen"),
    ("Egypt", "Cairo"),
    ("Finland", "Helsinki"),
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Ghana", "Accra"),
    ("Greece", "Athens"),
    ("Hungary", "Budapest"),
    ("Iceland", "Reykjavik"),
    ("India", "New Delhi"),
    ("Ireland", "Dublin"),
    ("Japan", "Tokyo"),
    ("Mexico", "Mexico City"),
    ("Morocco", "Rabat"),
    ("Netherlands", "Amsterdam"),
    ("New Zealand", "Wellington"),
    ("Nigeria", "Abuja"),
    ("Norway", "Oslo"),
    ("Peru", "Lima"),
    ("Philippines", "Manila"),
    ("Poland", "Warsaw"),
    ("Portugal", "Lisbon"),
    ("Romania", "Bucharest"),
    ("Saudi Arabia", "Riyadh"),
    ("Singapore", "Singapore"),
    ("Spain", "Madrid"),
    ("Sweden", "Stockholm"),
    ("Switzerland", "Bern"),
    ("Thailand", "Bangkok"),
    ("Turkey", "Ankara"),
    ("Uganda", "Kampala"),
    ("Ukraine", "Kyiv"),
    ("United Kingdom", "London"),
    ("United States", "Washington, D.C."),
    ("Vietnam", "Hanoi"),
    ("Zambia", "Lusaka"),
]
_CONCEPTS = [
    ("embedding", "vectors that represent tokens or items"),
    ("attention mechanism", "a way to weight relevant context"),
    ("training loss", "a signal used to update model parameters"),
    ("tokenizer", "a component that converts text into tokens"),
    ("gradient descent", "an optimization method that updates parameters to reduce loss"),
    ("inference", "using a trained model to produce predictions or outputs"),
    ("fine-tuning", "adapting a pretrained model using task-specific data"),
    ("overfitting", "learning training data too specifically and generalizing poorly"),
    ("validation set", "held-out data used to assess model behavior during development"),
    ("context window", "the tokens a model can consider in one request"),
    ("language model", "a model that estimates patterns and probabilities in sequences of tokens"),
    ("batch size", "the number of training examples processed before an optimizer update"),
    ("learning rate", "the step size used when updating model parameters"),
    ("epoch", "one pass through a training dataset"),
    ("parameter", "a learned numeric value inside a model"),
    ("activation function", "a function that introduces nonlinear behavior into a network"),
    ("classification", "predicting one label from a defined set of labels"),
    ("regression", "predicting a continuous numeric value"),
    ("supervised learning", "learning from examples paired with target outputs"),
    ("unsupervised learning", "learning patterns from data without target labels"),
]
_WORDS = [
    "dog", "blue", "sun", "code", "river", "apple", "book", "cloud", "drum", "eagle",
    "forest", "glass", "harbor", "island", "jacket", "kite", "lamp", "meadow", "north", "ocean",
    "piano", "quiet", "road", "stone", "train", "unity", "valley", "water", "yellow", "zebra",
    "amber", "bridge", "circle", "dawn", "earth", "field", "garden", "hill", "ink", "joy",
    "key", "leaf", "moon", "night", "orange", "paper", "queen", "rain", "star", "tree",
]
_LIST_POOLS = [
    ("colors", ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "brown", "gray"]),
    ("fruits", ["apple", "banana", "orange", "pear", "grape", "mango", "peach", "plum", "kiwi", "melon"]),
    ("animals", ["dog", "cat", "horse", "rabbit", "tiger", "lion", "zebra", "panda", "otter", "whale"]),
    ("tools", ["hammer", "saw", "wrench", "pliers", "drill", "level", "chisel", "clamp", "file", "vise"]),
    ("vehicles", ["car", "bus", "train", "bicycle", "truck", "boat", "scooter", "tram", "van", "taxi"]),
    ("instruments", ["piano", "guitar", "violin", "drum", "flute", "trumpet", "cello", "harp", "banjo", "oboe"]),
    ("shapes", ["circle", "square", "triangle", "rectangle", "oval", "hexagon", "pentagon", "cube", "sphere", "cone"]),
    ("furniture", ["chair", "table", "desk", "sofa", "stool", "shelf", "bed", "bench", "cabinet", "dresser"]),
    ("weather terms", ["rain", "snow", "wind", "fog", "hail", "cloud", "storm", "breeze", "frost", "sunshine"]),
    ("school supplies", ["pencil", "notebook", "eraser", "ruler", "marker", "folder", "paper", "crayon", "stapler", "binder"]),
]
_PRIVATE_COMPANIES = [
    "Anthropic", "OpenAI", "Stripe", "Databricks", "Scale AI", "Canva", "Discord", "Epic Games",
    "Figma", "Instacart", "Notion", "Plaid", "Reddit", "Rippling", "SpaceX", "Valve", "Waymo",
    "Chime", "Klarna", "Airtable",
]
_PRIVATE_METRICS = [
    "private revenue", "unpublished profit", "internal cash balance", "unreleased customer count",
    "confidential operating cost", "internal churn rate", "unannounced valuation", "private payroll total",
    "unpublished contract value", "confidential product margin",
]
_TIME_WINDOWS = ["last month", "last quarter", "the previous fiscal year", "this week", "yesterday"]
_FUNCTION_TASKS = [
    ("add_numbers", "Return the sum of two numbers."),
    ("is_even", "Return True when a number is even."),
    ("last_item", "Return the last item in a list."),
    ("count_positive", "Count the positive integers in a list."),
    ("first_word", "Return the first whitespace-delimited word in a string."),
    ("maximum_value", "Return the largest number in a non-empty list."),
    ("minimum_value", "Return the smallest number in a non-empty list."),
    ("reverse_text", "Return a string with its characters reversed."),
    ("sum_even", "Return the sum of the even integers in a list."),
    ("unique_items", "Return the input items with duplicates removed while preserving order."),
    ("word_count", "Return the number of whitespace-delimited words in a string."),
    ("clamp_value", "Clamp a number between a lower and upper bound."),
    ("average_value", "Return the arithmetic mean of a non-empty list of numbers."),
    ("starts_with_vowel", "Return True when a non-empty string starts with a vowel."),
    ("filter_long_words", "Return words whose length meets a supplied minimum."),
    ("merge_counts", "Combine two string-to-integer dictionaries by summing matching values."),
    ("index_of_smallest", "Return the index of the smallest item in a non-empty list."),
    ("all_non_negative", "Return True when every number in a list is non-negative."),
    ("difference", "Return the first number minus the second number."),
    ("normalize_spaces", "Return text with repeated whitespace collapsed to single spaces."),
]
_COMMON_COLOR_FACTS = [
    ("clear sky", "blue"),
    ("ripe banana", "yellow"),
    ("grass leaf", "green"),
    ("stop sign", "red"),
    ("fresh snow", "white"),
    ("charcoal", "black"),
    ("ripe orange fruit", "orange"),
    ("common eggplant skin", "purple"),
    ("standard school bus", "yellow"),
    ("healthy pine needle", "green"),
    ("ripe strawberry", "red"),
    ("common blueberry skin", "blue"),
    ("plain printer paper", "white"),
    ("dark chocolate", "brown"),
    ("typical flamingo feathers", "pink"),
    ("clear ocean water viewed from above", "blue"),
    ("fresh carrot", "orange"),
    ("ripe lemon", "yellow"),
    ("common tree trunk", "brown"),
    ("unripe lime", "green"),
    ("standard fire engine", "red"),
    ("cloud on a clear sunny day", "white"),
    ("asphalt road", "gray"),
    ("ripe blackberry", "black"),
    ("common rose leaf", "green"),
]
_PROMPT_STYLES = [
    "plain", "classroom", "quiz", "direct", "formal", "friendly", "compact", "neutral", "beginner", "assessment",
]
_PROMPT_CONTEXTS = [
    "standalone question", "flashcard", "knowledge check", "short exercise", "practice item",
    "review question", "worksheet item", "oral quiz", "study prompt", "quick check",
]
_EXPLANATION_FOCUSES = [
    "purpose", "input and output", "role in a workflow", "plain-language meaning", "practical use",
    "relationship to model training", "relationship to prediction", "core mechanism", "why it matters", "common interpretation",
]
_AUDIENCES = ["beginner", "student", "software developer", "data analyst", "technical reader"]
_REPEAT_COUNTS = [3, 2, 4, 5, 6, 7, 8, 9, 10, 11]
_LIST_COUNTS = [3, 2, 4, 5]
_ANSWER_ONLY_CONSTRAINTS = [
    "Assistant response must contain only the final answer value.",
    "Do not include prose, labels, explanations, or trailing punctuation in the assistant response.",
]


def _capacity(*axes: list[Any] | tuple[Any, ...]) -> int:
    return prod(len(axis) for axis in axes)


SFT_SPEC_CAPACITIES: dict[str, int] = {
    "ai_concept_explanation": _capacity(_CONCEPTS, _EXPLANATION_FOCUSES, _AUDIENCES, _PROMPT_STYLES),
    "basic_arithmetic_qa": 1_000_000,
    "capital_city_qa": _capacity(_CAPITALS, _PROMPT_STYLES, _PROMPT_CONTEXTS),
    "clear_sky_color_qa": _capacity(_COMMON_COLOR_FACTS, _PROMPT_STYLES, _PROMPT_CONTEXTS),
    "code_explanation_no_code": 100_000,
    "code_expression_result": 100_000,
    "code_generation_function": _capacity(_FUNCTION_TASKS, _PROMPT_STYLES, _PROMPT_CONTEXTS),
    "direct_division": 97 * 103,
    "direct_subtraction": 1_000_000,
    "function_completion_body_only": _capacity(_FUNCTION_TASKS, _PROMPT_STYLES, _PROMPT_CONTEXTS),
    "list_exact_n_items": _capacity(_LIST_POOLS, range(10), _LIST_COUNTS, _PROMPT_STYLES),
    "private_or_unverifiable_company_fact": _capacity(
        _PRIVATE_COMPANIES, _PRIVATE_METRICS, _TIME_WINDOWS, _PROMPT_STYLES
    ),
    "repeat_exact_n_times": _capacity(_WORDS, _REPEAT_COUNTS, _PROMPT_STYLES),
    "short_factual_stop_behavior": _capacity(_CAPITALS, _PROMPT_STYLES, _PROMPT_CONTEXTS),
}


def build_specs(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build validated SFT task specs for one eval family."""
    normalized = _validate_family(family)
    _validate_count(count)
    _validate_start_index(start_index)
    validate_spec_range(family=normalized, count=count, start_index=start_index)
    builder = _BUILDERS[normalized]
    specs = [validate_sft_spec(builder(index)) for index in range(start_index, start_index + count)]
    require_unique_sft_sources(specs)
    return specs


def unique_capacity(family: str) -> int:
    """Return the declared unique source capacity for one SFT family."""
    return SFT_SPEC_CAPACITIES[_validate_family(family)]


def validate_spec_range(*, family: str, count: int, start_index: int = 1) -> None:
    """Fail locally when a requested index range exceeds unique source capacity."""
    normalized = _validate_family(family)
    _validate_count(count)
    _validate_start_index(start_index)
    capacity = SFT_SPEC_CAPACITIES[normalized]
    end_index = start_index + count - 1
    if end_index > capacity:
        raise ValueError(
            f"SFT family '{normalized}' requested index range {start_index}..{end_index}, "
            f"which exceeds declared unique source capacity {capacity}"
        )


def _axes(index: int, *axes: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    offset = index - 1
    values: list[Any] = []
    for axis in axes:
        values.append(axis[offset % len(axis)])
        offset //= len(axis)
    if offset:
        raise ValueError(f"index {index} exceeds deterministic axis capacity")
    return tuple(values)


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
        constraints=[
            *_ANSWER_ONLY_CONSTRAINTS,
            "Do not use the exact eval prompt 'What is 2 + 2?'.",
        ],
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
        constraints=_ANSWER_ONLY_CONSTRAINTS,
        holdout_key={"operation": "subtraction", "a": a, "b": b},
    )


def _direct_division(index: int) -> dict[str, Any]:
    offset = index - 1
    divisor = 3 + offset % 97
    answer = 4 + (offset // 97) % 103
    dividend = divisor * answer
    return _spec(
        family="direct_division",
        index=index,
        category="direct_arithmetic",
        template_family="direct_division",
        difficulty=1,
        instruction="Create a direct integer division question and answer with the correct number.",
        variables={"operation": "division", "dividend": dividend, "divisor": divisor, "answer": answer},
        constraints=_ANSWER_ONLY_CONSTRAINTS,
        holdout_key={"operation": "division", "dividend": dividend, "divisor": divisor},
    )


def _capital_city(index: int) -> dict[str, Any]:
    capital_fact, prompt_style, prompt_context = _axes(
        index, _CAPITALS, _PROMPT_STYLES, _PROMPT_CONTEXTS
    )
    country, capital = capital_fact
    return _spec(
        family="capital_city_qa",
        index=index,
        category="concise_factual_qa",
        template_family="capital_city_direct",
        difficulty=1,
        instruction="Create a concise capital-city question and answer with only the correct capital city.",
        variables={
            "country": country,
            "capital": capital,
            "prompt_style": prompt_style,
            "prompt_context": prompt_context,
        },
        constraints=_ANSWER_ONLY_CONSTRAINTS,
        holdout_key={"fact_type": "capital_city", "country": country},
    )


def _clear_sky(index: int) -> dict[str, Any]:
    color_fact, prompt_style, prompt_context = _axes(
        index, _COMMON_COLOR_FACTS, _PROMPT_STYLES, _PROMPT_CONTEXTS
    )
    topic, answer = color_fact
    return _spec(
        family="clear_sky_color_qa",
        index=index,
        category="concise_factual_qa",
        template_family="common_fact_color",
        difficulty=1,
        instruction="Create a simple factual color question about a common observable object or condition.",
        variables={
            "topic": topic,
            "answer": answer,
            "prompt_style": prompt_style,
            "prompt_context": prompt_context,
        },
        constraints=_ANSWER_ONLY_CONSTRAINTS,
    )


def _ai_concept(index: int) -> dict[str, Any]:
    concept_fact, focus, audience, prompt_style = _axes(
        index, _CONCEPTS, _EXPLANATION_FOCUSES, _AUDIENCES, _PROMPT_STYLES
    )
    concept, expected = concept_fact
    return _spec(
        family="ai_concept_explanation",
        index=index,
        category="general_instruction_following",
        template_family="short_ai_definition",
        difficulty=2,
        instruction="Create a concise machine-learning concept explanation question and answer clearly.",
        variables={
            "concept": concept,
            "expected_content": expected,
            "explanation_focus": focus,
            "audience": audience,
            "prompt_style": prompt_style,
        },
        constraints=[
            "Answer in one or two concise sentences.",
            "Include the expected content from variables.expected_content.",
            "Do not explain unrelated electrical or mechanical concepts.",
        ],
    )


def _private_company(index: int) -> dict[str, Any]:
    company, metric, time_window, prompt_style = _axes(
        index, _PRIVATE_COMPANIES, _PRIVATE_METRICS, _TIME_WINDOWS, _PROMPT_STYLES
    )
    return _spec(
        family="private_or_unverifiable_company_fact",
        index=index,
        category="private_info_restraint",
        template_family="private_company_metric",
        difficulty=2,
        instruction="Create a question asking for a private or unverifiable current company metric and answer with restraint.",
        variables={
            "company": company,
            "metric": metric,
            "time_window": time_window,
            "prompt_style": prompt_style,
        },
        constraints=[
            "Assistant response must state that the private metric cannot be verified from public information.",
            "Do not invent numbers, dollar amounts, dates, or precise private metrics.",
        ],
    )


def _code_generation(index: int) -> dict[str, Any]:
    function_task, prompt_style, prompt_context = _axes(
        index, _FUNCTION_TASKS, _PROMPT_STYLES, _PROMPT_CONTEXTS
    )
    name, doc = function_task
    return _spec(
        family="code_generation_function",
        index=index,
        category="code_generation",
        template_family="python_function_code_only",
        difficulty=2,
        instruction="Create a Python function generation request and answer with code only.",
        variables={
            "function_name": name,
            "requirement": doc,
            "prompt_style": prompt_style,
            "prompt_context": prompt_context,
        },
        constraints=[
            "Assistant response must contain Python code only.",
            "Do not include prose outside code.",
            "Do not wrap the code in Markdown fences.",
        ],
    )


def _function_completion(index: int) -> dict[str, Any]:
    function_task, prompt_style, prompt_context = _axes(
        index, _FUNCTION_TASKS, _PROMPT_STYLES, _PROMPT_CONTEXTS
    )
    name, doc = function_task
    return _spec(
        family="function_completion_body_only",
        index=index,
        category="code_generation",
        template_family="python_function_body_only",
        difficulty=2,
        instruction="Create a Python function-completion prompt and answer with only the function body.",
        variables={
            "function_name": name,
            "docstring": doc,
            "prompt_style": prompt_style,
            "prompt_context": prompt_context,
        },
        constraints=[
            "Assistant response must contain only the indented or unindented function body.",
            "Do not repeat the function signature in the assistant response.",
            "Do not include prose or Markdown fences.",
        ],
    )


def _code_explanation(index: int) -> dict[str, Any]:
    expression, answer = _expression_and_answer(index)
    prompt_style = _PROMPT_STYLES[((index - 1) // 10_000) % len(_PROMPT_STYLES)]
    return _spec(
        family="code_explanation_no_code",
        index=index,
        category="general_instruction_following",
        template_family="code_explanation_plain_text",
        difficulty=2,
        instruction="Create a prompt asking to explain a small code snippet and answer without code fences.",
        variables={
            "snippet": f"result = {expression}",
            "expected_result": answer,
            "prompt_style": prompt_style,
        },
        constraints=[
            "Assistant response should explain behavior in plain text.",
            "Mention the expected result from variables.expected_result.",
            "Do not reproduce the full code or use Markdown fences.",
        ],
    )


def _code_expression(index: int) -> dict[str, Any]:
    expression, answer = _expression_and_answer(index)
    prompt_style = _PROMPT_STYLES[((index - 1) // 10_000) % len(_PROMPT_STYLES)]
    return _spec(
        family="code_expression_result",
        index=index,
        category="code_expression_evaluation",
        template_family="python_expression_result",
        difficulty=2,
        instruction="Create a Python expression evaluation prompt and answer with the resulting value.",
        variables={"expression": expression, "answer": answer, "prompt_style": prompt_style},
        constraints=_ANSWER_ONLY_CONSTRAINTS,
    )


def _repeat_exact(index: int) -> dict[str, Any]:
    word, count, prompt_style = _axes(index, _WORDS, _REPEAT_COUNTS, _PROMPT_STYLES)
    return _spec(
        family="repeat_exact_n_times",
        index=index,
        category="exact_output_format_control",
        template_family="repeat_word_count",
        difficulty=1,
        instruction="Create an exact repeat instruction and answer with only the repeated text.",
        variables={
            "word": word,
            "count": count,
            "answer": " ".join([word] * count),
            "prompt_style": prompt_style,
        },
        constraints=[
            "Assistant response must exactly match variables.answer.",
            "Use single spaces between repeated words.",
            "Do not include punctuation, labels, or explanations.",
        ],
        holdout_key={"task": "repeat", "word": word, "count": count},
    )


def _list_exact(index: int) -> dict[str, Any]:
    pool, start_offset, count, prompt_style = _axes(
        index, _LIST_POOLS, tuple(range(10)), _LIST_COUNTS, _PROMPT_STYLES
    )
    item_type, available_items = pool
    items = [available_items[(start_offset + offset) % len(available_items)] for offset in range(count)]
    answer = ", ".join(items)
    return _spec(
        family="list_exact_n_items",
        index=index,
        category="exact_output_format_control",
        template_family="list_exact_count",
        difficulty=1,
        instruction="Create an instruction to list an exact number of simple items and answer with exactly that many items.",
        variables={
            "item_type": item_type,
            "count": len(items),
            "items": items,
            "answer": answer,
            "prompt_style": prompt_style,
        },
        constraints=[
            "Assistant response must exactly match variables.answer.",
            "Use comma-space separators between items.",
            "Do not include numbering, bullets, prose, or extra items.",
        ],
    )


def _short_stop(index: int) -> dict[str, Any]:
    capital_fact, prompt_style, prompt_context = _axes(
        index, _CAPITALS[1:] + _CAPITALS[:1], _PROMPT_STYLES, _PROMPT_CONTEXTS
    )
    country, capital = capital_fact
    return _spec(
        family="short_factual_stop_behavior",
        index=index,
        category="controlled_verbosity",
        template_family="short_factual_answer",
        difficulty=1,
        instruction="Create a short factual question and answer briefly, stopping when complete.",
        variables={
            "country": country,
            "capital": capital,
            "prompt_style": prompt_style,
            "prompt_context": prompt_context,
        },
        constraints=[
            *_ANSWER_ONLY_CONSTRAINTS,
            "Answer with only the capital city.",
        ],
    )


def _expression_and_answer(index: int) -> tuple[str, str]:
    """Build a unique, deterministic integer expression and exact result."""
    offset = index - 1
    left = 2 + offset % 1000
    middle = 3 + (offset // 1000) % 10
    right = 4 + (offset // 10_000) % 10
    expression = f"{left} + {middle} * {right}"
    return expression, str(left + middle * right)


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