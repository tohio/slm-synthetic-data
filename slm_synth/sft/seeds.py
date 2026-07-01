"""Deterministic seed-row builders for synthetic SFT datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slm_synth.sft.schema import validate_sft_row
from slm_synth.taxonomy.holdouts import HoldoutRegistry, load_default_holdout_registry

SFT_SEED_FAMILIES = frozenset(
    {
        "ai_concept_explanation",
        "answer_only_arithmetic",
        "capital_city_qa",
        "clear_sky_color_qa",
        "code_generation_function",
        "code_explanation_no_code",
        "function_completion_body_only",
        "list_exact_n_items",
        "private_or_unverifiable_company_fact",
        "repeat_exact_n_times",
    }
)


def build_seed_rows(
    *,
    family: str,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic SFT seed rows for one supported family."""
    normalized_family = _validate_family(family)
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    if normalized_family == "ai_concept_explanation":
        return build_ai_concept_explanation_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "answer_only_arithmetic":
        return build_answer_only_arithmetic_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "capital_city_qa":
        return build_capital_city_qa_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "clear_sky_color_qa":
        return build_clear_sky_color_qa_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "repeat_exact_n_times":
        return build_repeat_exact_n_times_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "list_exact_n_items":
        return build_list_exact_n_items_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "code_generation_function":
        return build_code_generation_function_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "code_explanation_no_code":
        return build_code_explanation_no_code_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "function_completion_body_only":
        return build_function_completion_body_only_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    if normalized_family == "private_or_unverifiable_company_fact":
        return build_private_or_unverifiable_company_fact_rows(
            count=count,
            start_index=start_index,
            holdout_registry=registry,
        )
    raise AssertionError(f"unhandled SFT seed family: {normalized_family}")


def build_ai_concept_explanation_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build concise AI concept explanation SFT rows without using held-out items."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _ai_concept_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_ai_concept_explanation_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "general_instruction_following",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} AI concept SFT rows")


def build_answer_only_arithmetic_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build answer-only arithmetic SFT siblings without using held-out items."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _arithmetic_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_answer_only_arithmetic_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": str(candidate["answer"])},
            ],
            "metadata": {
                "category": "answer_only_compliance",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} answer-only arithmetic SFT rows")


def _ai_concept_candidates() -> Iterable[dict[str, Any]]:
    concepts = (
        {
            "concept": "attention_mechanism",
            "prompt": "In machine learning, what is an attention mechanism?",
            "answer": "It helps a model focus on the most relevant parts of the input when producing an output.",
            "difficulty": 1,
        },
        {
            "concept": "embedding_vector",
            "prompt": "What is an embedding vector in machine learning?",
            "answer": "It is a numeric representation of data, such as a word or item, that a model can process.",
            "difficulty": 1,
        },
        {
            "concept": "training_loss",
            "prompt": "In AI training, what does loss measure?",
            "answer": "Loss measures how far the model's predictions are from the desired answers.",
            "difficulty": 1,
        },
        {
            "concept": "classification_model",
            "prompt": "What does a classification model do?",
            "answer": "It assigns an input to one of a set of categories or labels.",
            "difficulty": 1,
        },
        {
            "concept": "language_model",
            "prompt": "What is a language model?",
            "answer": "It is a model trained to predict or generate text based on patterns in language.",
            "difficulty": 1,
        },
    )

    for concept in concepts:
        yield {
            **concept,
            "template_family": "short_ai_concept_definition",
            "eval_family": "ai_concept_explanation",
            "holdout_key": {
                "type": "ai_concept",
                "concept": concept["concept"],
            },
        }


def build_capital_city_qa_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build concise capital-city QA SFT siblings without using held-out items."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _capital_city_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_capital_city_qa_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "concise_factual_qa",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} capital-city SFT rows")


def build_clear_sky_color_qa_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build concise sky-color QA SFT siblings without using held-out items."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _clear_sky_color_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_clear_sky_color_qa_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "concise_factual_qa",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} sky-color SFT rows")


def build_repeat_exact_n_times_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build exact-repeat SFT siblings without using held-out items."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _repeat_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_repeat_exact_n_times_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "exact_output_format_control",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} exact-repeat SFT rows")


def build_list_exact_n_items_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build exact-list SFT siblings without using held-out items."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _list_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_list_exact_n_items_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "exact_output_format_control",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} exact-list SFT rows")


def build_private_or_unverifiable_company_fact_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build SFT restraint rows for private or unverifiable company facts."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _private_company_fact_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_private_or_unverifiable_company_fact_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": candidate["category"],
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} private-company fact SFT rows")


def build_code_generation_function_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build SFT rows for simple code generation without using held-out tasks."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _code_generation_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_code_generation_function_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "code_generation",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} code-generation SFT rows")


def build_function_completion_body_only_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build SFT rows for function-body-only completion tasks."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _function_completion_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_function_completion_body_only_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "code_generation",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} function-completion SFT rows")


def build_code_explanation_no_code_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build SFT rows for code explanations that should not repeat code."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    rows: list[dict[str, Any]] = []
    row_number = start_index
    for candidate in _code_explanation_candidates():
        try:
            registry.reject_if_holdout(prompt=candidate["prompt"], holdout_key=candidate["holdout_key"])
        except ValueError:
            continue

        row = {
            "id": f"sft_code_explanation_no_code_{row_number:06d}",
            "messages": [
                {"role": "user", "content": candidate["prompt"]},
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "metadata": {
                "category": "general_instruction_following",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_sft_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} code-explanation SFT rows")


def _arithmetic_candidates() -> Iterable[dict[str, Any]]:
    templates = (
        "Answer with only the number: What is {left} + {right}?",
        "Compute {left} + {right}. Answer with only the number.",
        "What is {left} + {right}? Answer with only the number.",
    )

    for left in range(3, 80):
        for right in range(4, 80):
            answer = left + right
            for template in templates:
                prompt = template.format(left=left, right=right)
                yield {
                    "prompt": prompt,
                    "answer": answer,
                    "difficulty": 1 if answer < 50 else 2,
                    "template_family": "direct_qa",
                    "eval_family": "basic_arithmetic_qa",
                    "holdout_key": {
                        "type": "arithmetic",
                        "operation": "add",
                        "operands": [left, right],
                    },
                }


def _capital_city_candidates() -> Iterable[dict[str, Any]]:
    facts = (
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Germany", "Berlin"),
        ("Canada", "Ottawa"),
        ("Brazil", "Brasilia"),
        ("Egypt", "Cairo"),
        ("Kenya", "Nairobi"),
        ("India", "New Delhi"),
        ("Australia", "Canberra"),
        ("South Korea", "Seoul"),
        ("Mexico", "Mexico City"),
        ("Argentina", "Buenos Aires"),
    )
    templates = (
        "What is the capital of {country}?",
        "Name the capital city of {country}.",
        "Which city is the capital of {country}?",
    )

    for country, capital in facts:
        for template in templates:
            prompt = template.format(country=country)
            yield {
                "prompt": prompt,
                "answer": capital,
                "difficulty": 1,
                "template_family": "capital_city_direct",
                "eval_family": "capital_city_qa",
                "holdout_key": {
                    "type": "capital_city",
                    "country": country,
                    "answer": capital,
                },
            }


def _clear_sky_color_candidates() -> Iterable[dict[str, Any]]:
    facts = (
        (
            "On a cloudless day, what color does the sky usually appear?",
            "blue",
            "cloudless_day_sky_color",
        ),
        (
            "What color is the sky usually at midday when there are no clouds?",
            "blue",
            "midday_clear_sky_color",
        ),
        (
            "During clear daytime weather, what color is the sky generally?",
            "blue",
            "clear_daytime_sky_color",
        ),
        (
            "In normal daylight with no storm clouds, what color does the sky look?",
            "blue",
            "normal_daylight_sky_color",
        ),
        (
            "What is the usual color of a cloud-free daytime sky?",
            "blue",
            "cloud_free_daytime_sky_color",
        ),
    )

    for prompt, answer, relation in facts:
        yield {
            "prompt": prompt,
            "answer": answer,
            "difficulty": 1,
            "template_family": "sky_color_direct",
            "eval_family": "clear_sky_color_qa",
            "holdout_key": {
                "type": "factual_relation",
                "relation": relation,
                "answer": answer,
            },
        }


def _repeat_candidates() -> Iterable[dict[str, Any]]:
    tokens = ("dog", "blue", "yes", "sun", "green", "book", "code", "river")
    templates = (
        "Repeat {token} exactly {count_word} times.",
        'Output "{token}" exactly {count_word} times.',
        "Write {token} exactly {count_word} times and nothing else.",
    )
    count_words = {
        2: "two",
        3: "three",
        4: "four",
        5: "five",
    }

    for token in tokens:
        for count, count_word in count_words.items():
            answer = " ".join([token] * count)
            for template in templates:
                prompt = template.format(token=token, count_word=count_word)
                yield {
                    "prompt": prompt,
                    "answer": answer,
                    "difficulty": 1 if count <= 3 else 2,
                    "template_family": "repeat_word_count",
                    "eval_family": "repeat_exact_n_times",
                    "holdout_key": {
                        "type": "repeat_exact_n_times",
                        "token": token,
                        "count": count,
                    },
                }


def _list_candidates() -> Iterable[dict[str, Any]]:
    item_sets = (
        ("animals", ("dog", "cat", "horse", "bird", "fish")),
        ("fruits", ("apple", "banana", "pear", "orange", "plum")),
        ("planets", ("Mercury", "Venus", "Earth", "Mars", "Jupiter")),
        ("shapes", ("circle", "square", "triangle", "rectangle", "oval")),
        ("tools", ("hammer", "saw", "wrench", "drill", "level")),
    )
    templates = (
        "List exactly {count_word} {item_type}.",
        "Name exactly {count_word} {item_type}.",
        "Give exactly {count_word} {item_type} and nothing else.",
    )
    count_words = {
        2: "two",
        3: "three",
        4: "four",
    }

    for item_type, items in item_sets:
        for count, count_word in count_words.items():
            answer = ", ".join(items[:count])
            for template in templates:
                prompt = template.format(count_word=count_word, item_type=item_type)
                yield {
                    "prompt": prompt,
                    "answer": answer,
                    "difficulty": 1 if count <= 3 else 2,
                    "template_family": "list_n_items",
                    "eval_family": "list_exact_n_items",
                    "holdout_key": {
                        "type": "list_exact_n_items",
                        "item_type": item_type,
                        "count": count,
                    },
                }


def _private_company_fact_candidates() -> Iterable[dict[str, Any]]:
    companies = ("Cohere", "Scale AI", "Perplexity", "Mistral AI", "Character.AI", "Runway")
    facts = (
        ("private_revenue", "private revenue", "last week", "private_info_restraint"),
        ("internal_payroll_total", "internal payroll total", "last month", "private_info_restraint"),
        ("unreleased_user_count", "unreleased user count", "yesterday", "unknown_fact_restraint"),
        ("non_public_customer_count", "non-public customer count", "last quarter", "unknown_fact_restraint"),
    )
    templates = (
        "What was {company}'s {fact_label} {time_window}?",
        "Can you tell me {company}'s {fact_label} from {time_window}?",
        "What is the exact {fact_label} for {company} {time_window}?",
    )

    for company in companies:
        for fact_key, fact_label, time_window, category in facts:
            for template in templates:
                prompt = template.format(company=company, fact_label=fact_label, time_window=time_window)
                yield {
                    "prompt": prompt,
                    "answer": "I do not have access to that private or non-public information.",
                    "category": category,
                    "difficulty": 2,
                    "template_family": "private_company_fact",
                    "eval_family": "private_or_unverifiable_company_fact",
                    "holdout_key": {
                        "type": "unverifiable_company_fact",
                        "company": company,
                        "fact": fact_key,
                        "time_window": time_window.replace(" ", "_"),
                    },
                }


def _code_generation_candidates() -> Iterable[dict[str, Any]]:
    tasks = (
        {
            "function_name": "multiply",
            "task_key": "multiply_two_numbers",
            "prompt": "Write a Python function named multiply that multiplies two numbers. Return only code.",
            "answer": "def multiply(a, b):\n    return a * b",
            "difficulty": 1,
        },
        {
            "function_name": "is_even",
            "task_key": "is_even_number",
            "prompt": "Write a Python function named is_even that returns True if a number is even. Return only code.",
            "answer": "def is_even(n):\n    return n % 2 == 0",
            "difficulty": 1,
        },
        {
            "function_name": "last_item",
            "task_key": "return_last_item",
            "prompt": "Write a Python function named last_item that returns the last item in a list. Return only code.",
            "answer": "def last_item(items):\n    return items[-1]",
            "difficulty": 1,
        },
        {
            "function_name": "count_items",
            "task_key": "count_items",
            "prompt": "Write a Python function named count_items that returns the length of a list. Return only code.",
            "answer": "def count_items(items):\n    return len(items)",
            "difficulty": 1,
        },
        {
            "function_name": "cube",
            "task_key": "cube_number",
            "prompt": "Write a Python function named cube that returns the cube of a number. Return only code.",
            "answer": "def cube(x):\n    return x * x * x",
            "difficulty": 1,
        },
    )

    for task in tasks:
        yield {
            **task,
            "template_family": "simple_function_generation",
            "eval_family": "code_generation_function",
            "holdout_key": {
                "type": "code_generation",
                "function_name": task["function_name"],
                "function_task": task["task_key"],
            },
        }


def _function_completion_candidates() -> Iterable[dict[str, Any]]:
    tasks = (
        {
            "function_name": "double",
            "signature": "def double(x):",
            "docstring": '"""Return x doubled."""',
            "answer": "return x * 2",
            "task_key": "double_number",
            "difficulty": 1,
        },
        {
            "function_name": "is_empty",
            "signature": "def is_empty(items):",
            "docstring": '"""Return True if the list has no items."""',
            "answer": "return len(items) == 0",
            "task_key": "list_is_empty",
            "difficulty": 1,
        },
        {
            "function_name": "total",
            "signature": "def total(values):",
            "docstring": '"""Return the sum of the values."""',
            "answer": "return sum(values)",
            "task_key": "sum_values",
            "difficulty": 1,
        },
        {
            "function_name": "starts_with_a",
            "signature": "def starts_with_a(text):",
            "docstring": '"""Return True if text starts with the letter a."""',
            "answer": "return text.startswith(\"a\")",
            "task_key": "starts_with_a",
            "difficulty": 1,
        },
    )

    for task in tasks:
        prompt = (
            "Complete this Python function. Return only the function body.\n\n"
            f"{task['signature']}\n"
            f"    {task['docstring']}"
        )
        yield {
            **task,
            "prompt": prompt,
            "template_family": "function_body_completion",
            "eval_family": "function_completion_body_only",
            "holdout_key": {
                "type": "function_completion",
                "function_name": task["function_name"],
            },
        }


def _code_explanation_candidates() -> Iterable[dict[str, Any]]:
    tasks = (
        {
            "function_name": "double",
            "code": "def double(x):\n    return x * 2",
            "answer": "It returns the input value multiplied by two.",
            "operation": "double_number",
            "difficulty": 1,
        },
        {
            "function_name": "is_even",
            "code": "def is_even(n):\n    return n % 2 == 0",
            "answer": "It returns True when the input number is even.",
            "operation": "check_even_number",
            "difficulty": 1,
        },
        {
            "function_name": "total",
            "code": "def total(values):\n    return sum(values)",
            "answer": "It returns the sum of all values in the input list.",
            "operation": "sum_values",
            "difficulty": 1,
        },
        {
            "function_name": "last_item",
            "code": "def last_item(items):\n    return items[-1]",
            "answer": "It returns the final item from the input list.",
            "operation": "return_last_item",
            "difficulty": 1,
        },
    )

    for task in tasks:
        prompt = f"Explain what this Python function does:\n\n{task['code']}"
        yield {
            **task,
            "prompt": prompt,
            "template_family": "plain_code_explanation",
            "eval_family": "code_explanation_no_code",
            "holdout_key": {
                "type": "code_explanation",
                "function_name": task["function_name"],
                "operation": task["operation"],
            },
        }


def _validate_family(family: str) -> str:
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    normalized = family.strip().lower()
    if normalized not in SFT_SEED_FAMILIES:
        supported = ", ".join(sorted(SFT_SEED_FAMILIES))
        raise ValueError(f"Unsupported SFT seed family '{family}'. Supported families: {supported}")
    return normalized
