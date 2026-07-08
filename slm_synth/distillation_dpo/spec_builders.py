"""Production pair-spec builders for distillation-DPO artifacts.

``distillation-dpo-*`` is intentionally isolated from generic DPO. Production
pairs use teacher-quality chosen responses and controlled-weak rejected
responses. Student-model sampling belongs in ``slm-distillation`` or a later
explicit workflow, not in this synthetic data repo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from slm_synth.distillation_dpo.seeds import validate_family

PairBuilder = Callable[[int], dict[str, Any]]


def build_production_rows(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build deterministic production distillation-DPO rows for one family."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    _validate_positive_int(start_index, "start_index")
    builder = _BUILDERS[normalized_family]

    rows: list[dict[str, Any]] = []
    for index in range(start_index, start_index + count):
        spec = builder(index)
        rows.append(
            {
                "id": f"distillation-dpo-{normalized_family}-production-{index:06d}",
                "prompt": [{"role": "user", "content": spec["prompt"]}],
                "chosen": [{"role": "assistant", "content": spec["chosen"]}],
                "rejected": [{"role": "assistant", "content": spec["rejected"]}],
                "metadata": {
                    "category": spec["category"],
                    "difficulty": spec["difficulty"],
                    "template_family": spec["template_family"],
                    "eval_family": spec["eval_family"],
                    "failure_mode": spec["failure_mode"],
                },
            }
        )
    return rows


def _teacher_response_preference(index: int) -> dict[str, Any]:
    templates = (
        _arithmetic_pair,
        _answer_only_pair,
        _repeat_exact_pair,
        _list_exact_pair,
        _code_function_pair,
        _code_expression_pair,
        _factual_restraint_pair,
        _concise_factual_pair,
        _subtraction_pair,
        _division_pair,
    )
    return templates[(index - 1) % len(templates)](index)


def _arithmetic_pair(index: int) -> dict[str, Any]:
    left = 17 + index
    right = 23 + (index * 7) % 91
    correct = left + right
    wrong = correct + 1 + (index % 3)
    return {
        "prompt": f"Answer with only the integer result: {left} + {right}.",
        "chosen": str(correct),
        "rejected": str(wrong),
        "category": "direct_arithmetic",
        "difficulty": 1,
        "template_family": "distillation_dpo_integer_addition",
        "eval_family": "basic_arithmetic_qa",
        "failure_mode": "wrong_numeric_answer",
    }


def _subtraction_pair(index: int) -> dict[str, Any]:
    left = 200 + index * 3
    right = 11 + index % 37
    correct = left - right
    wrong = correct - 2 - (index % 4)
    return {
        "prompt": f"Answer with only the integer result: {left} - {right}.",
        "chosen": str(correct),
        "rejected": str(wrong),
        "category": "direct_arithmetic",
        "difficulty": 1,
        "template_family": "distillation_dpo_integer_subtraction",
        "eval_family": "direct_subtraction",
        "failure_mode": "wrong_numeric_answer",
    }


def _division_pair(index: int) -> dict[str, Any]:
    divisor = 2 + index % 11
    quotient = 5 + (index * 3) % 29
    dividend = divisor * quotient
    wrong = quotient + 1
    return {
        "prompt": f"Answer with only the integer result: {dividend} / {divisor}.",
        "chosen": str(quotient),
        "rejected": str(wrong),
        "category": "direct_arithmetic",
        "difficulty": 1,
        "template_family": "distillation_dpo_integer_division",
        "eval_family": "direct_division",
        "failure_mode": "wrong_numeric_answer",
    }


def _answer_only_pair(index: int) -> dict[str, Any]:
    cities = (
        ("France", "Paris", "Lyon"),
        ("Japan", "Tokyo", "Osaka"),
        ("Canada", "Ottawa", "Toronto"),
        ("Brazil", "Brasília", "Rio de Janeiro"),
    )
    country, correct, wrong = cities[(index - 1) % len(cities)]
    return {
        "prompt": f"What is the capital city of {country}? Answer with only the city name.",
        "chosen": correct,
        "rejected": f"The capital city of {country} is {correct}.",
        "category": "answer_only_compliance",
        "difficulty": 1,
        "template_family": "distillation_dpo_answer_only_capital",
        "eval_family": "capital_city_qa",
        "failure_mode": "extra_explanation",
    }


def _repeat_exact_pair(index: int) -> dict[str, Any]:
    words = ("echo", "blue", "token", "river")
    count = 2 + index % 4
    word = words[(index - 1) % len(words)]
    chosen = " ".join([word] * count)
    rejected = " ".join([word] * (count + 1))
    return {
        "prompt": f"Repeat the word {word} exactly {count} times, separated by spaces.",
        "chosen": chosen,
        "rejected": rejected,
        "category": "exact_output_format_control",
        "difficulty": 1,
        "template_family": "distillation_dpo_repeat_exact",
        "eval_family": "repeat_exact_n_times",
        "failure_mode": "format_violation",
    }


def _list_exact_pair(index: int) -> dict[str, Any]:
    topics = ("colors", "animals", "fruits", "tools")
    values = {
        "colors": ["red", "blue", "green", "yellow", "orange"],
        "animals": ["cat", "dog", "bird", "horse", "fish"],
        "fruits": ["apple", "banana", "pear", "orange", "plum"],
        "tools": ["hammer", "saw", "wrench", "drill", "level"],
    }
    topic = topics[(index - 1) % len(topics)]
    count = 3
    chosen_items = values[topic][:count]
    rejected_items = values[topic][: count + 1]
    return {
        "prompt": f"List exactly {count} common {topic}, separated by commas.",
        "chosen": ", ".join(chosen_items),
        "rejected": ", ".join(rejected_items),
        "category": "exact_output_format_control",
        "difficulty": 1,
        "template_family": "distillation_dpo_list_exact",
        "eval_family": "list_exact_n_items",
        "failure_mode": "format_violation",
    }


def _code_function_pair(index: int) -> dict[str, Any]:
    suffix = 1 + index // 10
    return {
        "prompt": f"Write a Python function named clamp_{suffix} that returns a number constrained between a minimum and maximum. Return code only, no Markdown.",
        "chosen": f"def clamp_{suffix}(value, minimum, maximum):\n    return max(minimum, min(value, maximum))",
        "rejected": f"def clamp_{suffix}(value, minimum, maximum)\n    return value",
        "category": "code_generation",
        "difficulty": 2,
        "template_family": "distillation_dpo_python_function",
        "eval_family": "code_generation_function",
        "failure_mode": "code_syntax_error",
    }


def _code_expression_pair(index: int) -> dict[str, Any]:
    value = 3 + index % 9
    return {
        "prompt": f"What is the result of this Python expression? Answer with only the value: {value} * 2 + 1",
        "chosen": str(value * 2 + 1),
        "rejected": str(value * 2),
        "category": "code_expression_evaluation",
        "difficulty": 1,
        "template_family": "distillation_dpo_python_expression",
        "eval_family": "code_expression_result",
        "failure_mode": "wrong_numeric_answer",
    }


def _factual_restraint_pair(index: int) -> dict[str, Any]:
    subjects = (
        "the private home address of a company CEO",
        "the unreleased revenue number for a private startup next month",
        "the password for a specific user's account",
        "a future championship winner stated as certain",
    )
    subject = subjects[(index - 1) % len(subjects)]
    return {
        "prompt": f"Answer safely and do not invent unverifiable details: What is {subject}?",
        "chosen": "I cannot verify that from reliable public information, so I should not invent it.",
        "rejected": "The answer is definitely available, and I can provide the exact private detail.",
        "category": "private_info_restraint",
        "difficulty": 2,
        "template_family": "distillation_dpo_factual_restraint",
        "eval_family": "private_or_unverifiable_company_fact",
        "failure_mode": "unsafe_private_info_guess",
    }


def _concise_factual_pair(index: int) -> dict[str, Any]:
    facts = (
        ("What color is a clear daytime sky? Answer in one word.", "blue", "A clear daytime sky is usually blue because sunlight scatters in the atmosphere."),
        ("What do bees make? Answer in one word.", "honey", "Bees can make honey, wax, and other things depending on the context."),
        ("What gas do humans need to breathe? Answer in one word.", "oxygen", "Humans need oxygen, which is part of air."),
    )
    prompt, chosen, rejected = facts[(index - 1) % len(facts)]
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": "answer_only_compliance",
        "difficulty": 1,
        "template_family": "distillation_dpo_concise_factual",
        "eval_family": "clear_sky_color_qa",
        "failure_mode": "extra_explanation",
    }


_BUILDERS: dict[str, PairBuilder] = {
    "teacher_response_preference": _teacher_response_preference,
}


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
