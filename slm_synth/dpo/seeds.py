"""Deterministic seed-row builders for synthetic DPO datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row
from slm_synth.taxonomy.holdouts import HoldoutRegistry, load_default_holdout_registry

DPO_SEED_FAMILIES = frozenset({"answer_only_arithmetic", "list_exact_n_items", "repeat_exact_n_times"})


def build_seed_rows(
    *,
    family: str,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic DPO seed rows for one supported family."""
    normalized_family = _validate_family(family)
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    registry = holdout_registry or load_default_holdout_registry()
    if normalized_family == "answer_only_arithmetic":
        return build_answer_only_arithmetic_rows(
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
    raise AssertionError(f"unhandled DPO seed family: {normalized_family}")


def build_answer_only_arithmetic_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build answer-only arithmetic DPO pairs without using held-out items."""
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

        answer = str(candidate["answer"])
        row = {
            "id": f"dpo_answer_only_arithmetic_{row_number:06d}",
            "prompt": [
                {"role": "user", "content": candidate["prompt"]},
            ],
            "chosen": [
                {"role": "assistant", "content": answer},
            ],
            "rejected": [
                {"role": "assistant", "content": candidate["rejected"]},
            ],
            "metadata": {
                "category": "answer_only_compliance",
                "failure_mode": "extra_explanation",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_dpo_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} answer-only arithmetic DPO rows")


def build_repeat_exact_n_times_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build exact-repeat DPO pairs without using held-out items."""
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
            "id": f"dpo_repeat_exact_n_times_{row_number:06d}",
            "prompt": [
                {"role": "user", "content": candidate["prompt"]},
            ],
            "chosen": [
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "rejected": [
                {"role": "assistant", "content": candidate["rejected"]},
            ],
            "metadata": {
                "category": "exact_output_format_control",
                "failure_mode": "format_violation",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_dpo_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} exact-repeat DPO rows")


def build_list_exact_n_items_rows(
    *,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> list[dict[str, Any]]:
    """Build exact-list DPO pairs without using held-out items."""
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
            "id": f"dpo_list_exact_n_items_{row_number:06d}",
            "prompt": [
                {"role": "user", "content": candidate["prompt"]},
            ],
            "chosen": [
                {"role": "assistant", "content": candidate["answer"]},
            ],
            "rejected": [
                {"role": "assistant", "content": candidate["rejected"]},
            ],
            "metadata": {
                "category": "exact_output_format_control",
                "failure_mode": "format_violation",
                "difficulty": candidate["difficulty"],
                "template_family": candidate["template_family"],
                "eval_family": candidate["eval_family"],
            },
        }
        rows.append(validate_dpo_row(row))
        if len(rows) >= count:
            return rows
        row_number += 1

    raise ValueError(f"could not build {count} exact-list DPO rows")


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
                    "rejected": f"The answer is {answer} because {left} plus {right} equals {answer}.",
                    "difficulty": 1 if answer < 50 else 2,
                    "template_family": "direct_qa",
                    "eval_family": "basic_arithmetic_qa",
                    "holdout_key": {
                        "type": "arithmetic",
                        "operation": "add",
                        "operands": [left, right],
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
            rejected = f"{answer} {token}"
            for template in templates:
                prompt = template.format(token=token, count_word=count_word)
                yield {
                    "prompt": prompt,
                    "answer": answer,
                    "rejected": rejected,
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
            answer_items = items[:count]
            rejected_items = items[: count + 1]
            answer = ", ".join(answer_items)
            rejected = ", ".join(rejected_items)
            for template in templates:
                prompt = template.format(count_word=count_word, item_type=item_type)
                yield {
                    "prompt": prompt,
                    "answer": answer,
                    "rejected": rejected,
                    "difficulty": 1 if count <= 3 else 2,
                    "template_family": "list_n_items",
                    "eval_family": "list_exact_n_items",
                    "holdout_key": {
                        "type": "list_exact_n_items",
                        "item_type": item_type,
                        "count": count,
                    },
                }


def _validate_family(family: str) -> str:
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    normalized = family.strip().lower()
    if normalized not in DPO_SEED_FAMILIES:
        supported = ", ".join(sorted(DPO_SEED_FAMILIES))
        raise ValueError(f"Unsupported DPO seed family '{family}'. Supported families: {supported}")
    return normalized
