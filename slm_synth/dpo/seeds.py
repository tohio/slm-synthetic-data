"""Deterministic seed-row builders for synthetic DPO datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row
from slm_synth.taxonomy.holdouts import HoldoutRegistry, load_default_holdout_registry

DPO_SEED_FAMILIES = frozenset({"answer_only_arithmetic"})


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


def _validate_family(family: str) -> str:
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    normalized = family.strip().lower()
    if normalized not in DPO_SEED_FAMILIES:
        supported = ", ".join(sorted(DPO_SEED_FAMILIES))
        raise ValueError(f"Unsupported DPO seed family '{family}'. Supported families: {supported}")
    return normalized
