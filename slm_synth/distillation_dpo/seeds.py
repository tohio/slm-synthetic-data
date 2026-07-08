"""Smoke-only seed rows for distillation-DPO artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DISTILLATION_DPO_FAMILIES = frozenset({"teacher_response_preference"})

_SEED_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "prompt": [{"role": "user", "content": "What is 2 + 2? Answer with only the number."}],
        "chosen": [{"role": "assistant", "content": "4"}],
        "rejected": [{"role": "assistant", "content": "5"}],
        "metadata": {
            "category": "direct_arithmetic",
            "difficulty": 1,
            "template_family": "distillation_dpo_teacher_preference",
            "eval_family": "basic_arithmetic_qa",
            "failure_mode": "wrong_numeric_answer",
        },
    },
    {
        "prompt": [{"role": "user", "content": "Name the usual color of a clear daytime sky. Answer in one word."}],
        "chosen": [{"role": "assistant", "content": "blue"}],
        "rejected": [{"role": "assistant", "content": "The usual color of a clear daytime sky is blue."}],
        "metadata": {
            "category": "answer_only_compliance",
            "difficulty": 1,
            "template_family": "distillation_dpo_teacher_preference",
            "eval_family": "clear_sky_color_qa",
            "failure_mode": "extra_explanation",
        },
    },
    {
        "prompt": [{"role": "user", "content": "Repeat the word echo exactly three times, separated by spaces."}],
        "chosen": [{"role": "assistant", "content": "echo echo echo"}],
        "rejected": [{"role": "assistant", "content": "echo echo echo echo"}],
        "metadata": {
            "category": "exact_output_format_control",
            "difficulty": 1,
            "template_family": "distillation_dpo_teacher_preference",
            "eval_family": "repeat_exact_n_times",
            "failure_mode": "format_violation",
        },
    },
)


def validate_family(family: Any) -> str:
    """Return a normalized distillation-DPO family name or raise."""
    if not isinstance(family, str) or not family.strip():
        raise ValueError("distillation-DPO family must be a non-empty string")
    normalized = family.strip().lower()
    if normalized not in DISTILLATION_DPO_FAMILIES:
        supported = ", ".join(sorted(DISTILLATION_DPO_FAMILIES))
        raise ValueError(f"Unsupported distillation-DPO family '{family}'. Supported families: {supported}")
    return normalized


def build_seed_rows(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build deterministic smoke rows for one distillation-DPO family.

    These rows are intentionally smoke/control rows. Production distillation-DPO
    data should later replace the controlled-weak rejection source with sampled
    student responses or another explicitly approved weak-response source.
    """
    normalized_family = validate_family(family)
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    rows: list[dict[str, Any]] = []
    for offset in range(count):
        template = _SEED_TEMPLATES[offset % len(_SEED_TEMPLATES)]
        row_number = start_index + offset
        rows.append(
            {
                "id": f"distillation-dpo-{normalized_family}-{row_number:06d}",
                "prompt": _copy_messages(template["prompt"]),
                "chosen": _copy_messages(template["chosen"]),
                "rejected": _copy_messages(template["rejected"]),
                "metadata": dict(template["metadata"]),
            }
        )
    return rows


def _copy_messages(messages: list[Mapping[str, str]]) -> list[dict[str, str]]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]
