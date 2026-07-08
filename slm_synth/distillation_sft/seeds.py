"""Small built-in prompt seeds for response-distillation smoke/pilot runs.

These are local prompt seeds only. They are not public training rows until they
are merged with validated teacher outputs and stripped to the public schema.
"""

from __future__ import annotations

from itertools import islice
from typing import Iterator

from slm_synth.distillation_sft.prompts import build_prompt_record
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS, validate_signal

DISTILLATION_PROMPT_SEEDS: dict[str, tuple[str, ...]] = {
    "arithmetic": (
        "Solve the arithmetic problem and give the final answer: 47 + 28.",
        "Solve the arithmetic problem step by step: 18 * 7.",
        "A box has 96 pencils. They are packed evenly into 12 bags. How many pencils are in each bag?",
    ),
    "code": (
        "Write a Python function that returns the largest integer in a non-empty list.",
        "Write a Python function that counts how many strings in a list start with a given prefix.",
        "Write a Python function that normalizes whitespace in a sentence.",
    ),
    "debugging": (
        "Find and explain the bug in this Python snippet: for i in range(len(items)): total += item[i]",
        "A function returns None instead of a sorted list after calling values.sort(). Explain the issue and fix it.",
        "Explain why this loop may never terminate: while count != target: count += 2",
    ),
    "database": (
        "Write a SQL query to count orders per customer from an orders table.",
        "Explain when to use an index on a database column.",
        "Given users(id, email) and purchases(user_id, amount), write a query for total purchase amount by email.",
    ),
    "cloud": (
        "Explain one practical use of autoscaling in a cloud application.",
        "Describe the difference between object storage and block storage.",
        "Give a short deployment checklist for a small web service on cloud infrastructure.",
    ),
    "data_transform": (
        "Convert a list of records with first_name and last_name into records with full_name.",
        "Explain how to deduplicate rows by email while keeping the newest timestamp.",
        "Transform a CSV column named price from strings like '$12.50' into numeric values.",
    ),
    "educational_qa": (
        "Explain photosynthesis to a middle-school student.",
        "What is the difference between a noun and a verb? Give one example of each.",
        "Explain why fractions with the same denominator can be added by adding the numerators.",
    ),
    "factual_restraint": (
        "What is the private home address of a public figure?",
        "Who will definitely win the next World Cup?",
        "Give a confident answer to a question when the source information is not provided.",
    ),
    "planning": (
        "Create a concise plan for preparing a small dataset for model fine-tuning.",
        "Plan the steps to migrate a script into a package without changing behavior.",
        "Make a checklist for validating a generated JSONL dataset before training.",
    ),
    "instruction": (
        "Rewrite this sentence to be clearer: The thing was done by the team after the issue happened.",
        "Summarize this in one sentence: The service failed during peak traffic because the database connection pool was exhausted.",
        "Turn this rough note into a clear task: fix bad ids teacher output merge should reject missing duplicate unexpected.",
    ),
}

missing_signals = DISTILLATION_SIGNALS - set(DISTILLATION_PROMPT_SEEDS)
extra_signals = set(DISTILLATION_PROMPT_SEEDS) - DISTILLATION_SIGNALS
if missing_signals or extra_signals:
    raise RuntimeError(
        "distillation prompt seeds must exactly match supported signals; "
        f"missing={sorted(missing_signals)}, extra={sorted(extra_signals)}"
    )


def iter_seed_prompts(signal: str) -> Iterator[str]:
    """Yield built-in seed prompts for one supported distillation signal."""
    normalized_signal = validate_signal(signal)
    yield from DISTILLATION_PROMPT_SEEDS[normalized_signal]


def build_seed_prompt_records(*, signal: str, count: int, start_index: int = 1) -> list[dict[str, object]]:
    """Build deterministic local prompt records by cycling signal seed prompts."""
    normalized_signal = validate_signal(signal)
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")

    seeds = DISTILLATION_PROMPT_SEEDS[normalized_signal]
    cycled_prompts = (seeds[index % len(seeds)] for index in range(count))

    records: list[dict[str, object]] = []
    for offset, prompt in enumerate(islice(cycled_prompts, count)):
        records.append(
            build_prompt_record(
                signal=normalized_signal,
                prompt=prompt,
                index=start_index + offset,
                metadata={
                    "seed_source": "builtin",
                    "seed_index": offset % len(seeds),
                },
            )
        )
    return records
