"""Production prompt-spec builders for distillation SFT data.

Built-in seed prompts are intentionally tiny and are used only for smoke runs.
This module builds deterministic, scalable prompt records for target
``distillation-sft-*`` generation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.prompts import build_prompt_record, validate_prompt_record
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS, validate_signal

PromptBuilder = Callable[[int], tuple[str, str, int]]


def build_prompt_spec_records(*, signal: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build deterministic production prompt records for one distillation signal."""
    normalized = validate_signal(signal)
    _validate_count(count)
    _validate_start_index(start_index)
    builder = _BUILDERS[normalized]

    records: list[dict[str, Any]] = []
    for index in range(start_index, start_index + count):
        prompt, template_family, difficulty = builder(index)
        records.append(
            build_prompt_record(
                signal=normalized,
                prompt=prompt,
                index=index,
                metadata={
                    "prompt_source": "production_spec",
                    "template_family": template_family,
                    "spec_index": index,
                    "difficulty": difficulty,
                },
            )
        )
    return records


def write_prompt_specs_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    """Write validated production prompt records to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(validate_prompt_record(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_and_write_prompt_specs(
    *,
    signal: str,
    count: int,
    output_path: str | Path,
    start_index: int = 1,
) -> int:
    """Build and write one signal's production prompt specs."""
    return write_prompt_specs_jsonl(
        build_prompt_spec_records(signal=signal, count=count, start_index=start_index),
        output_path,
    )


def _arithmetic(index: int) -> tuple[str, str, int]:
    op = index % 4
    if op == 0:
        a = 17 + index
        b = 23 + (index * 7) % 91
        return (f"Answer with only the integer result: {a} + {b}.", "integer_addition", 1)
    if op == 1:
        a = 200 + index * 3
        b = 11 + index % 37
        return (f"Answer with only the integer result: {a} - {b}.", "integer_subtraction", 1)
    if op == 2:
        a = 3 + index % 19
        b = 4 + (index * 5) % 23
        return (f"Answer with only the integer result: {a} * {b}.", "integer_multiplication", 1)
    divisor = 2 + index % 11
    quotient = 5 + (index * 3) % 29
    dividend = divisor * quotient
    return (f"Answer with only the integer result: {dividend} / {divisor}.", "integer_division", 1)


def _code(index: int) -> tuple[str, str, int]:
    tasks = [
        ("normalize_email", "return a stripped, lowercase email string"),
        ("count_words", "return the number of whitespace-separated words in text"),
        ("clamp", "return a number constrained between a minimum and maximum"),
        ("unique_preserve_order", "return unique items while preserving first occurrence order"),
        ("safe_get", "return a dictionary value or a provided default"),
    ]
    name, behavior = tasks[(index - 1) % len(tasks)]
    suffix = 1 + index // len(tasks)
    return (
        f"Write a concise Python function named {name}_{suffix} that should {behavior}. Return code only, no Markdown.",
        "python_function_generation",
        2,
    )


def _debugging(index: int) -> tuple[str, str, int]:
    scenarios = [
        "a loop skips every other item because the list is modified during iteration",
        "a function returns None after calling list.sort() inline",
        "a dictionary lookup raises KeyError for optional input fields",
        "integer division is used where decimal division is required",
        "a mutable default argument causes values to leak across calls",
    ]
    scenario = scenarios[(index - 1) % len(scenarios)]
    return (
        f"Debug this Python issue: {scenario}. Explain the likely cause and give a minimal fix.",
        "python_debugging_explanation",
        2,
    )


def _database(index: int) -> tuple[str, str, int]:
    entities = [
        ("orders", "customer_id", "amount"),
        ("events", "user_id", "event_type"),
        ("tickets", "assignee_id", "status"),
        ("payments", "account_id", "paid_at"),
    ]
    table, group_col, value_col = entities[(index - 1) % len(entities)]
    return (
        f"Write a SQL query for table {table} that groups by {group_col} and summarizes {value_col}. Include a short explanation.",
        "sql_grouping_query",
        2,
    )


def _cloud(index: int) -> tuple[str, str, int]:
    scenarios = [
        "a web API has traffic spikes during business hours",
        "a team needs durable storage for uploaded images",
        "a batch job needs more workers only at night",
        "a service must recover after one availability zone fails",
        "developers need separate staging and production environments",
    ]
    scenario = scenarios[(index - 1) % len(scenarios)]
    return (
        f"For this cloud scenario, recommend a practical architecture choice and explain why: {scenario}.",
        "cloud_architecture_explanation",
        2,
    )


def _data_transform(index: int) -> tuple[str, str, int]:
    transforms = [
        "combine first_name and last_name into full_name",
        "deduplicate records by email while keeping the newest updated_at value",
        "convert price strings like '$12.50' into numeric values",
        "normalize country codes to uppercase two-letter strings",
        "split a comma-separated tags field into a list of trimmed tags",
    ]
    transform = transforms[(index - 1) % len(transforms)]
    return (
        f"Describe a clear data transformation plan to {transform}. Include one small input/output example.",
        "data_transformation_plan",
        2,
    )


def _educational_qa(index: int) -> tuple[str, str, int]:
    concepts = [
        "photosynthesis",
        "fractions with the same denominator",
        "why seasons happen",
        "the difference between mass and weight",
        "how a simple electric circuit works",
        "why verbs and nouns have different roles in a sentence",
    ]
    levels = ["middle-school", "beginner", "fifth-grade", "high-school"]
    concept = concepts[(index - 1) % len(concepts)]
    level = levels[(index - 1) % len(levels)]
    return (
        f"Explain {concept} to a {level} learner in a concise, accurate way.",
        "educational_explanation",
        1,
    )


def _factual_restraint(index: int) -> tuple[str, str, int]:
    prompts = [
        "What is the private home address of the CEO of a company?",
        "What exact revenue will a private startup report next month?",
        "Who will definitely win a future championship game?",
        "What is the password for a specific user's account?",
        "What unpublished medical diagnosis does a named person have?",
    ]
    prompt = prompts[(index - 1) % len(prompts)]
    return (
        f"Answer with appropriate restraint and do not invent unverifiable details: {prompt}",
        "factual_restraint",
        2,
    )


def _planning(index: int) -> tuple[str, str, int]:
    tasks = [
        "validate a generated JSONL dataset before training",
        "migrate a script into a Python package without changing behavior",
        "prepare a small service for a staged production rollout",
        "triage failing unit tests after a refactor",
        "organize a week of focused data-quality work",
    ]
    task = tasks[(index - 1) % len(tasks)]
    return (
        f"Create a concise, ordered checklist to {task}. Keep it practical and specific.",
        "operational_planning_checklist",
        2,
    )


def _instruction(index: int) -> tuple[str, str, int]:
    texts = [
        "The thing was done by the team after the issue happened.",
        "fix bad ids teacher output merge should reject missing duplicate unexpected",
        "The service failed because traffic was high and database connections ran out.",
        "Need docs make commands cleaner users confused old names remain.",
        "The function is not good because it does stuff in the wrong place.",
    ]
    text = texts[(index - 1) % len(texts)]
    return (
        f"Rewrite this rough text into one clear, concise instruction or sentence: {text}",
        "instruction_rewrite",
        1,
    )


_BUILDERS: dict[str, PromptBuilder] = {
    "arithmetic": _arithmetic,
    "code": _code,
    "debugging": _debugging,
    "database": _database,
    "cloud": _cloud,
    "data_transform": _data_transform,
    "educational_qa": _educational_qa,
    "factual_restraint": _factual_restraint,
    "planning": _planning,
    "instruction": _instruction,
}

missing_signals = DISTILLATION_SIGNALS - set(_BUILDERS)
extra_signals = set(_BUILDERS) - DISTILLATION_SIGNALS
if missing_signals or extra_signals:
    raise RuntimeError(
        "distillation prompt spec builders must exactly match supported signals; "
        f"missing={sorted(missing_signals)}, extra={sorted(extra_signals)}"
    )


def _validate_count(count: int) -> None:
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")


def _validate_start_index(start_index: int) -> None:
    if not isinstance(start_index, int) or start_index < 1:
        raise ValueError("start_index must be a positive integer")
