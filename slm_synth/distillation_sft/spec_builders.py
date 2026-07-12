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
    sequence = ((index - 1) // 4) + 1
    if op == 0:
        a = 17 + sequence * 11
        b = 23 + sequence * 7
        return (f"Answer with only the integer result: {a} + {b}.", "integer_addition", 1)
    if op == 1:
        b = 11 + sequence * 3
        a = b + 100 + sequence * 13
        return (f"Answer with only the integer result: {a} - {b}.", "integer_subtraction", 1)
    if op == 2:
        a = 3 + sequence
        b = 4 + sequence * 2
        return (f"Answer with only the integer result: {a} * {b}.", "integer_multiplication", 1)
    divisor = 2 + (sequence % 11)
    quotient = 5 + sequence * 3
    dividend = divisor * quotient
    return (f"Answer with only the integer result: {dividend} / {divisor}.", "integer_division", 1)


def _code(index: int) -> tuple[str, str, int]:
    tasks = (
        ("normalize_email", "return a stripped, lowercase email string", "python_string_normalization"),
        ("count_words", "return the number of whitespace-separated words in text", "python_text_aggregation"),
        ("clamp", "return a number constrained between a minimum and maximum", "python_boundary_validation"),
        (
            "unique_preserve_order",
            "return unique items while preserving first occurrence order",
            "python_order_preserving_collection",
        ),
        ("safe_get", "return a dictionary value or a provided default", "python_mapping_lookup"),
    )
    name, behavior, template_family = tasks[(index - 1) % len(tasks)]
    suffix = ((index - 1) // len(tasks)) + 1
    return (
        f"Write a concise Python function named {name}_{suffix} that should {behavior}. Return code only, no Markdown.",
        template_family,
        2,
    )


def _debugging(index: int) -> tuple[str, str, int]:
    scenarios = (
        ("a loop skips every other item because the list is modified during iteration", "python_iteration_mutation_bug"),
        ("a function returns None after calling list.sort() inline", "python_in_place_method_bug"),
        ("a dictionary lookup raises KeyError for optional input fields", "python_optional_key_bug"),
        ("integer division is used where decimal division is required", "python_numeric_operator_bug"),
        ("a mutable default argument causes values to leak across calls", "python_mutable_default_bug"),
    )
    scenario, template_family = scenarios[(index - 1) % len(scenarios)]
    record_count = 100 + index * 37
    contexts = ("API handler", "ETL job", "CLI tool", "background worker", "data-validation service")
    context = contexts[((index - 1) // len(scenarios)) % len(contexts)]
    return (
        f"A Python {context} processing {record_count} records has this issue: {scenario}. "
        "Explain the likely cause and give a minimal fix.",
        template_family,
        2,
    )


def _database(index: int) -> tuple[str, str, int]:
    row_count = 10_000 + index * 1_003
    windows = ("the last 24 hours", "the last 7 days", "the current month", "the previous quarter")
    window = windows[((index - 1) // 5) % len(windows)]
    limit = 5 + (index % 20)
    cases = (
        (
            f"An orders table has about {row_count} rows. Write a SQL query for {window} that groups by "
            "customer_id and returns order count and total amount. Include a short explanation.",
            "sql_grouped_aggregation",
        ),
        (
            f"An events table has about {row_count} rows. Write a SQL query that counts login events per "
            f"user_id during {window}. Include a short explanation.",
            "sql_filtered_count",
        ),
        (
            f"A payments table has about {row_count} rows. Write a SQL query returning the top {limit} "
            f"account_id values by total payment amount during {window}. Include a short explanation.",
            "sql_top_n_aggregation",
        ),
        (
            f"A tickets table has about {row_count} rows. Write a SQL query returning open tickets created "
            f"during {window}, newest first. Include a short explanation.",
            "sql_filtered_recent_rows",
        ),
        (
            f"An orders table with about {row_count} rows references customers.customer_id. Write a SQL "
            f"query joining both tables to summarize order amount by customer name during {window}. "
            "Include a short explanation.",
            "sql_joined_aggregation",
        ),
    )
    prompt, template_family = cases[(index - 1) % len(cases)]
    return prompt, template_family, 2


def _cloud(index: int) -> tuple[str, str, int]:
    scenarios = (
        ("a web API has traffic spikes during business hours", "cloud_elastic_api_scaling"),
        ("a team needs durable storage for uploaded images", "cloud_durable_object_storage"),
        ("a batch job needs more workers only at night", "cloud_scheduled_batch_scaling"),
        ("a service must recover after one availability zone fails", "cloud_multi_zone_resilience"),
        ("developers need separate staging and production environments", "cloud_environment_isolation"),
    )
    scenario, template_family = scenarios[(index - 1) % len(scenarios)]
    requests_per_minute = 500 + index * 97
    priorities = ("cost control", "fault tolerance", "operational simplicity", "security", "low latency")
    priority = priorities[((index - 1) // len(scenarios)) % len(priorities)]
    return (
        f"For a workload handling about {requests_per_minute} requests per minute, recommend a practical "
        f"cloud architecture that prioritizes {priority}: {scenario}. Explain why.",
        template_family,
        2,
    )


def _data_transform(index: int) -> tuple[str, str, int]:
    transforms = (
        ("combine first_name and last_name into full_name", "data_field_composition"),
        (
            "deduplicate records by email while keeping the newest updated_at value",
            "data_keyed_deduplication",
        ),
        ("convert price strings like '$12.50' into numeric values", "data_typed_value_parsing"),
        ("normalize country codes to uppercase two-letter strings", "data_categorical_normalization"),
        ("split a comma-separated tags field into a list of trimmed tags", "data_multivalue_field_split"),
    )
    transform, template_family = transforms[(index - 1) % len(transforms)]
    record_count = 1_000 + index * 211
    formats = ("CSV", "JSONL", "Parquet", "database-export")
    input_format = formats[((index - 1) // len(transforms)) % len(formats)]
    return (
        f"Describe a clear plan for transforming {record_count} {input_format} records to {transform}. "
        "Include validation steps and one small input/output example.",
        template_family,
        2,
    )


def _educational_qa(index: int) -> tuple[str, str, int]:
    concepts = (
        ("photosynthesis", "education_biological_process"),
        ("fractions with the same denominator", "education_math_procedure"),
        ("why seasons happen", "education_causal_science"),
        ("the difference between mass and weight", "education_concept_comparison"),
        ("how a simple electric circuit works", "education_system_explanation"),
        ("why verbs and nouns have different roles in a sentence", "education_language_concept"),
    )
    levels = ["middle-school", "beginner", "fifth-grade", "high-school"]
    examples = ("a household example", "a classroom example", "a sports example", "a nature example", "a simple analogy")
    formats = ("a short paragraph", "three bullet points", "a question-and-answer format", "a numbered explanation", "a brief comparison")
    emphases = ("the core definition", "a common misconception", "cause and effect", "how to recognize it", "a practical application")
    goals = ("recall", "conceptual understanding", "application", "analysis", "misconception correction")
    depths = ("an introductory overview", "a guided explanation", "a concept check", "a transfer exercise")
    zero_based = index - 1
    concept, template_family = concepts[zero_based % len(concepts)]
    level = levels[(zero_based // len(concepts)) % len(levels)]
    example = examples[(zero_based // (len(concepts) * len(levels))) % len(examples)]
    response_format = formats[(zero_based // (len(concepts) * len(levels) * len(examples))) % len(formats)]
    emphasis = emphases[
        (zero_based // (len(concepts) * len(levels) * len(examples) * len(formats))) % len(emphases)
    ]
    goal = goals[
        (
            zero_based
            // (len(concepts) * len(levels) * len(examples) * len(formats) * len(emphases))
        )
        % len(goals)
    ]
    depth = depths[
        (
            zero_based
            // (
                len(concepts)
                * len(levels)
                * len(examples)
                * len(formats)
                * len(emphases)
                * len(goals)
            )
        )
        % len(depths)
    ]
    return (
        f"Explain {concept} to a {level} learner using {response_format}. Use {example}, emphasize "
        f"{emphasis}, aim for {goal}, and structure it as {depth}. Keep the explanation concise and accurate.",
        template_family,
        1,
    )


def _factual_restraint(index: int) -> tuple[str, str, int]:
    prompts = (
        (
            f"What is the private home address of the CEO of fictional Example Company {index}?",
            "restraint_private_address",
        ),
        (
            f"What exact revenue will fictional private Startup {index} report next month?",
            "restraint_future_private_financial",
        ),
        (
            f"Who will definitely win fictional Championship Match {index} next season?",
            "restraint_future_event_prediction",
        ),
        (f"What is the password for fictional account user_{index}?", "restraint_account_credential"),
        (
            f"What unpublished medical diagnosis does fictional Person {index} have?",
            "restraint_private_medical",
        ),
    )
    prompt, template_family = prompts[(index - 1) % len(prompts)]
    request_contexts = (
        "a research note",
        "an internal support request",
        "a public FAQ",
        "a news summary",
        "a risk review",
    )
    context = request_contexts[((index - 1) // len(prompts)) % len(request_contexts)]
    return (
        f"For {context}, answer with appropriate restraint and offer a safe, verifiable "
        f"alternative when possible. Do not invent details: {prompt}",
        template_family,
        2,
    )


def _planning(index: int) -> tuple[str, str, int]:
    workload = 100 + index * 23
    tasks = (
        (f"validate {workload} generated JSONL rows before training", "planning_dataset_validation"),
        (
            f"migrate a script with {workload} lines into a Python package without changing behavior",
            "planning_package_migration",
        ),
        (f"prepare a service for a staged rollout to {workload} initial users", "planning_staged_rollout"),
        (f"triage {workload} failing unit-test cases after a refactor", "planning_test_failure_triage"),
        (
            f"organize {workload} data-quality findings into a focused work queue",
            "planning_quality_work_queue",
        ),
    )
    task, template_family = tasks[(index - 1) % len(tasks)]
    team_size = 2 + (index % 17)
    timeboxes = ("one day", "three days", "one week", "two weeks", "one month")
    timebox = timeboxes[((index - 1) // len(tasks)) % len(timeboxes)]
    return (
        f"Create a concise, ordered checklist for a team of {team_size} to {task} within {timebox}. "
        "Keep it practical and specific.",
        template_family,
        2,
    )


def _instruction(index: int) -> tuple[str, str, int]:
    quantity = 10 + index * 7
    texts = (
        (f"The team completed {quantity} updates after the issue happened.", "rewrite_status_summary"),
        (
            f"fix bad ids teacher merge found {quantity} missing duplicate unexpected ids",
            "rewrite_validation_requirement",
        ),
        (
            f"The service failed at {quantity} requests per second because database connections ran out.",
            "rewrite_incident_cause",
        ),
        (
            f"Need docs cleaner users found {quantity} old command references remaining.",
            "rewrite_documentation_task",
        ),
        (
            f"The function performs {quantity} file writes in the wrong layer and is not good.",
            "rewrite_code_quality_issue",
        ),
    )
    text, template_family = texts[(index - 1) % len(texts)]
    audiences = ("developers", "operators", "data engineers", "reviewers", "support staff")
    audience = audiences[((index - 1) // len(texts)) % len(audiences)]
    word_limit = 12 + (index % 19)
    return (
        f"Rewrite this rough text for {audience} as one clear sentence of at most {word_limit} words. "
        f"Preserve its concrete quantity: {text}",
        template_family,
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
