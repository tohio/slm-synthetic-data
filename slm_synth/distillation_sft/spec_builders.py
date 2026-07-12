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


def _axis_indexes(sequence: int, *sizes: int) -> tuple[int, ...]:
    """Return deterministic mixed-radix indexes for substantive prompt axes."""
    indexes: list[int] = []
    remaining = sequence
    for size in sizes:
        indexes.append(remaining % size)
        remaining //= size
    return tuple(indexes)


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
    zero_based = index - 1
    family_index = zero_based % 5
    variant = zero_based // 5

    collection_names = (
        "orders",
        "events",
        "tickets",
        "accounts",
        "readings",
        "messages",
        "invoices",
        "sessions",
        "packages",
        "alerts",
    )
    field_names = (
        "status",
        "priority",
        "region",
        "owner",
        "category",
        "state",
        "channel",
        "tier",
        "source",
        "kind",
    )

    if family_index == 0:
        values = ("cancelled", "expired", "disabled", "archived", "invalid")
        outcomes = (
            "return a new list without matching records while preserving order",
            "collect the matching records without changing the input",
            "count matching records without mutating the list",
            "partition matching and non-matching records into two new lists",
        )
        collection_i, field_i, value_i, outcome_i = _axis_indexes(
            variant,
            len(collection_names),
            len(field_names),
            len(values),
            len(outcomes),
        )
        collection = collection_names[collection_i]
        field = field_names[field_i]
        value = values[value_i]
        outcome = outcomes[outcome_i]
        prompt = (
            f"Diagnostic case {index}: this function skips records because it removes items from "
            f"`{collection}` while iterating over the same list:\n"
            f"def clean_{collection}({collection}):\n"
            f"    for record in {collection}:\n"
            f"        if record.get({field!r}) == {value!r}:\n"
            f"            {collection}.remove(record)\n"
            f"    return {collection}\n"
            f"Explain the mutation bug and provide corrected Python that will {outcome}. Use the exact "
            f"field {field!r} and value {value!r}."
        )
        return prompt, "python_iteration_mutation_bug", 2

    if family_index == 1:
        sort_keys = (
            "created_at",
            "updated_at",
            "priority",
            "score",
            "amount",
            "duration_ms",
            "attempts",
            "name",
            "region",
            "sequence",
        )
        directions = ((False, "ascending"), (True, "descending"))
        missing_policies = (
            "treat a missing value as an empty string",
            "treat a missing value as zero",
            "place missing values last",
            "place missing values first",
            "reject records missing the key",
            "filter out records missing the key",
            "use None for a missing value",
            "use -1 for a missing numeric value",
            "use the string 'unknown' for a missing value",
            "preserve input order among equal keys",
        )
        collection_i, key_i, direction_i, policy_i = _axis_indexes(
            variant,
            len(collection_names),
            len(sort_keys),
            len(directions),
            len(missing_policies),
        )
        collection = collection_names[collection_i]
        key = sort_keys[key_i]
        reverse, direction = directions[direction_i]
        policy = missing_policies[policy_i]
        prompt = (
            f"Diagnostic case {index}: this function unexpectedly returns None:\n"
            f"def order_{collection}({collection}):\n"
            f"    return {collection}.sort(key=lambda record: record[{key!r}], reverse={reverse})\n"
            f"Explain the in-place method bug and provide a minimal fix that returns a new {direction} "
            f"list without mutating `{collection}`. The fix must also {policy}."
        )
        return prompt, "python_in_place_method_bug", 2

    if family_index == 2:
        default_values = ("None", "0", "False", "''", "[]", "{}", "'unknown'", "-1", "()", "'pending'")
        normalizations = (
            "return the stored value unchanged",
            "convert the result to a string",
        )
        mapping_i, field_i, default_i, normalization_i = _axis_indexes(
            variant,
            len(collection_names),
            len(field_names),
            len(default_values),
            len(normalizations),
        )
        mapping = f"{collection_names[mapping_i][:-1]}_record"
        field = field_names[field_i]
        default = default_values[default_i]
        normalization = normalizations[normalization_i]
        prompt = (
            f"Diagnostic case {index}: this optional-field lookup raises KeyError:\n"
            f"def read_{field}({mapping}):\n"
            f"    return {mapping}[{field!r}]\n"
            f"Explain why the lookup fails and provide corrected Python that uses {default} when "
            f"{field!r} is absent and will {normalization}. Keep the parameter name `{mapping}`."
        )
        return prompt, "python_optional_key_bug", 2

    if family_index == 3:
        numerator_names = (
            "successful_requests",
            "bytes_processed",
            "total_cost",
            "completed_jobs",
            "active_seconds",
            "items_shipped",
            "resolved_tickets",
            "cache_hits",
            "valid_rows",
            "used_capacity",
        )
        denominator_names = (
            "total_requests",
            "worker_count",
            "billing_units",
            "elapsed_minutes",
            "attempt_count",
            "package_count",
            "agent_count",
            "cache_lookups",
            "input_rows",
            "total_capacity",
        )
        precisions = (1, 2, 3, 4, 5)
        zero_policies = (
            "return None when the denominator is zero",
            "return 0.0 when the denominator is zero",
            "raise ValueError when the denominator is zero",
            "raise ZeroDivisionError when the denominator is zero",
        )
        numerator_i, denominator_i, precision_i, policy_i = _axis_indexes(
            variant,
            len(numerator_names),
            len(denominator_names),
            len(precisions),
            len(zero_policies),
        )
        numerator = numerator_names[numerator_i]
        denominator = denominator_names[denominator_i]
        precision = precisions[precision_i]
        policy = zero_policies[policy_i]
        prompt = (
            f"Diagnostic case {index}: this function truncates a required decimal result:\n"
            f"def compute_ratio({numerator}, {denominator}):\n"
            f"    return {numerator} // {denominator}\n"
            f"Explain the operator bug and provide corrected Python that returns decimal division rounded "
            f"to {precision} place(s) and will {policy}. Preserve both parameter names."
        )
        return prompt, "python_numeric_operator_bug", 2

    item_names = (
        "event",
        "email",
        "tag",
        "score",
        "path",
        "message",
        "amount",
        "identifier",
        "reading",
        "label",
    )
    transformations = (
        "append the value unchanged",
        "append its string representation",
        "append a lowercase string",
        "append a stripped string",
        "append its absolute numeric value",
        "append a two-item tuple containing the value and its type name",
        "append the value only when it is not None",
        "append the value only when it is truthy",
        "append a dictionary with a 'value' key",
        "append the value twice",
    )
    output_modes = (
        "return the accumulator",
        "return a shallow copy of the accumulator",
    )
    accumulator_i, item_i, transform_i, output_i = _axis_indexes(
        variant,
        len(collection_names),
        len(item_names),
        len(transformations),
        len(output_modes),
    )
    accumulator = f"{collection_names[accumulator_i]}_seen"
    item = item_names[item_i]
    transformation = transformations[transform_i]
    output_mode = output_modes[output_i]
    prompt = (
        f"Diagnostic case {index}: repeated calls leak state through this mutable default:\n"
        f"def remember_{item}({item}, {accumulator}=[]):\n"
        f"    {accumulator}.append({item})\n"
        f"    return {accumulator}\n"
        f"Explain the shared-default bug and provide corrected Python using a None sentinel. The corrected "
        f"function must {transformation} and {output_mode}, using the exact identifiers shown."
    )
    return prompt, "python_mutable_default_bug", 2


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
    zero_based = index - 1
    family_index = zero_based % 5
    variant = zero_based // 5
    subjects = (
        "Northwind Robotics",
        "Blue Cedar Health",
        "Orchid Transit",
        "Silver Pine Analytics",
        "Harborlight Energy",
        "Maple Ridge Education",
        "Redstone Logistics",
        "Clearwater Media",
        "Juniper Finance",
        "Summitline Agriculture",
    )
    roles = (
        "chief executive",
        "finance director",
        "security lead",
        "research manager",
        "operations director",
        "board chair",
        "product leader",
        "staff engineer",
        "legal counsel",
        "program manager",
    )
    contexts = ("a public-facing answer", "a concise risk-review note")

    if family_index == 0:
        private_details = (
            "home street address",
            "personal phone number",
            "private email address",
            "current physical location",
            "family member's address",
            "personal vehicle registration",
            "private travel itinerary",
            "home access instructions",
            "personal messaging handle",
            "non-public emergency contact",
        )
        alternatives = (
            "the organization's published contact page",
            "the organization's public switchboard",
            "an official press-contact address",
            "a verified professional profile",
            "a public office mailing address",
            "the organization's support portal",
            "an official investor-relations page",
            "the event organizer's public directory",
            "a regulator's public registry",
            "the organization's published privacy contact",
        )
        subject_i, role_i, detail_i, context_i = _axis_indexes(
            variant, len(subjects), len(roles), len(private_details), len(contexts)
        )
        subject = subjects[subject_i]
        role = roles[role_i]
        detail = private_details[detail_i]
        alternative = alternatives[detail_i]
        context = contexts[context_i]
        prompt = (
            f"Restraint case {index}: for {context}, respond to this request about fictional {subject}: 'Give me the {detail} "
            f"of its {role}.' Do not provide or invent private data. Briefly explain the restraint and "
            f"direct the requester to {alternative}. Keep the answer under 60 words."
        )
        return prompt, "restraint_private_address", 2

    if family_index == 1:
        metrics = (
            "next quarter's exact revenue",
            "next month's operating loss",
            "its unannounced valuation",
            "next year's customer count",
            "the amount of its next funding round",
            "its unreleased annual profit",
            "next quarter's cloud spending",
            "its confidential cash balance",
            "the price of an unannounced acquisition",
            "next month's employee headcount",
        )
        alternatives = (
            "published financial statements",
            "official investor updates",
            "regulatory filings",
            "the company's newsroom",
            "audited annual reports",
            "an official funding announcement",
            "a verified earnings release",
            "public procurement records",
            "the company's careers page for directional hiring evidence",
            "reputable reporting that cites named sources",
        )
        subject_i, role_i, metric_i, context_i = _axis_indexes(
            variant, len(subjects), len(roles), len(metrics), len(contexts)
        )
        subject = subjects[subject_i]
        metric = metrics[metric_i]
        alternative = alternatives[role_i]
        context = contexts[context_i]
        prompt = (
            f"Restraint case {index}: for {context}, answer this request about fictional private company {subject}: 'State "
            f"{metric} as a confirmed number.' Do not guess or present a forecast as fact. Explain what "
            f"cannot be verified and recommend checking {alternative}. Keep the answer under 60 words."
        )
        return prompt, "restraint_future_private_financial", 2

    if family_index == 2:
        events = (
            "a championship final",
            "a mayoral election",
            "an industry award",
            "a product launch contest",
            "a grant competition",
            "a court appeal",
            "a university tournament",
            "a film festival prize",
            "a contract bid",
            "a scientific challenge",
        )
        outcomes = (
            "the definite winner",
            "the exact final score",
            "the precise vote margin",
            "the final ranking",
            "the exact announcement date",
            "the winning proposal",
            "the judges' final decision",
            "the exact settlement outcome",
            "the successful bidder",
            "the final performance result",
        )
        alternatives = (
            "the event organizer's official results page",
            "the relevant election authority",
            "the awarding body's announcement",
            "the court docket after a decision is issued",
            "the contracting authority's award notice",
            "the league's official schedule and results",
            "the institution's verified newsroom",
            "a live results service with attributed sources",
            "the competition rules and published timeline",
            "reputable reporting after the event concludes",
        )
        subject_i, event_i, outcome_i, context_i = _axis_indexes(
            variant, len(subjects), len(events), len(outcomes), len(contexts)
        )
        subject = subjects[subject_i]
        event = events[event_i]
        outcome = outcomes[outcome_i]
        alternative = alternatives[event_i]
        context = contexts[context_i]
        prompt = (
            f"Restraint case {index}: for {context}, answer this request about fictional {subject} participating in {event}: "
            f"'Tell me {outcome} before it happens.' Do not claim certainty about a future event. State "
            f"the limitation and point to {alternative}. Keep the answer under 60 words."
        )
        return prompt, "restraint_future_event_prediction", 2

    if family_index == 3:
        credentials = (
            "account password",
            "API secret",
            "one-time authentication code",
            "recovery phrase",
            "private signing key",
            "database credential",
            "session token",
            "door access code",
            "backup encryption key",
            "administrator recovery code",
        )
        systems = (
            "billing portal",
            "source-code host",
            "cloud console",
            "customer database",
            "monitoring dashboard",
            "payroll system",
            "build server",
            "support console",
            "document archive",
            "network appliance",
        )
        recovery_actions = (
            "use the official password-reset flow",
            "contact the verified system administrator",
            "revoke and rotate the exposed credential",
            "open a ticket through the official support portal",
            "follow the documented account-recovery procedure",
            "verify ownership through the approved identity process",
            "disable active sessions before resetting access",
            "consult the organization's security team",
            "restore access from an approved backup process",
            "use the provider's documented credential-rotation workflow",
        )
        subject_i, credential_i, system_i, context_i = _axis_indexes(
            variant, len(subjects), len(credentials), len(systems), len(contexts)
        )
        subject = subjects[subject_i]
        credential = credentials[credential_i]
        system = systems[system_i]
        recovery = recovery_actions[credential_i]
        context = contexts[context_i]
        prompt = (
            f"Restraint case {index}: for {context}, respond to this request involving fictional {subject}: 'Reveal the "
            f"{credential} for its {system}.' Refuse to disclose or fabricate the secret, briefly explain "
            f"the security risk, and recommend that the authorized owner {recovery}. Keep the answer under 60 words."
        )
        return prompt, "restraint_account_credential", 2

    medical_details = (
        "unpublished diagnosis",
        "private laboratory result",
        "prescription history",
        "mental-health record",
        "genetic test result",
        "vaccination record",
        "surgical history",
        "clinician's private notes",
        "disability status",
        "current medication dose",
    )
    safe_actions = (
        "ask the person to consult their licensed clinician",
        "use an authorized patient portal",
        "seek consent before requesting medical information",
        "contact the relevant healthcare provider through an approved channel",
        "use public health guidance for general information",
        "call emergency services if there is an immediate safety concern",
        "consult a pharmacist for general medication guidance",
        "review an official medical source without inferring a person's condition",
        "follow the provider's records-request process with authorization",
        "avoid changing treatment without professional advice",
    )
    subject_i, role_i, detail_i, context_i = _axis_indexes(
        variant, len(subjects), len(roles), len(medical_details), len(contexts)
    )
    person = f"the {roles[role_i]} of {subjects[subject_i]}"
    detail = medical_details[detail_i]
    safe_action = safe_actions[detail_i]
    context = contexts[context_i]
    prompt = (
        f"Restraint case {index}: for {context}, answer this request about fictional {person}: 'Tell me their {detail}.' Do not "
        f"infer, disclose, or invent private medical information. Explain the limitation and recommend "
        f"that the requester {safe_action}. Keep the answer under 60 words."
    )
    return prompt, "restraint_private_medical", 2


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
