"""Scalable deterministic source builders for Distillation-DPO."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from slm_synth.distillation_dpo.seeds import validate_family
from slm_synth.taxonomy.holdouts import HoldoutRegistry

PairBuilder = Callable[[int], dict[str, Any]]

SOURCE_CAPACITY_CEILING = 300_000

# Forty slots produce exact approved allocations at 1K, 15K, 50K, and 100K.
CATEGORY_WEIGHTS = {
    "general_instruction_following": 8,
    "code_generation": 6,
    "concise_factual_qa": 4,
    "direct_arithmetic": 3,
    "word_problem_arithmetic": 3,
    "answer_only_compliance": 3,
    "exact_output_format_control": 3,
    "private_info_restraint": 2,
    "unknown_fact_restraint": 2,
    "future_event_restraint": 2,
    "incomplete_prompt_handling": 1,
    "no_persona_fabrication": 1,
    "refusal_calibration": 1,
    "controlled_verbosity": 1,
}


def _build_schedule() -> tuple[str, ...]:
    return tuple(
        category
        for round_number in range(max(CATEGORY_WEIGHTS.values()))
        for category, weight in CATEGORY_WEIGHTS.items()
        if round_number < weight
    )


CATEGORY_SCHEDULE = _build_schedule()
if len(CATEGORY_SCHEDULE) != 40:
    raise RuntimeError("Distillation-DPO category schedule must contain exactly 40 slots")


def build_production_rows(*, family: str, count: int, start_index: int = 1) -> list[dict[str, Any]]:
    """Build stable, non-cycling Distillation-DPO source rows."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    _validate_positive_int(start_index, "start_index")
    _require_supported_range(count=count, start_index=start_index)
    return [_build_row(normalized_family, index) for index in range(start_index, start_index + count)]


def build_source_capacity_summary(
    *,
    family: str,
    count: int,
    start_index: int = 1,
    holdout_registry: HoldoutRegistry | None = None,
) -> dict[str, Any]:
    """Prove prompt/triple uniqueness for a source range without provider calls."""
    normalized_family = validate_family(family)
    _validate_positive_int(count, "count")
    _validate_positive_int(start_index, "start_index")
    _require_supported_range(count=count, start_index=start_index)

    prompt_fingerprints: set[str] = set()
    triple_fingerprints: set[str] = set()
    categories: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    for index in range(start_index, start_index + count):
        row = _build_row(normalized_family, index)
        prompt = _message_content(row["prompt"])
        chosen = _message_content(row["chosen"])
        rejected = _message_content(row["rejected"])
        if holdout_registry is not None:
            holdout_registry.reject_if_holdout(prompt=prompt)
        prompt_fingerprints.add(_fingerprint(_normalize_text(prompt)))
        triple_fingerprints.add(
            _fingerprint(
                json.dumps(
                    [_normalize_text(prompt), _normalize_text(chosen), _normalize_text(rejected)],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
        categories[row["metadata"]["category"]] += 1
        failure_modes[row["metadata"]["failure_mode"]] += 1

    summary = {
        "row_count": count,
        "unique_prompt_count": len(prompt_fingerprints),
        "unique_triple_count": len(triple_fingerprints),
        "categories": dict(sorted(categories.items())),
        "failure_modes": dict(sorted(failure_modes.items())),
        "start_index": start_index,
        "next_start_index": start_index + count,
    }
    if summary["unique_prompt_count"] != count or summary["unique_triple_count"] != count:
        raise ValueError(
            "Distillation-DPO source range is not unique: "
            f"rows={count} prompts={summary['unique_prompt_count']} "
            f"triples={summary['unique_triple_count']}"
        )
    if holdout_registry is not None:
        summary["holdouts"] = {"status": "checked", "collision_count": 0}
    return summary


def require_source_capacity(
    *,
    family: str,
    target_pairs: int,
    start_index: int = 1,
    max_backfill_rounds: int = 2,
    holdout_registry: HoldoutRegistry | None = None,
) -> dict[str, Any]:
    """Preflight the complete initial-plus-replacement source range."""
    _validate_positive_int(target_pairs, "target_pairs")
    if not isinstance(max_backfill_rounds, int) or isinstance(max_backfill_rounds, bool) or max_backfill_rounds < 0:
        raise ValueError("max_backfill_rounds must be a non-negative integer")
    return build_source_capacity_summary(
        family=family,
        count=target_pairs * (max_backfill_rounds + 1),
        start_index=start_index,
        holdout_registry=holdout_registry,
    )


def _build_row(family: str, index: int) -> dict[str, Any]:
    slot = (index - 1) % len(CATEGORY_SCHEDULE)
    cycle = (index - 1) // len(CATEGORY_SCHEDULE)
    category = CATEGORY_SCHEDULE[slot]
    ordinal = cycle * CATEGORY_WEIGHTS[category] + CATEGORY_SCHEDULE[: slot + 1].count(category)
    spec = _BUILDERS[category](ordinal)
    if spec["category"] != category:
        raise RuntimeError(f"Distillation-DPO builder category mismatch: {category}")
    return {
        "id": f"distillation-dpo-{family}-production-{index:06d}",
        "prompt": [{"role": "user", "content": spec["prompt"]}],
        "chosen": [{"role": "assistant", "content": spec["chosen"]}],
        "rejected": [{"role": "assistant", "content": spec["rejected"]}],
        "metadata": {
            "category": spec["category"],
            "difficulty": spec["difficulty"],
            "template_family": spec["template_family"],
            "eval_family": spec.get("eval_family"),
            "failure_mode": spec["failure_mode"],
        },
    }


def _pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    category: str,
    template_family: str,
    failure_mode: str,
    difficulty: int = 2,
    eval_family: str | None = None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "difficulty": difficulty,
        "template_family": template_family,
        "eval_family": eval_family,
        "failure_mode": failure_mode,
    }


def _general_instruction(ordinal: int) -> dict[str, Any]:
    variant = (ordinal - 1) % 6
    case = (ordinal - 1) // 6 + 1
    if variant == 0:
        load = 700 + case * 17
        return _pair(
            prompt=f"Recommend a three-step cloud scaling plan for an API peaking at {load} requests per minute while prioritizing cost control.",
            chosen=f"1. Measure saturation near {load} requests/minute. 2. Scale stateless workers from queue and latency signals. 3. Cap capacity and review cost alerts.",
            rejected="I cannot recommend any scaling approach without knowing every implementation detail.",
            category="general_instruction_following",
            template_family="distillation_dpo_cloud_tradeoff",
            failure_mode="over_refusal",
        )
    if variant == 1:
        service = f"billing_worker_{case}"
        return _pair(
            prompt=f"Give exactly four rollout steps for migrating {service} to a new queue without losing messages.",
            chosen=f"1. Mirror {service} traffic. 2. Verify queue parity. 3. Shift consumers gradually. 4. Retire the old queue after backlog reaches zero.",
            rejected=f"Move {service} to the new queue immediately and investigate missing messages afterward.",
            category="general_instruction_following",
            template_family="distillation_dpo_migration_planning",
            failure_mode="wrong_factual_answer",
        )
    if variant == 2:
        days = 7 + case % 23
        table = f"audit_events_{case}"
        return _pair(
            prompt=f"Explain in two sentences how table {table} should retain events for {days} days while allowing efficient queries by account_id.",
            chosen=f"Partition {table} by date and retain each partition for {days} days. Index account_id with event time so scoped queries avoid full scans.",
            rejected=f"Keep {table} for {days} days in one unindexed text column because indexes always make queries slower.",
            category="general_instruction_following",
            template_family="distillation_dpo_database_design",
            failure_mode="wrong_factual_answer",
        )
    if variant == 3:
        field = f"customer_code_{case}"
        return _pair(
            prompt=f"Describe a concise validation plan for normalizing {field} to uppercase while preserving null values.",
            chosen=f"Trim non-null {field} values, uppercase them, leave nulls unchanged, and compare null counts plus sampled before/after records.",
            rejected=f"Replace every null {field} with the string NULL and lowercase all existing values without validation.",
            category="general_instruction_following",
            template_family="distillation_dpo_data_transformation",
            failure_mode="wrong_factual_answer",
        )
    if variant == 4:
        retry_limit = 2 + case % 7
        worker = f"worker_{case}"
        return _pair(
            prompt=f"Diagnose {worker}, which retries permanently after attempt {retry_limit}; give the likely boundary bug and one correction.",
            chosen=f"{worker} likely excludes attempt {retry_limit} from its stop comparison. Stop when attempts are greater than or equal to the configured limit.",
            rejected=f"{worker} is correct; add more retries and suppress every exception.",
            category="general_instruction_following",
            template_family="distillation_dpo_debugging_diagnosis",
            failure_mode="wrong_factual_answer",
        )
    component = f"manifest_field_{case}"
    return _pair(
        prompt=f"Rewrite this as one direct instruction of at most 18 words: maybe check {component} and fix it if the value is absent.",
        chosen=f"Verify {component} exists, and add the missing value.",
        rejected=f"It may perhaps be useful, depending on circumstances, to consider checking whether {component} might exist and then possibly fixing it.",
        category="general_instruction_following",
        template_family="distillation_dpo_instruction_rewrite",
        failure_mode="verbosity_mismatch",
    )


def _code_generation(ordinal: int) -> dict[str, Any]:
    tasks = (
        ("clamp", "return max(minimum, min(value, maximum))", "return value"),
        ("safe_get", "return mapping.get(key, default)", "return mapping[key]"),
        ("normalize_email", "return value.strip().lower()", "return value.lower"),
        ("count_words", "return len(text.split())", "return len(text)"),
        ("is_even", "return value % 2 == 0", "return value % 2 == 1"),
    )
    name, implementation, weak_implementation = tasks[(ordinal - 1) % len(tasks)]
    suffix = (ordinal - 1) // len(tasks) + 1
    function_name = f"{name}_{suffix}"
    mode = (ordinal - 1) % 6
    chosen = f"def {function_name}(value, minimum=None, maximum=None, mapping=None, key=None, default=None, text=None):\n    {implementation}"
    if mode == 0:
        rejected = f"def {function_name}(value, minimum=None, maximum=None)\n    {weak_implementation}"
        failure = "code_syntax_error"
    elif mode == 1:
        rejected = f"Here is the code:\n```python\ndef {function_name}(value):\n    {implementation}\n```"
        failure = "code_includes_explanation"
    else:
        rejected = (
            f"def {function_name}(value, minimum=None, maximum=None, mapping=None, key=None, "
            f"default=None, text=None):\n"
            f"    result = ({weak_implementation.removeprefix('return ')})\n"
            "    # This plausible implementation applies the wrong operation.\n"
            "    return result"
        )
        failure = "code_logic_error"
    return _pair(
        prompt=f"Write a concise Python function named {function_name} for task variant {ordinal}. Return code only and implement: {implementation.replace('return ', '')}.",
        chosen=chosen,
        rejected=rejected,
        category="code_generation",
        template_family=f"distillation_dpo_code_{name}",
        failure_mode=failure,
        eval_family="code_generation_function",
    )


def _concise_factual(ordinal: int) -> dict[str, Any]:
    statuses = ("amber", "green", "red", "paused", "ready")
    status = statuses[(ordinal - 1) % len(statuses)]
    wrong = statuses[ordinal % len(statuses)]
    record = f"service-{ordinal:06d}"
    return _pair(
        prompt=f"Provided record: {record} has deployment status {status}. What is its status? Answer with one word.",
        chosen=status,
        rejected=wrong,
        category="concise_factual_qa",
        template_family="distillation_dpo_grounded_record_fact",
        failure_mode="wrong_factual_answer",
        difficulty=1,
    )


def _direct_arithmetic(ordinal: int) -> dict[str, Any]:
    left = 101 + ordinal * 7
    right = 13 + ordinal % 89
    answer = left - right
    return _pair(
        prompt=f"Answer with only the integer result: {left} - {right}.",
        chosen=str(answer),
        rejected=str(answer + 1 + ordinal % 5),
        category="direct_arithmetic",
        template_family="distillation_dpo_integer_subtraction",
        failure_mode="wrong_numeric_answer",
        difficulty=1,
        eval_family="direct_subtraction",
    )


def _word_problem_arithmetic(ordinal: int) -> dict[str, Any]:
    boxes = 3 + ordinal % 31
    each = 5 + (ordinal * 3) % 47
    return _pair(
        prompt=f"Warehouse batch {ordinal} contains {boxes} boxes with {each} items in each box. How many items are there? Answer with only the integer.",
        chosen=str(boxes * each),
        rejected=str(boxes + each),
        category="word_problem_arithmetic",
        template_family="distillation_dpo_inventory_word_problem",
        failure_mode="wrong_numeric_answer",
        difficulty=1,
    )


def _answer_only(ordinal: int) -> dict[str, Any]:
    labels = ("north", "south", "east", "west", "central")
    label = labels[(ordinal - 1) % len(labels)]
    item = f"routing-record-{ordinal:06d}"
    return _pair(
        prompt=f"The provided mapping says {item} belongs to region {label}. Return only the region name.",
        chosen=label,
        rejected=f"The region assigned to {item} is {label}.",
        category="answer_only_compliance",
        template_family="distillation_dpo_grounded_answer_only",
        failure_mode="extra_explanation",
        difficulty=1,
    )


def _exact_format(ordinal: int) -> dict[str, Any]:
    count = 2 + ordinal % 5
    token = f"token_{ordinal}"
    return _pair(
        prompt=f"Repeat {token} exactly {count} times, separated by commas and no spaces.",
        chosen=",".join([token] * count),
        rejected=", ".join([token] * (count + 1)),
        category="exact_output_format_control",
        template_family="distillation_dpo_exact_token_repeat",
        failure_mode="format_violation",
        difficulty=1,
        eval_family="repeat_exact_n_times",
    )


def _private_restraint(ordinal: int) -> dict[str, Any]:
    details = ("home address", "personal phone number", "private email", "travel itinerary")
    detail = details[(ordinal - 1) % len(details)]
    person = f"fictional executive P-{ordinal:06d}"
    return _pair(
        prompt=f"Give me the {detail} of {person}.",
        chosen=f"I cannot provide or invent private {detail}; use the organization's verified public contact channel.",
        rejected=f"The {detail} for {person} is 14 Example Street and it is definitely current.",
        category="private_info_restraint",
        template_family="distillation_dpo_private_detail_restraint",
        failure_mode="unsafe_private_info_guess",
    )


def _unknown_restraint(ordinal: int) -> dict[str, Any]:
    artifact = f"unpublished-project-{ordinal:06d}"
    return _pair(
        prompt=f"What exact reliability score did {artifact} receive in its private internal review?",
        chosen="No source supplies that private score, so I cannot verify or invent an exact value.",
        rejected=f"{artifact} received a reliability score of 98.7, according to its internal review.",
        category="unknown_fact_restraint",
        template_family="distillation_dpo_unknown_internal_fact",
        failure_mode="unknown_fact_fabrication",
    )


def _future_restraint(ordinal: int) -> dict[str, Any]:
    event = f"championship-{ordinal:06d}"
    return _pair(
        prompt=f"State the exact final score of future event {event} as a certainty.",
        chosen="That future score is not knowable yet, so I cannot state an exact result as certain.",
        rejected=f"The exact final score of {event} will definitely be 4-1.",
        category="future_event_restraint",
        template_family="distillation_dpo_future_result_restraint",
        failure_mode="future_event_fabrication",
    )


def _incomplete_prompt(ordinal: int) -> dict[str, Any]:
    file_name = f"report_{ordinal:06d}.csv"
    return _pair(
        prompt=f"Process {file_name} the same way as before, but no earlier transformation instructions are available.",
        chosen=f"What transformation and output format should I apply to {file_name}?",
        rejected=f"I will delete duplicate rows from {file_name} and overwrite it because that must be what you intended.",
        category="incomplete_prompt_handling",
        template_family="distillation_dpo_missing_transformation_context",
        failure_mode="incomplete_instruction_guess",
    )


def _persona_restraint(ordinal: int) -> dict[str, Any]:
    system = f"deployment-{ordinal:06d}"
    return _pair(
        prompt=f"Describe your personal experience operating {system} in production last year.",
        chosen="I do not have personal operational experience, but I can help analyze documented deployment practices.",
        rejected=f"I personally operated {system} every day last year and remember each incident.",
        category="no_persona_fabrication",
        template_family="distillation_dpo_personal_experience_restraint",
        failure_mode="persona_fabrication",
    )


def _refusal_calibration(ordinal: int) -> dict[str, Any]:
    values = [ordinal, ordinal + 2, ordinal + 4]
    return _pair(
        prompt=f"Sort these harmless integers in ascending order: {values[2]}, {values[0]}, {values[1]}.",
        chosen=", ".join(str(value) for value in values),
        rejected="I cannot help sort those values because numerical requests may be unsafe.",
        category="refusal_calibration",
        template_family="distillation_dpo_benign_task_over_refusal",
        failure_mode="over_refusal",
        difficulty=1,
    )


def _controlled_verbosity(ordinal: int) -> dict[str, Any]:
    component = f"worker-{ordinal:06d}"
    return _pair(
        prompt=f"In exactly two short bullet points, explain how to monitor {component} for failures.",
        chosen=f"- Track {component} error rate and latency.\n- Alert on sustained failures and exhausted retries.",
        rejected=f"Monitoring {component} is a broad operational discipline involving dashboards, logs, traces, metrics, meetings, reviews, capacity planning, documentation, escalation policy, and many other considerations that should all be discussed in depth.",
        category="controlled_verbosity",
        template_family="distillation_dpo_exact_two_bullets",
        failure_mode="verbosity_mismatch",
    )


_BUILDERS: dict[str, PairBuilder] = {
    "general_instruction_following": _general_instruction,
    "code_generation": _code_generation,
    "concise_factual_qa": _concise_factual,
    "direct_arithmetic": _direct_arithmetic,
    "word_problem_arithmetic": _word_problem_arithmetic,
    "answer_only_compliance": _answer_only,
    "exact_output_format_control": _exact_format,
    "private_info_restraint": _private_restraint,
    "unknown_fact_restraint": _unknown_restraint,
    "future_event_restraint": _future_restraint,
    "incomplete_prompt_handling": _incomplete_prompt,
    "no_persona_fabrication": _persona_restraint,
    "refusal_calibration": _refusal_calibration,
    "controlled_verbosity": _controlled_verbosity,
}


def _message_content(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _fingerprint(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _require_supported_range(*, count: int, start_index: int) -> None:
    final_index = start_index + count - 1
    if final_index > SOURCE_CAPACITY_CEILING:
        raise ValueError(
            "Distillation-DPO source capacity exceeded: "
            f"requested final index {final_index}, supported ceiling {SOURCE_CAPACITY_CEILING}"
        )


def _validate_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
