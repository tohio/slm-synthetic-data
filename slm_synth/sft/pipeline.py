#!/usr/bin/env python3
"""Production generic SFT pipeline migrated from the finalized one-off.

Stages:
1A derivation generation -> 1B concrete task generation -> deterministic novelty
filtering -> answer generation -> deterministic validation -> judge -> reviewer ->
final exact dedup. Model-facing contracts use OpenRouter strict JSON Schema.
Shared batching, retries/fault isolation, novelty, IO, and backend construction live
in ``slm_synth.runtime``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_synth.output_constraints import (
    OutputConstraintError,
    evaluate_sft_output_constraints,
)
from slm_synth.sft.acceptance import partition_unique_sft_rows
from slm_synth.sft.schema import validate_sft_row
from slm_synth.sft.spec_builders import SFT_SPEC_FAMILIES, build_specs
from slm_synth.sft.specs import validate_sft_spec

from slm_synth.runtime import (
    NoveltyFilter,
    append_jsonl,
    build_backend,
    chunked,
    fill_exact_count,
    reset_output_files,
    run_model_stage_with_isolation,
    split_sequence_batch,
    split_slot_batch,
    write_json,
)


@dataclass(frozen=True)
class Seed:
    index: int
    id: str
    family: str
    instruction: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Derivation:
    seed_id: str
    seed_index: int
    ordinal: int
    text: str


@dataclass(frozen=True)
class Task:
    seed_id: str
    seed_index: int
    derivation_ordinal: int
    task_ordinal: int
    text: str


def reset_outputs(out: Path) -> None:
    reset_output_files(
        out,
        (
            "derivations.generated.jsonl",
            "tasks.generated.jsonl",
            "tasks.accepted.jsonl",
            "answers.generated.jsonl",
            "answers.validated.jsonl",
            "judge.decisions.jsonl",
            "reviewer.decisions.jsonl",
            "quality.rejected.jsonl",
            "stage.failures.jsonl",
            "sft.accepted.jsonl",
            "summary.json",
            "plan.json",
        ),
    )


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def load_seeds(family: str, count: int) -> list[Seed]:
    specs = build_specs(family=family, count=count, start_index=1)
    seeds: list[Seed] = []
    for index, raw in enumerate(specs, start=1):
        spec = validate_sft_spec(raw)
        seeds.append(
            Seed(
                index=index,
                id=spec["id"],
                family=family,
                instruction=spec["instruction"],
                metadata=dict(spec["metadata"]),
            )
        )
    return seeds



# ---------------------------------------------------------------------------
# OpenRouter structured-output contract
# ---------------------------------------------------------------------------

def call_structured_object(
    backend: Any,
    *,
    prompt: str,
    schema: Mapping[str, Any],
    schema_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the repo's OpenRouter JSON-Schema path.

    The backend owns provider routing, retries, adaptive admission, telemetry,
    and response_format=json_schema with strict=true. This one-off script owns
    only the role schemas and local semantic/invariant validation.
    """
    method = getattr(backend, "generate_structured_object_with_metadata", None)
    if method is None:
        raise TypeError(
            "backend does not implement generate_structured_object_with_metadata"
        )

    result = method(
        prompt=prompt,
        schema=dict(schema),
        schema_name=schema_name,
    )
    if not isinstance(result, Mapping):
        raise TypeError("structured backend returned a non-object envelope")

    data = result.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("structured backend response missing object data")

    telemetry = result.get("telemetry")
    return dict(data), dict(telemetry) if isinstance(telemetry, Mapping) else {}


def exact_string_list_schema(
    *,
    field: str,
    count: int,
    description: str,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("structured list count must be positive")
    return {
        "type": "object",
        "properties": {
            field: {
                "type": "array",
                "description": description,
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
            },
        },
        "required": [field],
        "additionalProperties": False,
    }


def judge_schema(expected_ids: Sequence[str]) -> dict[str, Any]:
    if not expected_ids:
        raise ValueError("judge schema requires at least one id")

    decision = {
        "type": "object",
        "properties": {
            "assessable": {
                "type": "boolean",
                "description": "Whether the candidate can be meaningfully assessed.",
            },
            "accepted": {
                "type": "boolean",
                "description": (
                    "True only when the assistant response has no concrete "
                    "material defect."
                ),
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "description": "Concise evidence-based reason for the decision.",
            },
        },
        "required": ["assessable", "accepted", "reason"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {
                    row_id: decision
                    for row_id in expected_ids
                },
                "required": list(expected_ids),
                "additionalProperties": False,
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


def reviewer_schema(expected_ids: Sequence[str]) -> dict[str, Any]:
    if not expected_ids:
        raise ValueError("reviewer schema requires at least one id")

    decision = {
        "type": "object",
        "properties": {
            "agreed": {
                "type": "boolean",
                "description": "Whether the judge ACCEPT decision is justified.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Empty string when agreed=true; concise concrete defect "
                    "when agreed=false."
                ),
            },
        },
        "required": ["agreed", "reason"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {
                    row_id: decision
                    for row_id in expected_ids
                },
                "required": list(expected_ids),
                "additionalProperties": False,
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Generic parsed lists
# ---------------------------------------------------------------------------

def extract_string_list(parsed: Mapping[str, Any], field: str) -> list[str]:
    values = parsed.get(field)
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")

    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        result.append(value.strip())
    return result


# Model 1 — task generator
# ---------------------------------------------------------------------------

def generate_derivations(
    *,
    seed: Seed,
    count: int,
    backend: Any,
    max_fill_attempts: int,
) -> list[str]:
    def request_derivations(
        request_count: int,
        prior: Sequence[str],
    ) -> list[str]:
        prior_block = ""
        if prior:
            prior_block = (
                "\nAlready accepted derivations. Do not repeat or substantially "
                "overlap these:\n"
                + "\n".join(f"- {item}" for item in prior)
                + "\n"
            )

        prompt = f"""
You are generating semantic task classes for the SFT family: {seed.family}.

Seed:
{seed.instruction}

Generate exactly {request_count} DISTINCT {seed.family} problem classes that are
meaningfully different from one another.

The seed is only a starting point. Do NOT paraphrase it repeatedly.
{prior_block}
Each derivation must:
- represent a different capability, problem type, or failure mode;
- be broad enough to support multiple concrete tasks later;
- differ semantically, not just by variable names, entities, data types,
  numbers, or wording;
- stay within the {seed.family} family;
- avoid producing the final user task;
- avoid answers or solutions.

For programming, good derivations can differ in skills such as:
- parsing and validation
- graph traversal
- stateful data structures
- concurrency or async coordination
- retries and backoff
- serialization
- file/path handling
- dynamic programming
- caching and invalidation
- API contract handling
- SQL/data access
- testing and debugging

Bad derivations are cosmetic variations that exercise essentially the same
skill.

Before returning a derivation, compare it against all previously selected
derivations and reject it if the core solution strategy or capability
substantially overlaps.

Return exactly {request_count} derivations under the supplied
structured-output schema. Each derivation must be a concise description of
the semantic problem class, not a concrete task.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="derivations",
                count=request_count,
                description=(
                    f"Exactly {request_count} distinct semantic problem classes "
                    f"for the {seed.family} SFT family."
                ),
            ),
            schema_name="sft_task_derivations",
        )
        return extract_string_list(parsed, "derivations")

    initial = request_derivations(count, ())
    return fill_exact_count(
        field="derivations",
        requested=count,
        initial=initial,
        fetch_missing=request_derivations,
        max_fill_attempts=max_fill_attempts,
        stage_label="task-derivations",
    )


def generate_tasks_from_derivation(
    *,
    seed: Seed,
    derivation: str,
    count: int,
    backend: Any,
    max_fill_attempts: int,
) -> list[str]:
    family_task_guidance = {
        "programming": (
            "Keep each task centered on one primary programming behavior or a tightly "
            "related set of behaviors. Requirements must be mutually consistent and "
            "implementable from the supplied signature, state, and context. Make retry, "
            "timeout, cancellation, ordering, cleanup, and error semantics unambiguous "
            "when they are relevant. Required tests must agree with those semantics."
        ),
        "creative_writing": (
            "Use a small number of meaningful writing constraints. Do not stack many "
            "independent exact-count, exact-length, forbidden-word, structural-placement, "
            "and style constraints into the same task unless that combination is itself "
            "the capability being tested. Exact constraints must be mutually compatible "
            "and objectively checkable."
        ),
        "planning_brainstorming_recommendations": (
            "Keep hard constraints mutually feasible and clearly distinguish them from "
            "preferences. Do not require the assistant to prove travel time, accessibility, "
            "opening hours, prices, availability, or other external facts unless the task "
            "supplies the needed facts. Avoid stacking unrelated scheduling, budget, travel, "
            "staffing, accessibility, and contingency constraints into one task."
        ),
        "applied_math_and_reasoning": (
            "State all quantities, units, assumptions, objectives, and constraints needed "
            "for the calculation or proof. Avoid ambiguous probability dependencies, "
            "optimization rules, endpoint conventions, or mutually inconsistent constraints. "
            "The requested conclusion must be derivable from the supplied information."
        ),
    }.get(seed.family, "")

    def request_tasks(request_count: int, prior: Sequence[str]) -> list[str]:
        prior_block = ""
        if prior:
            prior_block = (
                "\nAlready accepted from this derivation in the current request. "
                "Do not repeat these:\n"
                + "\n".join(f"- {item}" for item in prior)
                + "\n"
            )

        prompt = f"""
EXACT OUTPUT COUNT: {request_count}

Generate exactly {request_count} COMPLETE, STANDALONE user tasks.
The JSON array named "tasks" MUST contain exactly {request_count} strings.
Stop after task {request_count}.

Family: {seed.family}

Curated seed:
{seed.instruction}

Derivation:
{derivation}
{prior_block}
Task requirements:
- Every task must be ready to hand directly to an assistant.
- Stay within {seed.family}.
- The tasks must be materially different from one another.
- Do not merely rename entities, variables, fields, dates, or numbers.
- Vary the real problem, objective, constraints, edge cases, and situation.
- Include all code, schemas, facts, labels, source text, or other material
  required to solve the task.
- Make the task internally consistent: no requirement, example, test expectation,
  function signature, or stated edge case may contradict another.
- Prefer a clear, realistic task over a dense checklist of loosely related constraints.
{family_task_guidance}
- Do not answer the tasks.
- Do not include numbering, labels, commentary, or extra fields outside the
  task strings themselves.

Before returning the JSON, verify internally that the "tasks" array has
exactly {request_count} entries.

Return one JSON object with exactly one field:
{{"tasks": ["task 1", "...", "task {request_count}"]}}

FINAL COUNT RULE: "tasks" must contain exactly {request_count} strings.
Do not generate task {request_count + 1}.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="tasks",
                count=request_count,
                description=(
                    f"Exactly {request_count} complete standalone "
                    f"{seed.family} user tasks."
                ),
            ),
            schema_name="sft_concrete_tasks",
        )
        return extract_string_list(parsed, "tasks")

    initial = request_tasks(count, ())
    return fill_exact_count(
        field="tasks",
        requested=count,
        initial=initial,
        fetch_missing=request_tasks,
        max_fill_attempts=max_fill_attempts,
        stage_label="task-generation",
    )


# ---------------------------------------------------------------------------
# Model 2 — answer generator
# ---------------------------------------------------------------------------

def generate_answers(
    *,
    family: str,
    tasks: Sequence[str],
    backend: Any,
    max_fill_attempts: int,
) -> list[str]:
    family_verification = {
        "programming": (
            "For programming tasks, verify the finished solution against every explicit "
            "behavioral requirement, edge case, API contract, and requested test. Check "
            "that tests actually exercise the behavior they claim to test, and correct "
            "material defects before returning the answer."
        ),
        "applied_math_and_reasoning": (
            "For applied-math and reasoning tasks, independently recompute numerical "
            "results, optimization claims, break-even values, probabilities, units, and "
            "constraint checks. Do not claim optimality, completeness, or feasibility "
            "without verifying the required alternatives or bounds."
        ),
        "creative_writing": (
            "For creative-writing tasks, treat exact word counts, line counts, required "
            "phrases, forbidden words, structural rules, point of view, and other explicit "
            "constraints as hard requirements. Verify them before returning the answer."
        ),
        "planning_brainstorming_recommendations": (
            "For planning tasks, cross-check schedules, overlaps, budgets, totals, timing, "
            "resource limits, requested alternatives, and feasibility. Do not invent "
            "distances, opening hours, accessibility, prices, availability, or other "
            "external facts that the task does not supply."
        ),
    }.get(family, "")

    def request_answers(
        selected_tasks: Sequence[str],
        prior_answers: Sequence[str],
    ) -> list[str]:
        body = "\n\n".join(
            f"--- TASK {index} ---\n{task}"
            for index, task in enumerate(selected_tasks, start=1)
        )
        request_count = len(selected_tasks)

        prompt = f"""
EXACT OUTPUT COUNT: {request_count}

Family: {family}

Answer every task below accurately and completely.

{body}

Requirements:
- Return exactly {request_count} answers.
- Return one answer per task in the same order.
- Answer each task as written.
- Before returning each answer, verify internally that every explicit deliverable,
  constraint, ordering rule, requested format, exclusion, count, length, and output
  contract in that task is satisfied.
- Recheck calculations and internally dependent claims for consistency instead of
  relying on the first result produced.
- Do not invent unsupported facts, measurements, prices, dates, distances,
  accessibility claims, APIs, files, environment state, or other external state.
- If a material requirement is not satisfied, correct the answer before returning it.
{family_verification}
- Do not rewrite the task.
- Do not emit IDs, roles, metadata, or synthetic-data commentary.
- The "answers" array must contain exactly {request_count} strings.

Return exactly:
{{"answers": ["answer 1", "...", "answer {request_count}"]}}

FINAL COUNT RULE: stop after answer {request_count}.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="answers",
                count=request_count,
                description=(
                    f"Exactly {request_count} assistant answers, in the same "
                    "order as the supplied tasks."
                ),
            ),
            schema_name="sft_answers",
        )
        return extract_string_list(parsed, "answers")

    initial = request_answers(tasks, ())

    if len(initial) >= len(tasks):
        if len(initial) > len(tasks):
            print(
                f"[cardinality:answers] requested={len(tasks)} observed={len(initial)} "
                f"action=trim excess={len(initial)-len(tasks)}",
                flush=True,
            )
        return initial[:len(tasks)]

    answers = list(initial)
    attempts = 0
    while len(answers) < len(tasks):
        missing = len(tasks) - len(answers)
        attempts += 1
        if attempts > max_fill_attempts:
            raise ValueError(
                f"answers remained underfilled after {max_fill_attempts} fill attempt(s): "
                f"requested={len(tasks)} observed={len(answers)}"
            )

        remaining_tasks = tasks[len(answers):]
        print(
            f"[cardinality:answers] requested={len(tasks)} observed={len(answers)} "
            f"missing={missing} action=request_missing attempt={attempts}/{max_fill_attempts}",
            flush=True,
        )

        extra = request_answers(remaining_tasks, answers)
        if not extra:
            raise ValueError(
                f"answers fill attempt returned no usable values: "
                f"requested={len(tasks)} observed={len(answers)}"
            )
        answers.extend(extra[:missing])

    return answers


# ---------------------------------------------------------------------------
# Judge / reviewer batch parsing
# ---------------------------------------------------------------------------

def public_conversation(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(row["messages"])



def parse_judge_batch(
    parsed: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    decisions = parsed.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("judge response must contain decisions object")

    expected = set(expected_ids)
    observed = set(decisions)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"judge decision keys mismatch: missing={missing} unexpected={unexpected}"
        )

    result: dict[str, dict[str, Any]] = {}
    for row_id in expected_ids:
        item = decisions[row_id]
        if not isinstance(item, Mapping):
            raise ValueError(f"judge decision must be object for {row_id}")

        assessable = item.get("assessable")
        accepted = item.get("accepted")
        reason = item.get("reason")

        if not isinstance(assessable, bool):
            raise ValueError(f"judge assessable must be boolean for {row_id}")
        if not isinstance(accepted, bool):
            raise ValueError(f"judge accepted must be boolean for {row_id}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"judge reason missing for {row_id}")

        # Code owns this invariant. An unassessable row cannot be accepted.
        accepted = bool(assessable and accepted)

        result[row_id] = {
            "assessable": assessable,
            "judge_accepted": accepted,
            "judge_reason": reason.strip(),
        }

    return result


# ---------------------------------------------------------------------------
# Model 3 — judge
# ---------------------------------------------------------------------------

def judge_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    backend: Any,
) -> dict[str, dict[str, Any]]:
    expected_ids = [row["id"] for row in rows]
    payload = [
        {
            "id": row["id"],
            "public_conversation": public_conversation(row),
            "deterministic_validation": "passed",
        }
        for row in rows
    ]

    prompt = f"""
Judge each SFT candidate below.

Evaluate the ASSISTANT response, not the user's request.

Reject only for a concrete material defect in correctness, grounding,
completeness, instruction adherence, or safety handling.

Do not invent stricter requirements.
Do not reject because another answer might be preferable.
Do not treat the user's unsafe, incorrect, or adversarial request as assistant behavior.
Deterministic structural checks already passed.

Candidates:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For each candidate id, decide:
- assessable: whether the row can be meaningfully assessed;
- accepted: whether the assistant response should be accepted;
- reason: one concise evidence-based reason.

The response structure is enforced by the supplied JSON Schema.
""".strip()

    parsed, _ = call_structured_object(
        backend,
        prompt=prompt,
        schema=judge_schema(expected_ids),
        schema_name="sft_judge_batch",
    )
    return parse_judge_batch(parsed, expected_ids)


def parse_reviewer_batch(
    parsed: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    decisions = parsed.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("reviewer response must contain decisions object")

    expected = set(expected_ids)
    observed = set(decisions)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"reviewer decision keys mismatch: missing={missing} unexpected={unexpected}"
        )

    result: dict[str, dict[str, Any]] = {}
    for row_id in expected_ids:
        item = decisions[row_id]
        if not isinstance(item, Mapping):
            raise ValueError(f"reviewer decision must be object for {row_id}")

        agreed = item.get("agreed")
        reason = item.get("reason")

        if not isinstance(agreed, bool):
            raise ValueError(f"reviewer agreed must be boolean for {row_id}")
        if not isinstance(reason, str):
            raise ValueError(f"reviewer reason must be string for {row_id}")
        if not agreed and not reason.strip():
            raise ValueError(
                f"reviewer disagreement requires a concrete reason for {row_id}"
            )

        result[row_id] = {
            "reviewed": True,
            "reviewer_agreed": agreed,
            "reviewer_reason": "AGREE" if agreed else reason.strip(),
            "accepted": agreed,
        }

    return result


# ---------------------------------------------------------------------------
# Model 4 — reviewer
# ---------------------------------------------------------------------------

def reviewer_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    judge_decisions: Mapping[str, Mapping[str, Any]],
    backend: Any,
) -> dict[str, dict[str, Any]]:
    expected_ids = [row["id"] for row in rows]
    payload = [
        {
            "id": row["id"],
            "public_conversation": public_conversation(row),
            "judge_reason": judge_decisions[row["id"]]["judge_reason"],
        }
        for row in rows
    ]

    prompt = f"""
Review the judge ACCEPT decisions below.

Your job is only to decide whether each acceptance is reasonably justified.

AGREE unless there is a clear material defect the judge missed.
Do not perform a stricter second perfection test.
Do not reject for style preferences, harmless ambiguity, or merely because
another answer could be better.

Items:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

For each supplied id:
- agreed=true when the judge acceptance is justified; use an empty reason;
- agreed=false only for a clear material defect; give one concise reason.

The response structure is enforced by the supplied JSON Schema.
""".strip()

    parsed, _ = call_structured_object(
        backend,
        prompt=prompt,
        schema=reviewer_schema(expected_ids),
        schema_name="sft_reviewer_batch",
    )
    return parse_reviewer_batch(parsed, expected_ids)


# ---------------------------------------------------------------------------
# Code-owned SFT structure
# ---------------------------------------------------------------------------

def make_spec(*, seed: Seed, task: Task, row_number: int) -> dict[str, Any]:
    metadata = dict(seed.metadata)
    metadata["task_family"] = seed.family
    metadata["interaction_modes"] = ["single_turn"]
    metadata["context_mode"] = "self_contained"

    return validate_sft_spec(
        {
            "id": f"sft_{seed.family}_{row_number:06d}",
            "instruction": task.text,
            "metadata": metadata,
            "constraints": [
                "The public task must contain every fact, code sample, schema, label set, or source needed to answer it.",
                "The assistant must answer the public task without relying on hidden seed or generation metadata.",
            ],
        }
    )


def make_row(spec: Mapping[str, Any], task: str, answer: str) -> dict[str, Any]:
    return validate_sft_row(
        {
            "id": spec["id"],
            "messages": [
                {"role": "user", "content": task},
                {"role": "assistant", "content": answer},
            ],
            "metadata": spec["metadata"],
        }
    )


def make_backend(
    *,
    model: str,
    max_tokens: int,
    concurrency: int,
    routing_mode: str,
    provider: str | None,
    temperature: float | None,
    top_p: float | None,
) -> Any:
    return build_backend(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        concurrency=concurrency,
        routing_mode=routing_mode,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_family(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out)

    seeds = load_seeds(args.family, args.seeds)

    plan = {
        "family": args.family,
        "seed_count": len(seeds),
        "derivations_per_seed": args.derivations_per_seed,
        "tasks_per_derivation": args.tasks_per_derivation,
        "planned_derivations": len(seeds) * args.derivations_per_seed,
        "planned_task_candidates": len(seeds) * args.derivations_per_seed * args.tasks_per_derivation,
        "answer_batch_size": args.answer_batch_size,
        "judge_batch_size": args.judge_batch_size,
        "reviewer_batch_size": args.reviewer_batch_size,
        "concurrency": args.concurrency,
        "cardinality_fill_attempts": args.cardinality_fill_attempts,
        "stage_batch_attempts": args.stage_batch_attempts,
        "models": {
            "derivation_generator": args.derivation_model,
            "task_generator": args.task_model,
            "answer_generator": args.answer_model,
            "judge": args.judge_model,
            "reviewer": args.reviewer_model,
        },
    }
    write_json(out / "plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)

    if args.dry_run:
        print("dry-run complete", flush=True)
        return 0

    derivation_backend = make_backend(
        model=args.derivation_model,
        max_tokens=args.derivation_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    task_backend = make_backend(
        model=args.task_model,
        max_tokens=args.task_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    answer_backend = make_backend(
        model=args.answer_model,
        max_tokens=args.answer_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    judge_backend = make_backend(
        model=args.judge_model,
        max_tokens=args.judge_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    reviewer_backend = make_backend(
        model=args.reviewer_model,
        max_tokens=args.reviewer_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Stage 1A: all derivations.
    derivation_batches = [
        (seed, list(range(1, args.derivations_per_seed + 1)))
        for seed in seeds
    ]

    def derive_seed_slots(
        batch: tuple[Seed, Sequence[int]],
    ) -> list[Derivation]:
        seed, slots = batch
        texts = generate_derivations(
            seed=seed,
            count=len(slots),
            backend=derivation_backend,
            max_fill_attempts=args.cardinality_fill_attempts,
        )
        return [
            Derivation(seed.id, seed.index, ordinal, derivation)
            for ordinal, derivation in zip(slots, texts, strict=True)
        ]

    derivation_results, derivation_failures = run_model_stage_with_isolation(
        stage="task-derivations",
        batches=derivation_batches,
        item_count=lambda batch: len(batch[1]),
        worker=derive_seed_slots,
        split_batch=split_slot_batch,
        concurrency=min(args.concurrency, max(1, len(seeds))),
        batch_size_display=args.derivations_per_seed,
        max_attempts=args.stage_batch_attempts,
    )

    derivations: list[Derivation] = []
    for result in derivation_results:
        derivations.extend(result)

    for failure in derivation_failures:
        seed, slots = failure["batch"]
        for ordinal in slots:
            append_jsonl(out / "stage.failures.jsonl", {
                "stage": "task-derivations",
                "seed_id": seed.id,
                "seed_index": seed.index,
                "derivation_ordinal": ordinal,
                "error": failure["error"],
            })

    derivations.sort(key=lambda x: (x.seed_index, x.ordinal))

    for item in derivations:
        append_jsonl(out / "derivations.generated.jsonl", {
            "seed_id": item.seed_id,
            "seed_index": item.seed_index,
            "ordinal": item.ordinal,
            "derivation": item.text,
        })

    # Stage 1B: all concrete tasks.
    seed_by_id = {seed.id: seed for seed in seeds}
    task_batches = [
        (item, list(range(1, args.tasks_per_derivation + 1)))
        for item in derivations
    ]

    def expand_derivation_slots(
        batch: tuple[Derivation, Sequence[int]],
    ) -> list[Task]:
        item, slots = batch
        seed = seed_by_id[item.seed_id]
        texts = generate_tasks_from_derivation(
            seed=seed,
            derivation=item.text,
            count=len(slots),
            backend=task_backend,
            max_fill_attempts=args.cardinality_fill_attempts,
        )
        return [
            Task(item.seed_id, item.seed_index, item.ordinal, ordinal, task_text)
            for ordinal, task_text in zip(slots, texts, strict=True)
        ]

    task_results, task_generation_failures = run_model_stage_with_isolation(
        stage="task-generation",
        batches=task_batches,
        item_count=lambda batch: len(batch[1]),
        worker=expand_derivation_slots,
        split_batch=split_slot_batch,
        concurrency=args.concurrency,
        batch_size_display=args.tasks_per_derivation,
        max_attempts=args.stage_batch_attempts,
    )

    tasks: list[Task] = []
    for result in task_results:
        tasks.extend(result)

    for failure in task_generation_failures:
        derivation, slots = failure["batch"]
        for ordinal in slots:
            append_jsonl(out / "stage.failures.jsonl", {
                "stage": "task-generation",
                "seed_id": derivation.seed_id,
                "seed_index": derivation.seed_index,
                "derivation_ordinal": derivation.ordinal,
                "task_ordinal": ordinal,
                "error": failure["error"],
            })

    tasks.sort(key=lambda x: (x.seed_index, x.derivation_ordinal, x.task_ordinal))

    # Stage 2: deterministic novelty / dedup.
    print(f"[task-dedup] start candidates={len(tasks)}", flush=True)
    dedup_started = time.monotonic()
    novelty = NoveltyFilter(
        jaccard_threshold=args.jaccard_threshold,
        sequence_threshold=args.sequence_threshold,
    )
    for seed in seeds:
        novelty.add(f"seed:{seed.id}", seed.instruction)

    accepted_tasks: list[Task] = []
    task_rejections: Counter[str] = Counter()

    for index, task in enumerate(tasks, 1):
        keep, reason, evidence = novelty.check(task.text)
        append_jsonl(out / "tasks.generated.jsonl", {
            "seed_id": task.seed_id,
            "seed_index": task.seed_index,
            "derivation_ordinal": task.derivation_ordinal,
            "task_ordinal": task.task_ordinal,
            "task": task.text,
            "keep": keep,
            "reason": reason,
            "evidence": evidence,
        })
        if keep:
            novelty.add(
                f"task:{task.seed_id}:{task.derivation_ordinal}:{task.task_ordinal}",
                task.text,
            )
            accepted_tasks.append(task)
            append_jsonl(out / "tasks.accepted.jsonl", {
                "seed_id": task.seed_id,
                "seed_index": task.seed_index,
                "derivation_ordinal": task.derivation_ordinal,
                "task_ordinal": task.task_ordinal,
                "task": task.text,
            })
        else:
            task_rejections[reason or "unknown"] += 1

        if index % 100 == 0 or index == len(tasks):
            print(
                f"[task-dedup] progress checked={index}/{len(tasks)} "
                f"accepted={len(accepted_tasks)} rejected={index-len(accepted_tasks)} "
                f"candidate_comparisons={novelty.candidate_comparisons} "
                f"sequence_comparisons={novelty.sequence_comparisons} "
                f"elapsed={time.monotonic()-dedup_started:.1f}s",
                flush=True,
            )

    print(
        f"[task-dedup] complete generated={len(tasks)} accepted={len(accepted_tasks)} "
        f"rejected={len(tasks)-len(accepted_tasks)} "
        f"candidate_comparisons={novelty.candidate_comparisons} "
        f"sequence_comparisons={novelty.sequence_comparisons} "
        f"elapsed={time.monotonic()-dedup_started:.1f}s",
        flush=True,
    )

    # Stage 3: ALL answers concurrently.
    answer_batches = chunked(accepted_tasks, args.answer_batch_size)

    def answer_worker(batch: list[Task]) -> list[tuple[Task, str]]:
        answers = generate_answers(
            family=args.family,
            tasks=[item.text for item in batch],
            backend=answer_backend,
            max_fill_attempts=args.cardinality_fill_attempts,
        )
        return list(zip(batch, answers, strict=True))

    answer_results, answer_failures = run_model_stage_with_isolation(
        stage="answers",
        batches=answer_batches,
        item_count=len,
        worker=answer_worker,
        split_batch=split_sequence_batch,
        concurrency=args.concurrency,
        batch_size_display=args.answer_batch_size,
        max_attempts=args.stage_batch_attempts,
    )

    for failure in answer_failures:
        for task in failure["batch"]:
            append_jsonl(out / "stage.failures.jsonl", {
                "stage": "answers",
                "seed_id": task.seed_id,
                "seed_index": task.seed_index,
                "derivation_ordinal": task.derivation_ordinal,
                "task_ordinal": task.task_ordinal,
                "task": task.text,
                "error": failure["error"],
            })

    answered: list[tuple[Task, str]] = []
    for result in answer_results:
        answered.extend(result)
    answered.sort(key=lambda pair: (
        pair[0].seed_index,
        pair[0].derivation_ordinal,
        pair[0].task_ordinal,
    ))

    generated_specs: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []

    for row_number, (task, answer) in enumerate(answered, 1):
        seed = seed_by_id[task.seed_id]
        spec = make_spec(seed=seed, task=task, row_number=row_number)
        row = make_row(spec, task.text, answer)
        generated_specs.append(spec)
        generated_rows.append(row)
        append_jsonl(out / "answers.generated.jsonl", {
            "id": spec["id"],
            "seed_id": task.seed_id,
            "derivation_ordinal": task.derivation_ordinal,
            "task": task.text,
            "answer": answer,
        })

    # Stage 4: deterministic validation.
    print(f"[deterministic-validation] start rows={len(generated_rows)}", flush=True)
    validation_started = time.monotonic()
    valid_specs: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    deterministic_rejections = 0

    for index, (spec, row) in enumerate(zip(generated_specs, generated_rows, strict=True), 1):
        try:
            evaluate_sft_output_constraints(specs=[spec], rows=[row])
        except OutputConstraintError as exc:
            deterministic_rejections += 1
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "deterministic",
                "id": spec["id"],
                "spec": spec,
                "row": row,
                "reason": str(exc),
            })
        else:
            valid_specs.append(spec)
            valid_rows.append(row)
            append_jsonl(out / "answers.validated.jsonl", {
                "id": spec["id"],
                "messages": row["messages"],
                "metadata": row["metadata"],
            })

        if index % 100 == 0 or index == len(generated_rows):
            print(
                f"[deterministic-validation] progress checked={index}/{len(generated_rows)} "
                f"valid={len(valid_rows)} rejected={deterministic_rejections} "
                f"elapsed={time.monotonic()-validation_started:.1f}s",
                flush=True,
            )

    print(
        f"[deterministic-validation] complete valid={len(valid_rows)} "
        f"rejected={deterministic_rejections} elapsed={time.monotonic()-validation_started:.1f}s",
        flush=True,
    )

    rows_by_id = {row["id"]: row for row in valid_rows}
    specs_by_id = {spec["id"]: spec for spec in valid_specs}

    # Stage 5: ALL judge batches concurrently.
    #
    # OpenRouter JSON Schema owns output shape. The repo backend already
    # retries provider/render failures; this stage wrapper retries only the
    # affected batch. Local parsing then validates the semantic invariants.
    judge_batches = chunked(valid_rows, args.judge_batch_size)

    judge_result_batches, judge_failures = run_model_stage_with_isolation(
        stage="judge",
        batches=judge_batches,
        item_count=len,
        worker=lambda batch: judge_batch(
            rows=batch,
            backend=judge_backend,
        ),
        split_batch=split_sequence_batch,
        concurrency=args.concurrency,
        batch_size_display=args.judge_batch_size,
        max_attempts=args.stage_batch_attempts,
    ) if judge_batches else ([], [])

    for failure in judge_failures:
        for row in failure["batch"]:
            append_jsonl(out / "stage.failures.jsonl", {
                "stage": "judge",
                "id": row["id"],
                "error": failure["error"],
            })
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "judge_unrecoverable",
                "id": row["id"],
                "spec": specs_by_id[row["id"]],
                "row": row,
                "reason": failure["error"],
            })


    judge_decisions: dict[str, dict[str, Any]] = {}
    for result in judge_result_batches:
        judge_decisions.update(result)

    missing_judge_ids = [
        row["id"] for row in valid_rows
        if row["id"] not in judge_decisions
    ]
    if missing_judge_ids:
        print(
            f"[judge] unrecoverable_items={len(missing_judge_ids)} "
            f"action=exclude_and_continue",
            flush=True,
        )

    judge_accepted_ids: list[str] = []
    for row in valid_rows:
        if row["id"] not in judge_decisions:
            continue
        decision = judge_decisions[row["id"]]
        append_jsonl(out / "judge.decisions.jsonl", {
            "id": row["id"],
            **decision,
        })
        if decision["judge_accepted"]:
            judge_accepted_ids.append(row["id"])
        else:
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "judge",
                "id": row["id"],
                "spec": specs_by_id[row["id"]],
                "row": row,
                "decision": decision,
            })

    print(
        f"[judge] result accepted={len(judge_accepted_ids)} "
        f"rejected={len(valid_rows)-len(judge_accepted_ids)}",
        flush=True,
    )

    # Stage 6: reviewer ONLY judge-accepted rows, concurrently.
    review_rows = [rows_by_id[row_id] for row_id in judge_accepted_ids]
    reviewer_batches = chunked(review_rows, args.reviewer_batch_size)

    reviewer_result_batches, reviewer_failures = run_model_stage_with_isolation(
        stage="reviewer",
        batches=reviewer_batches,
        item_count=len,
        worker=lambda batch: reviewer_batch(
            rows=batch,
            judge_decisions=judge_decisions,
            backend=reviewer_backend,
        ),
        split_batch=split_sequence_batch,
        concurrency=args.concurrency,
        batch_size_display=args.reviewer_batch_size,
        max_attempts=args.stage_batch_attempts,
    ) if reviewer_batches else ([], [])

    for failure in reviewer_failures:
        for row in failure["batch"]:
            append_jsonl(out / "stage.failures.jsonl", {
                "stage": "reviewer",
                "id": row["id"],
                "error": failure["error"],
            })
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "reviewer_unrecoverable",
                "id": row["id"],
                "spec": specs_by_id[row["id"]],
                "row": row,
                "judge": judge_decisions[row["id"]],
                "reason": failure["error"],
            })


    reviewer_decisions: dict[str, dict[str, Any]] = {}
    for result in reviewer_result_batches:
        reviewer_decisions.update(result)

    missing_reviewer_ids = [
        row_id for row_id in judge_accepted_ids
        if row_id not in reviewer_decisions
    ]
    if missing_reviewer_ids:
        print(
            f"[reviewer] unrecoverable_items={len(missing_reviewer_ids)} "
            f"action=exclude_and_continue",
            flush=True,
        )

    final_rows: list[dict[str, Any]] = []

    for row_id in judge_accepted_ids:
        if row_id not in reviewer_decisions:
            continue
        review = reviewer_decisions[row_id]
        append_jsonl(out / "reviewer.decisions.jsonl", {
            "id": row_id,
            **review,
        })

        if review["accepted"]:
            final_rows.append(rows_by_id[row_id])
        else:
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "reviewer",
                "id": row_id,
                "spec": specs_by_id[row_id],
                "row": rows_by_id[row_id],
                "judge": judge_decisions[row_id],
                "review": review,
            })

    print(
        f"[reviewer] result accepted={len(final_rows)} "
        f"rejected={len(judge_accepted_ids)-len(final_rows)}",
        flush=True,
    )

    # Stage 7: final exact dedup.
    unique_rows, final_dedup = partition_unique_sft_rows(final_rows)
    for row in unique_rows:
        append_jsonl(out / "sft.accepted.jsonl", row)

    summary = {
        **plan,
        "derivations_generated": len(derivations),
        "derivation_generation_failures": sum(
            failure["items"] for failure in derivation_failures
        ),
        "task_candidates_generated": len(tasks),
        "task_generation_failures": sum(
            failure["items"] for failure in task_generation_failures
        ),
        "task_candidates_after_dedup": len(accepted_tasks),
        "task_rejection_counts": dict(sorted(task_rejections.items())),
        "task_dedup_candidate_comparisons": novelty.candidate_comparisons,
        "task_dedup_sequence_comparisons": novelty.sequence_comparisons,
        "answers_generated": len(generated_rows),
        "answer_generation_failures": sum(
            failure["items"] for failure in answer_failures
        ),
        "answers_validated": len(valid_rows),
        "deterministic_rejections": deterministic_rejections,
        "judge_accepted": len(judge_accepted_ids),
        "judge_rejected": sum(
            1 for decision in judge_decisions.values()
            if not decision["judge_accepted"]
        ),
        "judge_unrecoverable": len(missing_judge_ids),
        "reviewer_accepted": len(final_rows),
        "reviewer_rejected": sum(
            1 for decision in reviewer_decisions.values()
            if not decision["accepted"]
        ),
        "reviewer_unrecoverable": len(missing_reviewer_ids),
        "final_accepted_rows": len(unique_rows),
        "final_dedup": final_dedup,
        "outputs": {
            "derivations": str(out / "derivations.generated.jsonl"),
            "generated_tasks": str(out / "tasks.generated.jsonl"),
            "accepted_tasks": str(out / "tasks.accepted.jsonl"),
            "generated_answers": str(out / "answers.generated.jsonl"),
            "validated_answers": str(out / "answers.validated.jsonl"),
            "judge_decisions": str(out / "judge.decisions.jsonl"),
            "reviewer_decisions": str(out / "reviewer.decisions.jsonl"),
            "rejections": str(out / "quality.rejected.jsonl"),
            "stage_failures": str(out / "stage.failures.jsonl"),
            "final_dataset": str(out / "sft.accepted.jsonl"),
        },
    }

    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0




def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"expected object in {path} line {line_number}")
            rows.append(dict(value))
    return rows


def _resolve_families(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(values)
    if not requested or requested == ("all",):
        return tuple(sorted(SFT_SPEC_FAMILIES))
    unknown = sorted(set(requested) - set(SFT_SPEC_FAMILIES))
    if unknown:
        raise ValueError(f"unknown SFT families: {unknown}")
    return tuple(dict.fromkeys(requested))


def _quality_evidence(work_dir: Path, accepted_ids: set[str]) -> dict[str, dict[str, Any]]:
    judges = {
        row["id"]: row
        for row in _read_jsonl(work_dir / "judge.decisions.jsonl")
        if isinstance(row.get("id"), str)
    }
    reviewers = {
        row["id"]: row
        for row in _read_jsonl(work_dir / "reviewer.decisions.jsonl")
        if isinstance(row.get("id"), str)
    }
    evidence: dict[str, dict[str, Any]] = {}
    for row_id in sorted(accepted_ids):
        judge = judges.get(row_id)
        review = reviewers.get(row_id)
        if judge is None or review is None:
            raise RuntimeError(f"missing judge/reviewer evidence for accepted SFT row {row_id}")
        evidence[row_id] = {
            "assessable": judge.get("assessable") is True,
            "judge_accepted": judge.get("judge_accepted") is True,
            "judge_reason": str(judge.get("judge_reason", "")),
            "reviewed": review.get("reviewed") is True,
            "reviewer_agreed": review.get("reviewer_agreed") is True,
            "reviewer_reason": str(review.get("reviewer_reason", "")),
            "accepted": review.get("accepted") is True,
        }
        if not (
            evidence[row_id]["assessable"]
            and evidence[row_id]["judge_accepted"]
            and evidence[row_id]["reviewed"]
            and evidence[row_id]["reviewer_agreed"]
            and evidence[row_id]["accepted"]
        ):
            raise RuntimeError(f"accepted SFT row has non-passing quality evidence: {row_id}")
    return evidence


def _write_family_evidence_manifest(
    *,
    path: Path,
    family: str,
    dataset_path: Path,
    rows: list[dict[str, Any]],
    generation_run: str,
    work_dir: Path,
    summary: Mapping[str, Any],
) -> Path:
    accepted_ids = {row["id"] for row in rows}
    payload = {
        "schema_version": 1,
        "dataset_type": "sft",
        "generation_run": generation_run,
        "family": family,
        "dataset_path": str(dataset_path),
        "row_count": len(rows),
        "metadata": {
            "pipeline": "derivation_task_novelty_answer_validate_judge_reviewer_dedup",
            "deterministic_output_validation": {
                row_id: {"status": "passed"}
                for row_id in sorted(accepted_ids)
            },
            "quality_adjudication": _quality_evidence(work_dir, accepted_ids),
            "summary_path": str(work_dir / "summary.json"),
            "summary": dict(summary),
        },
    }
    write_json(path, payload)
    return path


def run_production(args: argparse.Namespace) -> int:
    families = _resolve_families(args.families)
    run_dir = Path(args.output_dir)
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    work_root = run_dir / "work"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    dataset_entries: list[dict[str, Any]] = []
    family_summaries: dict[str, dict[str, Any]] = {}
    attempted_per_family: dict[str, int] = {}
    accepted_per_family: dict[str, int] = {}
    rejected_per_family: dict[str, int] = {}
    duplicate_per_family: dict[str, int] = {}

    for family in families:
        work_dir = work_root / family
        family_args = argparse.Namespace(**vars(args))
        family_args.family = family
        family_args.output_dir = str(work_dir)
        print(f"[sft-pipeline] family={family} start", flush=True)
        rc = run_family(family_args)
        if rc != 0:
            return rc
        if args.dry_run:
            continue

        summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
        accepted_rows = [validate_sft_row(row) for row in _read_jsonl(work_dir / "sft.accepted.jsonl")]
        if not accepted_rows:
            raise RuntimeError(f"SFT family produced no accepted rows: {family}")

        dataset_path = dataset_dir / f"{family}.jsonl"
        reset_output_files(dataset_dir, (dataset_path.name,))
        for row in accepted_rows:
            append_jsonl(dataset_path, row)

        evidence_manifest = _write_family_evidence_manifest(
            path=manifest_dir / f"{family}.{args.generation_run}.manifest.json",
            family=family,
            dataset_path=dataset_path,
            rows=accepted_rows,
            generation_run=args.generation_run,
            work_dir=work_dir,
            summary=summary,
        )
        dataset_entries.append(
            {
                "family": family,
                "dataset_path": str(dataset_path),
                "row_count": len(accepted_rows),
                "batch_count": 1,
                "batch_manifests": [str(evidence_manifest)],
            }
        )
        family_summaries[family] = dict(summary)
        attempted = int(summary.get("answers_generated", 0) or 0)
        final_dedup = summary.get("final_dedup")
        duplicates = (
            int(final_dedup.get("duplicate_rows", 0) or 0)
            if isinstance(final_dedup, Mapping)
            else 0
        )
        attempted_per_family[family] = attempted
        accepted_per_family[family] = len(accepted_rows)
        duplicate_per_family[family] = duplicates
        rejected_per_family[family] = max(attempted - len(accepted_rows) - duplicates, 0)
        print(
            f"[sft-pipeline] family={family} complete accepted={len(accepted_rows)} "
            f"dataset={dataset_path}",
            flush=True,
        )

    if args.dry_run:
        print("[sft-pipeline] dry-run complete", flush=True)
        return 0

    total_attempted = sum(attempted_per_family.values())
    total_accepted = sum(accepted_per_family.values())
    total_duplicates = sum(duplicate_per_family.values())
    total_rejected = sum(rejected_per_family.values())
    run_manifest = {
        "schema_version": 1,
        "dataset_type": "sft",
        "generation_run": args.generation_run,
        "generation_mode": "stage_oriented_pipeline",
        "teacher_model": args.answer_model,
        "teacher_provider": "openrouter",
        "families": list(families),
        "datasets": dataset_entries,
        "total_rows": total_accepted,
        "metadata": {
            "pipeline": "derivation_task_novelty_answer_validate_judge_reviewer_dedup",
            "derivation_model": args.derivation_model,
            "task_model": args.task_model,
            "answer_model": args.answer_model,
            "adjudicator_model": args.judge_model,
            "reviewer_model": args.reviewer_model,
            "candidate_rows": total_attempted,
            "attempted_rows": total_attempted,
            "accepted_rows": total_accepted,
            "rejected_rows": total_rejected,
            "duplicate_rows": total_duplicates,
            "candidate_rows_per_family": attempted_per_family,
            "attempted_rows_per_family": attempted_per_family,
            "accepted_rows_per_family": accepted_per_family,
            "rejected_rows_per_family": rejected_per_family,
            "duplicate_rows_per_family": duplicate_per_family,
            "generation_status": "complete",
            "publish_ready": True,
            "family_summaries": family_summaries,
        },
    }
    manifest_path = manifest_dir / f"{args.generation_run}.manifest.json"
    write_json(manifest_path, run_manifest)
    print(
        f"[sft-pipeline] complete families={len(families)} accepted={total_accepted} "
        f"manifest={manifest_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production generic SFT pipeline migrated from the finalized one-off."
    )
    p.add_argument("--families", nargs="+", default=["all"])
    p.add_argument("--generation-run", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--derivations-per-seed", type=int, default=30)
    p.add_argument("--tasks-per-derivation", type=int, default=15)
    p.add_argument("--answer-batch-size", type=int, default=4)
    p.add_argument("--judge-batch-size", type=int, default=10)
    p.add_argument("--reviewer-batch-size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cardinality-fill-attempts", type=int, default=3)
    p.add_argument("--stage-batch-attempts", type=int, default=3)
    p.add_argument("--derivation-model", default="openai/gpt-5.6-luna-pro")
    p.add_argument("--task-model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--answer-model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--judge-model", default="nvidia/nemotron-3.5-lightning")
    p.add_argument("--reviewer-model", default="google/gemma-4-31b-it")
    p.add_argument("--derivation-max-tokens", type=int, default=4096)
    p.add_argument("--task-max-tokens", type=int, default=4096)
    p.add_argument("--answer-max-tokens", type=int, default=4096)
    p.add_argument("--judge-max-tokens", type=int, default=4096)
    p.add_argument("--reviewer-max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--routing-mode", "--openrouter-routing-mode", dest="routing_mode", choices=["auto", "prefer", "strict"], default="auto")
    p.add_argument("--provider", "--openrouter-provider", dest="provider", default=None)
    p.add_argument("--jaccard-threshold", type=float, default=0.82)
    p.add_argument("--sequence-threshold", type=float, default=0.90)
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "seeds",
        "derivations_per_seed",
        "tasks_per_derivation",
        "answer_batch_size",
        "judge_batch_size",
        "reviewer_batch_size",
        "concurrency",
        "cardinality_fill_attempts",
        "stage_batch_attempts",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    return run_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
