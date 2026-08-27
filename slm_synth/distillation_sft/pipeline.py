#!/usr/bin/env python3
"""Production Distillation-SFT pipeline migrated from the finalized one-off.

Stages: derivation -> student-appropriate prompt -> task novelty -> teacher response
-> deterministic row/response validation + response novelty -> judge -> reviewer ->
final exact prompt/response dedup. Shared runtime owns backend construction,
batching/cardinality, bounded retries with recursive isolation, novelty, and IO.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any




from slm_synth.runtime import (
    NoveltyFilter,
    normalize,
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
from slm_synth.distillation_sft.io import write_manifest, write_run_manifest, write_signal_dataset
from slm_synth.distillation_sft.public_metadata import build_public_metadata
from slm_synth.distillation_sft.schema import validate_public_row
from slm_synth.distillation_sft.signals import DISTILLATION_SIGNALS

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


DISTILLATION_SFT_SEEDS: dict[str, list[str]] = {
    "arithmetic": [
        "Solve practical arithmetic and numerical reasoning tasks accurately, including units, percentages, ratios, rates, and multi-step calculations.",
        "Explain and verify arithmetic claims, catching common calculation, rounding, sign, unit, and proportional-reasoning errors.",
        "Apply arithmetic to realistic finance, measurement, scheduling, estimation, and everyday quantitative decisions.",
    ],
    "cloud": [
        "Answer practical cloud-infrastructure questions using grounded architecture, networking, security, reliability, scaling, and operations knowledge.",
        "Diagnose cloud configuration and deployment problems from self-contained evidence without inventing provider state or unavailable resources.",
        "Explain cloud design trade-offs and operational decisions with precise assumptions, constraints, and failure-mode awareness.",
    ],
    "code": [
        "Write, explain, transform, and reason about code for realistic programming tasks with correct behavior, edge-case handling, and maintainability.",
        "Implement self-contained programming solutions from explicit requirements, examples, schemas, interfaces, and constraints.",
        "Review code behavior and propose concrete corrections without inventing APIs, files, dependencies, or runtime state not supplied by the user.",
    ],
    "data_transform": [
        "Transform structured and semi-structured data accurately across tables, JSON, CSV, text records, schemas, units, and field mappings.",
        "Design robust data-cleaning and reshaping logic that handles missing values, malformed records, duplicate data, types, and validation constraints.",
        "Explain and implement reproducible data transformations while preserving meaning, ordering, keys, and requested output formats.",
    ],
    "database": [
        "Answer practical database questions covering SQL, schema design, transactions, indexing, constraints, query behavior, and data integrity.",
        "Diagnose database correctness and performance problems from self-contained schemas, queries, plans, and application constraints.",
        "Produce database solutions that distinguish logical correctness, concurrency behavior, operational trade-offs, and engine-specific assumptions.",
    ],
    "debugging": [
        "Diagnose realistic software failures from code, errors, logs, inputs, and observed behavior, then explain the root cause and targeted fix.",
        "Reason about subtle debugging failures such as state mutation, boundary conditions, concurrency, parsing, retries, APIs, configuration, and data shape.",
        "Separate evidence from hypotheses during debugging and avoid claiming environment facts that are not present in the task.",
    ],
    "educational_qa": [
        "Teach and answer self-contained educational questions accurately across science, mathematics, computing, history, language, and general knowledge.",
        "Explain concepts at an appropriate level using clear reasoning, examples, distinctions, and qualifications without unnecessary boilerplate.",
        "Correct misconceptions and verify claims while distinguishing established facts, assumptions, approximations, and uncertainty.",
    ],
    "factual_restraint": [
        "Respond accurately when information is missing, ambiguous, unverifiable, current, or source-dependent; do not fabricate facts or citations.",
        "Distinguish what can be concluded from supplied evidence from what would require external verification, live state, or additional context.",
        "Handle uncertain factual questions with calibrated confidence, explicit assumptions, and concise requests for genuinely necessary information.",
    ],
    "instruction": [
        "Follow explicit user instructions, constraints, formats, ordering requirements, transformations, and output contracts exactly while preserving correctness.",
        "Resolve multi-constraint instruction-following tasks without silently dropping requirements or adding unrequested content.",
        "Produce responses that respect scope, requested format, exclusions, source material, and stated assumptions.",
    ],
    "planning": [
        "Create realistic plans, procedures, troubleshooting sequences, implementation steps, and decision frameworks from stated goals and constraints.",
        "Prioritize dependencies, risks, alternatives, checkpoints, and verification steps in practical technical and non-technical planning tasks.",
        "Adapt plans to incomplete information without inventing unavailable resources, commitments, schedules, or external state.",
    ],
}
DISTILLATION_SFT_FAMILIES = tuple(DISTILLATION_SFT_SEEDS)


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


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def load_seeds(family: str, count: int) -> list[Seed]:
    seed_texts = DISTILLATION_SFT_SEEDS[family]
    if count > len(seed_texts):
        raise ValueError(
            f"--seeds={count} exceeds the {len(seed_texts)} curated seed(s) "
            f"available for distillation family {family!r}"
        )

    return [
        Seed(
            index=index,
            id=f"distill_seed_{family}_{index:03d}",
            family=family,
            instruction=instruction,
            metadata={
                "dataset_kind": "distillation_sft",
                "distillation_family": family,
                "seed_index": index,
            },
        )
        for index, instruction in enumerate(seed_texts[:count], start=1)
    ]


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
            "defect_category": {
                "type": "string",
                "enum": [
                    "none",
                    "incorrect_result",
                    "non_compiling_or_non_running",
                    "missed_explicit_requirement",
                    "prompt_response_mismatch",
                    "unsupported_claim",
                    "materially_incomplete",
                    "malformed_or_unusable",
                    "template_contamination",
                    "insufficient_prompt_conditioning",
                ],
                "description": (
                    "Use 'none' when agreed=true. When agreed=false, select the "
                    "single category that best describes the verified material defect."
                ),
            },
            "evidence": {
                "type": "string",
                "description": (
                    "When agreed=false, quote or precisely identify the exact code, "
                    "statement, control-flow behavior, or explicit prompt requirement "
                    "that proves the defect. Empty when agreed=true."
                ),
            },
            "verification": {
                "type": "string",
                "description": (
                    "When agreed=false, give the minimal concrete verification showing "
                    "why the cited evidence fails the explicit requirement. This must "
                    "be based only on the supplied prompt and response. Empty when "
                    "agreed=true."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "Empty string when agreed=true; concise concrete defect "
                    "when agreed=false."
                ),
            },
        },
        "required": ["agreed", "defect_category", "evidence", "verification", "reason"],
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
You are generating semantic task classes for a response-distillation SFT
family: {seed.family}.

Capability seed:
{seed.instruction}

Generate exactly {request_count} DISTINCT semantic derivations.
{prior_block}
A derivation is a concise capability/problem class that can later produce many
natural user prompts and high-quality teacher responses.

Requirements:
- Stay within the {seed.family} capability family.
- Make derivations materially different in concept, reasoning requirement,
  failure mode, objective, constraints, or evidence pattern.
- Do not create cosmetic variants by changing names, entities, numbers,
  products, dates, clouds, programming languages, or surface wording.
- Prefer reusable skills the student can learn from a response-distillation
  example.
- Do not depend primarily on live browsing, proprietary state, hidden files,
  enormous context, or unverifiable current facts.
- Do not produce final user prompts yet.
- Do not provide answers or solutions.

Before returning each derivation, compare its core skill and solution strategy
against the other selected derivations and reject substantial overlap.

Return exactly {request_count} derivations under the supplied structured-output
schema.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="derivations",
                count=request_count,
                description=(
                    f"Exactly {request_count} distinct response-distillation "
                    f"task classes for {seed.family}."
                ),
            ),
            schema_name="distillation_sft_task_derivations",
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
    def request_tasks(request_count: int, prior: Sequence[str]) -> list[str]:
        prior_block = ""
        if prior:
            prior_block = (
                "\nAlready accepted tasks from this derivation. Do not repeat "
                "or substantially overlap these:\n"
                + "\n".join(f"- {item}" for item in prior)
                + "\n"
            )

        prompt = f"""
EXACT OUTPUT COUNT: {request_count}

Generate exactly {request_count} COMPLETE, STANDALONE, NATURAL user prompts
for response distillation.

Distillation family: {seed.family}

Capability seed:
{seed.instruction}

Semantic derivation:
{derivation}
{prior_block}
Every prompt must:
- be something a real user could naturally ask an assistant;
- be directly answerable from the prompt and ordinary stable knowledge or
  reasoning;
- contain every required code sample, schema, table, source passage, value,
  constraint, label set, or local fact needed for the answer;
- elicit reusable teacher behavior rather than a one-off trick;
- be materially different from the other prompts in the batch;
- vary the actual problem, objective, constraints, evidence, edge cases, or
  reasoning strategy rather than merely changing entities or numbers;
- avoid requiring live web access, current account state, hidden files,
  proprietary systems, or unspecified external evidence;
- avoid benchmark/test-set language and avoid mentioning teachers, students,
  SFT, distillation, synthetic data, judges, reviewers, chosen/rejected data,
  or dataset generation;
- not reveal the expected answer or instruct the assistant to manufacture a
  mistake;
- not contain multiple unrelated requests accidentally fused into one task.

Do not answer the prompts.
Do not include numbering, labels, commentary, or extra fields outside the task
strings themselves.

Return exactly {request_count} strings in the "tasks" array.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="tasks",
                count=request_count,
                description=(
                    f"Exactly {request_count} complete standalone natural user "
                    f"prompts for distillation family {seed.family}."
                ),
            ),
            schema_name="distillation_sft_concrete_tasks",
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
# Model 2 — teacher response generator
# ---------------------------------------------------------------------------

def generate_answers(
    *,
    family: str,
    tasks: Sequence[str],
    backend: Any,
    max_fill_attempts: int,
) -> list[str]:
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

You are the teacher-response generator for response-distillation SFT.
Distillation family: {family}

Answer every task below as a strong assistant would answer the user directly.

{body}

Teacher-response requirements:
- Return exactly {request_count} answers, one per task in the same order.
- Answer the actual user request directly and completely.
- Be factually and logically correct.
- Follow every explicit instruction, requested format, scope limitation, and
  constraint in the task.
- Include enough explanation, intermediate work, evidence use, or reasoning
  summary for the response to teach reusable behavior, but do not add padding
  or unnecessary exposition.
- When the task is underspecified, preserve uncertainty, state the material
  assumption, or request genuinely necessary information instead of inventing
  facts.
- Do not invent citations, APIs, files, environment state, product behavior,
  cloud resources, study results, current facts, or source material.
- Do not expose hidden chain-of-thought. Give concise useful reasoning or
  calculations when needed for the answer.
- Do not use canned introductions or conclusions merely to make answers look
  different.
- Write each answer independently for its corresponding task. Never split one
  task's answer across neighboring array items and never reuse one generic
  response for unrelated tasks.
- Do not rewrite the task.
- Do not emit IDs, roles, metadata, teacher/student language, or synthetic-data
  commentary.

The "answers" array must contain exactly {request_count} non-empty strings.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="answers",
                count=request_count,
                description=(
                    f"Exactly {request_count} high-quality teacher responses, "
                    "in the same order as the supplied tasks."
                ),
            ),
            schema_name="distillation_sft_teacher_responses",
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
            "prompt": row["prompt"],
            "response": row["response"],
            "deterministic_validation": "passed",
        }
        for row in rows
    ]

    prompt = f"""
Judge each response-distillation SFT candidate below.

Evaluate the complete PROMPT + TEACHER RESPONSE pair.

Accept when the response is suitable as a high-quality response-distillation
example. Reject only for a concrete material defect such as:
- an incorrect factual, mathematical, logical, technical, or code claim;
- failure to answer the prompt or a prompt/response semantic mismatch;
- violation of an explicit user instruction or requested output constraint;
- materially incomplete treatment of the requested task;
- unsupported fabrication, invented source/state, or unjustified certainty;
- unsafe handling that is materially inappropriate for the request;
- obvious boilerplate/template contamination or a generic response that is not
  genuinely conditioned on the prompt.

Do not invent stricter requirements.
Do not reject merely because another wording could be better, because optional
detail is absent, or because of harmless stylistic preferences.
Deterministic structural and novelty checks already passed.

Candidates:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For each candidate id, decide:
- assessable: whether the prompt/response pair can be meaningfully assessed;
- accepted: whether the teacher response should be accepted;
- reason: one concise evidence-based reason.

The response structure is enforced by the supplied JSON Schema.
""".strip()

    parsed, _ = call_structured_object(
        backend,
        prompt=prompt,
        schema=judge_schema(expected_ids),
        schema_name="distillation_sft_judge_batch",
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
        defect_category = item.get("defect_category")
        evidence = item.get("evidence")
        verification = item.get("verification")
        reason = item.get("reason")

        if not isinstance(agreed, bool):
            raise ValueError(f"reviewer agreed must be boolean for {row_id}")
        if not isinstance(defect_category, str):
            raise ValueError(f"reviewer defect_category must be string for {row_id}")
        if not isinstance(evidence, str):
            raise ValueError(f"reviewer evidence must be string for {row_id}")
        if not isinstance(verification, str):
            raise ValueError(f"reviewer verification must be string for {row_id}")
        if not isinstance(reason, str):
            raise ValueError(f"reviewer reason must be string for {row_id}")

        if agreed:
            if defect_category != "none":
                raise ValueError(
                    f"reviewer agreement requires defect_category='none' for {row_id}"
                )
            if evidence.strip() or verification.strip() or reason.strip():
                raise ValueError(
                    f"reviewer agreement requires empty evidence/verification/reason for {row_id}"
                )
        else:
            if defect_category == "none":
                raise ValueError(
                    f"reviewer disagreement requires a defect category for {row_id}"
                )
            if not evidence.strip():
                raise ValueError(
                    f"reviewer disagreement requires concrete evidence for {row_id}"
                )
            if not verification.strip():
                raise ValueError(
                    f"reviewer disagreement requires verification for {row_id}"
                )
            if not reason.strip():
                raise ValueError(
                    f"reviewer disagreement requires a concrete reason for {row_id}"
                )

            contradiction_markers = (
                "no material defect",
                "no clear material defect",
                "judge's acceptance is justified",
                "judge acceptance is justified",
                "response is correct",
                "implementation works correctly",
            )
            combined = " ".join(
                [evidence.lower(), verification.lower(), reason.lower()]
            )
            if any(marker in combined for marker in contradiction_markers):
                raise ValueError(
                    f"reviewer disagreement contradicts its own verification for {row_id}"
                )

        result[row_id] = {
            "reviewed": True,
            "reviewer_agreed": agreed,
            "reviewer_defect_category": defect_category,
            "reviewer_evidence": evidence.strip(),
            "reviewer_verification": verification.strip(),
            "reviewer_reason": "AGREE" if agreed else reason.strip(),
            "accepted": agreed,
        }

    return result


# ---------------------------------------------------------------------------
# Model 4 — reviewer
# ---------------------------------------------------------------------------

def reviewer_batch(
    *,
    family: str,
    rows: Sequence[Mapping[str, Any]],
    judge_decisions: Mapping[str, Mapping[str, Any]],
    backend: Any,
) -> dict[str, dict[str, Any]]:
    expected_ids = [row["id"] for row in rows]
    payload = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "judge_reason": judge_decisions[row["id"]]["judge_reason"],
        }
        for row in rows
    ]

    code_calibration = ""
    if family == "code":
        code_calibration = """

Code-review calibration:
- Evaluate the response against the prompt's explicit requirements and the
  response's stated assumptions. Do not invent additional production-hardening,
  portability, API-shape, validation, CLI-ordering, or error-recovery requirements.
- Reject only for a concrete defect that makes the supplied solution materially
  incorrect, non-compiling/non-running under its stated environment, or unable to
  satisfy an explicit requirement for an input or case the prompt actually asks
  it to handle.
- A documented limitation, trade-off, implementation choice, dialect assumption,
  or conventional interface difference is not a defect when the requested
  behavior still works.
- Do not reject because an implementation could be more robust, more portable,
  more efficient, more production-ready, or more extensively tested unless the
  prompt explicitly requires that property.
- Before rejecting, verify the claimed defect against the supplied code. Do not
  rely on a hypothetical failure mode that the code's synchronization, control
  flow, data constraints, or stated assumptions already prevent.
- If the response explicitly notes a caveat and provides a usable solution within
  the prompt's requirements, do not reject merely for that caveat.
- Do not infer missing imports, missing calls, absent branches, or syntax errors
  without checking the full supplied response for them.
- Do not reject because a response uses an allowed dependency, format, API, or
  language construct that the prompt explicitly permits.
- If your own verification shows the response satisfies the requirement, you MUST
  return agreed=true. You may not describe the response as correct or say no
  material defect exists while returning agreed=false.
""".rstrip()

    prompt = f"""
Review the judge ACCEPT decisions for the response-distillation SFT candidates
below.

Your job is a final independent check for clear material defects, not a stricter
perfection test.

Judge only against what the user actually requested. Do not silently add new
requirements. A disagreement must identify a defect that materially changes
correctness, usability, or compliance with an explicit instruction.

Reject an accepted item only when you can identify a concrete material problem,
for example:
- incorrect answer or reasoning;
- prompt/response mismatch;
- missed explicit instruction;
- unsupported factual claim or invented evidence/state;
- materially incomplete answer;
- malformed or unusable teacher response;
- obvious template/repeated-response contamination;
- response not sufficiently conditioned on the supplied prompt to be useful as
  a teacher example.

Do not reject for stylistic preference, benign wording differences, optional
extra detail, merely because another answer could be better, or because the
response omits a requirement that is not present in the prompt.{code_calibration}

Items:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

For each supplied id:
- agreed=true when the judge acceptance is reasonably justified:
  defect_category="none", evidence="", verification="", reason="".
- agreed=false only for a VERIFIED clear material defect.
  1. defect_category: choose the single best category.
  2. evidence: identify the exact relevant prompt requirement and exact response
     code/text that conflicts with it.
  3. verification: demonstrate the failure concretely. For code, trace the relevant
     control flow, API behavior, syntax, or input case. Do not merely assert that it
     fails.
  4. reason: one concise summary of the verified material defect.
- If you cannot complete steps 2 and 3 from the supplied material, return
  agreed=true. Uncertainty is not grounds for rejection.

The response structure is enforced by the supplied JSON Schema.
""".strip()

    parsed, _ = call_structured_object(
        backend,
        prompt=prompt,
        schema=reviewer_schema(expected_ids),
        schema_name="distillation_sft_reviewer_batch",
    )
    return parse_reviewer_batch(parsed, expected_ids)

# ---------------------------------------------------------------------------
# Code-owned SFT structure
# ---------------------------------------------------------------------------

def make_spec(*, seed: Seed, task: Task, row_number: int) -> dict[str, Any]:
    return {
        "id": f"distill_{seed.family}_{row_number:06d}",
        "instruction": task.text,
        "metadata": {
            **seed.metadata,
            "distillation_family": seed.family,
            "interaction_mode": "single_turn",
            "context_mode": "self_contained",
            "seed_id": seed.id,
            "seed_index": task.seed_index,
            "derivation_ordinal": task.derivation_ordinal,
            "task_ordinal": task.task_ordinal,
        },
        "constraints": [
            "The prompt must contain every local fact, code sample, schema, label set, or source passage required to answer it.",
            "The teacher response must answer without relying on hidden generation metadata or unavailable external state.",
        ],
    }


def make_row(spec: Mapping[str, Any], task: str, answer: str) -> dict[str, Any]:
    prompt = task.strip()
    response = answer.strip()
    if not prompt:
        raise ValueError("distillation prompt must be non-empty")
    if not response:
        raise ValueError("distillation response must be non-empty")
    return {
        "id": spec["id"],
        "prompt": prompt,
        "response": response,
        "metadata": dict(spec["metadata"]),
    }


def validate_distillation_row(
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    if row.get("id") != spec.get("id"):
        raise ValueError("row/spec id mismatch")
    prompt = row.get("prompt")
    response = row.get("response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a non-empty string")
    if normalize(prompt) == normalize(response):
        raise ValueError("prompt and response are identical after normalization")

    # Code-owned leakage checks are intentionally narrow and objective.
    lowered_prompt = prompt.casefold()
    lowered_response = response.casefold()
    forbidden_prompt_markers = (
        "chosen response",
        "rejected response",
        "judge decision",
        "reviewer decision",
        "synthetic dataset",
        "response-distillation sft candidate",
    )
    for marker in forbidden_prompt_markers:
        if marker in lowered_prompt:
            raise ValueError(f"generation-control leakage in prompt: {marker}")

    forbidden_response_markers = (
        "as the teacher model",
        "as a teacher model",
        "for the student model",
        "synthetic-data commentary",
    )
    for marker in forbidden_response_markers:
        if marker in lowered_response:
            raise ValueError(f"generation-control leakage in response: {marker}")


def partition_unique_distillation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen_prompt: dict[str, str] = {}
    seen_response: dict[str, str] = {}
    seen_pair: dict[tuple[str, str], str] = {}
    accepted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for row in rows:
        rid = str(row["id"])
        pnorm = normalize(str(row["prompt"]))
        rnorm = normalize(str(row["response"]))
        pair = (pnorm, rnorm)

        if pair in seen_pair:
            reasons["exact_prompt_response_duplicate"] += 1
            continue
        if pnorm in seen_prompt:
            reasons["exact_prompt_duplicate"] += 1
            continue
        if rnorm in seen_response:
            reasons["exact_response_duplicate"] += 1
            continue

        seen_pair[pair] = rid
        seen_prompt[pnorm] = rid
        seen_response[rnorm] = rid
        accepted.append(dict(row))

    return accepted, {
        "attempted_rows": len(rows),
        "accepted_rows": len(accepted),
        "duplicate_rows": len(rows) - len(accepted),
        "duplicate_reason_counts": dict(sorted(reasons.items())),
    }

def run_family(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reset_output_files(out, (
        "derivations.generated.jsonl", "tasks.generated.jsonl", "tasks.accepted.jsonl",
        "responses.generated.jsonl", "responses.validated.jsonl", "judge.decisions.jsonl",
        "reviewer.decisions.jsonl", "quality.rejected.jsonl", "stage.failures.jsonl",
        "distillation_sft.accepted.jsonl", "summary.json", "plan.json",
    ))

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
            "teacher_response_generator": args.answer_model,
            "judge": args.judge_model,
            "reviewer": args.reviewer_model,
        },
    }
    write_json(out / "plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)

    if args.dry_run:
        print("dry-run complete", flush=True)
        return 0

    derivation_backend = build_backend(
        model=args.derivation_model,
        max_tokens=args.derivation_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    task_backend = build_backend(
        model=args.task_model,
        max_tokens=args.task_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    answer_backend = build_backend(
        model=args.answer_model,
        max_tokens=args.answer_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    judge_backend = build_backend(
        model=args.judge_model,
        max_tokens=args.judge_max_tokens,
        concurrency=args.concurrency,
        routing_mode=args.routing_mode,
        provider=args.provider,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    reviewer_backend = build_backend(
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
        append_jsonl(out / "responses.generated.jsonl", {
            "id": spec["id"],
            "seed_id": task.seed_id,
            "derivation_ordinal": task.derivation_ordinal,
            "task": task.text,
            "response": answer,
        })

    # Stage 4: deterministic validation + teacher-response novelty.
    print(f"[deterministic-validation] start rows={len(generated_rows)}", flush=True)
    validation_started = time.monotonic()
    valid_specs: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    deterministic_rejections = 0
    response_rejection_counts: Counter[str] = Counter()
    response_novelty = NoveltyFilter(
        jaccard_threshold=args.jaccard_threshold,
        sequence_threshold=args.sequence_threshold,
    )

    for index, (spec, row) in enumerate(zip(generated_specs, generated_rows, strict=True), 1):
        rejection_reason: str | None = None
        evidence: dict[str, Any] = {}
        try:
            validate_distillation_row(spec, row)
        except ValueError as exc:
            rejection_reason = str(exc)
        else:
            keep, novelty_reason, novelty_evidence = response_novelty.check(row["response"])
            if not keep:
                rejection_reason = f"response_{novelty_reason}"
                evidence = novelty_evidence
            else:
                response_novelty.add(f"response:{row['id']}", row["response"])

        if rejection_reason is not None:
            deterministic_rejections += 1
            response_rejection_counts[rejection_reason] += 1
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "deterministic",
                "id": spec["id"],
                "spec": spec,
                "row": row,
                "reason": rejection_reason,
                "evidence": evidence,
            })
        else:
            valid_specs.append(spec)
            valid_rows.append(row)
            append_jsonl(out / "responses.validated.jsonl", {
                "id": spec["id"],
                "prompt": row["prompt"],
                "response": row["response"],
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
            family=args.family,
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
    unique_rows, final_dedup = partition_unique_distillation_rows(final_rows)
    for row in unique_rows:
        append_jsonl(out / "distillation_sft.accepted.jsonl", row)

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
        "responses_generated": len(generated_rows),
        "response_generation_failures": sum(
            failure["items"] for failure in answer_failures
        ),
        "responses_validated": len(valid_rows),
        "response_rejection_counts": dict(sorted(response_rejection_counts.items())),
        "response_dedup_candidate_comparisons": response_novelty.candidate_comparisons,
        "response_dedup_sequence_comparisons": response_novelty.sequence_comparisons,
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
            "generated_responses": str(out / "responses.generated.jsonl"),
            "validated_responses": str(out / "responses.validated.jsonl"),
            "judge_decisions": str(out / "judge.decisions.jsonl"),
            "reviewer_decisions": str(out / "reviewer.decisions.jsonl"),
            "rejections": str(out / "quality.rejected.jsonl"),
            "stage_failures": str(out / "stage.failures.jsonl"),
            "final_dataset": str(out / "distillation_sft.accepted.jsonl"),
        },
    }

    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0




def _read_internal_accepted(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"invalid accepted row at {path}:{line_number}")
        rows.append(dict(value))
    return rows


def _public_row(*, signal: str, row: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"]).strip()
    response = str(row["response"]).strip()
    return validate_public_row(
        {
            "id": str(row["id"]),
            "prompt": prompt,
            "reasoning": None,
            "response": response,
            "metadata": build_public_metadata(signal=signal, prompt=prompt),
        }
    )


def _selected_signals(values: Sequence[str]) -> list[str]:
    requested = [value.strip().lower() for value in values if value.strip()]
    if not requested or requested == ["all"]:
        return sorted(DISTILLATION_SIGNALS)
    unknown = sorted(set(requested) - set(DISTILLATION_SIGNALS))
    if unknown:
        raise ValueError(f"unsupported Distillation-SFT signal(s): {unknown}")
    return list(dict.fromkeys(requested))


def run_production(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir)
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    internal_dir = run_dir / "internal"
    for path in (dataset_dir, manifest_dir, internal_dir):
        path.mkdir(parents=True, exist_ok=True)

    signals = _selected_signals(args.signals)
    datasets: list[dict[str, Any]] = []
    total_candidates = 0
    total_accepted = 0
    total_rejected = 0

    for signal in signals:
        family_out = internal_dir / signal
        family_args = SimpleNamespace(
            family=signal,
            seeds=args.seeds,
            derivations_per_seed=args.derivations_per_seed,
            tasks_per_derivation=args.tasks_per_derivation,
            answer_batch_size=args.answer_batch_size,
            judge_batch_size=args.judge_batch_size,
            reviewer_batch_size=args.reviewer_batch_size,
            concurrency=args.concurrency,
            cardinality_fill_attempts=args.cardinality_fill_attempts,
            stage_batch_attempts=args.stage_batch_attempts,
            derivation_model=args.derivation_model,
            task_model=args.task_model,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            reviewer_model=args.reviewer_model,
            derivation_max_tokens=args.derivation_max_tokens,
            task_max_tokens=args.task_max_tokens,
            answer_max_tokens=args.answer_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            reviewer_max_tokens=args.reviewer_max_tokens,
            routing_mode=args.routing_mode,
            provider=args.provider,
            temperature=args.temperature,
            top_p=args.top_p,
            jaccard_threshold=args.jaccard_threshold,
            sequence_threshold=args.sequence_threshold,
            output_dir=str(family_out),
            dry_run=args.dry_run,
        )
        run_family(family_args)
        if args.dry_run:
            continue

        summary_path = family_out / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accepted_internal = _read_internal_accepted(family_out / "distillation_sft.accepted.jsonl")
        public_rows = [_public_row(signal=signal, row=row) for row in accepted_internal]

        dataset_path = write_signal_dataset(
            signal=signal,
            rows=public_rows,
            output_dir=dataset_dir,
        )
        signal_manifest_path = manifest_dir / f"{signal}.{args.generation_run}.manifest.json"
        signal_manifest = write_manifest(
            manifest_path=signal_manifest_path,
            signal=signal,
            dataset_path=dataset_path,
            row_count=len(public_rows),
            teacher_model=args.answer_model,
            teacher_provider="openrouter",
            generation_run=args.generation_run,
            metadata={
                "pipeline": "derivation_task_novelty_response_validation_judge_reviewer_final_dedup",
                "deterministic_output_validation": "checked",
                "semantic_adjudication": "checked",
                "judge_model": args.judge_model,
                "reviewer_model": args.reviewer_model,
                "summary_path": str(summary_path),
                "final_dedup": summary.get("final_dedup", {}),
            },
        )
        datasets.append(
            {
                "signal": signal,
                "dataset_path": str(dataset_path),
                "manifest_path": str(signal_manifest),
                "row_count": len(public_rows),
            }
        )
        candidates = int(summary.get("task_candidates_after_dedup", 0) or 0)
        total_candidates += candidates
        total_accepted += len(public_rows)
        total_rejected += max(candidates - len(public_rows), 0)

    if args.dry_run:
        print(f"[distillation-sft] dry-run signals={signals}", flush=True)
        return 0

    run_manifest_path = manifest_dir / f"{args.generation_run}.manifest.json"
    write_run_manifest(
        manifest_path=run_manifest_path,
        generation_run=args.generation_run,
        teacher_model=args.answer_model,
        teacher_provider="openrouter",
        datasets=datasets,
        metadata={
            "pipeline": "derivation_task_novelty_response_validation_judge_reviewer_final_dedup",
            "candidate_rows": total_candidates,
            "planned_prompt_rows": total_candidates,
            "generated_accepted_rows": total_accepted,
            "accepted_rows": total_accepted,
            "rejected_rows": total_rejected,
            "curation_rejected_rows": total_rejected,
            "deterministic_output_validation": "checked",
            "semantic_adjudication": "checked",
            "judge_model": args.judge_model,
            "reviewer_model": args.reviewer_model,
        },
    )
    print(
        f"[distillation-sft] complete signals={len(signals)} accepted={total_accepted} "
        f"run_manifest={run_manifest_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Production Distillation-SFT pipeline.")
    p.add_argument("--signals", nargs="+", default=["all"])
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
    p.add_argument("--answer-model", "--teacher-model", dest="answer_model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--judge-model", default="nvidia/nemotron-3.5-lightning")
    p.add_argument("--reviewer-model", default="google/gemma-4-31b-it")
    p.add_argument("--derivation-max-tokens", type=int, default=4096)
    p.add_argument("--task-max-tokens", type=int, default=4096)
    p.add_argument("--answer-max-tokens", "--teacher-max-tokens", dest="answer_max_tokens", type=int, default=4096)
    p.add_argument("--judge-max-tokens", type=int, default=4096)
    p.add_argument("--reviewer-max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--routing-mode", choices=["auto", "prefer", "strict"], default="auto")
    p.add_argument("--provider", default=None)
    p.add_argument("--jaccard-threshold", type=float, default=0.82)
    p.add_argument("--sequence-threshold", type=float, default=0.90)
    p.add_argument("--dry-run", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "seeds", "derivations_per_seed", "tasks_per_derivation",
        "answer_batch_size", "judge_batch_size", "reviewer_batch_size",
        "concurrency", "cardinality_fill_attempts", "stage_batch_attempts",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return run_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
