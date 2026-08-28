#!/usr/bin/env python3
"""Production Distillation-DPO pipeline migrated from the finalized one-off.

Stages:
1A derivation generation -> 1B concrete task generation -> deterministic novelty
filtering -> pair generation -> deterministic pair validation -> judge -> reviewer ->
final exact triple dedup. Shared runtime owns batching, retries/fault isolation,
novelty, IO, and OpenRouter backend construction.

The model-facing semantics remain the finalized Distillation-DPO plain-text one-off contract.
Only final accepted rows are adapted to the repository public message-list schema.
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

from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row

from slm_synth.runtime import (
    NoveltyFilter,
    append_jsonl,
    canonical_exact,
    build_backend,
    chunked,
    fill_exact_count,
    normalize,
    reset_output_files,
    run_model_stage_with_isolation,
    split_sequence_batch,
    split_slot_batch,
    write_json,
)

DISTILLATION_DPO_DIMENSIONS = (
    "helpfulness_and_completeness",
    "factual_accuracy",
    "instruction_adherence",
    "detail",
    "organization",
    "style_and_tone",
    "tool_call_correctness",
    "groundedness",
    "safe_refusal",
    "code_correctness",
)

DISTILLATION_DPO_SEEDS: dict[str, str] = {
    "helpfulness_and_completeness": (
        "Create preference tasks where both responses are plausible, but one is materially "
        "more helpful and complete for the user's actual request."
    ),
    "factual_accuracy": (
        "Create preference tasks where both responses are plausible, but one is materially "
        "more factually accurate, internally consistent, and properly supported."
    ),
    "instruction_adherence": (
        "Create preference tasks where both responses are plausible, but one follows the "
        "user's explicit constraints, requested format, scope, and negative requirements better."
    ),
    "detail": (
        "Create preference tasks where both responses are plausible, but one provides the "
        "appropriate necessary detail while the other omits material information or adds "
        "unhelpful excess."
    ),
    "organization": (
        "Create preference tasks where both responses are plausible, but one organizes the "
        "information more clearly and usefully for the requested outcome."
    ),
    "style_and_tone": (
        "Create preference tasks where both responses are plausible, but one better matches "
        "the requested audience, register, style, and tone."
    ),
    "tool_call_correctness": (
        "Create preference tasks involving tool-use decisions where one response chooses and "
        "uses the appropriate tool or arguments correctly and the other makes a plausible but "
        "material tool-use mistake."
    ),
    "groundedness": (
        "Create preference tasks grounded in supplied evidence, passages, data, or context where "
        "one response stays supported by that material and the other makes a plausible unsupported "
        "inference, omission, or contradiction."
    ),
    "safe_refusal": (
        "Create preference tasks where one response handles safety boundaries correctly while "
        "remaining maximally useful, and the other response is a plausible but materially worse "
        "over-refusal, under-refusal, or unsafe handling."
    ),
    "code_correctness": (
        "Create programming preference tasks where both responses are plausible, but one is "
        "materially more correct against the stated code contract, edge cases, or execution behavior."
    ),
}

DERIVATION_SCOPE_GUIDANCE: dict[str, str] = {
    "helpfulness_and_completeness": (
        "Derivations must center on whether a response materially fulfills the user's requested "
        "outcome. Good classes include omitted requested artifacts, missing necessary steps, "
        "failure to address part of a multi-part request, or insufficiently actionable guidance. "
        "Do not use cases where the only difference would be factual correctness, style, or mere verbosity."
    ),
    "factual_accuracy": (
        "Derivations must center on objectively checkable truth or correctness: factual claims, numerical "
        "results, logical conclusions, scientific or technical behavior, dates, units, definitions with "
        "important qualifiers, version-specific facts, calculations, or comparisons. Supplied evidence may "
        "be used sometimes, but factual_accuracy must NOT collapse into groundedness. Prefer many classes "
        "whose correctness can be judged from the task itself or stable domain knowledge without requiring "
        "a supplied passage, record, table, abstract, log, or source excerpt. EXCLUDE creative-writing, tone, "
        "formatting, and subjective-preference classes with no factual proposition to adjudicate."
    ),
    "instruction_adherence": (
        "Derivations must contain explicit user constraints that can actually be followed or violated, "
        "such as required format, count, wording restrictions, scope limits, or required components. "
        "Do not rely on unstated preferences or generic response quality."
    ),
    "detail": (
        "Derivations must require a level of detail that is necessary for the requested outcome, such as "
        "implementation steps, explanation of mechanisms, examples, or operational instructions. "
        "The distinction must be useful detail versus materially insufficient detail, not simply long versus short."
    ),
    "organization": (
        "Derivations must make organization itself consequential: sequencing, prioritization, grouping, "
        "comparison structure, roadmap structure, or navigability. If the eventual task would require source "
        "material such as interview findings, ticket contents, benchmarks, feature definitions, or incident data, "
        "the derivation must explicitly require that the concrete task supply that material rather than invent it."
    ),
    "style_and_tone": (
        "Derivations must specify a meaningful audience, tone, register, voice, or stylistic constraint where "
        "two otherwise plausible responses can differ materially. Do not use correctness or completeness as the "
        "primary distinction."
    ),
    "tool_call_correctness": (
        "Derivations MUST involve an actual tool-use decision or tool interaction. The eventual task must provide "
        "or clearly define one or more tools/APIs/functions and enough information to judge tool selection, "
        "arguments, sequencing, or interpretation of tool results. Valid classes include choosing the wrong tool, "
        "wrong arguments, missing a required tool call, calling tools in the wrong order, or misusing returned "
        "tool evidence. EXCLUDE ordinary factual lookup questions, web-search requests with no tool schema, "
        "translation, arithmetic, generic coding, or normal question answering."
    ),
    "groundedness": (
        "Derivations MUST require the response to stay faithful to evidence supplied inside the eventual user task, "
        "such as a passage, table, abstract, review, policy excerpt, log, structured data, or quoted source. "
        "The task must contain the grounding material. EXCLUDE open-ended creative writing, general-knowledge "
        "questions, or tasks where no supplied evidence constrains the answer."
    ),
    "safe_refusal": (
        "Derivations must hinge on safety calibration, not refusal alone. The set must deliberately cover three "
        "distinct classes: (1) harmful or risky requests where refusal or safe redirection is appropriate; "
        "(2) benign requests that are easy to over-refuse, where the preferred response should comply normally; "
        "and (3) mixed requests where the safe portion should be answered while the unsafe portion is refused or "
        "bounded. Avoid producing an overwhelmingly harmful-only set. Benign calibration examples should include "
        "public legal information, routine medical or educational information, account-security help, safety "
        "research, defensive security, historical analysis, and other legitimate requests that should not be "
        "refused merely because they mention a sensitive domain. Mixed examples should require selective "
        "compliance rather than blanket refusal."
    ),
    "code_correctness": (
        "Derivations must describe self-contained programming tasks with a concrete behavioral contract that can "
        "support materially different correctness outcomes. Require the eventual task to contain all necessary "
        "function signatures, input/output behavior, constraints, examples, or API assumptions. Avoid derivations "
        "that depend on missing context or continuation from another task."
    ),
}

DISTILLATION_DPO_DEFECT_GUIDANCE: dict[str, str] = {
    "helpfulness_and_completeness": (
        "Make REJECTED materially less helpful or complete for the user's actual request."
    ),
    "factual_accuracy": (
        "Make REJECTED materially worse on factual accuracy, calculation, or factual qualification."
    ),
    "instruction_adherence": (
        "Make REJECTED materially worse at following an explicit user instruction or constraint."
    ),
    "detail": (
        "Make REJECTED materially worse in the amount or usefulness of detail for the requested outcome."
    ),
    "organization": (
        "Make REJECTED materially worse in organization, sequencing, prioritization, or navigability."
    ),
    "style_and_tone": (
        "Make REJECTED materially worse for the requested audience, tone, register, voice, or style."
    ),
    "tool_call_correctness": (
        "For a task where tool use is relevant, make REJECTED materially worse in tool choice, arguments, tool sequencing, or use of tool evidence."
    ),
    "groundedness": (
        "Make REJECTED materially less grounded in the evidence, passage, data, or context supplied by the user."
    ),
    "safe_refusal": (
        "Make REJECTED materially worse in safety calibration, such as over-refusing, under-refusing, or handling the risk less usefully."
    ),
    "code_correctness": (
        "Make REJECTED materially worse on correctness against the stated code contract, edge cases, API behavior, or execution behavior."
    ),
}


DISTILLATION_DPO_SEED_LENSES = (
    "Core case: isolate one clear material distinction on the selected preference dimension.",
    "Constraint interaction: use multiple explicit requirements while keeping the selected preference dimension as the primary reason one response wins.",
    "Boundary and exception case: exercise edge conditions, ambiguity, exceptions, or qualification that expose subtle preference failures.",
    "Evidence-sensitive case: use self-contained facts, code, structured data, or tool evidence when relevant so correctness and preference are independently checkable.",
    "Operational case: use a realistic multi-step workflow, decision, or troubleshooting scenario with a plausible but materially weaker rejected response.",
)


@dataclass(frozen=True)
class Seed:
    index: int
    id: str
    dimension: str
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
            "pairs.generated.jsonl",
            "pairs.validated.jsonl",
            "judge.decisions.jsonl",
            "reviewer.decisions.jsonl",
            "quality.rejected.jsonl",
            "stage.failures.jsonl",
            "distillation_dpo.accepted.jsonl",
            "summary.json",
            "plan.json",
        ),
    )


# ---------------------------------------------------------------------------
# Seed loading -- DPO-specific semantics only
# ---------------------------------------------------------------------------

def load_seeds(dimension: str, count: int) -> list[Seed]:
    if count < 1:
        raise ValueError("seed count must be positive")

    base = DISTILLATION_DPO_SEEDS[dimension]
    seeds: list[Seed] = []
    for index in range(1, count + 1):
        lens = DISTILLATION_DPO_SEED_LENSES[(index - 1) % len(DISTILLATION_DPO_SEED_LENSES)]
        cycle = (index - 1) // len(DISTILLATION_DPO_SEED_LENSES) + 1
        instruction = (
            f"{base}\n\nSeed lens: {lens}\n"
            f"This is seed variant {index} (lens cycle {cycle}). Preserve the "
            "Distillation-DPO dimension boundary while exploring a distinct semantic starting region."
        )
        seeds.append(
            Seed(
                index=index,
                id=f"distillation_dpo_{dimension}_seed_{index:03d}",
                dimension=dimension,
                instruction=instruction,
                metadata={
                    "dataset_kind": "distillation_dpo",
                    "preference_dimension": dimension,
                },
            )
        )
    return seeds


# ---------------------------------------------------------------------------
# OpenRouter structured-output contract -- copied from SFT
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
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": [field],
        "additionalProperties": False,
    }


def pair_list_schema(count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("pair list count must be positive")
    return {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "chosen": {"type": "string", "minLength": 1},
                        "rejected": {"type": "string", "minLength": 1},
                    },
                    "required": ["chosen", "rejected"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["pairs"],
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
                "description": "Whether the DPO pair can be meaningfully assessed.",
            },
            "chosen_complete": {
                "type": "boolean",
                "description": (
                    "Whether CHOSEN actually supplies every material deliverable and "
                    "requirement requested by the user, rather than merely promising, "
                    "describing, outlining, or partially answering the task."
                ),
            },
            "chosen_correct": {
                "type": "boolean",
                "description": (
                    "Whether CHOSEN's substantive content satisfies the task's explicit "
                    "contract: its calculations, code behavior, factual claims, internal "
                    "consistency, and edge-case handling must be materially correct."
                ),
            },
            "preference_valid": {
                "type": "boolean",
                "description": (
                    "Whether CHOSEN is genuinely and materially preferable to REJECTED, "
                    "with REJECTED still being a plausible response rather than nonsense "
                    "or an unrelated answer."
                ),
            },
            "dimension_aligned": {
                "type": "boolean",
                "description": (
                    "Whether the material reason CHOSEN is preferable is primarily the "
                    "stated preference dimension rather than another quality dimension."
                ),
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Concise evidence-based reason identifying any failed gate or, when "
                    "all pass, why the preference is valid."
                ),
            },
        },
        "required": [
            "assessable",
            "chosen_complete",
            "chosen_correct",
            "preference_valid",
            "dimension_aligned",
            "reason",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": {row_id: decision for row_id in expected_ids},
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
                "properties": {row_id: decision for row_id in expected_ids},
                "required": list(expected_ids),
                "additionalProperties": False,
            },
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }


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


def extract_pair_list(parsed: Mapping[str, Any]) -> list[dict[str, str]]:
    values = parsed.get("pairs")
    if not isinstance(values, list):
        raise ValueError("pairs must be a list")

    result: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"pairs[{index}] must be an object")
        chosen = value.get("chosen")
        rejected = value.get("rejected")
        if not isinstance(chosen, str) or not chosen.strip():
            raise ValueError(f"pairs[{index}].chosen must be a non-empty string")
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError(f"pairs[{index}].rejected must be a non-empty string")
        result.append({"chosen": chosen.strip(), "rejected": rejected.strip()})
    return result


# ---------------------------------------------------------------------------
# Model 1 — same derivation/task pattern as SFT; DPO wording only
# ---------------------------------------------------------------------------

def generate_derivations(
    *,
    seed: Seed,
    count: int,
    backend: Any,
    max_fill_attempts: int,
) -> list[str]:
    def request_derivations(request_count: int, prior: Sequence[str]) -> list[str]:
        prior_block = ""
        if prior:
            prior_block = (
                "\nAlready accepted derivations. Do not repeat or substantially "
                "overlap these:\n"
                + "\n".join(f"- {item}" for item in prior)
                + "\n"
            )

        dimension_scope = DERIVATION_SCOPE_GUIDANCE[seed.dimension]

        prompt = f"""
You are generating semantic task classes for the Distillation-DPO preference dimension:
{seed.dimension}.

Seed:
{seed.instruction}

Dimension boundary:
{dimension_scope}

Generate exactly {request_count} DISTINCT {seed.dimension} preference problem classes
that are meaningfully different from one another.

The seed defines the preference dimension. Your job is to stay inside that dimension,
not to maximize topical variety at the cost of relevance.
{prior_block}
Each derivation must:
- be intrinsically valid for the {seed.dimension} preference dimension;
- represent a different capability, problem type, or failure mode;
- be broad enough to support multiple concrete user tasks later;
- differ semantically, not just by variable names, entities, data types,
  numbers, wording, or swapping one factoid for another;
- describe a setting where two plausible assistant responses could differ
  materially on {seed.dimension};
- contain enough conceptual information that the later task generator can create
  a self-contained task without inventing missing source material or hidden facts;
- explicitly require supplied evidence/context in the later task whenever the
  derivation depends on evidence, records, source text, benchmark data, tickets,
  interviews, tool definitions, schemas, or similar inputs;
- avoid trivial "correct answer versus obvious nonsense" distinctions;
- avoid producing the final user task;
- avoid chosen/rejected responses.

Additional set-level balance requirements:
- For factual_accuracy, ensure a substantial majority of the returned derivations do NOT require supplied
  evidence or source material. Include intrinsically checkable classes such as calculations, unit conversion,
  dates/time, scientific or technical facts, logical conclusions, definitions with qualifiers, version-specific
  behavior, and numerical comparisons. Keep supplied-source verification as a minority.
- For safe_refusal, ensure the returned set visibly covers all three calibration classes: harmful requests
  requiring refusal, benign requests that should NOT be refused, and mixed requests requiring partial
  compliance plus refusal of the unsafe portion. Do not return a harmful-only set.

Reject a candidate derivation before returning it if:
- the preference could only be judged on a different DPO dimension;
- the eventual task would be under-specified without inventing facts or source data;
- it violates the dimension boundary above;
- its core preference distinction substantially overlaps a previously selected derivation.

Return exactly {request_count} derivations under the supplied structured-output
schema. Each derivation must be a concise description of the semantic preference
problem class, not a concrete task.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=exact_string_list_schema(
                field="derivations",
                count=request_count,
                description=(
                    f"Exactly {request_count} distinct semantic problem classes "
                    f"for the {seed.dimension} Distillation-DPO preference dimension."
                ),
            ),
            schema_name="distillation_dpo_task_derivations",
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

Distillation-DPO preference dimension: {seed.dimension}

Curated seed:
{seed.instruction}

Derivation:
{derivation}
{prior_block}
Task requirements:
- Every task must be a NORMAL USER REQUEST that could appear in a real assistant
  conversation.
- The user task must stand on its own and must NOT reveal that preference data
  will be generated from it.
- Do NOT ask the assistant to compare Response A vs Response B, choose between
  two candidate answers, generate one correct and one incorrect answer, create
  contrasting responses, or intentionally manufacture an error.
- Do NOT embed the desired answer, correction, verdict, or preference direction
  in the task unless that information is naturally part of source material the
  user is asking the assistant to analyze.
- Do NOT write meta-prompts such as "write a response that corrects this" when a
  natural user request can ask the underlying question directly.
- Make {seed.dimension} relevant through the substance of the task, not by
  naming the dimension.
- FAITHFULNESS: every task must directly instantiate the supplied derivation.
  The main quality distinction exercised by the task must be {seed.dimension}.
  If the task would primarily exercise a different preference dimension,
  discard it internally and generate a replacement.
- SELF-CONTAINMENT: every task must contain all information required for a
  capable assistant to answer it without inventing missing source material.
  If the task depends on a document, dataset, spreadsheet, tickets, interview
  findings, benchmark results, policy, contract, logs, source passage, schema,
  tool definition, API contract, jurisdiction, version, date, or other
  material context needed for a correct answer, embed that information
  directly in the task.
- Never refer to unavailable material with phrases such as "the provided
  document", "the attached report", "my spreadsheet", "the dataset", or
  similar wording unless the relevant contents are actually included in the
  task string.
- Reject and replace any task whose required source, records, evidence,
  jurisdiction, tool definition, version, or other necessary input is absent.
- Reject and replace any task with internally contradictory, mutually
  exclusive, or impossible constraints.
- SINGLE-TURN COMPLETION: the task must be fully answerable in one assistant
  response. Do not generate tasks that require asking the user questions and
  waiting for later replies, collecting information over multiple turns, or
  taking a future action before the answer can be completed.
- For tool_call_correctness ONLY: every task must define at least one concrete
  callable tool, function, or endpoint contract. Naming a database, API,
  service, application, or external system is not enough by itself. Include
  enough callable schema, arguments, result fields, or endpoint behavior to
  judge tool selection, arguments, sequencing, or interpretation of returned
  results. Ordinary factual lookup, SQL-writing, generic coding, CLI advice,
  calculations, data analysis, or "please look this up" requests without an
  explicit callable contract are INVALID and must be replaced.
- For groundedness ONLY: the evidence the answer must be grounded in must be
  supplied inside the task. External general knowledge or an unstated source
  must not be necessary to answer correctly.
- Avoid elementary trivia where the only possible negative is an obviously
  false fact.
- The tasks must be materially different from one another.
- Do not merely rename entities, variables, fields, dates, or numbers.
- Vary the real problem, objective, constraints, evidence, edge cases, and
  situation.
- Do not answer the tasks.
- Do not mention DPO, chosen, rejected, judge, reviewer, preference labels,
  "better response", or "worse response".
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
                    f"Exactly {request_count} complete standalone user tasks for "
                    f"the {seed.dimension} DPO preference dimension."
                ),
            ),
            schema_name="distillation_dpo_concrete_tasks",
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
# Model 2 — DPO pair generator (SFT answer stage analogue)
# ---------------------------------------------------------------------------

def generate_pairs(
    *,
    dimension: str,
    tasks: Sequence[str],
    backend: Any,
    max_fill_attempts: int,
) -> list[dict[str, str]]:
    def request_pairs(
        selected_tasks: Sequence[str],
        prior_pairs: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        body = "\n\n".join(
            f"--- TASK {index} ---\n{task}"
            for index, task in enumerate(selected_tasks, start=1)
        )
        request_count = len(selected_tasks)
        defect_guidance = DISTILLATION_DPO_DEFECT_GUIDANCE[dimension]

        prompt = f"""
EXACT OUTPUT COUNT: {request_count}

Preference dimension: {dimension}
Dimension guidance:
{defect_guidance}

Generate one CHOSEN and one REJECTED assistant response for every user task below.

{body}

Pair-generation role:
- CHOSEN should be a plausible strong response to the user task.
- REJECTED should be a plausible weaker response to the same user task.
- The intended preference should be about {dimension}, using the dimension guidance above.

Requirements:
- Return exactly {request_count} pairs, one per task, in the same order.
- Both responses must be standalone assistant responses to the same supplied user task.
- Keep both responses realistic and user-facing.
- REJECTED must be meaningfully worse on the target preference dimension, not merely
  different in wording or formatting.
- REJECTED must still read like a sincere answer that plausibly believes it satisfies
  the user. It must not describe, confess, announce, or explain its own shortcomings.
- Output only the candidate response itself. Neither CHOSEN nor REJECTED may contain
  commentary about whether the response follows the task, satisfies requirements, is
  correct, is incomplete, omits something, uses a shortcut, or contains a mistake.
- Do not append self-evaluation statements such as "I did not...", "This does not...",
  "I omitted...", "I included X even though...", "Sorry for not...",
  "This is incomplete...", "This implementation fails...", or "That should be enough...".
- The candidate must behave as though it believes its answer is appropriate.
- For code tasks, present each candidate implementation as a finished solution.
  Do not use placeholder or disclaimer language such as "quick and dirty",
  "placeholder", "not fully working", "might have issues", "starting point",
  "for simplicity", or "works for most cases".
- Do not use ellipses, stubs, or placeholder comments in place of implementation
  that the user explicitly requested. A weaker code candidate should still look
  like a sincere complete attempt; its weakness must come from the implementation
  itself, not from commentary announcing incompleteness or defects.
- Do not make REJECTED random nonsense, unrelated, empty, deliberately absurd, or a
  response to a different task.
- Do not include evaluation commentary, self-checks, preference labels, metadata,
  synthetic-data commentary, or explanations of why one response is better.
- Do not rewrite the task.
- The "pairs" array must contain exactly {request_count} objects.

Return exactly:
{{"pairs":[
  {{"chosen":"chosen response 1","rejected":"rejected response 1"}},
  ...
]}}

FINAL COUNT RULE: stop after pair {request_count}.
""".strip()

        parsed, _ = call_structured_object(
            backend,
            prompt=prompt,
            schema=pair_list_schema(request_count),
            schema_name="distillation_dpo_response_pairs",
        )
        return extract_pair_list(parsed)

    initial = request_pairs(tasks, ())

    if len(initial) >= len(tasks):
        if len(initial) > len(tasks):
            print(
                f"[cardinality:pairs] requested={len(tasks)} observed={len(initial)} "
                f"action=trim excess={len(initial)-len(tasks)}",
                flush=True,
            )
        return initial[:len(tasks)]

    pairs = list(initial)
    attempts = 0
    while len(pairs) < len(tasks):
        missing = len(tasks) - len(pairs)
        attempts += 1
        if attempts > max_fill_attempts:
            raise ValueError(
                f"pairs remained underfilled after {max_fill_attempts} fill attempt(s): "
                f"requested={len(tasks)} observed={len(pairs)}"
            )

        remaining_tasks = tasks[len(pairs):]
        print(
            f"[cardinality:pairs] requested={len(tasks)} observed={len(pairs)} "
            f"missing={missing} action=request_missing attempt={attempts}/{max_fill_attempts}",
            flush=True,
        )

        extra = request_pairs(remaining_tasks, pairs)
        if not extra:
            raise ValueError(
                f"pairs fill attempt returned no usable values: "
                f"requested={len(tasks)} observed={len(pairs)}"
            )
        pairs.extend(extra[:missing])

    return pairs


# ---------------------------------------------------------------------------
# Judge / reviewer parsing -- same keyed-ID contract as SFT
# ---------------------------------------------------------------------------

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
        chosen_complete = item.get("chosen_complete")
        chosen_correct = item.get("chosen_correct")
        preference_valid = item.get("preference_valid")
        dimension_aligned = item.get("dimension_aligned")
        reason = item.get("reason")

        if not isinstance(assessable, bool):
            raise ValueError(f"judge assessable must be boolean for {row_id}")
        if not isinstance(chosen_complete, bool):
            raise ValueError(f"judge chosen_complete must be boolean for {row_id}")
        if not isinstance(chosen_correct, bool):
            raise ValueError(f"judge chosen_correct must be boolean for {row_id}")
        if not isinstance(preference_valid, bool):
            raise ValueError(f"judge preference_valid must be boolean for {row_id}")
        if not isinstance(dimension_aligned, bool):
            raise ValueError(f"judge dimension_aligned must be boolean for {row_id}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"judge reason missing for {row_id}")

        accepted = bool(
            assessable
            and chosen_complete
            and chosen_correct
            and preference_valid
            and dimension_aligned
        )

        result[row_id] = {
            "assessable": assessable,
            "chosen_complete": chosen_complete,
            "chosen_correct": chosen_correct,
            "preference_valid": preference_valid,
            "dimension_aligned": dimension_aligned,
            "judge_accepted": accepted,
            "judge_reason": reason.strip(),
        }

    return result


def judge_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    backend: Any,
) -> dict[str, dict[str, Any]]:
    expected_ids = [row["id"] for row in rows]
    payload = [
        {
            "id": row["id"],
            "preference_dimension": row["metadata"]["preference_dimension"],
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "deterministic_validation": "passed",
        }
        for row in rows
    ]

    prompt = f"""
Judge each Distillation-DPO candidate pair below.

Evaluate the CHOSEN and REJECTED responses against the user's actual task and
the stated preference dimension.

Evaluate in this order:
1. CHOSEN validity: CHOSEN must be acceptable on its own, not merely better than
   REJECTED. It must substantially fulfill the user's task, satisfy the user's explicit
   requirements, and avoid material factual, technical, safety, or instruction-following
   defects. Reject the pair if CHOSEN:
   - refuses, omits, or substitutes for a required deliverable when the task is otherwise
     answerable from the supplied information;
   - fabricates necessary facts, data, measurements, prices, dates, sources, capabilities,
     tool results, or other task-specific information not supplied by the prompt;
   - is materially incomplete, internally inconsistent, or technically/factually wrong;
   - is only an introduction, stub, placeholder, outline, or description of the requested
     answer rather than the requested answer itself; or
   - would itself be unsuitable for inclusion as the preferred side of a high-quality DPO pair.
2. REJECTED plausibility: REJECTED must still be a plausible assistant response rather
   than nonsense, meta-commentary, or an answer to another task.
3. Material preference: CHOSEN must be genuinely preferable to REJECTED by a material
   margin, not merely longer, differently formatted, differently worded, or based on an
   equally valid alternative approach. Reject if both responses are materially defective.
4. Dimension attribution: the material reason CHOSEN wins must be specifically tied to
   the stated preference dimension. Reject if the preference is mainly caused by a
   different quality dimension.

Accept only when all four checks pass.
Do not accept a materially defective CHOSEN just because REJECTED is worse.
Do not reward extra detail, precision, citations, technical language, or confidence
unless those additions are themselves correct and supported by the task.
Do not invent stricter requirements.
Do not reject merely because another chosen answer might be even better.
Deterministic structural checks already passed.

Candidates:
{json.dumps(payload, ensure_ascii=False, indent=2)}

For each candidate id, decide each gate independently:
- assessable;
- chosen_complete;
- chosen_correct;
- preference_valid;
- dimension_aligned;
- reason: one concise evidence-based reason tied to this exact row.

For chosen_complete, check whether every material requested deliverable is actually
present in CHOSEN. A promise, introduction, outline, stub, or partial answer is not complete.

For chosen_correct, verify CHOSEN against the task's explicit contract rather than
trusting its self-description. Check calculations using the values it provides, compare
code behavior against stated requirements and edge cases, and look for contradictions
between what CHOSEN claims and what its implementation or numbers actually do.

Do not collapse the gates into a single overall preference judgment.
A row is accepted downstream only when all five boolean gates are true;
the final acceptance is computed deterministically by code, not by you.

The response structure is enforced by the supplied JSON Schema.
""".strip()

    parsed, _ = call_structured_object(
        backend,
        prompt=prompt,
        schema=judge_schema(expected_ids),
        schema_name="distillation_dpo_judge_batch",
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
            "preference_dimension": row["metadata"]["preference_dimension"],
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "judge_reason": judge_decisions[row["id"]]["judge_reason"],
        }
        for row in rows
    ]

    prompt = f"""
Review the judge ACCEPT decisions below.

Your job is only to decide whether each Distillation-DPO acceptance is reasonably justified.

AGREE unless there is a clear material defect the judge missed, such as:
- the rejected answer is actually preferable;
- the pair is not meaningfully different on the stated preference dimension;
- the rejected answer is trivial nonsense rather than a plausible negative;
- one response belongs to a different task;
- a concrete correctness, instruction, grounding, safety, or code defect makes
  the accepted preference unsuitable.

Do not perform a stricter second perfection test.
Do not reject for style preferences, harmless ambiguity, a minor imperfection that does
not undermine the preference, or merely because another pair could be better.
If you identify an imperfection but conclude that it is minor or does not undermine the
material preference, you MUST set agreed=true.

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
        schema_name="distillation_dpo_reviewer_batch",
    )
    return parse_reviewer_batch(parsed, expected_ids)


# ---------------------------------------------------------------------------
# DPO structure / deterministic validation
# ---------------------------------------------------------------------------

def make_row(*, seed: Seed, task: Task, row_number: int, chosen: str, rejected: str) -> dict[str, Any]:
    return {
        "id": f"distillation_dpo_{seed.dimension}_{row_number:06d}",
        "prompt": task.text,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            **seed.metadata,
            "seed_id": seed.id,
            "seed_index": seed.index,
            "derivation_ordinal": task.derivation_ordinal,
            "task_ordinal": task.task_ordinal,
        },
    }


def validate_pair_structure(row: Mapping[str, Any]) -> str | None:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")

    if not isinstance(prompt, str) or not prompt.strip():
        return "empty_prompt"
    if not isinstance(chosen, str) or not chosen.strip():
        return "empty_chosen"
    if not isinstance(rejected, str) or not rejected.strip():
        return "empty_rejected"
    if canonical_exact(chosen) == canonical_exact(rejected):
        return "chosen_rejected_identical"
    return None


def pair_exact_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        normalize(str(row["prompt"])),
        normalize(str(row["chosen"])),
        normalize(str(row["rejected"])),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_dimension(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out)

    seeds = load_seeds(args.dimension, args.seeds)

    plan = {
        "dimension": args.dimension,
        "seed_count": len(seeds),
        "derivations_per_seed": args.derivations_per_seed,
        "tasks_per_derivation": args.tasks_per_derivation,
        "planned_derivations": len(seeds) * args.derivations_per_seed,
        "planned_task_candidates": len(seeds) * args.derivations_per_seed * args.tasks_per_derivation,
        "pair_batch_size": args.pair_batch_size,
        "judge_batch_size": args.judge_batch_size,
        "reviewer_batch_size": args.reviewer_batch_size,
        "concurrency": args.concurrency,
        "cardinality_fill_attempts": args.cardinality_fill_attempts,
        "stage_batch_attempts": args.stage_batch_attempts,
        "models": {
            "derivation_generator": args.derivation_model,
            "task_generator": args.task_model,
            "pair_generator": args.pair_model,
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
    pair_backend = build_backend(
        model=args.pair_model,
        max_tokens=args.pair_max_tokens,
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

    # Stage 3: ALL DPO pairs concurrently.
    pair_batches = chunked(accepted_tasks, args.pair_batch_size)

    def pair_worker(batch: list[Task]) -> list[tuple[Task, dict[str, str]]]:
        pairs = generate_pairs(
            dimension=args.dimension,
            tasks=[item.text for item in batch],
            backend=pair_backend,
            max_fill_attempts=args.cardinality_fill_attempts,
        )
        return list(zip(batch, pairs, strict=True))

    pair_results, pair_failures = run_model_stage_with_isolation(
        stage="pairs",
        batches=pair_batches,
        item_count=len,
        worker=pair_worker,
        split_batch=split_sequence_batch,
        concurrency=args.concurrency,
        batch_size_display=args.pair_batch_size,
        max_attempts=args.stage_batch_attempts,
    ) if pair_batches else ([], [])

    for failure in pair_failures:
        for task in failure["batch"]:
            append_jsonl(out / "stage.failures.jsonl", {
                "stage": "pairs",
                "seed_id": task.seed_id,
                "seed_index": task.seed_index,
                "derivation_ordinal": task.derivation_ordinal,
                "task_ordinal": task.task_ordinal,
                "task": task.text,
                "error": failure["error"],
            })

    paired: list[tuple[Task, dict[str, str]]] = []
    for result in pair_results:
        paired.extend(result)
    paired.sort(key=lambda pair: (
        pair[0].seed_index,
        pair[0].derivation_ordinal,
        pair[0].task_ordinal,
    ))

    generated_rows: list[dict[str, Any]] = []

    for row_number, (task, pair) in enumerate(paired, 1):
        seed = seed_by_id[task.seed_id]
        row = make_row(
            seed=seed,
            task=task,
            row_number=row_number,
            chosen=pair["chosen"],
            rejected=pair["rejected"],
        )
        generated_rows.append(row)
        append_jsonl(out / "pairs.generated.jsonl", row)

    # Stage 4: deterministic validation.
    print(f"[deterministic-validation] start rows={len(generated_rows)}", flush=True)
    validation_started = time.monotonic()
    valid_rows: list[dict[str, Any]] = []
    deterministic_rejections = 0

    for index, row in enumerate(generated_rows, 1):
        reason = validate_pair_structure(row)
        if reason:
            deterministic_rejections += 1
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "deterministic",
                "id": row["id"],
                "row": row,
                "reason": reason,
            })
        else:
            valid_rows.append(row)
            append_jsonl(out / "pairs.validated.jsonl", row)

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

    # Stage 5: ALL judge batches concurrently.
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
                "row": rows_by_id[row_id],
                "judge": judge_decisions[row_id],
                "review": review,
            })

    print(
        f"[reviewer] result accepted={len(final_rows)} "
        f"rejected={len(judge_accepted_ids)-len(final_rows)}",
        flush=True,
    )

    # Stage 7: final exact prompt/chosen/rejected dedup.
    seen: dict[tuple[str, str, str], str] = {}
    unique_rows: list[dict[str, Any]] = []
    duplicate_rows = 0

    for row in final_rows:
        key = pair_exact_key(row)
        duplicate_of = seen.get(key)
        if duplicate_of is not None:
            duplicate_rows += 1
            append_jsonl(out / "quality.rejected.jsonl", {
                "stage": "final_dedup",
                "id": row["id"],
                "row": row,
                "reason": "exact_preference_triple_duplicate",
                "duplicate_of": duplicate_of,
            })
            continue
        seen[key] = row["id"]
        unique_rows.append(row)
        append_jsonl(out / "distillation_dpo.accepted.jsonl", row)

    final_dedup = {
        "attempted_rows": len(final_rows),
        "accepted_rows": len(unique_rows),
        "duplicate_rows": duplicate_rows,
        "duplicate_reason_counts": (
            {"exact_preference_triple_duplicate": duplicate_rows}
            if duplicate_rows else {}
        ),
    }

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
        "pairs_generated": len(generated_rows),
        "pair_generation_failures": sum(
            failure["items"] for failure in pair_failures
        ),
        "pairs_validated": len(valid_rows),
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
            "generated_pairs": str(out / "pairs.generated.jsonl"),
            "validated_pairs": str(out / "pairs.validated.jsonl"),
            "judge_decisions": str(out / "judge.decisions.jsonl"),
            "reviewer_decisions": str(out / "reviewer.decisions.jsonl"),
            "rejections": str(out / "quality.rejected.jsonl"),
            "stage_failures": str(out / "stage.failures.jsonl"),
            "final_dataset": str(out / "distillation_dpo.accepted.jsonl"),
        },
    }

    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0




# ---------------------------------------------------------------------------
# Production/publication adapter
# ---------------------------------------------------------------------------

PUBLIC_FAMILY = "teacher_response_preference"

CATEGORY_BY_DIMENSION = {
    "helpfulness_and_completeness": "general_instruction_following",
    "factual_accuracy": "concise_factual_qa",
    "instruction_adherence": "general_instruction_following",
    "detail": "controlled_verbosity",
    "organization": "general_instruction_following",
    "style_and_tone": "general_instruction_following",
    "tool_call_correctness": "general_instruction_following",
    "groundedness": "concise_factual_qa",
    "safe_refusal": "refusal_calibration",
    "code_correctness": "code_generation",
}

FAILURE_MODE_BY_DIMENSION = {
    "helpfulness_and_completeness": "incomplete_response",
    "factual_accuracy": "wrong_factual_answer",
    "instruction_adherence": "instruction_violation",
    "detail": "excessive_detail",
    "organization": "poor_organization",
    "style_and_tone": "tone_mismatch",
    "tool_call_correctness": "incorrect_tool_call",
    "groundedness": "ungrounded_response",
    "safe_refusal": "over_refusal",
    "code_correctness": "code_logic_error",
}


def _public_metadata(dimension: str) -> dict[str, Any]:
    return {
        "category": CATEGORY_BY_DIMENSION[dimension],
        "difficulty": 3,
        "template_family": f"distillation_dpo_{dimension}",
        "eval_family": None,
        "failure_mode": FAILURE_MODE_BY_DIMENSION[dimension],
    }


def _to_public_row(row: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    return validate_distillation_dpo_row(
        {
            "id": str(row["id"]),
            "prompt": [{"role": "user", "content": str(row["prompt"])}],
            "chosen": [{"role": "assistant", "content": str(row["chosen"])}],
            "rejected": [{"role": "assistant", "content": str(row["rejected"])}],
            "metadata": _public_metadata(dimension),
        }
    )


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


def _resolve_dimensions(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(values)
    if not requested or requested == ("all",):
        return tuple(DISTILLATION_DPO_DIMENSIONS)
    unknown = sorted(set(requested) - set(DISTILLATION_DPO_DIMENSIONS))
    if unknown:
        raise ValueError(f"unknown Distillation-DPO preference dimensions: {unknown}")
    return tuple(dict.fromkeys(requested))


def _quality_evidence(work_dir: Path, accepted_ids: set[str]) -> dict[str, dict[str, Any]]:
    judges = {row["id"]: row for row in _read_jsonl(work_dir / "judge.decisions.jsonl") if isinstance(row.get("id"), str)}
    reviewers = {row["id"]: row for row in _read_jsonl(work_dir / "reviewer.decisions.jsonl") if isinstance(row.get("id"), str)}
    evidence: dict[str, dict[str, Any]] = {}
    for row_id in sorted(accepted_ids):
        judge = judges.get(row_id)
        review = reviewers.get(row_id)
        if judge is None or review is None:
            raise RuntimeError(f"missing judge/reviewer evidence for accepted Distillation-DPO row {row_id}")
        evidence[row_id] = {
            "assessable": judge.get("assessable") is True,
            "chosen_complete": judge.get("chosen_complete") is True,
            "chosen_correct": judge.get("chosen_correct") is True,
            "preference_valid": judge.get("preference_valid") is True,
            "dimension_aligned": judge.get("dimension_aligned") is True,
            "judge_accepted": judge.get("judge_accepted") is True,
            "judge_reason": str(judge.get("judge_reason", "")),
            "reviewed": review.get("reviewed") is True,
            "reviewer_agreed": review.get("reviewer_agreed") is True,
            "reviewer_reason": str(review.get("reviewer_reason", "")),
            "accepted": review.get("accepted") is True,
        }
        gates = evidence[row_id]
        if not all(gates[key] for key in (
            "assessable", "chosen_complete", "chosen_correct", "preference_valid",
            "dimension_aligned", "judge_accepted", "reviewed", "reviewer_agreed", "accepted"
        )):
            raise RuntimeError(f"accepted Distillation-DPO row has non-passing quality evidence: {row_id}")
    return evidence


def run_production(args: argparse.Namespace) -> int:
    dimensions = _resolve_dimensions(args.dimensions)
    run_dir = Path(args.output_dir)
    dataset_dir = run_dir / "datasets"
    manifest_dir = run_dir / "manifests"
    work_root = run_dir / "work"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    dimension_summaries: dict[str, dict[str, Any]] = {}
    quality_by_id: dict[str, dict[str, Any]] = {}

    for dimension in dimensions:
        work_dir = work_root / dimension
        dimension_args = argparse.Namespace(**vars(args))
        dimension_args.dimension = dimension
        dimension_args.output_dir = str(work_dir)
        print(f"[distillation-dpo-pipeline] dimension={dimension} start", flush=True)
        rc = run_dimension(dimension_args)
        if rc != 0:
            return rc
        if args.dry_run:
            continue

        summary = json.loads((work_dir / "summary.json").read_text(encoding="utf-8"))
        internal_rows = _read_jsonl(work_dir / "distillation_dpo.accepted.jsonl")
        public_rows = [_to_public_row(row, dimension) for row in internal_rows]
        if not public_rows:
            raise RuntimeError(f"Distillation-DPO dimension produced no accepted rows: {dimension}")
        ids = {row["id"] for row in public_rows}
        quality_by_id.update(_quality_evidence(work_dir, ids))
        all_rows.extend(public_rows)
        dimension_summaries[dimension] = dict(summary)
        print(f"[distillation-dpo-pipeline] dimension={dimension} complete accepted={len(public_rows)}", flush=True)

    if args.dry_run:
        print("[distillation-dpo-pipeline] dry-run complete", flush=True)
        return 0

    # Final dataset-level exact dedup across all preference dimensions.
    seen: dict[tuple[str, str, str], str] = {}
    unique_rows: list[dict[str, Any]] = []
    cross_dimension_duplicates = 0
    for row in all_rows:
        key = (
            normalize(row["prompt"][-1]["content"]),
            normalize(row["chosen"][0]["content"]),
            normalize(row["rejected"][0]["content"]),
        )
        if key in seen:
            cross_dimension_duplicates += 1
            continue
        seen[key] = row["id"]
        unique_rows.append(row)

    dataset_path = dataset_dir / f"{PUBLIC_FAMILY}.jsonl"
    reset_output_files(dataset_dir, (dataset_path.name,))
    for row in unique_rows:
        append_jsonl(dataset_path, row)

    family_manifest = {
        "schema_version": 1,
        "dataset_type": "distillation-dpo",
        "family": PUBLIC_FAMILY,
        "dataset_path": str(dataset_path),
        "row_count": len(unique_rows),
        "teacher_model": args.pair_model,
        "teacher_provider": "openrouter",
        "generation_run": args.generation_run,
        "chosen_source": "teacher",
        "rejected_source": "controlled_weak",
        "target_consumer": "slm-distillation",
        "metadata": {
            "pipeline": "derivation_task_novelty_pair_validate_five_gate_judge_reviewer_dedup",
            "dimensions": list(dimensions),
            "quality_adjudication": {row["id"]: quality_by_id[row["id"]] for row in unique_rows},
            "cross_dimension_exact_duplicates": cross_dimension_duplicates,
            "dimension_summaries": dimension_summaries,
            "generation_status": "complete",
            "publish_ready": True,
        },
    }
    family_manifest_path = manifest_dir / f"{PUBLIC_FAMILY}.{args.generation_run}.manifest.json"
    write_json(family_manifest_path, family_manifest)

    run_manifest = {
        "schema_version": 1,
        "dataset_type": "distillation-dpo",
        "generation_run": args.generation_run,
        "teacher_model": args.pair_model,
        "teacher_provider": "openrouter",
        "chosen_source": "teacher",
        "rejected_source": "controlled_weak",
        "target_consumer": "slm-distillation",
        "families": [PUBLIC_FAMILY],
        "datasets": [{
            "family": PUBLIC_FAMILY,
            "dataset_path": str(dataset_path),
            "manifest_path": str(family_manifest_path),
            "row_count": len(unique_rows),
        }],
        "total_rows": len(unique_rows),
        "metadata": {
            "generation_mode": "stage_oriented_pipeline",
            "pipeline": "derivation_task_novelty_pair_validate_five_gate_judge_reviewer_dedup",
            "derivation_model": args.derivation_model,
            "task_model": args.task_model,
            "pair_model": args.pair_model,
            "judge_model": args.judge_model,
            "reviewer_model": args.reviewer_model,
            "dimensions": list(dimensions),
            "cross_dimension_exact_duplicates": cross_dimension_duplicates,
            "dimension_summaries": dimension_summaries,
            "generation_status": "complete",
            "publish_ready": True,
        },
    }
    manifest_path = manifest_dir / f"{args.generation_run}.manifest.json"
    write_json(manifest_path, run_manifest)
    print(
        f"[distillation-dpo-pipeline] complete dimensions={len(dimensions)} "
        f"accepted={len(unique_rows)} manifest={manifest_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production Distillation-DPO pipeline migrated from the finalized one-off."
    )
    p.add_argument("--dimensions", nargs="+", default=["all"])
    p.add_argument("--generation-run", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--derivations-per-seed", type=int, default=30)
    p.add_argument("--tasks-per-derivation", type=int, default=15)
    p.add_argument("--pair-batch-size", type=int, default=4)
    p.add_argument("--judge-batch-size", type=int, default=10)
    p.add_argument("--reviewer-batch-size", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cardinality-fill-attempts", type=int, default=3)
    p.add_argument("--stage-batch-attempts", type=int, default=3)
    p.add_argument("--derivation-model", default="openai/gpt-5.6-luna-pro")
    p.add_argument("--task-model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--pair-model", default="deepseek/deepseek-v4-flash")
    p.add_argument("--judge-model", default="google/gemma-4-31b-it")
    p.add_argument("--reviewer-model", default="openai/gpt-5.6-luna-pro")
    p.add_argument("--derivation-max-tokens", type=int, default=4096)
    p.add_argument("--task-max-tokens", type=int, default=4096)
    p.add_argument("--pair-max-tokens", type=int, default=4096)
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
        "seeds", "derivations_per_seed", "tasks_per_derivation", "pair_batch_size",
        "judge_batch_size", "reviewer_batch_size", "concurrency",
        "cardinality_fill_attempts", "stage_batch_attempts",
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
