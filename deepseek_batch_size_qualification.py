#!/usr/bin/env python3
"""
Standalone DeepSeek batch-size qualification harness.

Purpose:
    Test application-level batching for the already-qualified grounded-artifact
    architecture without modifying the production repository.

Phase 1:
    Batch sizes: 4, 8, 16, 24, 32
    Signals:     arithmetic, task_code, educational_qa_mcq_math,
                 educational_qa_mcq_general, factual_restraint
    Requests:    25 total (one request per batch-size/signal combination)
    Rows:        420 rendered records total

Phase 2 (run only after reviewing Phase 1):
    Repeat one selected batch size across all five signals to test stability.

The script reads the frozen 150-artifact manifest created by:
    deepseek_grounded_150_qualification.py --prepare-only ...

For batch size 32, it adds two TEST-ONLY, preflight-validated grounded artifacts
per signal. These are written to the output directory and never modify the
source manifest or production code.

Requirements:
    pip install httpx python-dotenv

Environment:
    OPENROUTER_API_KEY in .env or exported in the shell.

Examples:
    # No API calls. Validates input and writes the 32-per-signal test pool.
    python deepseek_batch_size_qualification.py \
        --manifest logs/deepseek_grounded_150/artifact_manifest.jsonl \
        --prepare-only \
        --output-dir logs/deepseek_batch_size_test

    # Phase 1: 25 paid requests / 420 rows.
    python deepseek_batch_size_qualification.py \
        --manifest logs/deepseek_grounded_150/artifact_manifest.jsonl \
        --output-dir logs/deepseek_batch_size_test

    # Safely resume Phase 1 after interruption.
    python deepseek_batch_size_qualification.py \
        --manifest logs/deepseek_grounded_150/artifact_manifest.jsonl \
        --output-dir logs/deepseek_batch_size_test \
        --resume

    # Phase 2 after selecting a winning batch size from Phase 1.
    python deepseek_batch_size_qualification.py \
        --manifest logs/deepseek_grounded_150/artifact_manifest.jsonl \
        --output-dir logs/deepseek_batch_size_test \
        --stress-batch-size 16 \
        --stress-repetitions 4
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import operator
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
SIGNALS = [
    "arithmetic",
    "task_code",
    "educational_qa_mcq_math",
    "educational_qa_mcq_general",
    "factual_restraint",
]
BATCH_SIZES = [4, 8, 16, 24, 32]


@dataclass(frozen=True)
class Artifact:
    signal: str
    family: str
    artifact_id: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Safe checks and test-only artifact extension
# ---------------------------------------------------------------------------

def safe_eval(expression: str) -> int:
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if type(node.op) in operators:
                return operators[type(node.op)](left, right)
            if isinstance(node.op, ast.Div) and right != 0 and left % right == 0:
                return left // right
        raise ValueError(f"Unsupported or non-exact expression: {expression!r}")

    return evaluate(ast.parse(expression, mode="eval"))


def numeric_literals(text: str) -> list[str]:
    """Extract integer literals while allowing terminal punctuation such as '7.'."""
    return re.findall(r"(?<![\w])-?\d+(?![\w])", text)


def extra_test_artifacts() -> list[Artifact]:
    return [
        Artifact("arithmetic", "direct_expression", "batchtest_arith_expr_31", {
            "instruction": "Create a direct integer-expression question.",
            "expression": "(64 - 19) * 3 + 8",
            "required_numeric_literals": ["64", "19", "3", "8"],
            "answer": "143",
        }),
        Artifact("arithmetic", "exact_allocation", "batchtest_arith_division_32", {
            "instruction": "Create an exact-allocation question asking for the number of containers required.",
            "setting": "equipment totes",
            "facts": ["3120 items total", "48 items per container"],
            "expression": "3120 / 48",
            "required_numeric_literals": ["3120", "48"],
            "answer": "65",
        }),

        Artifact("task_code", "nested_list_transformation", "batchtest_code_scale_31", {
            "code": """def triple_positive_rows(rows):
    return [[value * 3 for value in row if value > 0] for row in rows]""",
            "behavior_contract": "Return a new nested list retaining only positive integers in each row and multiplying retained values by 3, while preserving order and not mutating inputs.",
        }),
        Artifact("task_code", "dictionary_keywise_sum", "batchtest_code_merge_32", {
            "code": """def merge_defect_counts(first, second):
    totals = {}
    for label in set(first) | set(second):
        totals[label] = first.get(label, 0) + second.get(label, 0)
    return totals""",
            "behavior_contract": "Return a new dictionary over the union of two defect-count dictionaries, summing counts and treating absent labels as zero without mutating inputs.",
        }),

        Artifact("educational_qa_mcq_math", "exact_division", "batchtest_math_division_31", {
            "prompt_fact": "Create a question where 3024 labels are placed equally into 54 containers and ask how many are in each container.",
            "required_numeric_literals": ["3024", "54"],
            "choices": ["52", "56", "58", "64"],
            "answer": "56",
            "expression": "3024 / 54",
        }),
        Artifact("educational_qa_mcq_math", "two_step_quantity", "batchtest_math_quantity_32", {
            "prompt_fact": "Create a remaining-quantity question about workshop kits: begin with 690, remove 218, then remove 147 more.",
            "required_numeric_literals": ["690", "218", "147"],
            "choices": ["315", "325", "335", "345"],
            "answer": "325",
            "expression": "690 - 218 - 147",
        }),

        Artifact("educational_qa_mcq_general", "reading_comprehension", "batchtest_general_reading_31", {
            "evidence": "Passage: Theo stored the signed forms in the green cabinet. Later, he placed blank forms on the reception desk.",
            "question": "Where did Theo store the signed forms?",
            "choices": ["in the green cabinet", "on the reception desk", "in the mailroom", "inside his backpack"],
            "answer": "in the green cabinet",
        }),
        Artifact("educational_qa_mcq_general", "policy_application", "batchtest_general_policy_32", {
            "evidence": "Policy: Only sealed containers may be placed on the archive shelf.",
            "question": "Which action violates the policy?",
            "choices": [
                "A sealed container is placed on the archive shelf.",
                "An open container is placed on the archive shelf.",
                "An open container is kept on a worktable.",
                "A worker seals a container before shelving it.",
            ],
            "answer": "An open container is placed on the archive shelf.",
        }),

        Artifact("factual_restraint", "future_uncertainty", "batchtest_restraint_future_31", {
            "question": "What will the exact attendance be at the Lakeside Expo scheduled for September 2027?",
            "expected_safe_behavior": "The event is in the future, so exact attendance cannot yet be known; official totals may be checked after the event.",
            "forbidden_behavior": "Do not predict a total or state an exact attendance number.",
        }),
        Artifact("factual_restraint", "ambiguous_entity", "batchtest_restraint_ambiguous_32", {
            "question": "Why did Taylor Morgan's organization withdraw its proposal?",
            "expected_safe_behavior": "The person, organization, and proposal are not sufficiently identified; request clarifying context.",
            "forbidden_behavior": "Do not invent a withdrawal reason or organization.",
        }),
    ]


def load_manifest(path: Path) -> list[Artifact]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    artifacts: list[Artifact] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            artifacts.append(Artifact(
                signal=row["signal"],
                family=row["family"],
                artifact_id=row["artifact_id"],
                payload=row["payload"],
            ))
        except Exception as exc:
            raise SystemExit(f"Invalid manifest record on line {line_no}: {exc}") from exc
    return artifacts


def preflight_validate(artifacts: list[Artifact], expected_per_signal: int) -> None:
    failures: list[str] = []
    ids = [artifact.artifact_id for artifact in artifacts]
    duplicates = [artifact_id for artifact_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        failures.append(f"duplicate artifact IDs: {duplicates}")

    counts = Counter(artifact.signal for artifact in artifacts)
    expected = {signal: expected_per_signal for signal in SIGNALS}
    if dict(counts) != expected:
        failures.append(f"signal counts expected {expected}, received {dict(counts)}")

    for artifact in artifacts:
        payload = artifact.payload
        if artifact.signal in {"arithmetic", "educational_qa_mcq_math"}:
            try:
                result = str(safe_eval(payload["expression"]))
                if result != payload["answer"]:
                    failures.append(
                        f"{artifact.artifact_id}: expression gives {result}, expected {payload['answer']}"
                    )
            except Exception as exc:
                failures.append(f"{artifact.artifact_id}: invalid expression: {exc}")
        if artifact.signal == "educational_qa_mcq_math":
            choices = payload["choices"]
            if len(choices) != 4 or len(set(choices)) != 4 or payload["answer"] not in choices:
                failures.append(f"{artifact.artifact_id}: invalid math MCQ choices/answer")
        if artifact.signal == "task_code":
            try:
                tree = ast.parse(payload["code"])
                if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                    failures.append(f"{artifact.artifact_id}: code is not exactly one Python function")
            except SyntaxError as exc:
                failures.append(f"{artifact.artifact_id}: code does not parse: {exc}")
        if artifact.signal == "educational_qa_mcq_general":
            choices = payload["choices"]
            if len(choices) != 4 or len(set(choices)) != 4 or payload["answer"] not in choices:
                failures.append(f"{artifact.artifact_id}: invalid general MCQ choices/answer")
        if artifact.signal == "factual_restraint":
            if not payload.get("question") or not payload.get("expected_safe_behavior"):
                failures.append(f"{artifact.artifact_id}: incomplete restraint artifact")

    if failures:
        raise SystemExit("Preflight failed before any API calls:\n- " + "\n- ".join(failures))


def balanced_pools(artifacts: list[Artifact]) -> dict[str, list[Artifact]]:
    """
    Round-robin family ordering means the smaller batch tests do not consist
    only of the first family in the source manifest.
    """
    result: dict[str, list[Artifact]] = {}
    for signal in SIGNALS:
        by_family: dict[str, list[Artifact]] = defaultdict(list)
        for artifact in artifacts:
            if artifact.signal == signal:
                by_family[artifact.family].append(artifact)
        ordered: list[Artifact] = []
        family_names = list(by_family.keys())
        while any(by_family[family] for family in family_names):
            for family in family_names:
                if by_family[family]:
                    ordered.append(by_family[family].pop(0))
        result[signal] = ordered
    return result


# ---------------------------------------------------------------------------
# Array schemas and prompts
# ---------------------------------------------------------------------------

def item_schema(signal: str) -> dict[str, Any]:
    if signal == "arithmetic":
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "type": {"const": "arithmetic"},
                "question": {"type": "string"},
                "solution": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["artifact_id", "type", "question", "solution", "answer"],
            "additionalProperties": False,
        }
    if signal == "task_code":
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "type": {"const": "task_code"},
                "task": {"type": "string"},
            },
            "required": ["artifact_id", "type", "task"],
            "additionalProperties": False,
        }
    if signal == "educational_qa_mcq_math":
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "type": {"const": "educational_qa_mcq_math"},
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["artifact_id", "type", "question", "choices", "answer", "explanation"],
            "additionalProperties": False,
        }
    if signal == "educational_qa_mcq_general":
        return {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "type": {"const": "educational_qa_mcq_general"},
                "evidence": {"type": "string"},
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["artifact_id", "type", "evidence", "question", "choices", "answer", "explanation"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "type": {"const": "factual_restraint"},
            "question": {"type": "string"},
            "safe_answer": {"type": "string"},
        },
        "required": ["artifact_id", "type", "question", "safe_answer"],
        "additionalProperties": False,
    }


def batch_schema(signal: str, batch_size: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": item_schema(signal),
                "minItems": batch_size,
                "maxItems": batch_size,
            },
        },
        "required": ["records"],
        "additionalProperties": False,
    }


def prompt_for(signal: str, artifacts: list[Artifact]) -> str:
    batch_payload = [
        {
            "artifact_id": artifact.artifact_id,
            "family": artifact.family,
            "payload": artifact.payload,
        }
        for artifact in artifacts
    ]
    header = (
        "Generate one final synthetic PRETRAINING record for each grounded artifact in the JSON array below. "
        "Each artifact is authoritative. Do not change facts, code behavior, supplied choices, known answers, "
        "questions that must be copied, or required restraint behavior. "
        "Return exactly one record for every artifact, in the same order, preserving artifact_id exactly. "
        "Return only a JSON object with a single `records` array matching the supplied schema; no markdown or extra text.\n\n"
        f"SIGNAL: {signal}\nBATCH_SIZE: {len(artifacts)}\n\n"
    )
    if signal == "arithmetic":
        rules = """For each record:
- Write a natural learner-facing question.
- Write a concise worked solution.
- Copy the grounded answer exactly.
- Use each required numeric literal in the question as decimal numerals.
- Do not introduce additional numeric quantities or reveal the answer in the question.
"""
    elif signal == "task_code":
        rules = """For each record:
- Generate only a task specification corresponding to the valid code artifact.
- Start the task with "Write a Python function that".
- State the input shape, output shape, required operations, and that inputs must not be mutated.
- Do not include code, pseudocode, solution steps, or the supplied function name.
"""
    elif signal == "educational_qa_mcq_math":
        rules = """For each record:
- Write a self-contained natural question preserving all grounded mathematical facts.
- Copy choices exactly and in the supplied order.
- Copy the grounded answer exactly.
- Give a concise explanation.
- Do not alter or omit numeric values.
"""
    elif signal == "educational_qa_mcq_general":
        rules = """For each record:
- Copy evidence, question, choices, and answer exactly as supplied.
- Generate only a concise explanation using the supplied evidence.
- Do not rewrite or add to the evidence or question.
"""
    else:
        rules = """For each record:
- Copy the grounded question exactly as supplied.
- Generate a concise safe_answer following expected_safe_behavior.
- Avoid forbidden_behavior and do not add unsupported facts, predictions, disclosures,
  diagnoses, legal conclusions, or definitive financial recommendations.
"""
    return header + rules + "\nGROUNDED ARTIFACTS:\n" + json.dumps(batch_payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Per-row and request validation
# ---------------------------------------------------------------------------

def validate_row(artifact: Artifact, output: dict[str, Any]) -> dict[str, Any]:
    payload = artifact.payload
    failures: list[str] = []
    warnings: list[str] = []
    manual_required = artifact.signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"}

    if output.get("artifact_id") != artifact.artifact_id:
        failures.append("artifact_id mismatch")

    if artifact.signal == "arithmetic":
        if output.get("answer") != payload["answer"]:
            failures.append(f"answer differs from grounded answer {payload['answer']!r}")
        observed = numeric_literals(output.get("question", ""))
        required = payload["required_numeric_literals"]
        if Counter(observed) != Counter(required):
            failures.append(f"question numeric literals changed: expected {required}, observed {observed}")
        if payload["answer"] in observed and payload["answer"] not in required:
            failures.append("question reveals the grounded answer")
        if not output.get("solution", "").strip():
            failures.append("empty solution")

    elif artifact.signal == "task_code":
        task = output.get("task", "")
        lower = task.lower()
        if len(task.split()) < 18:
            failures.append("task too short for fidelity review")
        if not lower.startswith("write a python function that"):
            failures.append("task does not start with required instruction phrase")
        if any(marker in lower for marker in ("```", "\ndef ", "lambda ", "import ")):
            failures.append("task includes code/implementation leakage")
        warnings.append("manual instruction-to-code fidelity review required")

    elif artifact.signal == "educational_qa_mcq_math":
        if output.get("choices") != payload["choices"]:
            failures.append("choices differ from grounded choices")
        if output.get("answer") != payload["answer"]:
            failures.append("answer differs from grounded answer")
        observed = numeric_literals(output.get("question", ""))
        required = payload["required_numeric_literals"]
        if Counter(observed) != Counter(required):
            failures.append(f"question numeric literals changed: expected {required}, observed {observed}")
        if not output.get("explanation", "").strip():
            failures.append("empty explanation")

    elif artifact.signal == "educational_qa_mcq_general":
        for key in ("evidence", "question", "choices", "answer"):
            if output.get(key) != payload[key]:
                failures.append(f"{key} differs from grounded value")
        if not output.get("explanation", "").strip():
            failures.append("empty explanation")
        warnings.append("manual explanation fidelity review required")

    else:
        if output.get("question") != payload["question"]:
            failures.append("question differs from grounded question")
        if not output.get("safe_answer", "").strip():
            failures.append("empty safe_answer")
        warnings.append("manual restraint-behavior review required")

    return {
        "automatic_screen_pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "manual_review_required": manual_required,
    }


def validate_batch(expected: list[Artifact], records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = [artifact.artifact_id for artifact in expected]
    returned_ids = [record.get("artifact_id") for record in records]
    duplicate_ids = [item for item, count in Counter(returned_ids).items() if count > 1]
    missing_ids = [item for item in expected_ids if item not in returned_ids]
    unexpected_ids = [item for item in returned_ids if item not in expected_ids]

    batch_failures: list[str] = []
    if len(records) != len(expected):
        batch_failures.append(f"expected {len(expected)} rows, received {len(records)}")
    if duplicate_ids:
        batch_failures.append(f"duplicate returned artifact_ids: {duplicate_ids}")
    if missing_ids:
        batch_failures.append(f"missing artifact_ids: {missing_ids}")
    if unexpected_ids:
        batch_failures.append(f"unexpected artifact_ids: {unexpected_ids}")

    expected_by_id = {artifact.artifact_id: artifact for artifact in expected}
    row_reviews: list[dict[str, Any]] = []
    for row in records:
        artifact_id = row.get("artifact_id")
        if artifact_id in expected_by_id and artifact_id not in duplicate_ids:
            row_reviews.append({
                "artifact_id": artifact_id,
                "output": row,
                "validation": validate_row(expected_by_id[artifact_id], row),
            })

    row_failures = sum(
        1 for review in row_reviews if not review["validation"]["automatic_screen_pass"]
    )
    return {
        "batch_integrity_pass": not batch_failures,
        "batch_failures": batch_failures,
        "expected_ids": expected_ids,
        "returned_ids": returned_ids,
        "row_reviews": row_reviews,
        "row_automatic_screen_pass": len(row_reviews) - row_failures,
        "row_automatic_screen_fail": row_failures,
    }


# ---------------------------------------------------------------------------
# API calls, run plans, resume and reports
# ---------------------------------------------------------------------------

def call_openrouter(
    client: httpx.Client,
    *,
    api_key: str,
    model: str,
    signal: str,
    batch_size: int,
    artifacts: list[Artifact],
    max_retries: int,
) -> dict[str, Any]:
    max_tokens = max(2048, batch_size * 320)
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(signal, artifacts)}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"batch_{signal}_{batch_size}",
                "strict": True,
                "schema": batch_schema(signal, batch_size),
            },
        },
        "provider": {"require_parameters": True, "allow_fallbacks": False},
        "temperature": 0.35,
        "max_tokens": max_tokens,
    }
    started = time.monotonic()
    for attempt in range(max_retries + 1):
        response = client.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tohio/slm-synthetic-data",
                "X-Title": "DeepSeek grounded batch-size qualification",
            },
            json=request,
        )
        elapsed = time.monotonic() - started
        if response.status_code == 200:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            base = {
                "model_returned": body.get("model"),
                "provider": body.get("provider"),
                "usage": body.get("usage", {}),
                "retry_count": attempt,
                "elapsed_seconds": round(elapsed, 3),
                "max_tokens": max_tokens,
            }
            try:
                parsed = json.loads(raw)
                return {"status": "completed", "records": parsed["records"], **base}
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                return {"status": "malformed_output", "error": str(exc), "raw_output": raw, **base}

        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            seconds = 3 * (2 ** attempt)
            print(f"    transient HTTP {response.status_code}; retrying after {seconds}s")
            time.sleep(seconds)
            continue
        return {
            "status": "api_error",
            "error": f"HTTP {response.status_code}: {response.text}",
            "elapsed_seconds": round(elapsed, 3),
            "max_tokens": max_tokens,
        }

    return {"status": "api_error", "error": "retry budget exhausted"}


def phase1_plan(pools: dict[str, list[Artifact]]) -> list[dict[str, Any]]:
    plan = []
    for batch_size in BATCH_SIZES:
        for signal in SIGNALS:
            artifacts = pools[signal][:batch_size]
            plan.append({
                "request_key": f"phase1_bs{batch_size}_{signal}",
                "phase": "phase1",
                "batch_size": batch_size,
                "signal": signal,
                "artifacts": artifacts,
            })
    return plan


def stress_plan(pools: dict[str, list[Artifact]], batch_size: int, repetitions: int) -> list[dict[str, Any]]:
    plan = []
    for repetition in range(1, repetitions + 1):
        for signal in SIGNALS:
            plan.append({
                "request_key": f"stress_bs{batch_size}_rep{repetition}_{signal}",
                "phase": "stress",
                "batch_size": batch_size,
                "signal": signal,
                "repetition": repetition,
                "artifacts": pools[signal][:batch_size],
            })
    return plan


def existing_request_keys(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    keys = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.add(json.loads(line)["request_key"])
    return keys


def read_request_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_pool(path: Path, pools: dict[str, list[Artifact]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for signal in SIGNALS:
            for artifact in pools[signal]:
                handle.write(json.dumps(asdict(artifact), ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]], model: str, mode: str) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["batch_size"], record["signal"])].append(record)

    by_batch_size: dict[str, Any] = {}
    for (batch_size, signal), rows in sorted(grouped.items()):
        item = by_batch_size.setdefault(str(batch_size), {
            "requests": 0, "completed_array_responses": 0, "failed_responses": 0,
            "rendered_rows_received": 0, "batch_integrity_pass_requests": 0,
            "automatic_screen_pass_rows": 0, "automatic_screen_fail_rows": 0,
            "total_tokens": 0, "cost": 0.0, "elapsed_seconds": 0.0,
            "by_signal": {},
        })
        completed = [row for row in rows if row["status"] == "completed"]
        signal_summary = {
            "requests": len(rows),
            "completed_array_responses": len(completed),
            "failed_responses": len(rows) - len(completed),
            "rendered_rows_received": sum(len(row.get("records", [])) for row in completed),
            "batch_integrity_pass_requests": sum(
                1 for row in completed if row.get("validation", {}).get("batch_integrity_pass")
            ),
            "automatic_screen_pass_rows": sum(
                row.get("validation", {}).get("row_automatic_screen_pass", 0) for row in completed
            ),
            "automatic_screen_fail_rows": sum(
                row.get("validation", {}).get("row_automatic_screen_fail", 0) for row in completed
            ),
            "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in rows),
            "cost": sum(row.get("usage", {}).get("cost", 0.0) for row in rows),
            "elapsed_seconds": sum(row.get("elapsed_seconds", 0.0) for row in rows),
        }
        item["by_signal"][signal] = signal_summary
        for key in (
            "requests", "completed_array_responses", "failed_responses",
            "rendered_rows_received", "batch_integrity_pass_requests",
            "automatic_screen_pass_rows", "automatic_screen_fail_rows",
            "total_tokens", "cost", "elapsed_seconds",
        ):
            item[key] += signal_summary[key]

    for batch_size, item in by_batch_size.items():
        passed = item["automatic_screen_pass_rows"]
        item["cost_per_automatic_pass_row"] = item["cost"] / passed if passed else None
        item["tokens_per_automatic_pass_row"] = item["total_tokens"] / passed if passed else None

    return {
        "experiment": "deepseek_grounded_batch_size_qualification",
        "model": model,
        "mode": mode,
        "batch_sizes_tested": sorted(int(size) for size in by_batch_size),
        "by_batch_size": by_batch_size,
        "eligibility_rule": (
            "A batch size remains eligible only if every signal request returns a valid array, "
            "all expected artifact IDs occur exactly once, and review reveals no new fidelity pattern."
        ),
    }


def write_review(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# DeepSeek Grounded Batch-Size Qualification Review", ""]
    for record in records:
        lines.extend([
            f"## {record['request_key']}",
            "",
            f"- Signal: `{record['signal']}`",
            f"- Batch size: `{record['batch_size']}`",
            f"- Status: `{record['status']}`",
            f"- Cost: `{record.get('usage', {}).get('cost', 0.0)}`",
            f"- Elapsed seconds: `{record.get('elapsed_seconds', '')}`",
            "",
        ])
        if record["status"] != "completed":
            lines.extend(["```json", json.dumps(record, indent=2, ensure_ascii=False), "```", ""])
            continue
        lines.extend([
            "### Batch validation",
            "",
            "```json",
            json.dumps({
                "batch_integrity_pass": record["validation"]["batch_integrity_pass"],
                "batch_failures": record["validation"]["batch_failures"],
                "row_automatic_screen_pass": record["validation"]["row_automatic_screen_pass"],
                "row_automatic_screen_fail": record["validation"]["row_automatic_screen_fail"],
            }, indent=2, ensure_ascii=False),
            "```",
            "",
        ])
        for review in record["validation"]["row_reviews"]:
            lines.extend([
                f"### {review['artifact_id']}",
                "",
                "```json",
                json.dumps(review, indent=2, ensure_ascii=False),
                "```",
                "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone DeepSeek grounded batch-size qualification.")
    parser.add_argument("--manifest", required=True, help="Path to the frozen 150-artifact JSONL manifest.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="logs/deepseek_batch_size_qualification")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--stress-batch-size", type=int, choices=BATCH_SIZES)
    parser.add_argument("--stress-repetitions", type=int, default=4)
    args = parser.parse_args()

    source_artifacts = load_manifest(Path(args.manifest))
    preflight_validate(source_artifacts, expected_per_signal=30)

    test_artifacts = source_artifacts + extra_test_artifacts()
    preflight_validate(test_artifacts, expected_per_signal=32)
    pools = balanced_pools(test_artifacts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "batch_test_pool_32_per_signal.jsonl"
    write_pool(pool_path, pools)

    if args.stress_batch_size:
        mode = f"stress_bs{args.stress_batch_size}_reps{args.stress_repetitions}"
        plan = stress_plan(pools, args.stress_batch_size, args.stress_repetitions)
    else:
        mode = "phase1"
        plan = phase1_plan(pools)

    plan_json = {
        "model": args.model,
        "source_manifest": args.manifest,
        "mode": mode,
        "architecture": "signal-homogeneous grounded artifact array -> one structured DeepSeek array response",
        "batch_sizes": [args.stress_batch_size] if args.stress_batch_size else BATCH_SIZES,
        "request_count": len(plan),
        "expected_rendered_rows": sum(item["batch_size"] for item in plan),
        "test_pool_counts": {signal: len(pools[signal]) for signal in SIGNALS},
        "source_manifest_unchanged": True,
        "preflight": "passed",
    }
    (output_dir / f"{mode}_plan.json").write_text(json.dumps(plan_json, indent=2), encoding="utf-8")
    print(json.dumps(plan_json, indent=2))

    if args.prepare_only:
        print(f"\nPrepared and validated test pool without API calls: {pool_path}")
        return 0

    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing from .env or shell environment")

    records_path = output_dir / f"{mode}_requests.jsonl"
    if records_path.exists() and not args.resume:
        raise SystemExit(
            f"{records_path} already exists. Choose a new output directory or use --resume."
        )
    done = existing_request_keys(records_path) if args.resume else set()
    pending = [request for request in plan if request["request_key"] not in done]
    print(f"Pending API requests: {len(pending)}; already recorded: {len(done)}")

    with httpx.Client(timeout=300.0) as client:
        for index, request in enumerate(pending, 1):
            print(
                f"[{index}/{len(pending)}] {request['request_key']} | "
                f"{request['signal']} | batch={request['batch_size']}"
            )
            result = call_openrouter(
                client,
                api_key=api_key,
                model=args.model,
                signal=request["signal"],
                batch_size=request["batch_size"],
                artifacts=request["artifacts"],
                max_retries=args.max_retries,
            )
            record: dict[str, Any] = {
                "request_key": request["request_key"],
                "phase": request["phase"],
                "signal": request["signal"],
                "batch_size": request["batch_size"],
                "expected_artifact_ids": [artifact.artifact_id for artifact in request["artifacts"]],
                "model_requested": args.model,
                **result,
            }
            if result["status"] == "completed":
                record["validation"] = validate_batch(request["artifacts"], result["records"])
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    records = read_request_records(records_path)
    summary = summarize(records, args.model, mode)
    summary_path = output_dir / f"{mode}_summary.json"
    review_path = output_dir / f"{mode}_review.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_review(review_path, records)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {records_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
