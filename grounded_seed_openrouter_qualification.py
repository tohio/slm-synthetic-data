#!/usr/bin/env python3
"""
Grounded-seed OpenRouter qualification harness for synthetic pretraining records.

This is a standalone experiment. It does not import or modify the repo pipeline.

Thesis being tested:
    Instead of asking a model to invent both a candidate and its answer/code,
    supply a coherent grounded artifact for each signal and ask the model to
    render a natural training record from that artifact.

Default run:
    - 5 synthetic signals
    - 34 grounded seed artifacts per model
    - 2 generated samples per seed to expose within-seed repetition
    - 2 models: Ling-2.6-Flash and DeepSeek-V4-Flash
    - 136 OpenRouter calls total before retries, the same request count as the
      previous two-call 1-sample-per-seed comparison.

Requirements:
    pip install httpx python-dotenv

Environment:
    OPENROUTER_API_KEY in .env or exported in the shell.

Usage:
    python grounded_seed_openrouter_qualification.py --dry-run
    python grounded_seed_openrouter_qualification.py
    python grounded_seed_openrouter_qualification.py --samples-per-seed 1
    python grounded_seed_openrouter_qualification.py --models inclusionai/ling-2.6-flash
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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "inclusionai/ling-2.6-flash",
    "deepseek/deepseek-v4-flash",
]
ALL_SIGNALS = [
    "arithmetic",
    "task_code",
    "educational_qa_mcq_math",
    "educational_qa_mcq_general",
    "factual_restraint",
]


@dataclass(frozen=True)
class Seed:
    signal: str
    family: str
    seed_id: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Grounded artifacts
# ---------------------------------------------------------------------------

SEEDS: dict[str, list[Seed]] = {
    "arithmetic": [
        Seed("arithmetic", "direct_expression", "arith_expr_01", {
            "setting": "direct expression",
            "question_instruction": "Ask the learner to compute the integer value of the supplied expression.",
            "expression": "(37 - 12) * 4 + 18",
            "numbers": ["37", "12", "4", "18"],
            "answer": "118",
        }),
        Seed("arithmetic", "missing_operand", "arith_missing_01", {
            "setting": "art studio paint jars",
            "question_instruction": "Create a short word problem asking for the unknown starting number of jars.",
            "facts": ["12 jars were added", "the final total is 35 jars"],
            "expression": "35 - 12",
            "numbers": ["12", "35"],
            "answer": "23",
        }),
        Seed("arithmetic", "two_step_remaining_quantity", "arith_remaining_01", {
            "setting": "museum admission tickets",
            "question_instruction": "Create a short word problem asking how many tickets remain.",
            "facts": ["start with 460 tickets", "sell 138 tickets", "sell 77 more tickets"],
            "expression": "460 - 138 - 77",
            "numbers": ["460", "138", "77"],
            "answer": "245",
        }),
        Seed("arithmetic", "exact_allocation", "arith_division_01", {
            "setting": "shipping trays",
            "question_instruction": "Create an exact-allocation question asking for the number of trays required.",
            "facts": ["864 components", "24 components per tray"],
            "expression": "864 / 24",
            "numbers": ["864", "24"],
            "answer": "36",
        }),
        Seed("arithmetic", "unique_numeric_comparison", "arith_max_01", {
            "setting": "numeric comparison",
            "question_instruction": "Ask for the unique largest numeric value among the three supplied expressions.",
            "expressions": ["17 * 6", "94 + 11", "160 - 48"],
            "expression": "160 - 48",
            "numbers": ["17", "6", "94", "11", "160", "48"],
            "answer": "112",
        }),
    ],
    "task_code": [
        Seed("task_code", "normalized_token_counting", "code_tags_01", {
            "code": """def count_clean_tags(tags):
    counts = {}
    for tag in tags:
        cleaned = tag.strip().lower()
        if cleaned:
            counts[cleaned] = counts.get(cleaned, 0) + 1
    return counts""",
            "must_describe": ["list of strings", "strip", "lowercase", "dictionary", "count", "not mutate"],
            "summary": "Counts normalized non-empty tag strings as full strings; it does not split them into words.",
        }),
        Seed("task_code", "filter_sort_projection", "code_titles_01", {
            "code": """def select_top_titles(entries):
    kept = [entry for entry in entries if entry["rating"] >= 80]
    kept = sorted(kept, key=lambda entry: entry["rating"], reverse=True)
    return [entry["title"] for entry in kept]""",
            "must_describe": ["list of dictionaries", "title", "rating", "80", "descending", "not mutate"],
            "summary": "Filters by the literal threshold 80, sorts by descending rating, and returns titles.",
        }),
        Seed("task_code", "grouped_totals", "code_hours_01", {
            "code": """def total_hours_by_team(logs):
    totals = {}
    for log in logs:
        team = log["team"]
        totals[team] = totals.get(team, 0) + log["hours"]
    return totals""",
            "must_describe": ["list of dictionaries", "team", "hours", "sum", "dictionary", "not mutate"],
            "summary": "Sums hours by team without filtering or sorting.",
        }),
        Seed("task_code", "grouped_average_with_threshold", "code_readings_01", {
            "code": """def qualifying_sensor_averages(readings):
    totals = {}
    counts = {}
    for row in readings:
        sensor = row["sensor"]
        totals[sensor] = totals.get(sensor, 0) + row["reading"]
        counts[sensor] = counts.get(sensor, 0) + 1
    averages = {s: totals[s] / counts[s] for s in totals}
    return {s: avg for s, avg in averages.items() if avg >= 70}""",
            "must_describe": ["list of dictionaries", "sensor", "reading", "average", "70", "dictionary", "not mutate"],
            "summary": "Computes per-sensor averages and retains averages at least 70.",
        }),
        Seed("task_code", "paired_comparison_counts", "code_compare_01", {
            "code": """def compare_readings(first, second):
    result = {"first_higher": 0, "second_higher": 0, "equal": 0}
    for left, right in zip(first, second):
        if left > right:
            result["first_higher"] += 1
        elif right > left:
            result["second_higher"] += 1
        else:
            result["equal"] += 1
    return result""",
            "must_describe": ["two", "lists of integers", "matching", "first_higher", "second_higher", "equal", "dictionary", "not mutate"],
            "summary": "Compares paired positions without sorting and returns exactly three count keys.",
        }),
        Seed("task_code", "nested_list_transformation", "code_rows_01", {
            "code": """def keep_positive_rows(rows):
    return [[value for value in row if value > 0] for row in rows]""",
            "must_describe": ["list of lists", "positive", "original order", "new list", "not mutate"],
            "summary": "Retains positive integers independently within each row; it does not flatten or sort.",
        }),
        Seed("task_code", "selection_by_total", "code_batches_01", {
            "code": """def batches_over_limit(records):
    return [
        record["batch"]
        for record in records
        if sum(record["values"]) > 120
    ]""",
            "must_describe": ["list of dictionaries", "batch", "values", "sum", "120", "original order", "not mutate"],
            "summary": "Returns batch names whose list of values sums to more than 120, preserving order.",
        }),
        Seed("task_code", "dictionary_keywise_sum", "code_inventory_01", {
            "code": """def combine_inventory(first, second):
    result = {}
    for key in set(first) | set(second):
        result[key] = first.get(key, 0) + second.get(key, 0)
    return result""",
            "must_describe": ["two dictionaries", "union", "sum", "missing", "zero", "new dictionary", "not mutate"],
            "summary": "Combines the union of keys, treating absent values as zero.",
        }),
    ],
    "educational_qa_mcq_math": [
        Seed("educational_qa_mcq_math", "integer_expression", "math_expr_01", {
            "question_facts": "Evaluate (14 + 9) * 3 - 11.",
            "choices": ["58", "47", "69", "80"],
            "answer": "58",
            "expression": "(14 + 9) * 3 - 11",
            "required_numbers": ["14", "9", "3", "11"],
        }),
        Seed("educational_qa_mcq_math", "missing_operand", "math_missing_01", {
            "question_facts": "Find the missing integer in 6 * ? + 5 = 47.",
            "choices": ["5", "6", "7", "8"],
            "answer": "7",
            "expression": "(47 - 5) / 6",
            "required_numbers": ["6", "5", "47"],
        }),
        Seed("educational_qa_mcq_math", "exact_division", "math_division_01", {
            "question_facts": "A school places 756 folders equally into 21 cabinets. Ask how many folders are in each cabinet.",
            "choices": ["32", "34", "36", "38"],
            "answer": "36",
            "expression": "756 / 21",
            "required_numbers": ["756", "21"],
        }),
        Seed("educational_qa_mcq_math", "two_step_quantity", "math_two_step_01", {
            "question_facts": "A theater has 325 seats. 118 are reserved, then 47 more are reserved. Ask how many seats remain.",
            "choices": ["150", "160", "170", "180"],
            "answer": "160",
            "expression": "325 - 118 - 47",
            "required_numbers": ["325", "118", "47"],
        }),
        Seed("educational_qa_mcq_math", "unique_numeric_comparison", "math_max_01", {
            "question_facts": "Ask for the largest numeric value among 16 * 7, 97 + 10, and 138 - 19.",
            "choices": ["107", "112", "119", "121"],
            "answer": "119",
            "expression": "138 - 19",
            "required_numbers": ["16", "7", "97", "10", "138", "19"],
        }),
    ],
    "educational_qa_mcq_general": [
        Seed("educational_qa_mcq_general", "python_collection_behavior", "general_python_01", {
            "evidence": """values = ["red", "blue", "red"]
counts = {}
for value in values:
    counts[value] = counts.get(value, 0) + 1
result = counts["red"]""",
            "instruction": "Ask for the final value of result after the supplied Python code runs.",
            "choices": ["1", "2", "3", "\"red\""],
            "answer": "2",
        }),
        Seed("educational_qa_mcq_general", "grammar", "general_grammar_01", {
            "evidence": 'Sentence: "The careful artist quietly finished her sketch."',
            "instruction": 'Ask which listed word is an adverb.',
            "choices": ["careful", "artist", "quietly", "sketch"],
            "answer": "quietly",
        }),
        Seed("educational_qa_mcq_general", "vocabulary_in_context", "general_vocab_01", {
            "evidence": 'Sentence: "After running the marathon, Devin was exhausted and needed to rest for hours."',
            "instruction": 'Ask what "exhausted" means in this sentence.',
            "choices": ["very tired", "very excited", "confused", "unprepared"],
            "answer": "very tired",
        }),
        Seed("educational_qa_mcq_general", "reading_comprehension", "general_reading_01", {
            "evidence": "Passage: Mira placed the spare key in the blue drawer. She then left a note for her brother on the kitchen table.",
            "instruction": "Ask where Mira placed the spare key.",
            "choices": ["in the blue drawer", "on the kitchen table", "under the front door", "inside her backpack"],
            "answer": "in the blue drawer",
        }),
        Seed("educational_qa_mcq_general", "fictional_rule_application", "general_rule_01", {
            "evidence": 'Rule: In Orin, any lantern with a silver handle is called a "Velora." The lantern by the gate has a silver handle.',
            "instruction": "Ask which fictional label applies to the lantern.",
            "choices": ["Velora", "Sunwick", "Glassmere", "Nightcoil"],
            "answer": "Velora",
        }),
        Seed("educational_qa_mcq_general", "policy_application", "general_policy_01", {
            "evidence": "Policy: Only employees with an active badge may enter the records room.",
            "instruction": "Ask which action violates the policy.",
            "choices": [
                "An employee with an active badge enters the records room.",
                "A visitor without an active badge enters the records room.",
                "A visitor waits outside the records room.",
                "An employee checks that their badge is active before entering."
            ],
            "answer": "A visitor without an active badge enters the records room.",
        }),
        Seed("educational_qa_mcq_general", "scientific_method", "general_science_01", {
            "evidence": "Experiment: Group A listens to classical music while completing a puzzle. Group B listens to jazz music while completing the same puzzle. Puzzle, room, time limit, and instructions are identical.",
            "instruction": "Ask which variable was deliberately changed.",
            "choices": ["type of music", "puzzle used", "time limit", "instructions"],
            "answer": "type of music",
        }),
        Seed("educational_qa_mcq_general", "stated_ordering_rule", "general_order_01", {
            "evidence": "Rule: The banners are arranged in this order: Luma before Taro, Taro before Venn, and Venn before Sora.",
            "instruction": "Ask which banner comes first.",
            "choices": ["Luma", "Taro", "Venn", "Sora"],
            "answer": "Luma",
        }),
    ],
    "factual_restraint": [
        Seed("factual_restraint", "future_uncertainty", "restraint_future_01", {
            "scenario": "A city festival is scheduled for July 2027, and a user asks for its exact final attendance.",
            "question_target": "Ask for the exact attendance at the 2027 festival.",
            "response_requirement": "Say that the exact attendance cannot be known before the event occurs and official totals are available.",
            "required_any": ["cannot be known", "can't be known", "cannot determine", "not yet known", "not possible to know"],
            "forbidden": ["will be reported as", "will be exactly"],
        }),
        Seed("factual_restraint", "ambiguous_entity", "restraint_ambiguous_01", {
            "scenario": "A user asks what caused the lawsuit involving Jordan Lee's company, but gives no identifying details for Jordan Lee or the company.",
            "question_target": "Ask what caused the lawsuit involving Jordan Lee's company.",
            "response_requirement": "Request identifying details before answering; do not invent a lawsuit story.",
            "required_any": ["which jordan", "identify", "more information", "additional details", "clarify"],
            "forbidden": ["caused by", "was filed because"],
        }),
        Seed("factual_restraint", "private_information", "restraint_private_01", {
            "scenario": "A user asks for the home address of a private fictional person named Dana Rowe.",
            "question_target": "Ask for Dana Rowe's home address.",
            "response_requirement": "Decline to provide private residence information.",
            "required_any": ["cannot provide", "can't provide", "private", "home address"],
            "forbidden": [],
        }),
        Seed("factual_restraint", "unannounced_information", "restraint_unannounced_01", {
            "scenario": "A user asks for the release date and features of a company's unannounced product, without naming a real product.",
            "question_target": "Ask for release timing and features of the unannounced product.",
            "response_requirement": "Say that unannounced product details cannot be confirmed.",
            "required_any": ["unannounced", "cannot confirm", "no confirmed", "not publicly announced"],
            "forbidden": ["release date is", "will launch"],
        }),
        Seed("factual_restraint", "rumor_without_evidence", "restraint_rumor_01", {
            "scenario": "A user asks whether a rumor that a local business will close next month is true, without providing evidence.",
            "question_target": "Ask whether the closure rumor is true.",
            "response_requirement": "State that reliable evidence or official confirmation is needed.",
            "required_any": ["evidence", "official", "verify", "confirmed"],
            "forbidden": ["will close", "is closing"],
        }),
        Seed("factual_restraint", "missing_medical_context", "restraint_medical_01", {
            "scenario": "A user asks whether they should begin taking a prescription blood thinner, but provides no medical history or clinician advice.",
            "question_target": "Ask whether to start the blood thinner.",
            "response_requirement": "Avoid a medication decision and recommend consultation with a qualified clinician.",
            "required_any": ["doctor", "clinician", "healthcare professional", "prescriber"],
            "forbidden": ["you should start", "safe to start"],
        }),
        Seed("factual_restraint", "missing_legal_context", "restraint_legal_01", {
            "scenario": "A user asks whether a contract clause is enforceable, but does not supply the clause or jurisdiction.",
            "question_target": "Ask whether the contract clause is enforceable.",
            "response_requirement": "Identify missing clause text and jurisdiction; avoid declaring enforceability.",
            "required_any": ["jurisdiction", "clause", "legal professional", "lawyer"],
            "forbidden": ["is enforceable", "is not enforceable"],
        }),
        Seed("factual_restraint", "missing_financial_context", "restraint_financial_01", {
            "scenario": "A user asks whether to move retirement savings into bonds, but supplies no time horizon, risk tolerance, or goals.",
            "question_target": "Ask whether to move retirement savings into bonds.",
            "response_requirement": "Identify missing investment context and avoid a definitive allocation recommendation.",
            "required_any": ["risk tolerance", "time horizon", "goals", "financial advisor"],
            "forbidden": ["you should move", "move your"],
        }),
    ],
}


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

def schema_for(signal: str) -> dict[str, Any]:
    if signal == "arithmetic":
        return {
            "type": "object",
            "properties": {
                "type": {"const": "arithmetic"},
                "question": {"type": "string"},
                "solution": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["type", "question", "solution", "answer"],
            "additionalProperties": False,
        }
    if signal == "task_code":
        return {
            "type": "object",
            "properties": {
                "type": {"const": "task_code"},
                "task": {"type": "string"},
            },
            "required": ["type", "task"],
            "additionalProperties": False,
        }
    if signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        return {
            "type": "object",
            "properties": {
                "type": {"const": signal},
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                "answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["type", "question", "choices", "answer", "explanation"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "type": {"const": "factual_restraint"},
            "question": {"type": "string"},
            "safe_answer": {"type": "string"},
        },
        "required": ["type", "question", "safe_answer"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def prompt_for(seed: Seed, sample_index: int, current_date: str) -> str:
    payload = seed.payload
    common = (
        "Generate one natural synthetic PRETRAINING record from the grounded artifact below. "
        "The artifact is authoritative: do not change its facts, answer, choices, code behavior, "
        "or required safety behavior. Vary only natural wording and explanation style. "
        "Return only one JSON object matching the supplied schema. "
        "Do not output reasoning notes, markdown fences, or extra text.\n\n"
        f"SIGNAL: {seed.signal}\nFAMILY: {seed.family}\nVARIATION INDEX: {sample_index + 1}\n\n"
    )
    if seed.signal == "arithmetic":
        return common + f"""GROUNDED ARITHMETIC BACKBONE:
{json.dumps(payload, indent=2)}

REQUIREMENTS:
- Write a clear learner-facing question using every supplied numeric quantity as decimal numerals.
- Do not introduce any additional numeric quantities.
- Do not reveal the answer in the question.
- Write a concise worked solution.
- The JSON answer must be exactly "{payload['answer']}".
"""
    if seed.signal == "task_code":
        return common + f"""GROUNDED VALID PYTHON CODE:
```python
{payload['code']}
```

BEHAVIOR SUMMARY:
{payload['summary']}

REQUIREMENTS:
- Generate only a natural-language task specification faithfully describing the supplied code.
- The task must state the input and output contracts and that inputs are not mutated.
- Do not include code, pseudocode, a function name, an implementation, or a solution.
- Do not add behavior that is absent from the code.
"""
    if seed.signal == "educational_qa_mcq_math":
        return common + f"""GROUNDED MATH MCQ BACKBONE:
{json.dumps(payload, indent=2)}

REQUIREMENTS:
- Write a natural question that preserves the supplied mathematical facts.
- Copy the four supplied choices exactly and in the supplied order.
- Copy the answer exactly as "{payload['answer']}".
- Give a concise explanation consistent with expression "{payload['expression']}".
- Do not add numeric facts or alter the values.
"""
    if seed.signal == "educational_qa_mcq_general":
        return common + f"""GROUNDED EVIDENCE AND ANSWER RELATIONSHIP:
{json.dumps(payload, indent=2)}

REQUIREMENTS:
- Include the supplied evidence verbatim in the question, followed by a natural form of the requested question.
- Copy the four choices exactly and in the supplied order.
- Copy the answer exactly as supplied.
- Explain the answer using only the supplied evidence.
"""
    return common + f"""CURRENT DATE: {current_date}
GROUNDED RESTRAINT SCENARIO:
{json.dumps(payload, indent=2)}

REQUIREMENTS:
- Generate the natural user question requested by question_target.
- Generate a concise safe answer that follows response_requirement.
- Do not add unsupported facts, predictions, disclosures, diagnoses, legal conclusions, or financial recommendations.
"""


# ---------------------------------------------------------------------------
# Local validation
# ---------------------------------------------------------------------------

def safe_eval(expression: str) -> int:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if type(node.op) in ops:
                return ops[type(node.op)](left, right)
            if isinstance(node.op, ast.Div) and right != 0 and left % right == 0:
                return left // right
        raise ValueError("unsupported expression")
    return visit(ast.parse(expression, mode="eval"))


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def contains_all_text(text: str, values: list[str]) -> bool:
    lower = text.lower()
    return all(value.lower() in lower for value in values)


def validate(seed: Seed, output: dict[str, Any]) -> dict[str, Any]:
    p = seed.payload
    failures: list[str] = []
    manual = seed.signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"}

    if seed.signal == "arithmetic":
        question = output.get("question", "")
        answer = output.get("answer", "")
        solution = output.get("solution", "")
        if answer != p["answer"]:
            failures.append(f'answer mismatch: expected {p["answer"]!r}, got {answer!r}')
        try:
            if str(safe_eval(p["expression"])) != p["answer"]:
                failures.append("internal seed expression/answer mismatch")
        except Exception as exc:
            failures.append(f"invalid internal seed expression: {exc}")
        question_numbers = re.findall(r"(?<![\w.])-?\d+(?![\w.])", question)
        if Counter(question_numbers) != Counter(p["numbers"]):
            failures.append(f'question numeric quantities changed: expected {p["numbers"]}, got {question_numbers}')
        if not solution.strip():
            failures.append("empty solution")

    elif seed.signal == "task_code":
        task = output.get("task", "")
        task_lower = task.lower()
        if len(task.split()) < 18:
            failures.append("task specification too short")
        if any(marker in task_lower for marker in ("```", "\ndef ", "\nreturn ", "lambda ", "import ")):
            failures.append("task contains implementation/code leakage")
        if "function" not in task_lower:
            failures.append("task does not request a function")
        missing = [term for term in p["must_describe"] if term.lower() not in task_lower]
        if missing:
            failures.append(f"task is missing behavior terms: {missing}")

    elif seed.signal == "educational_qa_mcq_math":
        question = output.get("question", "")
        if output.get("choices") != p["choices"]:
            failures.append("choices differ from grounded choices")
        if output.get("answer") != p["answer"]:
            failures.append(f'answer mismatch: expected {p["answer"]!r}, got {output.get("answer")!r}')
        if not contains_all_text(question, p["required_numbers"]):
            failures.append("question omits supplied numeric facts")
        try:
            if str(safe_eval(p["expression"])) != p["answer"]:
                failures.append("internal seed expression/answer mismatch")
        except Exception as exc:
            failures.append(f"invalid internal seed expression: {exc}")

    elif seed.signal == "educational_qa_mcq_general":
        question = output.get("question", "")
        if p["evidence"] not in question:
            failures.append("question does not include grounded evidence verbatim")
        if output.get("choices") != p["choices"]:
            failures.append("choices differ from grounded choices")
        if output.get("answer") != p["answer"]:
            failures.append(f'answer mismatch: expected {p["answer"]!r}, got {output.get("answer")!r}')
        if not output.get("explanation", "").strip():
            failures.append("empty explanation")

    elif seed.signal == "factual_restraint":
        question = output.get("question", "")
        answer = output.get("safe_answer", "")
        if not question.strip():
            failures.append("empty question")
        if not answer.strip():
            failures.append("empty safe answer")
        answer_lower = answer.lower()
        if not any(term.lower() in answer_lower for term in p["required_any"]):
            failures.append("safe answer does not demonstrate required restraint behavior")
        found_forbidden = [term for term in p["forbidden"] if term.lower() in answer_lower]
        if found_forbidden:
            failures.append(f"safe answer contains forbidden assertion: {found_forbidden}")

    return {
        "deterministic_pass": not failures,
        "failures": failures,
        "requires_manual_review": manual,
    }


# ---------------------------------------------------------------------------
# OpenRouter call and report
# ---------------------------------------------------------------------------

def call_model(
    client: httpx.Client,
    api_key: str,
    model: str,
    seed: Seed,
    sample_index: int,
    current_date: str,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_for(seed, sample_index, current_date)}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"{seed.signal}_{seed.family}",
                "strict": True,
                "schema": copy.deepcopy(schema_for(seed.signal)),
            },
        },
        "provider": {"require_parameters": True, "allow_fallbacks": False},
        "temperature": 0.6,
        "max_tokens": 900 if seed.signal == "task_code" else 650,
    }
    for attempt in range(max_retries + 1):
        response = client.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tohio/slm-synthetic-data",
                "X-Title": "grounded seed synthetic qualification",
            },
            json=payload,
        )
        if response.status_code == 200:
            body = response.json()
            raw = body["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                return {
                    "status": "malformed_output",
                    "error": str(exc),
                    "raw_output": raw,
                    "usage": body.get("usage", {}),
                    "model_returned": body.get("model"),
                    "provider": body.get("provider"),
                    "retry_count": attempt,
                }
            return {
                "status": "completed",
                "output": parsed,
                "usage": body.get("usage", {}),
                "model_returned": body.get("model"),
                "provider": body.get("provider"),
                "retry_count": attempt,
            }
        if response.status_code in {429, 502, 503, 504} and attempt < max_retries:
            wait_seconds = 3 * (2 ** attempt)
            print(f"    transient HTTP {response.status_code}; waiting {wait_seconds}s")
            time.sleep(wait_seconds)
            continue
        return {"status": "api_error", "error": f"HTTP {response.status_code}: {response.text}"}
    return {"status": "api_error", "error": "retries exhausted"}


def write_review(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    out = ["# Grounded-Seed OpenRouter Qualification Review", "", "## Summary", "", "```json",
           json.dumps(summary, indent=2), "```", "", "## Generated Records", ""]
    for record in records:
        out.extend([
            f"### {record['model_requested']} — {record['signal']} — {record['family']} — sample {record['sample_index'] + 1}",
            "",
            f"Status: `{record['status']}`",
            "",
        ])
        if record["status"] != "completed":
            out.extend(["```json", json.dumps(record, indent=2, ensure_ascii=False), "```", ""])
            continue
        out.extend([
            "**Grounded Seed**", "", "```json",
            json.dumps(record["seed"], indent=2, ensure_ascii=False), "```", "",
            "**Generated Output**", "", "```json",
            json.dumps(record["output"], indent=2, ensure_ascii=False), "```", "",
            "**Local Validation**", "", "```json",
            json.dumps(record["validation"], indent=2, ensure_ascii=False), "```", "",
        ])
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--signals", nargs="+", choices=ALL_SIGNALS, default=ALL_SIGNALS)
    parser.add_argument("--samples-per-seed", type=int, default=2, choices=range(1, 6), metavar="[1-5]")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-root", default="logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    per_model = sum(len(SEEDS[signal]) for signal in args.signals) * args.samples_per_seed
    plan = {
        "models": args.models,
        "signals": args.signals,
        "seed_counts": {signal: len(SEEDS[signal]) for signal in args.signals},
        "samples_per_seed": args.samples_per_seed,
        "records_per_model": per_model,
        "api_requests_without_retries": per_model * len(args.models),
        "design": "grounded artifact -> one model-generated final record -> local validation",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing from .env or environment")

    current_date = date.today().isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{stamp}_grounded_seed_openrouter_qualification"
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    review_path = output_dir / "review.md"

    records: list[dict[str, Any]] = []
    total = plan["api_requests_without_retries"]
    index = 0

    with httpx.Client(timeout=180.0) as client:
        for model in args.models:
            for signal in args.signals:
                for seed in SEEDS[signal]:
                    for sample_index in range(args.samples_per_seed):
                        index += 1
                        print(f"[{index}/{total}] {model} | {signal} | {seed.family} | variation {sample_index + 1}")
                        result = call_model(client, api_key, model, seed, sample_index, current_date, args.max_retries)
                        record = {
                            "model_requested": model,
                            "signal": signal,
                            "family": seed.family,
                            "seed_id": seed.seed_id,
                            "sample_index": sample_index,
                            "seed": seed.payload,
                            **result,
                        }
                        if result["status"] == "completed":
                            record["validation"] = validate(seed, result["output"])
                            record["normalized_output"] = normalized(
                                json.dumps(result["output"], sort_keys=True, ensure_ascii=False)
                            )
                        records.append(record)
                        with records_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_by_model: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_requested"], record["signal"])].append(record)

    for (model, signal), rows in grouped.items():
        completed = [row for row in rows if row["status"] == "completed"]
        normalized_outputs = [row.get("normalized_output", "") for row in completed]
        duplicate_rows = sum(v - 1 for v in Counter(normalized_outputs).values() if v > 1)
        summary_by_model.setdefault(model, {})[signal] = {
            "attempted": len(rows),
            "completed": len(completed),
            "malformed_or_api_failed": len(rows) - len(completed),
            "deterministic_pass": sum(
                1 for row in completed if row["validation"]["deterministic_pass"]
            ),
            "exact_duplicate_output_rows": duplicate_rows,
            "unique_outputs": len(set(normalized_outputs)),
            "cost": sum(row.get("usage", {}).get("cost", 0.0) for row in completed),
            "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in completed),
            "manual_review_needed": signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"},
        }

    overall: dict[str, Any] = {}
    for model, signals in summary_by_model.items():
        overall[model] = {
            "attempted": sum(row["attempted"] for row in signals.values()),
            "completed": sum(row["completed"] for row in signals.values()),
            "deterministic_pass": sum(row["deterministic_pass"] for row in signals.values()),
            "cost": sum(row["cost"] for row in signals.values()),
            "total_tokens": sum(row["total_tokens"] for row in signals.values()),
        }

    summary = {
        "experiment": "grounded_seed_openrouter_qualification",
        "timestamp_utc": stamp,
        "current_date_injected": current_date,
        "plan": plan,
        "overall_by_model": overall,
        "by_model_and_signal": summary_by_model,
        "notes": [
            "Standalone test only; does not modify the repo pipeline.",
            "One model call per record: the model renders text from a coherent grounded artifact.",
            "No chain-of-thought examples and no candidate->answer free-generation architecture.",
            "Task-code outputs are instructions derived from valid code; code is held in the seed.",
            "Manual review is still required for semantic signals even when local checks pass.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_review(review_path, summary, records)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", records_path)
    print("Saved:", summary_path)
    print("Saved:", review_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
