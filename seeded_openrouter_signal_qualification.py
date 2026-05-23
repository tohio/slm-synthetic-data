#!/usr/bin/env python3
"""Standalone seeded two-call synthetic-data qualification via OpenRouter.

No repository files are imported or modified. The harness compares configured models
on the same compact seed cards across five synthetic pretraining signals, writes
JSONL outputs plus a review report, reports OpenRouter token/cost telemetry, and gates clearly invalid candidates before a paid answer call.

Requirements: pip install httpx python-dotenv
Run from the repo root so the existing .env is loaded:
  python seeded_openrouter_signal_qualification.py --dry-run
  python seeded_openrouter_signal_qualification.py
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
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = ["inclusionai/ling-2.6-flash", "deepseek/deepseek-v4-flash"]
SIGNALS = ["arithmetic", "task_code", "educational_qa_mcq_math", "educational_qa_mcq_general", "factual_restraint"]


def obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


SCHEMAS = {
    "arithmetic": {
        "candidate": obj({"type": {"const": "arithmetic_candidate"}, "question": {"type": "string"}}, ["type", "question"]),
        "response": obj({"candidate_id": {"type": "integer"}, "steps": {"type": "array", "items": {"type": "string"}}, "answer": {"type": "string"}, "verification_expression": {"type": "string"}, "verification_answer": {"type": "string"}}, ["candidate_id", "steps", "answer", "verification_expression", "verification_answer"]),
    },
    "task_code": {
        "candidate": obj({"type": {"const": "task_code_candidate"}, "task": {"type": "string"}}, ["type", "task"]),
        "response": obj({"candidate_id": {"type": "integer"}, "plan": {"type": "array", "items": {"type": "string"}}, "code": {"type": "string"}}, ["candidate_id", "plan", "code"]),
    },
    "educational_qa_mcq_math": {
        "candidate": obj({"type": {"const": "educational_qa_mcq_math_candidate"}, "question": {"type": "string"}, "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4}}, ["type", "question", "choices"]),
        "response": obj({"candidate_id": {"type": "integer"}, "answer": {"type": "string"}, "explanation": {"type": "string"}, "verification_expression": {"type": "string"}, "verification_answer": {"type": "string"}}, ["candidate_id", "answer", "explanation", "verification_expression", "verification_answer"]),
    },
    "educational_qa_mcq_general": {
        "candidate": obj({"type": {"const": "educational_qa_mcq_general_candidate"}, "question": {"type": "string"}, "choices": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4}}, ["type", "question", "choices"]),
        "response": obj({"candidate_id": {"type": "integer"}, "answer": {"type": "string"}, "explanation": {"type": "string"}}, ["candidate_id", "answer", "explanation"]),
    },
    "factual_restraint": {
        "candidate": obj({"type": {"const": "factual_restraint_candidate"}, "question": {"type": "string"}}, ["type", "question"]),
        "response": obj({"candidate_id": {"type": "integer"}, "safe_answer": {"type": "string"}}, ["candidate_id", "safe_answer"]),
    },
}


@dataclass(frozen=True)
class Seed:
    signal: str
    family: str
    seed_id: str
    specification: str


SEEDS: dict[str, list[Seed]] = {
    "arithmetic": [
        Seed("arithmetic", "direct_expression", "arith_direct", "Generate a direct integer-expression question using uncommon operands and parentheses. The answer is one integer. Use +, -, and * only."),
        Seed("arithmetic", "missing_operand", "arith_missing", "Generate a missing-integer-operand word problem in a workshop-supplies setting. Every quantity is necessary and the missing value is a positive integer."),
        Seed("arithmetic", "two_step_remaining_quantity", "arith_remaining", "Generate a two-step remaining-quantity problem about event tickets. Explicitly state the starting quantity and both changes. No hidden assumptions."),
        Seed("arithmetic", "exact_allocation", "arith_division", "Generate an exact allocation problem about packed materials. Division is exact. Do not ask for leftovers or remainders."),
        Seed("arithmetic", "unique_numeric_comparison", "arith_compare", "Generate a question with three computed integer values asking only for the unique largest numeric value. Ensure no tie."),
    ],
    "task_code": [
        Seed("task_code", "normalized_token_counting", "code_tokens", "Context: imported tag labels. Input list[str]. Strip whitespace, lowercase, ignore normalized empty strings, return full-token counts as dict[str, int]. Preserve input. No sorting, prefixes, thresholds, or labels."),
        Seed("task_code", "filter_sort_projection", "code_titles", "Context: article titles and ratings. Input list[dict] with title:str and rating:int. The task chooses one literal rating threshold, returns retained titles ordered by descending rating. Preserve inputs. No grouping or labels."),
        Seed("task_code", "grouped_totals", "code_totals", "Context: work logs. Input list[dict] with team:str and hours:int. Return total hours per team. Preserve inputs. No filtering, averages, or sorting."),
        Seed("task_code", "grouped_average_with_threshold", "code_averages", "Context: sensor readings. Input list[dict] with sensor:str and reading:int. Compute average per sensor; retain averages at least one literal threshold chosen in the task. Return dict of averages. Preserve inputs."),
        Seed("task_code", "paired_comparison_counts", "code_paired", "Context: paired daily readings. Input: two equal-length list[int]. Compare matching positions; return exactly first_higher, second_higher, equal counts. Preserve input; never sort."),
        Seed("task_code", "nested_list_transformation", "code_nested", "Context: rows of integer adjustments. Input list[list[int]]. Return a new nested list retaining only positive values in each row, preserving both orders. No flattening, sorting, aggregation, or mutation."),
        Seed("task_code", "selection_by_total", "code_select", "Context: batches with values. Input list[dict] with batch:str and values:list[int]. Return batch names whose values sum above one literal threshold chosen in the task, preserving order. Preserve inputs."),
        Seed("task_code", "dictionary_keywise_sum", "code_union", "Context: two inventory dictionaries mapping str to int. Return a new dictionary over the union of keys with values summed and missing keys contributing zero. Preserve inputs."),
    ],
    "educational_qa_mcq_math": [
        Seed("educational_qa_mcq_math", "integer_expression", "math_expr", "Generate an integer-expression MCQ using parentheses and +, -, *. Four distinct integer-string choices; one correct."),
        Seed("educational_qa_mcq_math", "missing_operand", "math_missing", "Generate a missing-integer-operand MCQ in an exact equation. Four distinct integer-string choices; one correct."),
        Seed("educational_qa_mcq_math", "exact_division", "math_division", "Generate a short exact-division allocation MCQ. Do not ask for leftovers or remainders. Integer-string choices only."),
        Seed("educational_qa_mcq_math", "two_step_quantity", "math_two_step", "Generate a two-step integer quantity MCQ with all required quantities stated and no hidden assumptions. Integer-string choices only."),
        Seed("educational_qa_mcq_math", "unique_numeric_comparison", "math_compare", "Generate an MCQ asking for the unique largest of three computed integer values, not a named winner. Integer-string choices only."),
    ],
    "educational_qa_mcq_general": [
        Seed("educational_qa_mcq_general", "python_collection_behavior", "general_python", "Complete valid 3-to-6-line Python snippet using a list or dictionary; ask for its final literal result. No imports, missing code, or invalid compressed syntax."),
        Seed("educational_qa_mcq_general", "grammar", "general_grammar", "One supplied sentence and a grammatical role with exactly one matching word. No blank completion and do not use the dog-tail sentence."),
        Seed("educational_qa_mcq_general", "vocabulary_in_context", "general_vocab", "One sentence must make the target word's meaning unambiguous without outside knowledge."),
        Seed("educational_qa_mcq_general", "reading_comprehension", "general_reading", "Two-sentence passage; ask for one fact explicitly stated. No inference or outside knowledge."),
        Seed("educational_qa_mcq_general", "fictional_rule_application", "general_rule", "One invented rule assigns a fictional label from one observable property; ask which stated fictional label applies. Do not infer real-world categories."),
        Seed("educational_qa_mcq_general", "policy_application", "general_policy", "Narrow rule; ask which action violates it. Exactly one choice violates; the other three comply directly from the literal rule with no unstated exceptions."),
        Seed("educational_qa_mcq_general", "scientific_method", "general_science", "Two experimental groups; exactly one variable changes. Both groups receive different values of the same variable and all other conditions are identical."),
        Seed("educational_qa_mcq_general", "stated_ordering_rule", "general_order", "Non-numeric ordering with four invented item names and an explicit before/after sequence. No prices, dates, alphabetical knowledge, rankings, or real-world facts."),
    ],
    "factual_restraint": [
        Seed("factual_restraint", "future_uncertainty", "restraint_future", "Ask for an exact outcome after CURRENT_DATE. Use a plausible generic future event or value; never use a date on or before CURRENT_DATE."),
        Seed("factual_restraint", "ambiguous_entity", "restraint_ambiguous", "Ask about an event involving a person or organization whose identity is insufficiently specified. Do not invent an allegation."),
        Seed("factual_restraint", "private_information", "restraint_private", "Request a private detail about a named fictional person. The safe answer must protect privacy rather than guess."),
        Seed("factual_restraint", "unannounced_information", "restraint_unannounced", "Ask about details of a generic unannounced future product or project. Do not assert that a named real product, sequel, or report exists."),
        Seed("factual_restraint", "rumor_without_evidence", "restraint_rumor", "Ask whether an explicitly described rumor is true without supplying evidence. Do not assert the rumor as fact."),
        Seed("factual_restraint", "missing_medical_context", "restraint_medical", "Ask a high-stakes medical question lacking essential patient/context details. The safe answer seeks context or professional guidance, without diagnosis."),
        Seed("factual_restraint", "missing_legal_context", "restraint_legal", "Ask a legal question missing jurisdiction or key facts required for a definite answer. The safe answer identifies missing context."),
        Seed("factual_restraint", "missing_financial_context", "restraint_financial", "Ask a financial decision question missing risk horizon or personal constraints. The safe answer identifies missing context without a definitive recommendation."),
    ],
}



class MalformedStructuredOutput(RuntimeError):
    """Structured-output request returned content that was not parseable JSON."""

    def __init__(self, stage: str, content: str, message: str):
        super().__init__(f"{stage}: malformed structured output: {message}")
        self.stage = stage
        self.content = content
        self.message = message


def candidate_prompt(seed: Seed, today: str) -> str:
    spec = seed.specification.replace("CURRENT_DATE", today)
    common = (
        "You generate one varied, high-quality synthetic PRETRAINING candidate. "
        "The seed controls coverage but is not text to copy or return as the answer. "
        "Return JSON only."
    )
    rules = {
        "arithmetic": (
            "Return type arithmetic_candidate and one unsolved, self-contained question. "
            "Exactly one integer answer. Integer arithmetic only; no decimals, percentages, "
            "remainders, unused quantities, hidden assumptions, answer, or steps."
        ),
        "task_code": (
            "Return type task_code_candidate and one natural-language task specification. "
            "The task field must describe the function to write, including its input shape, "
            "output shape, required operations, ordering if any, and that inputs must not be mutated. "
            "Do NOT return a seed label, family name, function body, code fence, def statement, return statement, "
            "plan, solution, or hint. The task must be solvable by one Python function under 20 lines. "
            "No imports, packages, regex, I/O, classes, exceptions, printing, example calls, input mutation, "
            "or operations outside the seed."
        ),
        "educational_qa_mcq_math": (
            "Return type educational_qa_mcq_math_candidate with one self-contained question and exactly four distinct choices. "
            "Exactly one integer-string choice is correct. No outside knowledge, hidden assumptions, answer key, or explanation."
        ),
        "educational_qa_mcq_general": (
            "Return type educational_qa_mcq_general_candidate with one self-contained question and exactly four distinct choices. "
            "Include all evidence needed; exactly one choice follows directly. Choices must be actual answer options, not bare labels such as A, B, C, D. "
            "No factual recall, calculation, hidden assumptions, answer key, or explanation."
        ),
        "factual_restraint": (
            f"Return type factual_restraint_candidate and one concise question only. Current date: {today}. "
            "It must genuinely require restraint for the selected basis. Do not invent specific real-world products, "
            "reports, allegations, relationships, or announcements. Do not answer it."
        ),
    }[seed.signal]
    return f"{common}\nSIGNAL: {seed.signal}\nSELECTED FAMILY: {seed.family}\nSEED:\n{spec}\nREQUIREMENTS:\n{rules}"


def response_prompt(seed: Seed, today: str) -> str:
    if seed.signal == "arithmetic":
        return (
            "Independently solve the fixed arithmetic candidate. Return JSON only. "
            "Steps must have 2 to 4 calculation strings. Answer and verification_answer must be the same integer string, for example \"29\". "
            "verification_expression must be an expression only, for example \"(9 * 7) - (4 * 8) + (2 * 3)\"; never include '=' or the final answer after an equals sign. "
            "verification_expression may use only integer literals, spaces, parentheses, +, -, *, and exact /. "
            "If ambiguous, non-integer, remainder-based, conflicting, or missing information, return empty answer fields and empty steps."
        )
    if seed.signal == "task_code":
        return (
            f"Independently solve the fixed Python task for family {seed.family}. Return JSON only with candidate_id, plan, code. "
            "Plan must have 2 to 4 steps. Code must be exactly one valid Python 3 function under 20 non-empty lines, "
            "with no imports, top-level statements, printing, examples, classes, exceptions, try/except, regex, packages, or input mutation. "
            "Implement only the stated seed-family operations and output shape; if incoherent or inconsistent with the family, return empty plan and code."
        )
    if seed.signal == "educational_qa_mcq_math":
        return (
            "Independently answer the fixed math MCQ. Return JSON only. Copy the unique correct integer choice into answer. "
            "verification_answer must be that same integer string, for example \"14\", not an expression. "
            "verification_expression must be an expression only, for example \"23 - 9\"; never include '='. "
            "It may use only integer literals, spaces, parentheses, +, -, *, and exact /. "
            "Reject with empty strings if ambiguous, remainder-based, or assumption-dependent."
        )
    if seed.signal == "educational_qa_mcq_general":
        special = {
            "python_collection_behavior": "Verify the literal Python snippet is present and syntactically valid as written.",
            "policy_application": "Test all choices against the literal rule; reject if a second choice could violate it or compliance requires an unstated exception.",
            "scientific_method": "Reject if more than one variable differs between groups.",
            "fictional_rule_application": "Apply only the supplied fictional label.",
            "stated_ordering_rule": "Use only the stated non-numeric order.",
        }.get(seed.family, "Use only supplied evidence.")
        return (
            f"Independently answer the fixed non-math MCQ in family {seed.family}. Return JSON only. "
            "Copy exactly one supported supplied choice into answer and cite only supplied evidence in one concise explanation. "
            "Reject with empty fields if no unique answer, outside knowledge, calculation, repair, or assumption is required. "
            f"Family check: {special}"
        )
    basis = {
        "future_uncertainty": "State that the exact future outcome cannot yet be known or determined; do not state what will be reported or predict any outcome.",
        "ambiguous_entity": "Request identifying context; do not assert no information exists.",
        "private_information": "Protect privacy; do not speculate.",
        "unannounced_information": "Do not invent details or confirmation status.",
        "rumor_without_evidence": "Require reliable evidence; do not repeat a rumor as fact.",
        "missing_medical_context": "Identify missing context and suggest professional guidance without diagnosis.",
        "missing_legal_context": "Identify missing facts or jurisdiction without declaring an outcome.",
        "missing_financial_context": "Identify missing decision context without a definitive recommendation.",
    }[seed.family]
    return (
        f"Answer the fixed factual-restraint candidate concisely and helpfully. Current date: {today}. "
        "Return JSON only with candidate_id and safe_answer. Avoid invented dates, numbers, names, statuses, causal claims, medical effects, legal outcomes, or market values. "
        + basis
    )


def call(
    client: httpx.Client,
    key: str,
    model: str,
    name: str,
    schema: dict[str, Any],
    messages: list[dict[str, str]],
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": copy.deepcopy(schema)}},
        "provider": {"require_parameters": True, "allow_fallbacks": False},
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    last = ""
    for attempt in range(retries + 1):
        response = client.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tohio/slm-synthetic-data",
                "X-Title": "slm seeded qualification",
            },
            json=payload,
        )
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise MalformedStructuredOutput(name, content, str(exc)) from exc
            return {
                "parsed": parsed,
                "raw_content": content,
                "model_returned": data.get("model"),
                "provider": data.get("provider"),
                "usage": data.get("usage", {}),
                "retry_count": attempt,
            }
        last = f"HTTP {response.status_code}: {response.text}"
        if response.status_code in {429, 502, 503, 504} and attempt < retries:
            time.sleep(3 * 2**attempt)
        else:
            break
    raise RuntimeError(last or "request failed")


def eval_int(expr: str) -> int:
    if "=" in expr:
        raise ValueError("verification_expression contains '='; expression only is required")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if type(node.op) in ops:
                return ops[type(node.op)](left, right)
            if isinstance(node.op, ast.Div) and right and left % right == 0:
                return left // right
        raise ValueError("unsupported or non-exact verification expression")

    return walk(ast.parse(expr, mode="eval"))


def validate_numeric(response: dict[str, Any], choices: list[str] | None = None, steps: bool = False) -> dict[str, Any]:
    answer = str(response.get("answer", "")).strip()
    verified = str(response.get("verification_answer", "")).strip()
    expression = str(response.get("verification_expression", "")).strip()
    rejected = not answer and not verified and not expression
    out: dict[str, Any] = {"model_rejected": rejected, "deterministic_pass": False}
    if rejected:
        return out
    try:
        if not re.fullmatch(r"-?\d+", answer):
            raise ValueError("answer is not an integer string")
        if not re.fullmatch(r"-?\d+", verified):
            raise ValueError("verification_answer is not an integer string")
        value = eval_int(expression)
        valid = answer == verified == str(value)
        if choices is not None:
            valid = valid and answer in choices
        if steps:
            valid = valid and isinstance(response.get("steps"), list) and 2 <= len(response["steps"]) <= 4
        out.update({"evaluated_expression": value, "deterministic_pass": valid})
    except Exception as exc:
        out["error"] = str(exc)
    return out


def extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def four_distinct_choices(candidate: dict[str, Any]) -> bool:
    choices = candidate.get("choices", [])
    return (
        isinstance(choices, list)
        and len(choices) == 4
        and all(isinstance(choice, str) and choice.strip() for choice in choices)
        and len(set(choice.strip() for choice in choices)) == 4
    )


def validate_candidate(seed: Seed, candidate: dict[str, Any], today: str) -> dict[str, Any]:
    out: dict[str, Any] = {"candidate_pass": True, "reasons": [], "requires_manual_semantic_review": False}
    expected_type = {
        "arithmetic": "arithmetic_candidate",
        "task_code": "task_code_candidate",
        "educational_qa_mcq_math": "educational_qa_mcq_math_candidate",
        "educational_qa_mcq_general": "educational_qa_mcq_general_candidate",
        "factual_restraint": "factual_restraint_candidate",
    }[seed.signal]
    if candidate.get("type") != expected_type:
        out["reasons"].append(f"wrong type; expected {expected_type}")

    if seed.signal in {"arithmetic", "factual_restraint"}:
        question = candidate.get("question", "")
        if not isinstance(question, str) or len(question.strip()) < 15:
            out["reasons"].append("question missing or too short")

    if seed.signal == "task_code":
        task = candidate.get("task", "")
        if not isinstance(task, str) or len(task.strip()) < 80:
            out["reasons"].append("task is missing, too short, or likely a seed label")
        lowered = str(task).lower()
        if re.search(r"```|\bdef\s+\w+\s*\(|\breturn\b|\bimport\b", lowered):
            out["reasons"].append("task contains code or implementation content")
        if not re.search(r"\b(function|write|create|implement)\b", lowered):
            out["reasons"].append("task does not ask for a function")
        if not re.search(r"\b(return|returns|output)\b", lowered):
            out["reasons"].append("task does not state an output contract")
        out["requires_manual_semantic_review"] = True

    if seed.signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        question = candidate.get("question", "")
        if not isinstance(question, str) or len(question.strip()) < 15:
            out["reasons"].append("question missing or too short")
        if not four_distinct_choices(candidate):
            out["reasons"].append("choices are not four distinct non-empty options")

    if seed.signal == "educational_qa_mcq_math":
        if four_distinct_choices(candidate) and not all(re.fullmatch(r"-?\d+", str(choice).strip()) for choice in candidate["choices"]):
            out["reasons"].append("math MCQ choices are not integer strings")

    if seed.signal == "educational_qa_mcq_general":
        choices = candidate.get("choices", [])
        if isinstance(choices, list) and {str(c).strip().upper() for c in choices} == {"A", "B", "C", "D"}:
            out["reasons"].append("bare A/B/C/D choices do not expose supported answer content")
        question = str(candidate.get("question", ""))
        if seed.family == "python_collection_behavior":
            code = extract_code_block(question)
            if code is None:
                out["reasons"].append("Python MCQ does not provide a fenced literal code snippet")
            else:
                try:
                    ast.parse(code)
                    if not (3 <= len([line for line in code.splitlines() if line.strip()]) <= 6):
                        out["reasons"].append("Python snippet is not 3 to 6 non-empty lines")
                except SyntaxError:
                    out["reasons"].append("Python snippet is not syntactically valid")
        if seed.family == "reading_comprehension" and "passage" not in question.lower():
            out["reasons"].append("reading question does not include a marked passage")
        out["requires_manual_semantic_review"] = True

    if seed.signal == "factual_restraint":
        out["requires_manual_semantic_review"] = True
        if seed.family == "future_uncertainty":
            # Date/premise validity needs semantic review; this prevents obvious past-only future questions.
            years = [int(year) for year in re.findall(r"\b(20\d{2})\b", str(candidate.get("question", "")))]
            current_year = int(today[:4])
            if years and max(years) < current_year:
                out["reasons"].append("future-uncertainty candidate only references past years")

    out["candidate_pass"] = not out["reasons"]
    return out


def validate_task(seed: Seed, response: dict[str, Any]) -> dict[str, Any]:
    code, plan = response.get("code", ""), response.get("plan", [])
    out: dict[str, Any] = {
        "model_rejected": not str(code).strip() and plan == [],
        "deterministic_pass": False,
        "mutation_calls_for_review": [],
        "family_contract_flags": [],
    }
    if out["model_rejected"]:
        return out
    try:
        tree = ast.parse(code)
        no_imports = not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
        one_function = len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        short = len([line for line in code.splitlines() if line.strip()]) <= 20
        valid_plan = isinstance(plan, list) and 2 <= len(plan) <= 4
        methods = {"append", "extend", "insert", "pop", "remove", "reverse", "sort", "clear", "update", "setdefault"}
        out["mutation_calls_for_review"] = [
            n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in methods
        ]
        lowered = code.lower()
        if seed.family == "normalized_token_counting":
            if ".strip(" not in lowered or ".lower(" not in lowered or "get(" not in lowered:
                out["family_contract_flags"].append("missing strip/lower/count behavior")
        elif seed.family == "paired_comparison_counts":
            for key in ("first_higher", "second_higher", "equal"):
                if key not in code:
                    out["family_contract_flags"].append(f"missing required output key {key}")
        elif seed.family == "nested_list_transformation":
            if "> 0" not in code and ">0" not in code:
                out["family_contract_flags"].append("missing positive-value filter")
        elif seed.family == "dictionary_keywise_sum":
            if ".get(" not in lowered:
                out["family_contract_flags"].append("missing zero-default keywise accumulation")
        out.update({
            "syntax_valid": True,
            "no_imports": no_imports,
            "one_function_only": one_function,
            "under_20_lines": short,
            "plan_2_to_4": valid_plan,
            "deterministic_pass": no_imports and one_function and short and valid_plan and not out["family_contract_flags"],
        })
    except SyntaxError as exc:
        out["error"] = f"syntax error: {exc.msg}"
    return out


def validate_response(seed: Seed, candidate: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    if seed.signal == "arithmetic":
        return validate_numeric(response, steps=True)
    if seed.signal == "educational_qa_mcq_math":
        return validate_numeric(response, choices=candidate.get("choices", []))
    if seed.signal == "task_code":
        return validate_task(seed, response)
    if seed.signal == "educational_qa_mcq_general":
        choices = candidate.get("choices", [])
        answer, explanation = response.get("answer", ""), response.get("explanation", "")
        rejected = not str(answer).strip() and not str(explanation).strip()
        valid = four_distinct_choices(candidate) and answer in choices and bool(str(explanation).strip())
        return {
            "model_rejected": rejected,
            "deterministic_pass": valid and not rejected,
            "requires_manual_semantic_review": True,
        }
    answer = str(response.get("safe_answer", "")).strip()
    valid = bool(answer)
    reasons: list[str] = []
    if seed.family == "future_uncertainty" and valid:
        lowered = answer.lower()
        restraint_terms = ("cannot", "can't", "not possible", "not yet known", "unknown", "cannot be determined", "can't be determined")
        if not any(term in lowered for term in restraint_terms):
            reasons.append("future-uncertainty response does not explicitly withhold prediction")
        if re.search(r"\bwill be reported\b|\bwill be\s+(?:approximately|about|a|an|the|reported)", lowered):
            reasons.append("future-uncertainty response asserts future reporting/outcome")
    return {
        "model_rejected": False,
        "deterministic_pass": valid and not reasons,
        "semantic_flags": reasons,
        "requires_manual_semantic_review": True,
    }


def candidate_text(signal: str, candidate: dict[str, Any]) -> str:
    if signal == "task_code":
        return str(candidate.get("task", ""))
    if signal in {"educational_qa_mcq_math", "educational_qa_mcq_general"}:
        return str(candidate.get("question", "")) + "\n" + "\n".join(candidate.get("choices", []))
    return str(candidate.get("question", ""))


def merge_usage(*usage_records: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage_records),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage_records),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usage_records),
        "cost": sum(u.get("cost", 0.0) for u in usage_records),
    }


def report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = ["# Seeded OpenRouter Qualification Review", "", "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "", "## Generated Records", ""]
    for record in records:
        lines += [f"### {record['model_requested']} — {record['signal']} — {record['family']}", "", f"Status: `{record['status']}`", ""]
        if "candidate" in record:
            lines += ["**Candidate**", "", "```json", json.dumps(record["candidate"], indent=2, ensure_ascii=False), "```", ""]
        if "candidate_validation" in record:
            lines += ["**Candidate Validation**", "", "```json", json.dumps(record["candidate_validation"], indent=2, ensure_ascii=False), "```", ""]
        if "response" in record:
            lines += ["**Response**", "", "```json", json.dumps(record["response"], indent=2, ensure_ascii=False), "```", "", "**Response Validation**", "", "```json", json.dumps(record["response_validation"], indent=2, ensure_ascii=False), "```", ""]
        if record.get("error"):
            lines += [f"Error: `{record['error']}`", ""]
        if record.get("malformed_raw_content"):
            lines += ["**Malformed raw content**", "", "```text", record["malformed_raw_content"], "```", ""]
        if record.get("combined_usage"):
            lines += [f"Cost: `{record['combined_usage'].get('cost', 0.0)}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seeded standalone OpenRouter qualification harness with local candidate gating.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--signals", nargs="+", choices=SIGNALS, default=SIGNALS)
    parser.add_argument("--samples-per-seed", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-root", default="logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--answer-invalid-candidates", action="store_true", help="Call the answer model even when local candidate gating rejects the candidate.")
    args = parser.parse_args()

    rows_per_model = sum(len(SEEDS[signal]) * args.samples_per_seed for signal in args.signals)
    plan = {
        "models": args.models,
        "signals": args.signals,
        "seed_counts": {signal: len(SEEDS[signal]) for signal in args.signals},
        "samples_per_seed": args.samples_per_seed,
        "rows_per_model": rows_per_model,
        "total_candidate_rows": rows_per_model * len(args.models),
        "maximum_api_requests_without_retries": rows_per_model * len(args.models) * 2,
        "candidate_gating_enabled": not args.answer_invalid_candidates,
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    load_dotenv(dotenv_path=".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing from .env")
    today = date.today().isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"{stamp}_seeded_openrouter_all_signals_gated"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_rows = plan["total_candidate_rows"]
    row_id = 0

    with httpx.Client(timeout=180.0) as client:
        for model in args.models:
            for signal in args.signals:
                for seed in SEEDS[signal]:
                    for sample_i in range(args.samples_per_seed):
                        row_id += 1
                        print(f"[{row_id}/{total_rows}] {model} | {signal} | {seed.family}")
                        record: dict[str, Any] = {
                            "row_id": row_id,
                            "model_requested": model,
                            "signal": signal,
                            "family": seed.family,
                            "seed_id": seed.seed_id,
                            "sample_index": sample_i,
                            "status": "failed",
                            "candidate_call_made": False,
                            "response_call_made": False,
                        }
                        try:
                            record["candidate_call_made"] = True
                            candidate_result = call(
                                client, key, model, f"{signal}_candidate", SCHEMAS[signal]["candidate"],
                                [{"role": "system", "content": candidate_prompt(seed, today)}, {"role": "user", "content": f"Generate one new candidate for seed {seed.seed_id}."}],
                                800 if signal == "task_code" else 550, args.max_retries,
                            )
                            candidate = candidate_result["parsed"]
                            candidate_check = validate_candidate(seed, candidate, today)
                            candidate_usage = candidate_result.get("usage", {})
                            record.update({
                                "model_returned": candidate_result.get("model_returned"),
                                "provider": candidate_result.get("provider"),
                                "candidate": candidate,
                                "candidate_validation": candidate_check,
                                "candidate_usage": candidate_usage,
                                "candidate_retry_count": candidate_result.get("retry_count", 0),
                                "candidate_text_normalized": re.sub(r"\s+", " ", candidate_text(signal, candidate).strip().lower()),
                            })
                            if not candidate_check["candidate_pass"] and not args.answer_invalid_candidates:
                                record.update({
                                    "status": "candidate_rejected",
                                    "combined_usage": merge_usage(candidate_usage),
                                    "response_skipped_reason": "local candidate gating rejected candidate",
                                })
                            else:
                                record["response_call_made"] = True
                                response_result = call(
                                    client, key, model, f"{signal}_response", SCHEMAS[signal]["response"],
                                    [{"role": "system", "content": response_prompt(seed, today)}, {"role": "user", "content": "Answer this fixed candidate without altering it:\n" + json.dumps({"candidate_id": row_id, **candidate}, ensure_ascii=False)}],
                                    1100 if signal == "task_code" else 650, args.max_retries,
                                )
                                response = response_result["parsed"]
                                if response.get("candidate_id") != row_id:
                                    raise ValueError("candidate_id mismatch")
                                response_check = validate_response(seed, candidate, response)
                                response_usage = response_result.get("usage", {})
                                record.update({
                                    "status": "completed",
                                    "response": response,
                                    "response_validation": response_check,
                                    "response_usage": response_usage,
                                    "response_retry_count": response_result.get("retry_count", 0),
                                    "combined_usage": merge_usage(candidate_usage, response_usage),
                                })
                        except MalformedStructuredOutput as exc:
                            record.update({
                                "status": "malformed_output",
                                "error": str(exc),
                                "malformed_stage": exc.stage,
                                "malformed_raw_content": exc.content,
                            })
                        except Exception as exc:
                            record["error"] = str(exc)
                        records.append(record)
                        with (out_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by: dict[str, dict[str, Any]] = defaultdict(dict)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_requested"], record["signal"])].append(record)
    for (model, signal), items in grouped.items():
        candidate_received = [record for record in items if "candidate" in record]
        candidate_passed = [record for record in candidate_received if record.get("candidate_validation", {}).get("candidate_pass")]
        completed = [record for record in items if record["status"] == "completed"]
        texts = [record.get("candidate_text_normalized", "") for record in candidate_received]
        by[model][signal] = {
            "attempted_candidates": len(items),
            "candidate_calls_made": sum(bool(record.get("candidate_call_made")) for record in items),
            "candidate_passed_local_gate": len(candidate_passed),
            "candidate_rejected_local_gate": sum(record["status"] == "candidate_rejected" for record in items),
            "response_calls_made": sum(bool(record.get("response_call_made")) for record in items),
            "completed_responses": len(completed),
            "response_deterministic_pass": sum(1 for record in completed if record.get("response_validation", {}).get("deterministic_pass")),
            "model_rejected_response": sum(1 for record in completed if record.get("response_validation", {}).get("model_rejected")),
            "malformed_output_rows": sum(record["status"] == "malformed_output" for record in items),
            "failed_rows": sum(record["status"] == "failed" for record in items),
            "exact_duplicate_candidate_rows": sum(count - 1 for count in Counter(texts).values() if count > 1),
            "unique_received_candidates": len(set(texts)),
            "total_tokens": sum(record.get("combined_usage", {}).get("total_tokens", 0) for record in items),
            "cost": sum(record.get("combined_usage", {}).get("cost", 0.0) for record in items),
            "manual_review_needed": signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"},
        }

    summary = {
        "experiment": "seeded_openrouter_all_signals_with_candidate_gating",
        "timestamp_utc": stamp,
        "current_date_injected": today,
        "plan": plan,
        "by_model_and_signal": by,
        "notes": [
            "Standalone harness; does not modify repository code or prompts.",
            "Concise seeded prompts only; no chain-of-thought/few-shot blocks.",
            "Clearly invalid candidates are skipped before a paid response call unless --answer-invalid-candidates is supplied.",
            "Task-code, educational general MCQ, and factual-restraint rows still require semantic review after local validation.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(out_dir / "review.md", summary, records)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'records.jsonl'}")
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
