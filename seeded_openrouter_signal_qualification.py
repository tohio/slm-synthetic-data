#!/usr/bin/env python3
"""Standalone seeded two-call synthetic-data qualification via OpenRouter.

No repository files are imported or modified. The harness compares configured models
on the same compact seed cards across five synthetic pretraining signals, writes
JSONL outputs plus a review report, and reports OpenRouter token/cost telemetry.

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


def candidate_prompt(seed: Seed, today: str) -> str:
    spec = seed.specification.replace("CURRENT_DATE", today)
    common = "You generate one varied, high-quality synthetic PRETRAINING candidate. The seed controls coverage but is not text to copy. Return JSON only."
    rules = {
        "arithmetic": "Return type arithmetic_candidate and one unsolved question. Exactly one integer answer. Integer arithmetic only; no decimals, percentages, remainders, unused quantities, hidden assumptions, answer, or steps.",
        "task_code": "Return type task_code_candidate and one short complete function task. State clear input/output contracts. It must be solvable by one Python function under 20 lines. No imports, packages, regex, I/O, classes, exceptions, printing, example calls, input mutation, or operations outside the seed.",
        "educational_qa_mcq_math": "Return type educational_qa_mcq_math_candidate with one self-contained question and exactly four distinct choices. Exactly one integer choice is correct. No outside knowledge, hidden assumptions, answer key, or explanation.",
        "educational_qa_mcq_general": "Return type educational_qa_mcq_general_candidate with one self-contained question and exactly four distinct choices. Include all evidence needed; exactly one choice follows directly. No factual recall, calculation, hidden assumptions, answer key, or explanation.",
        "factual_restraint": f"Return type factual_restraint_candidate and one concise question only. Current date: {today}. It must genuinely require restraint for the selected basis. Do not invent specific real-world products, reports, allegations, relationships, or announcements. Do not answer it.",
    }[seed.signal]
    return f"{common}\nSIGNAL: {seed.signal}\nSELECTED FAMILY: {seed.family}\nSEED:\n{spec}\nREQUIREMENTS:\n{rules}"


def response_prompt(seed: Seed, today: str) -> str:
    if seed.signal == "arithmetic":
        return "Independently solve the fixed arithmetic candidate. Return JSON only. Steps must have 2 to 4 calculation strings. Answer and verification_answer must be the same integer string. verification_expression must compute the final answer using only integer literals, spaces, parentheses, +, -, *, and exact /. If ambiguous, non-integer, remainder-based, conflicting, or missing information, return empty answer fields and empty steps."
    if seed.signal == "task_code":
        return f"Independently solve the fixed Python task for family {seed.family}. Return JSON only with candidate_id, plan, code. Plan must have 2 to 4 steps. Code must be exactly one valid Python 3 function under 20 non-empty lines, with no imports, top-level statements, printing, examples, classes, exceptions, try/except, regex, packages, or input mutation. Implement only the stated seed-family operations; if incoherent, return empty plan and code."
    if seed.signal == "educational_qa_mcq_math":
        return "Independently answer the fixed math MCQ. Return JSON only. Copy the unique correct integer choice into answer. explanation must be concise. verification_expression must compute answer using only integer literals, spaces, parentheses, +, -, *, and exact /. verification_answer must equal answer. Reject with empty strings if ambiguous, remainder-based, or assumption-dependent."
    if seed.signal == "educational_qa_mcq_general":
        special = {"python_collection_behavior": "Verify the Python snippet is syntactically valid as written.", "policy_application": "Test all choices against the literal rule; reject if a second choice could violate it or compliance requires an unstated exception.", "scientific_method": "Reject if more than one variable differs between groups.", "fictional_rule_application": "Apply only the supplied fictional label.", "stated_ordering_rule": "Use only the stated non-numeric order."}.get(seed.family, "Use only supplied evidence.")
        return f"Independently answer the fixed non-math MCQ in family {seed.family}. Return JSON only. Copy exactly one supported supplied choice into answer and cite only supplied evidence in one concise explanation. Reject with empty fields if no unique answer, outside knowledge, calculation, repair, or assumption is required. Family check: {special}"
    basis = {"future_uncertainty": "Do not predict an exact future outcome.", "ambiguous_entity": "Request identifying context; do not assert no information exists.", "private_information": "Protect privacy; do not speculate.", "unannounced_information": "Do not invent details or confirmation status.", "rumor_without_evidence": "Require reliable evidence; do not repeat a rumor as fact.", "missing_medical_context": "Identify missing context and suggest professional guidance without diagnosis.", "missing_legal_context": "Identify missing facts or jurisdiction without declaring an outcome.", "missing_financial_context": "Identify missing decision context without a definitive recommendation."}[seed.family]
    return f"Answer the fixed factual-restraint candidate concisely and helpfully. Current date: {today}. Return JSON only with candidate_id and safe_answer. Avoid invented dates, numbers, names, statuses, causal claims, medical effects, legal outcomes, or market values. {basis}"


def call(client: httpx.Client, key: str, model: str, name: str, schema: dict[str, Any], messages: list[dict[str, str]], max_tokens: int, retries: int) -> dict[str, Any]:
    payload = {"model": model, "messages": messages, "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": copy.deepcopy(schema)}}, "provider": {"require_parameters": True, "allow_fallbacks": False}, "temperature": 0.3, "max_tokens": max_tokens}
    last = ""
    for attempt in range(retries + 1):
        r = client.post(ENDPOINT, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/tohio/slm-synthetic-data", "X-Title": "slm seeded qualification"}, json=payload)
        if r.status_code == 200:
            d = r.json()
            return {"parsed": json.loads(d["choices"][0]["message"]["content"]), "model_returned": d.get("model"), "provider": d.get("provider"), "usage": d.get("usage", {}), "retry_count": attempt}
        last = f"HTTP {r.status_code}: {r.text}"
        if r.status_code in {429, 502, 503, 504} and attempt < retries:
            time.sleep(3 * 2**attempt)
        else:
            break
    raise RuntimeError(last or "request failed")


def eval_int(expr: str) -> int:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool): return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return -walk(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if type(node.op) in ops: return ops[type(node.op)](left, right)
            if isinstance(node.op, ast.Div) and right and left % right == 0: return left // right
        raise ValueError("unsupported or non-exact verification expression")
    return walk(ast.parse(expr, mode="eval"))


def validate_numeric(response: dict[str, Any], choices: list[str] | None = None, steps: bool = False) -> dict[str, Any]:
    a, v, e = str(response.get("answer", "")), str(response.get("verification_answer", "")), str(response.get("verification_expression", ""))
    rejected = not a.strip() and not v.strip() and not e.strip()
    out: dict[str, Any] = {"model_rejected": rejected, "deterministic_pass": False}
    if rejected: return out
    try:
        value = eval_int(e)
        valid = str(int(a)) == a and str(int(v)) == v and a == v == str(value)
        if choices is not None: valid = valid and a in choices
        if steps: valid = valid and isinstance(response.get("steps"), list) and 2 <= len(response["steps"]) <= 4
        out.update({"evaluated_expression": value, "deterministic_pass": valid})
    except Exception as exc:
        out["error"] = str(exc)
    return out


def validate_task(response: dict[str, Any]) -> dict[str, Any]:
    code, plan = response.get("code", ""), response.get("plan", [])
    out: dict[str, Any] = {"model_rejected": not str(code).strip() and plan == [], "deterministic_pass": False, "mutation_calls_for_review": []}
    if out["model_rejected"]: return out
    try:
        tree = ast.parse(code)
        no_imports = not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
        one_function = len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        short = len([line for line in code.splitlines() if line.strip()]) <= 20
        valid_plan = isinstance(plan, list) and 2 <= len(plan) <= 4
        methods = {"append", "extend", "insert", "pop", "remove", "reverse", "sort", "clear", "update", "setdefault"}
        out["mutation_calls_for_review"] = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in methods]
        out.update({"syntax_valid": True, "no_imports": no_imports, "one_function_only": one_function, "under_20_lines": short, "plan_2_to_4": valid_plan, "deterministic_pass": no_imports and one_function and short and valid_plan})
    except SyntaxError as exc:
        out["error"] = f"syntax error: {exc.msg}"
    return out


def validate(signal: str, candidate: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    if signal == "arithmetic": return validate_numeric(response, steps=True)
    if signal == "educational_qa_mcq_math": return validate_numeric(response, choices=candidate.get("choices", []))
    if signal == "task_code": return validate_task(response)
    if signal == "educational_qa_mcq_general":
        choices, answer, explanation = candidate.get("choices", []), response.get("answer", ""), response.get("explanation", "")
        valid = isinstance(choices, list) and len(choices) == 4 and len(set(choices)) == 4 and answer in choices and bool(str(explanation).strip())
        return {"model_rejected": not str(answer).strip() and not str(explanation).strip(), "deterministic_pass": valid, "requires_manual_semantic_review": True}
    answer = response.get("safe_answer", "")
    return {"model_rejected": False, "deterministic_pass": bool(str(answer).strip()), "requires_manual_semantic_review": True}


def candidate_text(signal: str, c: dict[str, Any]) -> str:
    return c.get("task", "") if signal == "task_code" else c.get("question", "") + "\n" + "\n".join(c.get("choices", []))


def report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = ["# Seeded OpenRouter Qualification Review", "", "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "", "## Generated Records", ""]
    for r in records:
        lines += [f"### {r['model_requested']} — {r['signal']} — {r['family']}", "", f"Status: `{r['status']}`", ""]
        if r["status"] == "completed":
            lines += ["**Candidate**", "", "```json", json.dumps(r["candidate"], indent=2, ensure_ascii=False), "```", "", "**Response**", "", "```json", json.dumps(r["response"], indent=2, ensure_ascii=False), "```", "", "**Validation**", "", "```json", json.dumps(r["validation"], indent=2), "```", ""]
        else:
            lines += [f"Error: `{r.get('error', '')}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--signals", nargs="+", choices=SIGNALS, default=SIGNALS)
    p.add_argument("--samples-per-seed", type=int, default=1, choices=range(1, 6))
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--output-root", default="logs")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    rows_per_model = sum(len(SEEDS[s]) * a.samples_per_seed for s in a.signals)
    plan = {"models": a.models, "signals": a.signals, "seed_counts": {s: len(SEEDS[s]) for s in a.signals}, "samples_per_seed": a.samples_per_seed, "rows_per_model": rows_per_model, "total_two_call_rows": rows_per_model * len(a.models), "api_requests_without_retries": rows_per_model * len(a.models) * 2}
    print(json.dumps(plan, indent=2))
    if a.dry_run: return 0
    load_dotenv(dotenv_path=".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key: raise SystemExit("OPENROUTER_API_KEY missing from .env")
    today, stamp = date.today().isoformat(), datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(a.output_root) / f"{stamp}_seeded_openrouter_all_signals"
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_rows, row_id = plan["total_two_call_rows"], 0
    with httpx.Client(timeout=180.0) as client:
        for model in a.models:
            for signal in a.signals:
                for seed in SEEDS[signal]:
                    for sample_i in range(a.samples_per_seed):
                        row_id += 1
                        print(f"[{row_id}/{total_rows}] {model} | {signal} | {seed.family}")
                        r: dict[str, Any] = {"row_id": row_id, "model_requested": model, "signal": signal, "family": seed.family, "seed_id": seed.seed_id, "sample_index": sample_i, "status": "failed"}
                        try:
                            cr = call(client, key, model, f"{signal}_candidate", SCHEMAS[signal]["candidate"], [{"role": "system", "content": candidate_prompt(seed, today)}, {"role": "user", "content": f"Generate one new candidate for seed {seed.seed_id}."}], 800 if signal == "task_code" else 550, a.max_retries)
                            candidate = cr["parsed"]
                            rr = call(client, key, model, f"{signal}_response", SCHEMAS[signal]["response"], [{"role": "system", "content": response_prompt(seed, today)}, {"role": "user", "content": "Answer this fixed candidate without altering it:\n" + json.dumps({"candidate_id": row_id, **candidate}, ensure_ascii=False)}], 1100 if signal == "task_code" else 650, a.max_retries)
                            response = rr["parsed"]
                            if response.get("candidate_id") != row_id: raise ValueError("candidate_id mismatch")
                            cu, ru = cr.get("usage", {}), rr.get("usage", {})
                            usage = {"prompt_tokens": cu.get("prompt_tokens", 0)+ru.get("prompt_tokens", 0), "completion_tokens": cu.get("completion_tokens", 0)+ru.get("completion_tokens", 0), "total_tokens": cu.get("total_tokens", 0)+ru.get("total_tokens", 0), "cost": cu.get("cost", 0.0)+ru.get("cost", 0.0)}
                            r.update({"status": "completed", "model_returned": cr.get("model_returned"), "provider": cr.get("provider"), "candidate": candidate, "response": response, "validation": validate(signal, candidate, response), "combined_usage": usage, "candidate_retry_count": cr.get("retry_count", 0), "response_retry_count": rr.get("retry_count", 0), "candidate_text_normalized": re.sub(r"\s+", " ", candidate_text(signal, candidate).strip().lower())})
                        except Exception as exc:
                            r["error"] = str(exc)
                        records.append(r)
                        with (out / "records.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by: dict[str, dict[str, Any]] = defaultdict(dict)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records: grouped[(r["model_requested"], r["signal"])].append(r)
    for (model, signal), items in grouped.items():
        done = [r for r in items if r["status"] == "completed"]
        texts = [r.get("candidate_text_normalized", "") for r in done]
        by[model][signal] = {"attempted": len(items), "completed": len(done), "deterministic_pass": sum(1 for r in done if r["validation"].get("deterministic_pass")), "model_rejected": sum(1 for r in done if r["validation"].get("model_rejected")), "exact_duplicate_rows": sum(n-1 for n in Counter(texts).values() if n > 1), "unique_candidates": len(set(texts)), "total_tokens": sum(r["combined_usage"]["total_tokens"] for r in done), "cost": sum(r["combined_usage"]["cost"] for r in done), "manual_review_needed": signal in {"task_code", "educational_qa_mcq_general", "factual_restraint"}}
    summary = {"experiment": "seeded_openrouter_all_signals", "timestamp_utc": stamp, "current_date_injected": today, "plan": plan, "by_model_and_signal": by, "notes": ["Standalone harness; does not modify repository code or prompts.", "Concise seeded prompts only; no chain-of-thought/few-shot blocks.", "Deterministic checks are not semantic approval for task_code, educational_qa_mcq_general, or factual_restraint; inspect review.md."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(out / "review.md", summary, records)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out / 'records.jsonl'}")
    print(f"Saved: {out / 'summary.json'}")
    print(f"Saved: {out / 'review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
