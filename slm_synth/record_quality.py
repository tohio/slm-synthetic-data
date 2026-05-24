from __future__ import annotations

import ast
from fractions import Fraction
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

EXPECTED_KEYS: dict[str, set[str]] = {
    "arithmetic": {"type", "question", "steps", "answer"},
    "task_code": {"type", "task", "plan", "code"},
    "educational_qa_mcq_math": {"type", "question", "choices", "correct_index", "explanation"},
    "educational_qa_mcq_general": {"type", "evidence", "question", "choices", "correct_index", "explanation"},
    "factual_restraint": {"type", "question", "safe_answer"},
}

ARITHMETIC_VERIFICATION_KEYS = {"verification_expression", "verification_answer"}
MCQ_VERIFICATION_KEYS = {"verification_expression", "verification_answer"}

ARITHMETIC_META_STEP_MARKERS = (
    "cannot determine",
    "can't determine",
    "without knowing",
    "not enough information",
    "assuming",
    "if initially",
    "question implies",
    "correct the understanding",
    "ambiguous",
    "unclear",
)

MCQ_MATH_META_EXPLANATION_MARKERS = (
    "provided choices",
    "provided options",
    "answer should be based on",
    "different interpretation",
    "error in the question",
    "error in question",
    "question generation",
    "incorrectly generated",
    "seems to be an error",
)

SIGNAL_FROM_FILE = {
    "arithmetic.jsonl": "arithmetic",
    "task_code.jsonl": "task_code",
    "educational_qa_mcq_math.jsonl": "educational_qa_mcq_math",
    "educational_qa_mcq_general.jsonl": "educational_qa_mcq_general",
    "factual_restraint.jsonl": "factual_restraint",
}


def signal_to_filename(signal: str) -> str:
    return f"{signal}.jsonl"


def strip_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_code(code: str) -> str:
    """Normalize code strings emitted by the LLM."""
    code = code.strip()
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n")
    code = code.replace("\\t", "\t")
    return code.strip()


def parse_python_code_cleanly(code: str) -> tuple[bool, str | None]:
    """Return whether generated Python code parses without SyntaxError or SyntaxWarning."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            ast.parse(code)
        if any(issubclass(w.category, SyntaxWarning) for w in caught):
            return False, "task_code_syntax_warning"
        return True, None
    except SyntaxError as exc:
        return False, f"task_code_syntax_error:{exc.msg}"


@dataclass
class ValidationResult:
    ok: bool
    record: dict[str, Any] | None = None
    issues: list[str] = field(default_factory=list)


def _validate_exact_keys(signal: str, row: dict[str, Any], *, extra_expected: set[str] | None = None) -> list[str]:
    expected = set(EXPECTED_KEYS[signal])
    if extra_expected:
        expected.update(extra_expected)
    keys = set(row)
    issues: list[str] = []
    extra = sorted(keys - expected)
    missing = sorted(expected - keys)
    if extra:
        issues.append("unexpected_fields:" + ",".join(extra))
    if missing:
        issues.append("missing_fields:" + ",".join(missing))
    return issues


def validate_arithmetic(row: dict[str, Any], *, require_verification: bool = False) -> ValidationResult:
    extra_expected = ARITHMETIC_VERIFICATION_KEYS if require_verification else None
    issues = _validate_exact_keys("arithmetic", row, extra_expected=extra_expected)
    if row.get("type") != "arithmetic":
        issues.append("bad_type")
    if not strip_text(row.get("question")):
        issues.append("empty_question")

    steps = row.get("steps")
    cleaned_steps: list[str] | None = None
    if not isinstance(steps, list) or not steps:
        issues.append("bad_steps")
    elif not all(isinstance(step, str) and step.strip() for step in steps):
        issues.append("bad_steps")
    else:
        cleaned_steps = [step.strip() for step in steps]
        step_text = normalize_space(" ".join(cleaned_steps))
        if any(marker in step_text for marker in ARITHMETIC_META_STEP_MARKERS):
            issues.append("arithmetic_meta_commentary")

    answer_text = strip_text(row.get("answer"))
    if not answer_text:
        issues.append("empty_answer")

    if require_verification:
        expression = strip_text(row.get("verification_expression"))
        verification_answer = _parse_integer_text(row.get("verification_answer"))
        answer = _parse_integer_text(answer_text)
        computed = _eval_numeric_expression(expression)

        if not expression:
            issues.append("arithmetic_missing_verification_expression")
        elif computed is None:
            issues.append("arithmetic_invalid_verification_expression")
        elif computed.denominator != 1:
            issues.append("arithmetic_non_integer_verified_answer")

        if verification_answer is None:
            issues.append("arithmetic_bad_verification_answer")
        if answer is None:
            issues.append("arithmetic_bad_answer")
        if computed is not None and computed.denominator == 1 and verification_answer is not None:
            if computed.numerator != verification_answer:
                issues.append("arithmetic_verification_answer_mismatch")
        if answer is not None and verification_answer is not None and answer != verification_answer:
            issues.append("arithmetic_answer_mismatch")

    if issues or cleaned_steps is None:
        return ValidationResult(False, None, issues)
    return ValidationResult(True, {
        "type": "arithmetic",
        "question": row["question"].strip(),
        "steps": cleaned_steps,
        "answer": answer_text,
    }, [])


def validate_task_code(row: dict[str, Any], *, require_syntax: bool = True) -> ValidationResult:
    issues = _validate_exact_keys("task_code", row)
    if row.get("type") != "task_code":
        issues.append("bad_type")
    if not strip_text(row.get("task")):
        issues.append("empty_task")
    if not isinstance(row.get("plan"), list) or not row.get("plan"):
        issues.append("bad_plan")
    elif not all(isinstance(step, str) and step.strip() for step in row["plan"]):
        issues.append("bad_plan")
    if not isinstance(row.get("code"), str) or not row.get("code", "").strip():
        issues.append("empty_code")
        normalized_code = ""
    else:
        normalized_code = normalize_code(row["code"])
    if require_syntax and normalized_code:
        syntax_ok, syntax_issue = parse_python_code_cleanly(normalized_code)
        if not syntax_ok and syntax_issue:
            issues.append(syntax_issue)
    if issues:
        return ValidationResult(False, None, issues)
    return ValidationResult(True, {
        "type": "task_code",
        "task": row["task"].strip(),
        "plan": [step.strip() for step in row["plan"]],
        "code": normalized_code,
    }, [])


def _validate_mcq_base(
    signal: str,
    row: dict[str, Any],
    *,
    extra_expected: set[str] | None = None,
    validate_index: bool = True,
) -> tuple[list[str], list[str] | None, int | None, str]:
    issues = _validate_exact_keys(signal, row, extra_expected=extra_expected)
    if row.get("type") != signal:
        issues.append("bad_type")
    if not strip_text(row.get("question")):
        issues.append("empty_question")
    choices = row.get("choices")
    cleaned_choices: list[str] | None = None
    if not isinstance(choices, list):
        issues.append("bad_choices")
    elif len(choices) != 4:
        issues.append("choices_not_four")
    elif not all(isinstance(choice, str) and choice.strip() for choice in choices):
        issues.append("empty_choice")
    else:
        cleaned_choices = [choice.strip() for choice in choices]
        if len({normalize_space(choice) for choice in cleaned_choices}) != 4:
            issues.append("duplicate_choices")
    idx = row.get("correct_index")
    cleaned_index: int | None = None
    if validate_index:
        if isinstance(idx, bool) or not isinstance(idx, int):
            issues.append("bad_correct_index_type")
        elif not (0 <= idx < 4):
            issues.append("correct_index_out_of_bounds")
        else:
            cleaned_index = idx
    explanation = strip_text(row.get("explanation"))
    if not explanation:
        issues.append("empty_explanation")
    return issues, cleaned_choices, cleaned_index, explanation


def _parse_integer_text(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d+", normalized):
        return None
    return int(normalized)


def _eval_numeric_expression(expression: Any) -> Fraction | None:
    """Safely evaluate +, -, *, and / over integer literals only."""
    if not isinstance(expression, str):
        return None
    expression = expression.strip()
    if not expression or len(expression) > 120:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError("unsupported expression")

    try:
        return evaluate(tree)
    except (ValueError, ZeroDivisionError):
        return None


def _contains_integer(text: str, value: int) -> bool:
    pattern = rf"(?<!\d){re.escape(str(value))}(?!\d)"
    return bool(re.search(pattern, text.replace(",", "")))


def _contains_math_meta_commentary(explanation: str) -> bool:
    normalized = normalize_space(explanation)
    return any(marker in normalized for marker in MCQ_MATH_META_EXPLANATION_MARKERS)


def validate_educational_qa_mcq_math(row: dict[str, Any], *, require_verification: bool = False) -> ValidationResult:
    extra_expected = MCQ_VERIFICATION_KEYS if require_verification else None
    issues, choices, supplied_index, explanation = _validate_mcq_base(
        "educational_qa_mcq_math",
        row,
        extra_expected=extra_expected,
        validate_index=not require_verification,
    )
    corrected_index = supplied_index

    if require_verification:
        expression = strip_text(row.get("verification_expression"))
        verification_answer = _parse_integer_text(row.get("verification_answer"))
        computed = _eval_numeric_expression(expression)
        if not expression:
            issues.append("mcq_missing_verification_expression")
        elif computed is None:
            issues.append("mcq_invalid_verification_expression")
        elif computed.denominator != 1:
            issues.append("mcq_non_integer_verified_answer")
        if verification_answer is None:
            issues.append("mcq_bad_verification_answer")
        if computed is not None and computed.denominator == 1 and verification_answer is not None and computed.numerator != verification_answer:
            issues.append("mcq_verification_answer_mismatch")
        if choices is not None:
            numeric_choices = [_parse_integer_text(choice) for choice in choices]
            if any(choice is None for choice in numeric_choices):
                issues.append("mcq_non_integer_choice")
            elif verification_answer is not None:
                matching_indices = [index for index, choice in enumerate(numeric_choices) if choice == verification_answer]
                if not matching_indices:
                    issues.append("mcq_verified_answer_missing_from_choices")
                elif len(matching_indices) > 1:
                    issues.append("mcq_verified_answer_not_unique")
                else:
                    corrected_index = matching_indices[0]
        if verification_answer is not None and explanation and not _contains_integer(explanation, verification_answer):
            issues.append("mcq_explanation_missing_answer")

    if explanation and _contains_math_meta_commentary(explanation):
        issues.append("mcq_math_meta_commentary")

    if issues or choices is None or corrected_index is None:
        return ValidationResult(False, None, issues)
    return ValidationResult(True, {
        "type": "educational_qa_mcq_math",
        "question": row["question"].strip(),
        "choices": choices,
        "correct_index": corrected_index,
        "explanation": explanation,
    }, [])


def validate_educational_qa_mcq_general(row: dict[str, Any]) -> ValidationResult:
    issues, choices, idx, explanation = _validate_mcq_base("educational_qa_mcq_general", row)
    evidence = strip_text(row.get("evidence"))
    if not evidence:
        issues.append("empty_evidence")
    if issues or choices is None or idx is None:
        return ValidationResult(False, None, issues)
    return ValidationResult(True, {
        "type": "educational_qa_mcq_general",
        "evidence": evidence,
        "question": row["question"].strip(),
        "choices": choices,
        "correct_index": idx,
        "explanation": explanation,
    }, [])


def validate_factual_restraint(row: dict[str, Any]) -> ValidationResult:
    issues = _validate_exact_keys("factual_restraint", row)
    if row.get("type") != "factual_restraint":
        issues.append("bad_type")
    if not strip_text(row.get("question")):
        issues.append("empty_question")
    if not strip_text(row.get("safe_answer")):
        issues.append("empty_safe_answer")
    if issues:
        return ValidationResult(False, None, issues)
    return ValidationResult(True, {
        "type": "factual_restraint",
        "question": row["question"].strip(),
        "safe_answer": row["safe_answer"].strip(),
    }, [])


def validate_record(
    signal: str,
    row: dict[str, Any],
    *,
    require_arithmetic_verification: bool = False,
    require_mcq_verification: bool = False,
) -> ValidationResult:
    if not isinstance(row, dict):
        return ValidationResult(False, None, ["not_object"])
    if signal == "arithmetic":
        has_metadata = bool(ARITHMETIC_VERIFICATION_KEYS.intersection(row))
        return validate_arithmetic(row, require_verification=require_arithmetic_verification or has_metadata)
    if signal == "task_code":
        return validate_task_code(row)
    if signal == "educational_qa_mcq_math":
        has_metadata = bool(MCQ_VERIFICATION_KEYS.intersection(row))
        return validate_educational_qa_mcq_math(row, require_verification=require_mcq_verification or has_metadata)
    if signal == "educational_qa_mcq_general":
        return validate_educational_qa_mcq_general(row)
    if signal == "factual_restraint":
        return validate_factual_restraint(row)
    return ValidationResult(False, None, [f"unknown_signal:{signal}"])


def normalized_factual_qa_key(row: dict[str, Any]) -> tuple[str, str]:
    return (normalize_space(str(row.get("question", ""))), normalize_space(str(row.get("safe_answer", ""))))


def canonical_exact_key(signal: str, row: dict[str, Any]) -> str:
    if signal == "factual_restraint":
        return json.dumps(normalized_factual_qa_key(row), ensure_ascii=False, sort_keys=True)
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class FileAudit:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    bad_json: int = 0
    issues: dict[str, int] = field(default_factory=dict)

    def add_issue(self, issue: str) -> None:
        self.issues[issue] = self.issues.get(issue, 0) + 1


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, list[str]]]:
    with path.open("r", encoding="utf-8") as file:
        for lineno, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                yield lineno, None, ["empty_line"]
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                yield lineno, None, [f"bad_json:{type(exc).__name__}"]
                continue
            if not isinstance(row, dict):
                yield lineno, None, ["not_object"]
                continue
            yield lineno, row, []


def audit_jsonl_file(path: Path, signal: str) -> FileAudit:
    audit = FileAudit()
    for _, row, parse_issues in iter_jsonl(path):
        audit.total += 1
        if parse_issues:
            audit.invalid += 1
            if any(issue.startswith("bad_json") for issue in parse_issues):
                audit.bad_json += 1
            for issue in parse_issues:
                audit.add_issue(issue)
            continue
        assert row is not None
        result = validate_record(signal, row)
        if result.ok:
            audit.valid += 1
        else:
            audit.invalid += 1
            for issue in result.issues:
                audit.add_issue(issue)
    return audit
