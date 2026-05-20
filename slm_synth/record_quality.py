from __future__ import annotations

import ast
import json
import warnings
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

EXPECTED_KEYS: dict[str, set[str]] = {
    "arithmetic": {"type", "question", "steps", "answer"},
    "task_code": {"type", "task", "plan", "code"},
    "educational_qa_mcq": {"type", "question", "choices", "correct_index", "explanation"},
    "factual_restraint": {"type", "question", "safe_answer"},
}

SIGNAL_FROM_FILE = {
    "arithmetic.jsonl": "arithmetic",
    "task_code.jsonl": "task_code",
    "educational_qa_mcq.jsonl": "educational_qa_mcq",
    "factual_restraint.jsonl": "factual_restraint",
}


def signal_to_filename(signal: str) -> str:
    return f"{signal}.jsonl"


def strip_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_code(code: str) -> str:
    """Normalize code strings emitted by the LLM.

    Some generations store escaped newlines literally ("\\n") instead of real
    newlines. Those are valid JSON strings but bad Python source until normalized.
    """
    code = code.strip()
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n")
    code = code.replace("\\t", "\t")
    return code.strip()


def parse_python_code_cleanly(code: str) -> tuple[bool, str | None]:
    """Return whether generated Python code parses without SyntaxError or SyntaxWarning.

    SyntaxWarning is treated as invalid for the published task_code signal so
    warning-producing code, such as non-raw regex strings with invalid escape
    sequences, is not exported.
    """
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


def _validate_exact_keys(signal: str, row: dict[str, Any]) -> list[str]:
    expected = EXPECTED_KEYS[signal]
    keys = set(row)
    issues: list[str] = []
    extra = sorted(keys - expected)
    missing = sorted(expected - keys)
    if extra:
        issues.append("unexpected_fields:" + ",".join(extra))
    if missing:
        issues.append("missing_fields:" + ",".join(missing))
    return issues


def validate_arithmetic(row: dict[str, Any]) -> ValidationResult:
    issues = _validate_exact_keys("arithmetic", row)
    if row.get("type") != "arithmetic":
        issues.append("bad_type")
    if not strip_text(row.get("question")):
        issues.append("empty_question")
    if not isinstance(row.get("steps"), list) or not row.get("steps"):
        issues.append("bad_steps")
    elif not all(isinstance(s, str) and s.strip() for s in row["steps"]):
        issues.append("bad_steps")
    if not strip_text(row.get("answer")):
        issues.append("empty_answer")

    if issues:
        return ValidationResult(False, None, issues)
    return ValidationResult(
        True,
        {
            "type": "arithmetic",
            "question": row["question"].strip(),
            "steps": [s.strip() for s in row["steps"]],
            "answer": row["answer"].strip(),
        },
        [],
    )


def validate_task_code(row: dict[str, Any], *, require_syntax: bool = True) -> ValidationResult:
    issues = _validate_exact_keys("task_code", row)
    if row.get("type") != "task_code":
        issues.append("bad_type")
    if not strip_text(row.get("task")):
        issues.append("empty_task")
    if not isinstance(row.get("plan"), list) or not row.get("plan"):
        issues.append("bad_plan")
    elif not all(isinstance(s, str) and s.strip() for s in row["plan"]):
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
    return ValidationResult(
        True,
        {
            "type": "task_code",
            "task": row["task"].strip(),
            "plan": [s.strip() for s in row["plan"]],
            "code": normalized_code,
        },
        [],
    )


def validate_educational_qa_mcq(row: dict[str, Any]) -> ValidationResult:
    issues = _validate_exact_keys("educational_qa_mcq", row)
    if row.get("type") != "educational_qa_mcq":
        issues.append("bad_type")
    if not strip_text(row.get("question")):
        issues.append("empty_question")
    choices = row.get("choices")
    if not isinstance(choices, list):
        issues.append("bad_choices")
    elif len(choices) != 4:
        issues.append("choices_not_four")
    elif not all(isinstance(c, str) and c.strip() for c in choices):
        issues.append("empty_choice")
    idx = row.get("correct_index")
    if not isinstance(idx, int):
        issues.append("bad_correct_index_type")
    elif isinstance(choices, list) and not (0 <= idx < len(choices)):
        issues.append("correct_index_out_of_bounds")
    if not strip_text(row.get("explanation")):
        issues.append("empty_explanation")

    if issues:
        return ValidationResult(False, None, issues)
    return ValidationResult(
        True,
        {
            "type": "educational_qa_mcq",
            "question": row["question"].strip(),
            "choices": [c.strip() for c in row["choices"]],
            "correct_index": int(row["correct_index"]),
            "explanation": row["explanation"].strip(),
        },
        [],
    )


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
    return ValidationResult(
        True,
        {
            "type": "factual_restraint",
            "question": row["question"].strip(),
            "safe_answer": row["safe_answer"].strip(),
        },
        [],
    )


def validate_record(signal: str, row: dict[str, Any]) -> ValidationResult:
    if not isinstance(row, dict):
        return ValidationResult(False, None, ["not_object"])
    if signal == "arithmetic":
        return validate_arithmetic(row)
    if signal == "task_code":
        return validate_task_code(row)
    if signal == "educational_qa_mcq":
        return validate_educational_qa_mcq(row)
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
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
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
            if any(i.startswith("bad_json") for i in parse_issues):
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
