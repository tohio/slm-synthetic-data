"""Lightweight response gates for distillation SFT rows."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slm_synth.distillation_sft.prompt_quality import normalize_prompt_text
from slm_synth.distillation_sft.signals import validate_signal

RESPONSE_QUALITY_CHECKS = (
    "non_empty_response",
    "minimum_response_length",
    "prompt_repeat",
    "schema_leakage",
    "unexpected_refusal",
    "arithmetic_answer",
    "code_function_shape",
    "code_python_syntax",
    "code_required_function_name",
    "database_query_shape",
    "factual_restraint",
)

_SCHEMA_FIELD_RE = re.compile(r'"(?:id|prompt|reasoning|response|metadata)"\s*:')
_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"i\s+can(?:not|'?t)|"
    r"i\s+am\s+unable|i'm\s+unable|"
    r"unable\s+to|"
    r"can(?:not|'?t)\s+provide|"
    r"i\s+do\s+not\s+have\s+access|i\s+don't\s+have\s+access|"
    r"not\s+enough\s+information|insufficient\s+information|"
    r"as\s+an\s+ai"
    r")\b",
)
_RESTRAINT_RE = re.compile(
    r"\b(?:"
    r"can(?:not|'?t)|do\s+not|don't|should\s+not|will\s+not|won't|"
    r"unverifiable|unknown|not\s+publicly|not\s+available|unavailable|"
    r"not\s+enough|insufficient|no\s+reliable|avoid\s+inventing|"
    r"cannot\s+determine|private|no\s+source"
    r")\b",
)
_ARITHMETIC_RE = re.compile(r"(-?\d+)\s*([+\-*/x×÷])\s*(-?\d+)")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_NAMED_FUNCTION_RE = re.compile(r"\bfunction\s+named\s+([A-Za-z_]\w*)\b", re.IGNORECASE)
_WORD_PROBLEM_PATTERNS = (
    (re.compile(r"\bhas\s+(\d+)\s+pencils\b.*\binto\s+(\d+)\s+bags\b", re.IGNORECASE), "/"),
    (re.compile(r"\bhas\s+(\d+)\s+shelves\b.*\bwith\s+(\d+)\s+books\b", re.IGNORECASE), "*"),
    (re.compile(r"\bbuys\s+(\d+)\s+packs\b.*\bwith\s+(\d+)\s+markers\b", re.IGNORECASE), "*"),
    (re.compile(r"\bhas\s+(\d+)\s+cars\b.*\bwith\s+(\d+)\s+seats\b", re.IGNORECASE), "*"),
    (re.compile(r"\buses\s+(\d+)\s+cups\b.*\bfor\s+(\d+)\s+loaves\b", re.IGNORECASE), "*"),
    (re.compile(r"\bcompletes\s+(\d+)\s+laps\b.*\bof\s+(\d+)\s+meters\b", re.IGNORECASE), "*"),
    (re.compile(r"\bships\s+(\d+)\s+boxes\b.*\bwith\s+(\d+)\s+items\b", re.IGNORECASE), "*"),
    (re.compile(r"\bhas\s+(\d+)\s+students\b.*\binto\s+(\d+)\s+groups\b", re.IGNORECASE), "/"),
    (
        re.compile(
            r"\bsold\s+(\d+)\s+morning\s+tickets\b.*\band\s+(\d+)\s+afternoon\s+tickets\b",
            re.IGNORECASE,
        ),
        "+",
    ),
)


@dataclass(frozen=True)
class ResponseQualitySummary:
    """Summary of lightweight response-quality checks."""

    checked_rows: int
    accepted_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for manifests."""
        return {
            "checked_rows": self.checked_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "checks": list(RESPONSE_QUALITY_CHECKS),
        }


def filter_public_rows_by_response_quality(
    *,
    signal: str,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], ResponseQualitySummary]:
    """Return accepted public rows plus a rejection summary.

    These gates run after teacher output merging and before final public JSONL
    writing. They are intentionally cheap and deterministic; they do not call a
    model and they do not retry rejected rows.
    """
    normalized_signal = validate_signal(signal)
    accepted: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    checked_rows = 0

    for row in rows:
        checked_rows += 1
        normalized_row = dict(row)
        reasons = validate_response_quality(signal=normalized_signal, row=normalized_row)
        if reasons:
            reason_counts.update(reasons)
            continue
        accepted.append(normalized_row)

    summary = ResponseQualitySummary(
        checked_rows=checked_rows,
        accepted_rows=len(accepted),
        rejected_rows=checked_rows - len(accepted),
        rejection_reasons=dict(reason_counts),
    )
    return accepted, summary


def validate_response_quality(*, signal: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return rejection reasons for one merged public row candidate."""
    normalized_signal = validate_signal(signal)
    prompt = _string_value(row, "prompt")
    response = _string_value(row, "response")
    stripped = response.strip()

    reasons: list[str] = []
    if not stripped:
        reasons.append("empty_response")
    elif _is_too_short_response(signal=normalized_signal, response=stripped):
        reasons.append("too_short_response")

    if prompt and stripped and _repeats_prompt(prompt=prompt, response=stripped):
        reasons.append("repeats_prompt")

    if stripped and _has_schema_leakage(stripped):
        reasons.append("schema_leakage")

    if normalized_signal != "factual_restraint" and stripped and _contains_unexpected_refusal(stripped):
        reasons.append("unexpected_refusal")

    if normalized_signal == "arithmetic" and prompt and stripped:
        reasons.extend(_validate_arithmetic_response(prompt=prompt, response=stripped))
    elif normalized_signal == "code" and prompt and stripped:
        reasons.extend(_validate_code_response(prompt=prompt, response=stripped))
    elif normalized_signal == "database" and prompt and stripped:
        reasons.extend(_validate_database_response(prompt=prompt, response=stripped))
    elif normalized_signal == "factual_restraint" and stripped:
        reasons.extend(_validate_factual_restraint_response(stripped))

    return tuple(dict.fromkeys(reasons))


def is_response_machine_verified(*, signal: str, row: Mapping[str, Any]) -> bool:
    """Return whether a response is independently proven correct.

    This is deliberately narrower than the response-quality gates. Passing a
    heuristic quality check is not proof of correctness. At present, only an
    exact integer answer to a parseable arithmetic expression is eligible.
    """
    normalized_signal = validate_signal(signal)
    if normalized_signal != "arithmetic":
        return False

    prompt = _string_value(row, "prompt")
    response = _string_value(row, "response").strip()
    expected = _expected_arithmetic_result(prompt)
    return expected is not None and response == str(expected)


def aggregate_rejection_reasons(summaries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Aggregate rejection-reason counters from manifest metadata summaries."""
    counts: Counter[str] = Counter()
    for summary in summaries:
        raw_counts = summary.get("rejection_reasons", {})
        if not isinstance(raw_counts, Mapping):
            continue
        for reason, count in raw_counts.items():
            if isinstance(reason, str) and isinstance(count, int) and count > 0:
                counts[reason] += count
    return dict(sorted(counts.items()))


def _string_value(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    return value if isinstance(value, str) else ""


def _is_too_short_response(*, signal: str, response: str) -> bool:
    if signal == "arithmetic" and _INTEGER_RE.fullmatch(response):
        return False
    return len(response.strip()) < 3


def _repeats_prompt(*, prompt: str, response: str) -> bool:
    normalized_prompt = normalize_prompt_text(prompt)
    normalized_response = normalize_prompt_text(response)
    if not normalized_prompt or not normalized_response:
        return False
    if normalized_prompt == normalized_response:
        return True
    if len(normalized_prompt) >= 24 and normalized_response.startswith(normalized_prompt):
        remainder = normalized_response[len(normalized_prompt) :].strip()
        return len(remainder) < max(12, len(normalized_prompt) // 5)
    return False


def _has_schema_leakage(response: str) -> bool:
    stripped = response.strip()
    if _SCHEMA_FIELD_RE.search(stripped):
        return True
    if stripped[:1] in {"{", "["}:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return True
        return isinstance(parsed, (dict, list))
    return False


def _contains_unexpected_refusal(response: str) -> bool:
    return bool(_REFUSAL_RE.search(response.casefold()))


def _validate_arithmetic_response(*, prompt: str, response: str) -> list[str]:
    expected = _expected_arithmetic_result(prompt)
    if expected is None:
        return []

    reasons: list[str] = []
    answer_only = "only the integer result" in prompt.casefold()
    if answer_only and not _INTEGER_RE.fullmatch(response.strip()):
        reasons.append("arithmetic_non_integer_answer")
        return reasons

    if str(expected) not in re.findall(r"[+-]?\d+", response):
        reasons.append("arithmetic_wrong_answer")
    return reasons


def _expected_arithmetic_result(prompt: str) -> int | None:
    match = _ARITHMETIC_RE.search(prompt)
    if match:
        return _apply_integer_operation(
            left=int(match.group(1)),
            op=match.group(2),
            right=int(match.group(3)),
        )
    for pattern, op in _WORD_PROBLEM_PATTERNS:
        word_match = pattern.search(prompt)
        if word_match:
            return _apply_integer_operation(
                left=int(word_match.group(1)),
                op=op,
                right=int(word_match.group(2)),
            )
    return None


def _apply_integer_operation(*, left: int, op: str, right: int) -> int | None:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op in {"*", "x", "×"}:
        return left * right
    if op in {"/", "÷"}:
        if right == 0 or left % right != 0:
            return None
        return left // right
    return None


def _validate_code_response(*, prompt: str, response: str) -> list[str]:
    prompt_lower = prompt.casefold()
    reasons: list[str] = []
    has_markdown_fence = "```" in response
    if has_markdown_fence:
        reasons.append("markdown_fence")
    asks_for_function = "python function" in prompt_lower
    has_function_shape = bool(re.search(r"\bdef\s+\w+\s*\(|\blambda\b", response))
    if asks_for_function and not has_function_shape:
        reasons.append("code_missing_function_definition")
        return reasons
    if has_function_shape and not has_markdown_fence:
        try:
            tree = ast.parse(response)
        except SyntaxError:
            reasons.append("code_invalid_python_syntax")
            return reasons
        required_name = _NAMED_FUNCTION_RE.search(prompt)
        if required_name:
            defined_names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if required_name.group(1) not in defined_names:
                reasons.append("code_wrong_function_name")
    return reasons


def _validate_database_response(*, prompt: str, response: str) -> list[str]:
    prompt_lower = prompt.casefold()
    response_lower = response.casefold()
    asks_for_query = any(
        marker in prompt_lower
        for marker in ("sql query", "write a sql", "write a query", "write a select", "write a sql query")
    )
    if asks_for_query and not ("select" in response_lower and "from" in response_lower):
        return ["database_missing_sql_query"]
    return []


def _validate_factual_restraint_response(response: str) -> list[str]:
    if _RESTRAINT_RE.search(response.casefold()):
        return []
    return ["factual_restraint_missing_restraint"]
