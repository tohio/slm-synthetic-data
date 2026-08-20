"""Structured, deterministic output constraints for alignment candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

OUTPUT_CONSTRAINT_FIELDS = frozenset(
    {
        "min_words",
        "max_words",
        "exact_nonempty_lines",
        "min_list_items",
        "max_list_items",
        "exact_list_items",
        "required_terms",
        "forbidden_terms",
        "required_headings",
        "exact_json_keys",
    }
)


class OutputConstraintError(ValueError):
    """Raised when a rendered candidate violates a deterministic constraint."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any]):
        super().__init__(message)
        self.evidence = dict(evidence)


def validate_output_constraints(value: Any) -> dict[str, Any]:
    """Validate and normalize one internal output-constraint declaration."""
    if not isinstance(value, Mapping):
        raise TypeError("output_constraints must be an object")
    extra = sorted(set(value) - OUTPUT_CONSTRAINT_FIELDS)
    if extra:
        raise ValueError(f"output_constraints contains unsupported field(s): {extra}")
    normalized: dict[str, Any] = {}
    for field in (
        "min_words",
        "max_words",
        "exact_nonempty_lines",
        "min_list_items",
        "max_list_items",
        "exact_list_items",
    ):
        if field in value:
            normalized[field] = _positive_int(value[field], f"output_constraints.{field}")
    for field in ("required_terms", "forbidden_terms", "required_headings", "exact_json_keys"):
        if field in value:
            normalized[field] = _string_list(value[field], f"output_constraints.{field}")
    _validate_bounds(normalized, "min_words", "max_words")
    _validate_bounds(normalized, "min_list_items", "max_list_items")
    if "exact_list_items" in normalized and (
        "min_list_items" in normalized or "max_list_items" in normalized
    ):
        raise ValueError(
            "output_constraints.exact_list_items cannot be combined with list-item bounds"
        )
    return normalized


def evaluate_sft_output_constraints(
    *, specs: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Evaluate every SFT row and raise when any declared constraint fails."""
    rows_by_id = {str(row["id"]): row for row in rows}
    evidence: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for spec in specs:
        row_id = str(spec["id"])
        result = evaluate_output_constraints(
            _final_assistant_content(rows_by_id[row_id]["messages"], row_id=row_id),
            spec.get("output_constraints", {}),
        )
        evidence[row_id] = result
        if result["status"] != "passed":
            failures.append(_failure_text(row_id, result))
    if failures:
        raise OutputConstraintError(
            "deterministic output constraint failure(s): " + " | ".join(failures),
            evidence=evidence,
        )
    return evidence


def evaluate_dpo_output_constraints(
    *, specs: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Require the chosen branch to pass and record rejected-branch observations."""
    rows_by_id = {str(row["id"]): row for row in rows}
    evidence: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for spec in specs:
        row_id = str(spec["id"])
        constraints = spec.get("output_constraints", {})
        row = rows_by_id[row_id]
        chosen = evaluate_output_constraints(
            _final_assistant_content(row["chosen"], row_id=row_id), constraints
        )
        rejected = evaluate_output_constraints(
            _final_assistant_content(row["rejected"], row_id=row_id), constraints
        )
        result = {
            "status": chosen["status"],
            "declared_constraint_count": chosen["declared_constraint_count"],
            "chosen": chosen,
            "rejected": rejected,
        }
        evidence[row_id] = result
        if chosen["status"] != "passed":
            failures.append(_failure_text(row_id, chosen))
    if failures:
        raise OutputConstraintError(
            "deterministic chosen-output constraint failure(s): " + " | ".join(failures),
            evidence=evidence,
        )
    return evidence


def evaluate_output_constraints(content: str, raw_constraints: Any) -> dict[str, Any]:
    """Return auditable observations for one final assistant response."""
    constraints = validate_output_constraints(raw_constraints)
    checks: list[dict[str, Any]] = []
    words = _words(content)
    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    list_items = _list_items(content)
    normalized = content.casefold()

    for field, comparison in (
        ("min_words", lambda actual, expected: actual >= expected),
        ("max_words", lambda actual, expected: actual <= expected),
        ("exact_nonempty_lines", lambda actual, expected: actual == expected),
        ("min_list_items", lambda actual, expected: actual >= expected),
        ("max_list_items", lambda actual, expected: actual <= expected),
        ("exact_list_items", lambda actual, expected: actual == expected),
    ):
        if field not in constraints:
            continue
        actual = (
            len(words)
            if field.endswith("words")
            else len(nonempty_lines)
            if field == "exact_nonempty_lines"
            else len(list_items)
        )
        checks.append(_check(field, constraints[field], actual, comparison(actual, constraints[field])))

    for term in constraints.get("required_terms", []):
        checks.append(_check("required_term", term, term.casefold() in normalized, term.casefold() in normalized))
    for term in constraints.get("forbidden_terms", []):
        present = _contains_term(content, term)
        checks.append(_check("forbidden_term", term, present, not present))
    headings = {_normalize_heading(line) for line in nonempty_lines}
    for heading in constraints.get("required_headings", []):
        present = heading.casefold().rstrip(":") in headings
        checks.append(_check("required_heading", heading, present, present))
    if "exact_json_keys" in constraints:
        expected = constraints["exact_json_keys"]
        try:
            parsed = json.loads(_strip_json_fence(content.strip()))
            actual = sorted(parsed) if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            actual = None
        checks.append(_check("exact_json_keys", sorted(expected), actual, actual == sorted(expected)))

    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "declared_constraint_count": len(constraints),
        "checks": checks,
    }


def _final_assistant_content(messages: Sequence[Mapping[str, Any]], *, row_id: str) -> str:
    values = [
        message.get("content")
        for message in messages
        if message.get("role") == "assistant" and isinstance(message.get("content"), str)
    ]
    if not values or not values[-1].strip():
        raise ValueError(f"rendered row {row_id} has no final assistant content")
    return values[-1].strip()


def _words(value: str) -> list[str]:
    return re.findall(r"\b[^\W_]+(?:['’\-][^\W_]+)*\b", value, flags=re.UNICODE)


def _list_items(value: str) -> list[str]:
    stripped = _strip_json_fence(value.strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [
        line
        for line in value.splitlines()
        if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line)
    ]


def _contains_term(content: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, content, flags=re.IGNORECASE) is not None


def _normalize_heading(line: str) -> str:
    return re.sub(r"^\s*#{1,6}\s*", "", line).strip().casefold().rstrip(":")


def _strip_json_fence(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return value


def _check(kind: str, expected: Any, observed: Any, passed: bool) -> dict[str, Any]:
    return {"constraint": kind, "expected": expected, "observed": observed, "passed": passed}


def _failure_text(row_id: str, result: Mapping[str, Any]) -> str:
    failed = [check for check in result["checks"] if check["passed"] is not True]
    details = ", ".join(
        f"{check['constraint']} expected={check['expected']!r} observed={check['observed']!r}"
        for check in failed
    )
    return f"{row_id}: {details}"


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        result.append(item.strip())
    if len(result) != len(set(item.casefold() for item in result)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _validate_bounds(value: Mapping[str, Any], minimum: str, maximum: str) -> None:
    if minimum in value and maximum in value and value[minimum] > value[maximum]:
        raise ValueError(f"output_constraints.{minimum} must not exceed {maximum}")
