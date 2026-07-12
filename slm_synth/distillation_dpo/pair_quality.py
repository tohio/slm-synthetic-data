"""Lightweight pair-quality gates for distillation-DPO rows."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slm_synth.distillation_dpo.schema import validate_distillation_dpo_row
from slm_synth.distillation_dpo.seeds import validate_family

PAIR_QUALITY_CHECKS = (
    "valid_public_schema",
    "non_empty_pair_text",
    "minimum_pair_text_length",
    "identical_pair",
    "non_contrastive_pair",
    "prompt_copy_pair",
    "schema_leakage",
    "chosen_code_complete",
    "rejected_code_failure_mode",
    "chosen_restraint_concise",
)

MAX_PRIVATE_RESTRAINT_WORDS = 60

_SCHEMA_FIELD_RE = re.compile(r'"(?:id|prompt|chosen|rejected|metadata|role|content)"\s*:')
_WORD_RE = re.compile(r"[a-z0-9]+")
_SIGNATURE_ONLY_RE = re.compile(r"^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^\n]*\)\s*:\s*$")
_EXPECTED_FUNCTION_RE = re.compile(r"\bnamed\s+([A-Za-z_]\w*)\b", re.IGNORECASE)
_FUNCTION_DEF_RE = re.compile(r"\b(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(")


@dataclass(frozen=True)
class PairQualitySummary:
    """Summary of deterministic distillation-DPO pair-quality checks."""

    checked_pairs: int
    accepted_pairs: int
    rejected_pairs: int
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_pairs": self.checked_pairs,
            "accepted_pairs": self.accepted_pairs,
            "rejected_pairs": self.rejected_pairs,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "checks": list(PAIR_QUALITY_CHECKS),
        }


def filter_pairs_by_quality(
    *,
    family: str,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], PairQualitySummary]:
    """Return accepted public rows plus a pair-quality summary."""
    validate_family(family)
    accepted: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    checked_pairs = 0

    for row in rows:
        checked_pairs += 1
        try:
            validated = validate_distillation_dpo_row(row)
        except (TypeError, ValueError):
            reason_counts.update(["malformed_row"])
            continue

        reasons = validate_pair_quality(validated)
        if reasons:
            reason_counts.update(reasons)
            continue
        accepted.append(validated)

    summary = PairQualitySummary(
        checked_pairs=checked_pairs,
        accepted_pairs=len(accepted),
        rejected_pairs=checked_pairs - len(accepted),
        rejection_reasons=dict(reason_counts),
    )
    return accepted, summary


def validate_pair_quality(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return rejection reasons for one validated distillation-DPO row."""
    prompt = _last_user_content(row.get("prompt", []))
    chosen = _assistant_content(row.get("chosen", []))
    rejected = _assistant_content(row.get("rejected", []))

    reasons: list[str] = []
    if not chosen.strip() or not rejected.strip():
        reasons.append("empty_pair_text")
    if _too_short(chosen) or _too_short(rejected):
        reasons.append("too_short_pair_text")

    chosen_norm = normalize_pair_text(chosen)
    rejected_norm = normalize_pair_text(rejected)
    prompt_norm = normalize_pair_text(prompt)

    if chosen_norm and rejected_norm and chosen_norm == rejected_norm:
        reasons.append("identical_pair")
    elif _is_non_contrastive(chosen_norm, rejected_norm):
        reasons.append("non_contrastive_pair")

    if prompt_norm and (_copies_prompt(prompt_norm, chosen_norm) or _copies_prompt(prompt_norm, rejected_norm)):
        reasons.append("prompt_copy_pair")

    if _has_schema_leakage(chosen) or _has_schema_leakage(rejected):
        reasons.append("schema_leakage")

    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping):
        category = metadata.get("category")
        failure_mode = metadata.get("failure_mode")
        if category == "code_generation":
            reasons.extend(
                _validate_code_generation_pair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    failure_mode=failure_mode,
                )
            )
        if category == "private_info_restraint" and _word_count(chosen) > MAX_PRIVATE_RESTRAINT_WORDS:
            reasons.append("chosen_restraint_too_verbose")

    return tuple(dict.fromkeys(reasons))


def aggregate_rejection_reasons(summaries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Aggregate rejection-reason counters from pair-quality summaries."""
    counts: Counter[str] = Counter()
    for summary in summaries:
        raw_counts = summary.get("rejection_reasons", {})
        if not isinstance(raw_counts, Mapping):
            continue
        for reason, count in raw_counts.items():
            if isinstance(reason, str) and isinstance(count, int) and count > 0:
                counts[reason] += count
    return dict(sorted(counts.items()))


def normalize_pair_text(text: str) -> str:
    """Normalize response text for cheap contrast checks."""
    if not isinstance(text, str):
        return ""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[`*_#>\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _assistant_content(messages: Any) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    contents = [message.get("content", "") for message in messages if isinstance(message, Mapping)]
    return "\n".join(content for content in contents if isinstance(content, str))


def _last_user_content(messages: Any) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _too_short(text: str) -> bool:
    stripped = text.strip()
    if re.fullmatch(r"[+-]?\d+", stripped):
        return False
    return len(stripped) < 2


def _is_non_contrastive(chosen_norm: str, rejected_norm: str) -> bool:
    if not chosen_norm or not rejected_norm:
        return False
    chosen_words = set(_WORD_RE.findall(chosen_norm))
    rejected_words = set(_WORD_RE.findall(rejected_norm))
    if len(chosen_words | rejected_words) < 6:
        return False
    overlap = len(chosen_words & rejected_words) / len(chosen_words | rejected_words)
    length_delta = abs(len(chosen_norm) - len(rejected_norm)) / max(len(chosen_norm), len(rejected_norm))
    return overlap >= 0.92 and length_delta <= 0.08


def _copies_prompt(prompt_norm: str, response_norm: str) -> bool:
    if not prompt_norm or not response_norm:
        return False
    if prompt_norm == response_norm:
        return True
    if len(prompt_norm) >= 24 and response_norm.startswith(prompt_norm):
        remainder = response_norm[len(prompt_norm) :].strip()
        return len(remainder) < max(12, len(prompt_norm) // 5)
    return False


def _has_schema_leakage(text: str) -> bool:
    stripped = text.strip()
    if _SCHEMA_FIELD_RE.search(stripped):
        return True
    if stripped[:1] in {"{", "["}:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return True
        return isinstance(parsed, (dict, list))
    return False


def _validate_code_generation_pair(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    failure_mode: Any,
) -> list[str]:
    reasons: list[str] = []
    stripped_chosen = chosen.strip()
    if _SIGNATURE_ONLY_RE.fullmatch(stripped_chosen):
        return ["chosen_code_signature_only"]

    try:
        chosen_tree = ast.parse(stripped_chosen)
    except SyntaxError:
        return ["chosen_code_syntax_error"]

    functions = [
        node
        for node in ast.walk(chosen_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    expected_match = _EXPECTED_FUNCTION_RE.search(prompt)
    if not functions:
        reasons.append("chosen_code_missing_function")
    elif expected_match is not None:
        expected_name = expected_match.group(1)
        expected_functions = [function for function in functions if function.name == expected_name]
        if not expected_functions:
            reasons.append("chosen_code_wrong_function_name")
        elif not any(_function_has_implementation(function) for function in expected_functions):
            reasons.append("chosen_code_incomplete_function")
    elif not any(_function_has_implementation(function) for function in functions):
        reasons.append("chosen_code_incomplete_function")

    if failure_mode == "code_syntax_error":
        stripped_rejected = rejected.strip()
        if not _FUNCTION_DEF_RE.search(stripped_rejected):
            reasons.append("rejected_code_not_python_function")
        else:
            try:
                ast.parse(stripped_rejected)
            except SyntaxError:
                pass
            else:
                reasons.append("rejected_code_missing_expected_syntax_error")
    return reasons


def _function_has_implementation(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for statement in function.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr):
            value = statement.value
            if isinstance(value, ast.Constant) and (isinstance(value.value, str) or value.value is Ellipsis):
                continue
        if isinstance(statement, ast.Raise) and _raises_not_implemented(statement):
            continue
        return True
    return False


def _raises_not_implemented(statement: ast.Raise) -> bool:
    value = statement.exc
    if isinstance(value, ast.Name):
        return value.id == "NotImplementedError"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id == "NotImplementedError"
    return False


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text.casefold()))
