"""Dataset-wide content acceptance helpers for generic DPO rows."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any

from slm_synth.dpo.schema import validate_dpo_row


def normalize_dpo_text(value: str) -> str:
    """Normalize message text for exact content comparisons."""
    if not isinstance(value, str):
        raise TypeError("DPO message content must be a string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip()).casefold()


def dpo_prompt_fingerprint(row: Mapping[str, Any]) -> str:
    validated = validate_dpo_row(row)
    return _row_part_fingerprint(validated["prompt"], validated.get("tools"))


def dpo_triple_fingerprint(row: Mapping[str, Any]) -> str:
    validated = validate_dpo_row(row)
    return json.dumps(
        {
            "prompt": _messages_payload(validated["prompt"]),
            "chosen": _messages_payload(validated["chosen"]),
            "rejected": _messages_payload(validated["rejected"]),
            "tools": _normalize_value(validated.get("tools")),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def partition_unique_dpo_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve the first row for each normalized ID, prompt, and preference triple."""
    validated_rows = [validate_dpo_row(row) for row in rows]
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_triples: set[str] = set()
    reason_counts: Counter[str] = Counter()

    for row in validated_rows:
        normalized_id = normalize_dpo_text(row["id"])
        prompt = dpo_prompt_fingerprint(row)
        triple = dpo_triple_fingerprint(row)
        reasons: list[str] = []
        if normalized_id in seen_ids:
            reasons.append("duplicate_id")
        if prompt in seen_prompts:
            reasons.append("duplicate_prompt")
        if triple in seen_triples:
            reasons.append("duplicate_triple")
        if reasons:
            reason_counts.update(reasons)
            continue
        accepted.append(row)
        seen_ids.add(normalized_id)
        seen_prompts.add(prompt)
        seen_triples.add(triple)

    return accepted, {
        "attempted_pairs": len(validated_rows),
        "accepted_pairs": len(accepted),
        "duplicate_pairs": len(validated_rows) - len(accepted),
        "duplicate_reason_counts": {
            reason: reason_counts[reason] for reason in sorted(reason_counts)
        },
    }


def build_dpo_content_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize uniqueness, response similarity, and negative construction patterns."""
    validated_rows = [validate_dpo_row(row) for row in rows]
    similarities = [_chosen_rejected_similarity(row) for row in validated_rows]
    patterns = Counter(_negative_pattern(row) for row in validated_rows)
    return {
        "ids": _uniqueness_summary(normalize_dpo_text(row["id"]) for row in validated_rows),
        "prompts": _uniqueness_summary(dpo_prompt_fingerprint(row) for row in validated_rows),
        "triples": _uniqueness_summary(dpo_triple_fingerprint(row) for row in validated_rows),
        "chosen_responses": _uniqueness_summary(
            _messages_fingerprint(row["chosen"]) for row in validated_rows
        ),
        "rejected_responses": _uniqueness_summary(
            _messages_fingerprint(row["rejected"]) for row in validated_rows
        ),
        "chosen_rejected_similarity": {
            "minimum": min(similarities, default=0.0),
            "mean": sum(similarities) / len(similarities) if similarities else 0.0,
            "maximum": max(similarities, default=0.0),
            "at_or_above_0_90": sum(value >= 0.90 for value in similarities),
            "at_or_above_0_98": sum(value >= 0.98 for value in similarities),
        },
        "negative_patterns": {
            "counts": {pattern: patterns[pattern] for pattern in sorted(patterns)},
            "repeated_pattern_count": sum(count > 1 for count in patterns.values()),
            "maximum_repetition": max(patterns.values(), default=0),
        },
    }


def _messages_payload(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _normalize_value(value) for key, value in message.items()} for message in messages]


def _messages_fingerprint(messages: Iterable[Mapping[str, Any]]) -> str:
    return json.dumps(
        _messages_payload(messages),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _chosen_rejected_similarity(row: Mapping[str, Any]) -> float:
    chosen = _branch_text(row["chosen"])
    rejected = _branch_text(row["rejected"])
    return SequenceMatcher(None, chosen, rejected, autojunk=False).ratio()


def _negative_pattern(row: Mapping[str, Any]) -> str:
    if any(message.get("tool_calls") or message["role"] == "tool" for branch in (row["chosen"], row["rejected"]) for message in branch):
        return "tool_use_branch"
    chosen = _branch_text(row["chosen"])
    rejected = _branch_text(row["rejected"])
    if chosen in rejected:
        return "chosen_verbatim_with_extra"
    if rejected in chosen:
        return "truncated_chosen"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", chosen) and re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?", rejected
    ):
        return "numeric_substitution"
    if rejected.startswith(("i cannot", "i can't", "sorry", "as an ai")):
        return "refusal_or_disclaimer"
    return "other"


def _row_part_fingerprint(messages: Iterable[Mapping[str, Any]], tools: Any) -> str:
    return json.dumps(
        {"messages": _messages_payload(messages), "tools": _normalize_value(tools)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _branch_text(messages: Iterable[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return normalize_dpo_text(" ".join(parts))


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_dpo_text(value)
    if isinstance(value, Mapping):
        return {key: _normalize_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _uniqueness_summary(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    unique = len(counts)
    return {
        "total": total,
        "unique": unique,
        "duplicate_count": total - unique,
        "unique_ratio": unique / total if total else 1.0,
        "repeated_value_count": sum(count > 1 for count in counts.values()),
        "maximum_repetition": max(counts.values(), default=0),
    }
