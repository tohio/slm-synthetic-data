"""Dataset-wide content acceptance helpers for generic SFT rows."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from slm_synth.sft.schema import validate_sft_row


def normalize_sft_text(value: str) -> str:
    """Normalize message text for exact content comparisons."""
    if not isinstance(value, str):
        raise TypeError("SFT message content must be a string")
    return re.sub(r"\s+", " ", value.strip().lower())


def sft_prompt_fingerprint(row: Mapping[str, Any]) -> str:
    """Return a normalized fingerprint of all non-assistant prompt messages."""
    validated = validate_sft_row(row)
    return _messages_fingerprint(
        message for message in validated["messages"] if message["role"] != "assistant"
    )


def sft_conversation_fingerprint(row: Mapping[str, Any]) -> str:
    """Return a normalized fingerprint of the complete conversation."""
    validated = validate_sft_row(row)
    return _messages_fingerprint(validated["messages"])


def sft_response_fingerprint(row: Mapping[str, Any]) -> str:
    """Return a normalized fingerprint of the assistant response."""
    validated = validate_sft_row(row)
    assistant_messages = [
        message for message in validated["messages"] if message["role"] == "assistant"
    ]
    return _messages_fingerprint(assistant_messages)


def partition_unique_sft_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve the first row for each ID, prompt, and conversation fingerprint."""
    validated_rows = [validate_sft_row(row) for row in rows]
    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_conversations: set[str] = set()
    duplicate_reason_counts: Counter[str] = Counter()

    for row in validated_rows:
        prompt = sft_prompt_fingerprint(row)
        conversation = sft_conversation_fingerprint(row)
        reasons: list[str] = []
        if row["id"] in seen_ids:
            reasons.append("duplicate_id")
        if prompt in seen_prompts:
            reasons.append("duplicate_prompt")
        if conversation in seen_conversations:
            reasons.append("duplicate_conversation")

        if reasons:
            duplicate_reason_counts.update(reasons)
            continue

        accepted.append(row)
        seen_ids.add(row["id"])
        seen_prompts.add(prompt)
        seen_conversations.add(conversation)

    return accepted, {
        "attempted_rows": len(validated_rows),
        "accepted_rows": len(accepted),
        "duplicate_rows": len(validated_rows) - len(accepted),
        "duplicate_reason_counts": {
            reason: duplicate_reason_counts[reason]
            for reason in sorted(duplicate_reason_counts)
        },
    }


def build_sft_content_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize normalized ID, prompt, conversation, and response repetition."""
    validated_rows = [validate_sft_row(row) for row in rows]
    return {
        "ids": _uniqueness_summary(row["id"] for row in validated_rows),
        "prompts": _uniqueness_summary(sft_prompt_fingerprint(row) for row in validated_rows),
        "conversations": _uniqueness_summary(
            sft_conversation_fingerprint(row) for row in validated_rows
        ),
        "responses": _uniqueness_summary(sft_response_fingerprint(row) for row in validated_rows),
    }


def _messages_fingerprint(messages: Iterable[Mapping[str, str]]) -> str:
    payload = [
        {"role": message["role"], "content": normalize_sft_text(message["content"])}
        for message in messages
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _uniqueness_summary(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(values)
    total = sum(counts.values())
    unique = len(counts)
    repeated_values = sum(1 for count in counts.values() if count > 1)
    return {
        "total": total,
        "unique": unique,
        "duplicate_count": total - unique,
        "unique_ratio": unique / total if total else 1.0,
        "repeated_value_count": repeated_values,
        "maximum_repetition": max(counts.values(), default=0),
    }
