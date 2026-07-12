"""Lightweight pair-quality gates for distillation-DPO rows."""

from __future__ import annotations

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
)

_SCHEMA_FIELD_RE = re.compile(r'"(?:id|prompt|chosen|rejected|metadata|role|content)"\s*:')
_WORD_RE = re.compile(r"[a-z0-9]+")


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
