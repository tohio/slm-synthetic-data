"""Prompt preflight checks for distillation SFT teacher calls."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slm_synth.distillation.prompts import validate_prompt_record


@dataclass(frozen=True)
class PromptPreflightSummary:
    """Summary of local prompt checks run before teacher calls."""

    prompt_count: int
    duplicate_id_count: int
    duplicate_prompt_text_count: int
    near_duplicate_prompt_count: int
    require_unique_prompt_text: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for manifests."""
        checks = ["id"]
        if self.require_unique_prompt_text:
            checks.append("normalized_prompt_text")
        return {
            "prompt_count": self.prompt_count,
            "duplicate_id_count": self.duplicate_id_count,
            "duplicate_prompt_text_count": self.duplicate_prompt_text_count,
            "near_duplicate_prompt_count": self.near_duplicate_prompt_count,
            "require_unique_prompt_text": self.require_unique_prompt_text,
            "checks": checks,
        }


def normalize_prompt_text(prompt: str) -> str:
    """Return a stable, cheap near-duplicate key for prompt text.

    This intentionally avoids semantic or numeric rewriting so production prompts
    that share a template but vary the actual task are not treated as duplicates.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    normalized = normalized.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([([{])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([)\]}])", r"\1", normalized)
    normalized = re.sub(r"\s*([+*/=])\s*", r"\1", normalized)
    return normalized.strip(" \t\r\n\"'`.,;:!?")


def validate_prompt_preflight(
    records: Sequence[Mapping[str, Any]],
    *,
    require_unique_prompt_text: bool,
) -> PromptPreflightSummary:
    """Validate prompt ids and, when required, duplicate/near-duplicate prompts.

    The checks are local-only and must run before teacher calls. Built-in smoke
    seeds may intentionally cycle prompt text, so prompt-text uniqueness is an
    explicit caller decision.
    """
    validated_records = [validate_prompt_record(record) for record in records]

    id_groups = _group_record_ids_by_key(validated_records, key_field="id")
    exact_prompt_groups = _group_record_ids_by_key(validated_records, key_field="prompt")
    normalized_prompt_groups = _group_record_ids_by_normalized_prompt(validated_records)

    duplicate_id_groups = _duplicates_only(id_groups)
    duplicate_exact_prompt_groups = _duplicates_only(exact_prompt_groups)
    near_duplicate_prompt_groups = _duplicates_only(normalized_prompt_groups)

    summary = PromptPreflightSummary(
        prompt_count=len(validated_records),
        duplicate_id_count=_extra_duplicate_count(duplicate_id_groups),
        duplicate_prompt_text_count=_extra_duplicate_count(duplicate_exact_prompt_groups),
        near_duplicate_prompt_count=_extra_duplicate_count(near_duplicate_prompt_groups),
        require_unique_prompt_text=require_unique_prompt_text,
    )

    problems: list[str] = []
    if duplicate_id_groups:
        problems.append(f"duplicate id(s): {_format_examples(duplicate_id_groups)}")
    if require_unique_prompt_text and near_duplicate_prompt_groups:
        problems.append(
            "duplicate or near-duplicate prompt text: "
            f"{_format_examples(near_duplicate_prompt_groups)}"
        )
    if problems:
        raise ValueError("distillation prompt preflight failed; " + "; ".join(problems))

    return summary


def _group_record_ids_by_key(records: Sequence[Mapping[str, Any]], *, key_field: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(record[key_field])].append(str(record["id"]))
    return dict(grouped)


def _group_record_ids_by_normalized_prompt(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[normalize_prompt_text(str(record["prompt"]))].append(str(record["id"]))
    return dict(grouped)


def _duplicates_only(groups: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    return {key: list(ids) for key, ids in groups.items() if len(ids) > 1}


def _extra_duplicate_count(groups: Mapping[str, Sequence[str]]) -> int:
    return sum(len(ids) - 1 for ids in groups.values())


def _format_examples(groups: Mapping[str, Sequence[str]], *, limit: int = 5) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for key in sorted(groups)[:limit]:
        display_key = key if len(key) <= 120 else key[:117] + "..."
        examples.append({"key": display_key, "ids": list(groups[key])})
    return examples
