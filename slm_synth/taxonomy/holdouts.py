"""Eval holdout registry loading and matching."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from slm_synth.taxonomy.eval_families import validate_eval_family


def normalize_text(text: str) -> str:
    """Normalize text for exact holdout prompt matching."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", text.strip().lower())


def holdout_key_fingerprint(holdout_key: Mapping[str, Any]) -> str:
    """Return a deterministic fingerprint for a structured holdout key."""
    if not isinstance(holdout_key, Mapping):
        raise TypeError("holdout_key must be an object")
    return json.dumps(holdout_key, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class HoldoutRecord:
    """One exact eval holdout item."""

    id: str
    eval_family: str
    prompt: str
    answer: str | None
    holdout_key: Mapping[str, Any] | None = None

    @property
    def normalized_prompt(self) -> str:
        return normalize_text(self.prompt)

    @property
    def key_fingerprint(self) -> str | None:
        if self.holdout_key is None:
            return None
        return holdout_key_fingerprint(self.holdout_key)


class HoldoutRegistry:
    """Exact prompt and structured-key holdout checker."""

    def __init__(self, records: Iterable[HoldoutRecord]):
        self.records = tuple(records)
        self._prompts = {record.normalized_prompt for record in self.records}
        self._keys = {
            fingerprint
            for record in self.records
            for fingerprint in [record.key_fingerprint]
            if fingerprint is not None
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "HoldoutRegistry":
        """Load a registry from configs/eval_holdouts.yaml."""
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("eval holdout registry must be a mapping")
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HoldoutRegistry":
        """Load a registry from a family -> records mapping."""
        records: list[HoldoutRecord] = []
        for raw_family, items in value.items():
            eval_family = validate_eval_family(raw_family)
            if eval_family is None:
                raise ValueError("eval holdout family must be non-empty")
            if not isinstance(items, list):
                raise ValueError(f"holdouts for {eval_family} must be a list")
            for item in items:
                records.append(_record_from_mapping(eval_family, item))
        return cls(records)

    def contains_prompt(self, prompt: str) -> bool:
        """Return True if prompt exactly matches a normalized holdout prompt."""
        return normalize_text(prompt) in self._prompts

    def contains_holdout_key(self, holdout_key: Mapping[str, Any] | None) -> bool:
        """Return True if a structured key exactly matches a held-out item."""
        if holdout_key is None:
            return False
        return holdout_key_fingerprint(holdout_key) in self._keys

    def reject_if_holdout(
        self,
        *,
        prompt: str,
        holdout_key: Mapping[str, Any] | None = None,
    ) -> None:
        """Raise if a candidate matches a held-out prompt or structured key."""
        if self.contains_prompt(prompt):
            raise ValueError("candidate prompt matches eval holdout prompt")
        if self.contains_holdout_key(holdout_key):
            raise ValueError("candidate holdout_key matches eval holdout key")


def default_holdout_registry_path() -> Path:
    """Return the default eval holdout registry path."""
    return Path(__file__).resolve().parents[2] / "configs" / "eval_holdouts.yaml"


def load_default_holdout_registry() -> HoldoutRegistry:
    """Load configs/eval_holdouts.yaml."""
    return HoldoutRegistry.from_file(default_holdout_registry_path())


def _record_from_mapping(eval_family: str, item: Any) -> HoldoutRecord:
    if not isinstance(item, Mapping):
        raise ValueError("holdout record must be an object")

    record_id = _require_string(item.get("id"), "id")
    prompt = _require_string(item.get("prompt"), "prompt")
    answer = item.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise TypeError("answer must be a string or null")

    holdout_key = item.get("holdout_key")
    if holdout_key is not None and not isinstance(holdout_key, Mapping):
        raise TypeError("holdout_key must be an object when present")

    return HoldoutRecord(
        id=record_id,
        eval_family=eval_family,
        prompt=prompt,
        answer=answer,
        holdout_key=holdout_key,
    )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()
