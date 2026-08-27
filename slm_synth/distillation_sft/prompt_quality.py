"""Prompt normalization used by Distillation-SFT publication checks."""

from __future__ import annotations

import re
import unicodedata


def normalize_prompt_text(prompt: str) -> str:
    """Return a stable normalized key for prompt uniqueness checks."""
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
