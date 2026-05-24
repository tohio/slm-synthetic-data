"""Validated model guidance for synthetic generation."""
from __future__ import annotations

import sys
from typing import Optional, TextIO

SUPPORTED_MODELS = {
    "deepseek/deepseek-v4-flash": "openrouter",
    "llama-3.1-8b-instant": "groq",
    "llama-3.3-70b-versatile": "groq",
}
_WARNED: set[str] = set()


def is_supported_model(model: Optional[str]) -> bool:
    return bool(model) and str(model) in SUPPORTED_MODELS


def warn_if_unsupported_model(model: Optional[str], *, context: str = "synthetic generation", stream: Optional[TextIO] = None) -> None:
    if not model or is_supported_model(model):
        return
    key = f"{context}:{model}"
    if key in _WARNED:
        return
    _WARNED.add(key)
    stream = stream or sys.stderr
    supported = ", ".join(sorted(SUPPORTED_MODELS))
    print(
        "[model] WARNING: "
        f"model '{model}' is not validated for {context}. "
        f"Validated models: {supported}. "
        "Grounded generation requires reliable strict structured output.",
        file=stream,
    )
