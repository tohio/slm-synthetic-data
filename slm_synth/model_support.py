"""Supported-model guidance for synthetic generation.

This module intentionally implements a lightweight warning rather than a hard
capability-profile system. The public supported path is small and explicit, but
experimentation remains possible.
"""
from __future__ import annotations

import sys
from typing import Optional, TextIO

SUPPORTED_GROQ_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
}

_WARNED: set[str] = set()


def is_supported_model(model: Optional[str]) -> bool:
    return bool(model) and str(model) in SUPPORTED_GROQ_MODELS


def warn_if_unsupported_model(
    model: Optional[str],
    *,
    context: str = "synthetic generation",
    stream: Optional[TextIO] = None,
) -> None:
    """Print a one-time warning for models outside the validated support set."""
    if not model or is_supported_model(model):
        return

    key = f"{context}:{model}"
    if key in _WARNED:
        return
    _WARNED.add(key)

    stream = stream or sys.stderr
    supported = ", ".join(sorted(SUPPORTED_GROQ_MODELS))
    print(
        "[model] WARNING: "
        f"model '{model}' is not validated for {context}. "
        f"Validated Groq models: {supported}. "
        "The pipeline requires reliable JSON object output and strict schema following.",
        file=stream,
    )
