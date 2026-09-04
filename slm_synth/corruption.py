"""Narrow deterministic corruption checks shared by publication schemas."""

from __future__ import annotations

import re
from typing import Any


HIGH_CONFIDENCE_CORRUPTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lone_o_line", re.compile(r"(?m)^\s*o\s*$")),
    ("defintions", re.compile(r"\bdefintions\b", re.IGNORECASE)),
)


class HighConfidenceCorruptionError(ValueError):
    """Raised when a public row contains a narrowly defined corruption artifact."""


def high_confidence_corruption_reasons(*values: Any) -> tuple[str, ...]:
    """Return stable reason names for the deliberately narrow defect patterns."""

    text = "\n".join(value for value in values if isinstance(value, str))
    return tuple(
        name
        for name, pattern in HIGH_CONFIDENCE_CORRUPTION_PATTERNS
        if pattern.search(text)
    )


def require_no_high_confidence_corruption(
    *values: Any,
    artifact_name: str = "public row",
) -> None:
    """Reject only the exact corruption classes established by the offline audit."""

    reasons = high_confidence_corruption_reasons(*values)
    if reasons:
        raise HighConfidenceCorruptionError(
            f"{artifact_name} contains high-confidence corruption artifact(s): "
            + ", ".join(reasons)
        )
