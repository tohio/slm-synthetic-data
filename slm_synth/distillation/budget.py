"""Token-target planning helpers for response distillation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from slm_synth.distillation.orchestration import normalize_signal_sequence

TOKEN_TARGETS = {
    "smoke": 100_000,
    "pilot": 1_000_000,
    "scale-check": 10_000_000,
    "final": 100_000_000,
}

DEFAULT_ESTIMATED_TOKENS_PER_ROW = 512


@dataclass(frozen=True)
class TokenBudgetPlan:
    """Approximate prompt-count plan for a token target."""

    target_name: str
    total_tokens: int
    estimated_tokens_per_row: int
    counts_by_signal: dict[str, int]

    @property
    def total_rows(self) -> int:
        """Return the planned number of rows across all signals."""
        return sum(self.counts_by_signal.values())

    @property
    def estimated_total_tokens(self) -> int:
        """Return the approximate token coverage from the planned row count."""
        return self.total_rows * self.estimated_tokens_per_row

    @property
    def target_label(self) -> str:
        """Return a compact human-readable target label for manifests."""
        return format_token_count(self.total_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the plan."""
        return {
            "target_name": self.target_name,
            "target_tokens": self.total_tokens,
            "target_label": self.target_label,
            "estimated_tokens_per_row": self.estimated_tokens_per_row,
            "total_rows": self.total_rows,
            "estimated_total_tokens": self.estimated_total_tokens,
            "counts_by_signal": dict(self.counts_by_signal),
        }


def build_token_budget_plan(
    *,
    target: str | int,
    signals: list[str] | tuple[str, ...] | None = None,
    estimated_tokens_per_row: int = DEFAULT_ESTIMATED_TOKENS_PER_ROW,
) -> TokenBudgetPlan:
    """Estimate per-signal row counts for a named or numeric token target.

    This is intentionally approximate. It plans the number of prompt records to
    request per signal based on a fixed average token estimate; it does not enforce
    the token budget after teacher generation.
    """
    target_name, total_tokens = normalize_token_target(target)
    if not isinstance(estimated_tokens_per_row, int) or estimated_tokens_per_row < 1:
        raise ValueError("estimated_tokens_per_row must be a positive integer")

    normalized_signals = normalize_signal_sequence(signals)
    if not normalized_signals:
        raise ValueError("at least one signal is required")

    total_rows = max(len(normalized_signals), ceil(total_tokens / estimated_tokens_per_row))
    base_count, remainder = divmod(total_rows, len(normalized_signals))
    counts_by_signal: dict[str, int] = {}
    for index, signal in enumerate(normalized_signals):
        counts_by_signal[signal] = base_count + (1 if index < remainder else 0)

    return TokenBudgetPlan(
        target_name=target_name,
        total_tokens=total_tokens,
        estimated_tokens_per_row=estimated_tokens_per_row,
        counts_by_signal=counts_by_signal,
    )


def normalize_token_target(target: str | int) -> tuple[str, int]:
    """Return a normalized target name and token count."""
    if isinstance(target, int):
        if target < 1:
            raise ValueError("numeric token target must be positive")
        return str(target), target

    if not isinstance(target, str) or not target.strip():
        raise ValueError("token target must be a non-empty string or positive integer")

    normalized = target.strip().lower().replace("_", "-")
    if normalized in TOKEN_TARGETS:
        return normalized, TOKEN_TARGETS[normalized]

    parsed = parse_token_count(normalized)
    return format_token_count(parsed), parsed


def parse_token_count(value: str) -> int:
    """Parse token counts like 100K, 1M, 10M, or 100000."""
    text = value.strip().lower().replace(",", "")
    if not text:
        raise ValueError("token count must be non-empty")

    multiplier = 1
    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1]

    try:
        numeric = float(text)
    except ValueError as exc:
        supported = ", ".join(sorted(TOKEN_TARGETS))
        raise ValueError(f"unsupported token target '{value}'. Use one of: {supported}; or a count like 100K") from exc

    tokens = int(numeric * multiplier)
    if tokens < 1:
        raise ValueError("token count must be positive")
    return tokens


def format_token_count(tokens: int) -> str:
    """Format common token targets as compact labels."""
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}B"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}K"
    return str(tokens)
