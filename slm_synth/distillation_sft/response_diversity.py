"""Deterministic response-diversity reporting for Distillation-SFT datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from slm_synth.distillation_sft.schema import validate_public_row


def normalize_response_text(value: str) -> str:
    """Normalize response text for exact diversity comparisons."""
    return " ".join(value.casefold().split())


def build_response_diversity_summary(files: Iterable[str | Path]) -> dict[str, Any]:
    """Build aggregate and per-signal exact response-diversity statistics."""
    counts_by_signal: dict[str, Counter[str]] = {}

    for raw_path in files:
        path = Path(raw_path)
        signal = path.stem.split(".batch", 1)[0]
        response_counts = counts_by_signal.setdefault(signal, Counter())
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
                row = validate_public_row(value)
                response_counts[normalize_response_text(row["response"])] += 1

    aggregate_counts: Counter[str] = Counter()
    signals: dict[str, dict[str, Any]] = {}
    for signal in sorted(counts_by_signal):
        response_counts = counts_by_signal[signal]
        aggregate_counts.update(response_counts)
        signals[signal] = _summarize_counts(response_counts)

    summary = _summarize_counts(aggregate_counts)
    summary["signals"] = signals
    return summary


def _summarize_counts(response_counts: Mapping[str, int]) -> dict[str, Any]:
    row_count = sum(response_counts.values())
    unique_response_count = len(response_counts)
    duplicate_response_count = row_count - unique_response_count
    repeated = sorted(
        (
            {"response": response[:160], "count": count}
            for response, count in response_counts.items()
            if count > 1
        ),
        key=lambda item: (-item["count"], item["response"]),
    )
    return {
        "row_count": row_count,
        "unique_response_count": unique_response_count,
        "duplicate_response_count": duplicate_response_count,
        "unique_response_ratio": unique_response_count / row_count if row_count else 0.0,
        "duplicate_examples": repeated[:10],
    }
