from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
O = TypeVar("O")


def chunked(values: Sequence[T], size: int) -> list[list[T]]:
    """Return deterministic contiguous batches."""
    if size < 1:
        raise ValueError("batch size must be positive")
    return [list(values[index:index + size]) for index in range(0, len(values), size)]


def split_sequence_batch(
    batch: Sequence[T],
) -> tuple[list[T], list[T]] | None:
    """Split a sequence batch for recursive fault isolation."""
    if len(batch) <= 1:
        return None
    midpoint = len(batch) // 2
    return list(batch[:midpoint]), list(batch[midpoint:])


def split_slot_batch(
    batch: tuple[O, Sequence[int]],
) -> tuple[tuple[O, list[int]], tuple[O, list[int]]] | None:
    """Split an owner+slot batch while preserving its owner."""
    owner, slots = batch
    if len(slots) <= 1:
        return None
    midpoint = len(slots) // 2
    return (
        (owner, list(slots[:midpoint])),
        (owner, list(slots[midpoint:])),
    )


def fill_exact_count(
    *,
    field: str,
    requested: int,
    initial: list[T],
    fetch_missing: Callable[[int, Sequence[T]], list[T]],
    max_fill_attempts: int,
    stage_label: str,
) -> list[T]:
    """Enforce code-owned cardinality without repairing semantic content.

    This is the finalized one-off behavior:
    - over-returned items are trimmed deterministically;
    - under-returned items are requested again, bounded by max_fill_attempts;
    - semantic content is never locally invented or repaired.
    """
    if requested < 1:
        raise ValueError("requested count must be positive")
    if max_fill_attempts < 1:
        raise ValueError("max_fill_attempts must be positive")

    values = list(initial)

    if len(values) > requested:
        print(
            f"[cardinality:{stage_label}] requested={requested} "
            f"observed={len(values)} action=trim excess={len(values)-requested}",
            flush=True,
        )
        return values[:requested]

    attempts = 0
    while len(values) < requested:
        missing = requested - len(values)
        attempts += 1
        if attempts > max_fill_attempts:
            raise ValueError(
                f"{field} remained underfilled after {max_fill_attempts} "
                f"fill attempt(s): requested={requested} observed={len(values)}"
            )

        print(
            f"[cardinality:{stage_label}] requested={requested} "
            f"observed={len(values)} missing={missing} "
            f"action=request_missing attempt={attempts}/{max_fill_attempts}",
            flush=True,
        )

        extra = fetch_missing(missing, values)
        if not extra:
            raise ValueError(
                f"{field} fill attempt returned no usable values: "
                f"requested={requested} observed={len(values)}"
            )

        values.extend(extra)

        if len(values) > requested:
            print(
                f"[cardinality:{stage_label}] fill_overflow "
                f"observed={len(values)} requested={requested} action=trim",
                flush=True,
            )
            values = values[:requested]

    return values
