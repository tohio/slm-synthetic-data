"""Small planning helpers for production dataset row targets."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class CountPlan:
    """A deterministic count allocation across named generation families."""

    planning_mode: str
    counts_by_key: dict[str, int]
    target_count: int | None = None
    count_per_key: int | None = None

    @property
    def planned_count(self) -> int:
        """Return the total planned rows or pairs."""
        return sum(self.counts_by_key.values())


def build_count_plan(
    *,
    keys: Sequence[str],
    count_per_key: int | None = None,
    target_count: int | None = None,
    key_name: str = "key",
    count_per_key_name: str = "count_per_key",
    target_count_name: str = "target_count",
    target_mode: str | None = None,
) -> CountPlan:
    """Allocate either a per-key count or a total target count across keys.

    Exactly one of count_per_key or target_count must be provided. Target counts are
    distributed in the order supplied by keys, with any remainder assigned to the
    earliest keys. This keeps production plans deterministic and reproducible.
    """
    normalized_keys = tuple(keys)
    if not normalized_keys:
        raise ValueError(f"at least one {key_name} is required")

    provided = sum(value is not None for value in (count_per_key, target_count))
    if provided != 1:
        raise ValueError(f"provide exactly one of {count_per_key_name} or {target_count_name}")

    if count_per_key is not None:
        count = _validate_positive_int(count_per_key, count_per_key_name)
        return CountPlan(
            planning_mode=count_per_key_name,
            counts_by_key={key: count for key in normalized_keys},
            count_per_key=count,
        )

    target = _validate_positive_int(target_count, target_count_name)
    if target < len(normalized_keys):
        raise ValueError(f"{target_count_name} must be at least the number of requested {key_name}s")

    base_count, remainder = divmod(target, len(normalized_keys))
    counts = {
        key: base_count + (1 if index < remainder else 0)
        for index, key in enumerate(normalized_keys)
    }
    return CountPlan(
        planning_mode=target_mode or target_count_name,
        counts_by_key=counts,
        target_count=target,
    )


def _validate_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value
