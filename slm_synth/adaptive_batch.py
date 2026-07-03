"""Shared adaptive batch-size control for live generation workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveBatchSizeController:
    """AIMD-style controller for rows requested per provider call."""

    maximum: int
    minimum: int = 1
    increase_successes: int = 4
    decrease_factor: float = 0.5

    def __post_init__(self) -> None:
        self.maximum = max(1, int(self.maximum))
        self.minimum = min(self.maximum, max(1, int(self.minimum)))
        self.increase_successes = max(1, int(self.increase_successes))
        self.decrease_factor = min(1.0, max(0.01, float(self.decrease_factor)))
        self.current = self.maximum
        self.observed_minimum = self.current
        self.observed_peak = self.current
        self.increases = 0
        self.decreases = 0
        self.failures = 0
        self.successes = 0
        self._successes_since_change = 0

    def record_success(self) -> None:
        self.successes += 1
        self._successes_since_change += 1
        if self.current >= self.maximum or self._successes_since_change < self.increase_successes:
            return

        previous = self.current
        self.current = min(self.maximum, self.current * 2)
        self.observed_peak = max(self.observed_peak, self.current)
        self._successes_since_change = 0
        if self.current != previous:
            self.increases += 1

    def record_failure(self) -> None:
        self.failures += 1
        previous = self.current
        reduced = max(self.minimum, int(previous * self.decrease_factor))
        if reduced >= previous and previous > self.minimum:
            reduced = previous - 1
        self.current = reduced
        self.observed_minimum = min(self.observed_minimum, self.current)
        self._successes_since_change = 0
        if self.current != previous:
            self.decreases += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "adaptive_batch_size_current": self.current,
            "adaptive_batch_size_maximum": self.maximum,
            "adaptive_batch_size_minimum": self.minimum,
            "adaptive_batch_size_observed_minimum": self.observed_minimum,
            "adaptive_batch_size_observed_peak": self.observed_peak,
            "adaptive_batch_size_increases": self.increases,
            "adaptive_batch_size_decreases": self.decreases,
            "adaptive_batch_size_successes": self.successes,
            "adaptive_batch_size_failures": self.failures,
        }
