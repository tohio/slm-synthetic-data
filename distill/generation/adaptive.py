from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Any


class RetryableProviderExhaustedError(RuntimeError):
    def __init__(self, message: str, *, telemetry: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.telemetry = telemetry or {}


class AdaptiveRequestController:
    def __init__(
        self,
        *,
        enabled: bool = True,
        maximum_in_flight: int = 1,
        initial_in_flight: int = 8,
        minimum_in_flight: int = 1,
        slow_start_enabled: bool = True,
        slow_start_multiplier: float = 2.0,
        increase_successes_per_step: int = 64,
        increase_step: int = 16,
        rate_limit_burst_threshold: int = 4,
        rate_limit_window_seconds: float = 2.0,
        rate_limit_decrease_factor: float = 0.50,
        sustained_rate_limit_attempt_window: int = 60,
        sustained_rate_limit_threshold: int = 20,
        cooldown_initial_seconds: float = 5.0,
        cooldown_max_seconds: float = 60.0,
        cooldown_multiplier: float = 2.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.maximum_in_flight = max(1, int(maximum_in_flight))
        self.minimum_in_flight = min(self.maximum_in_flight, max(1, int(minimum_in_flight)))
        self.initial_in_flight = min(self.maximum_in_flight, max(self.minimum_in_flight, int(initial_in_flight)))
        self.slow_start_enabled = bool(slow_start_enabled)
        self.slow_start_multiplier = max(1.0, float(slow_start_multiplier))
        self.increase_successes_per_step = max(1, int(increase_successes_per_step))
        self.increase_step = max(1, int(increase_step))
        self.rate_limit_burst_threshold = max(1, int(rate_limit_burst_threshold))
        self.rate_limit_window_seconds = max(0.0, float(rate_limit_window_seconds))
        self.rate_limit_decrease_factor = min(1.0, max(0.01, float(rate_limit_decrease_factor)))
        self.sustained_rate_limit_attempt_window = max(1, int(sustained_rate_limit_attempt_window))
        self.sustained_rate_limit_threshold = min(self.sustained_rate_limit_attempt_window, max(1, int(sustained_rate_limit_threshold)))
        self.cooldown_initial_seconds = max(0.0, float(cooldown_initial_seconds))
        self.cooldown_max_seconds = max(self.cooldown_initial_seconds, float(cooldown_max_seconds))
        self.cooldown_multiplier = max(1.0, float(cooldown_multiplier))
        self._condition = threading.Condition()
        self._events: deque[float] = deque()
        self._attempt_outcomes: deque[bool] = deque(maxlen=self.sustained_rate_limit_attempt_window)
        self._current_limit = self.initial_in_flight if self.enabled else self.maximum_in_flight
        self._active_requests = 0
        self._peak_limit = self._current_limit
        self._minimum_observed_limit = self._current_limit
        self._cooldown_until = 0.0
        self._next_cooldown_seconds = self.cooldown_initial_seconds
        self._successes_since_adjustment = 0
        self._slow_start = self.enabled and self.slow_start_enabled
        self._admission_generation = 0

    def acquire(self) -> tuple[float, int]:
        started = monotonic()
        with self._condition:
            while True:
                now = monotonic()
                remaining = self._cooldown_until - now
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                if self._active_requests < self._current_limit:
                    self._active_requests += 1
                    return max(0.0, monotonic() - started), self._admission_generation
                self._condition.wait()

    def release(self) -> None:
        with self._condition:
            if self._active_requests <= 0:
                raise RuntimeError('Adaptive request admission released without an active request')
            self._active_requests -= 1
            self._condition.notify_all()

    def record_rate_limit(self, model: str, admission_generation: int | None = None) -> tuple[bool, int, int, float]:
        if not self.enabled:
            return False, self.maximum_in_flight, self.maximum_in_flight, 0.0
        now = monotonic()
        with self._condition:
            if admission_generation is not None and admission_generation != self._admission_generation:
                return False, self._current_limit, self._current_limit, 0.0
            if now < self._cooldown_until:
                return False, self._current_limit, self._current_limit, 0.0
            self._events.append(now)
            self._attempt_outcomes.append(True)
            cutoff = now - self.rate_limit_window_seconds
            while self._events and self._events[0] < cutoff:
                self._events.popleft()
            burst_threshold = min(self.rate_limit_burst_threshold, self._current_limit)
            burst = len(self._events) >= burst_threshold
            sustained_events = sum(self._attempt_outcomes)
            sustained = len(self._attempt_outcomes) >= self.sustained_rate_limit_attempt_window and sustained_events >= self.sustained_rate_limit_threshold
            if not burst and not sustained:
                return False, self._current_limit, self._current_limit, 0.0
            previous = self._current_limit
            reduced = max(self.minimum_in_flight, int(previous * self.rate_limit_decrease_factor))
            if reduced >= previous and previous > self.minimum_in_flight:
                reduced = previous - 1
            self._current_limit = reduced
            self._minimum_observed_limit = min(self._minimum_observed_limit, reduced)
            self._admission_generation += 1
            self._slow_start = False
            cooldown = min(self._next_cooldown_seconds, self.cooldown_max_seconds)
            self._cooldown_until = now + cooldown
            self._next_cooldown_seconds = min(cooldown * self.cooldown_multiplier, self.cooldown_max_seconds)
            self._successes_since_adjustment = 0
            self._events.clear()
            self._attempt_outcomes.clear()
            self._condition.notify_all()
        print(f'[hosted] Adaptive admission decreased: model={model} in_flight_limit={previous}->{reduced} cooldown={cooldown:.2f}s', flush=True)
        return True, previous, reduced, cooldown

    def record_success(self, model: str, admission_generation: int | None = None) -> tuple[bool, int, int]:
        if not self.enabled:
            return False, self.maximum_in_flight, self.maximum_in_flight
        with self._condition:
            if admission_generation is not None and admission_generation != self._admission_generation:
                return False, self._current_limit, self._current_limit
            if monotonic() < self._cooldown_until:
                return False, self._current_limit, self._current_limit
            self._attempt_outcomes.append(False)
            self._successes_since_adjustment += 1
            required = self._current_limit if self._slow_start else self.increase_successes_per_step
            if self._successes_since_adjustment < required or self._current_limit >= self.maximum_in_flight:
                return False, self._current_limit, self._current_limit
            previous = self._current_limit
            if self._slow_start:
                increased = min(self.maximum_in_flight, max(previous + 1, int(previous * self.slow_start_multiplier)))
                mode = 'slow_start'
            else:
                increased = min(self.maximum_in_flight, previous + self.increase_step)
                mode = 'aimd'
            self._current_limit = increased
            self._peak_limit = max(self._peak_limit, increased)
            self._successes_since_adjustment = 0
            self._next_cooldown_seconds = self.cooldown_initial_seconds
            self._events.clear()
            self._attempt_outcomes.clear()
            self._condition.notify_all()
        print(f'[hosted] Adaptive admission increased: model={model} mode={mode} in_flight_limit={previous}->{increased}', flush=True)
        return True, previous, increased

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                'adaptive_current_in_flight_limit': self._current_limit,
                'adaptive_peak_in_flight_limit': self._peak_limit,
                'adaptive_min_in_flight_limit': self._minimum_observed_limit,
            }


def _error_text(exc: Exception) -> str:
    return str(exc).lower()


def is_capacity_or_rate_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(marker in text for marker in ('capacity_exceeded', 'rate_limit', 'rate limit', 'too many requests', 'error code: 429', 'error code: 498', ' 429 ', ' 498 '))


def is_transient_transport_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return any(marker in text for marker in ('timeout', 'timed out', 'temporarily unavailable', 'connection error', 'connection reset', 'remoteprotocolerror', 'peer closed connection', 'incomplete chunked read', 'error code: 500', 'error code: 502', 'error code: 503', 'error code: 504', ' 500 ', ' 502 ', ' 503 ', ' 504 '))


def is_retryable_provider_error(exc: Exception) -> bool:
    return is_capacity_or_rate_error(exc) or is_transient_transport_error(exc)


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', None) or {}
    value = headers.get('retry-after') or headers.get('Retry-After')
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def backoff_delay(*, attempt: int, exc: Exception, initial_seconds: float, max_seconds: float, multiplier: float, jitter_ratio: float, remaining_seconds: float) -> float:
    retry_after = retry_after_seconds(exc)
    if retry_after is not None:
        return min(max(0.0, retry_after), max(0.0, remaining_seconds))
    base = min(max_seconds, initial_seconds * (multiplier ** max(0, attempt - 1)))
    delay = base + base * max(0.0, jitter_ratio) * random.random()
    return min(delay, max(0.0, remaining_seconds))


def sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
