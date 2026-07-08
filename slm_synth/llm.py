from __future__ import annotations

import json
import os
from dataclasses import dataclass
import random
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class StructuredRenderedResponseError(RuntimeError):
    """Renderer returned responses that could not be parsed into a persistable object."""

    def __init__(self, message: str, *, telemetry: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.telemetry = telemetry or {}


class RetryableProviderExhaustedError(RuntimeError):
    """A transient provider failure exhausted the per-request retry window."""

    def __init__(self, message: str, *, telemetry: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.telemetry = telemetry or {}


class AdaptiveRequestController:
    """Admission control for variable-capacity upstream providers.

    Executor workers may exist up to the configured concurrency ceiling, but
    only ``current_limit`` requests can enter the provider call at once. The
    window grows on sustained success and shrinks after a burst of 429s.
    """

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
    ):
        self.enabled = bool(enabled)
        self.maximum_in_flight = max(1, int(maximum_in_flight))
        self.minimum_in_flight = min(self.maximum_in_flight, max(1, int(minimum_in_flight)))
        self.initial_in_flight = min(
            self.maximum_in_flight, max(self.minimum_in_flight, int(initial_in_flight))
        )
        self.slow_start_enabled = bool(slow_start_enabled)
        self.slow_start_multiplier = max(1.0, float(slow_start_multiplier))
        self.increase_successes_per_step = max(1, int(increase_successes_per_step))
        self.increase_step = max(1, int(increase_step))
        self.rate_limit_burst_threshold = max(1, int(rate_limit_burst_threshold))
        self.rate_limit_window_seconds = max(0.0, float(rate_limit_window_seconds))
        self.rate_limit_decrease_factor = min(1.0, max(0.01, float(rate_limit_decrease_factor)))
        self.sustained_rate_limit_attempt_window = max(1, int(sustained_rate_limit_attempt_window))
        self.sustained_rate_limit_threshold = min(
            self.sustained_rate_limit_attempt_window, max(1, int(sustained_rate_limit_threshold))
        )
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
        """Wait for admission, then claim one outbound provider slot and its generation."""
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
                raise RuntimeError("Adaptive request admission released without an acquired slot")
            self._active_requests -= 1
            self._condition.notify_all()

    def record_rate_limit(
        self, model: str, admission_generation: int | None = None
    ) -> tuple[bool, int, int, float]:
        """Shrink on burst or sustained 429 pressure; ignore stale in-flight fallout."""
        if not self.enabled:
            return False, self.maximum_in_flight, self.maximum_in_flight, 0.0
        now = monotonic()
        with self._condition:
            if (
                admission_generation is not None
                and admission_generation != self._admission_generation
            ):
                return False, self._current_limit, self._current_limit, 0.0
            if now < self._cooldown_until:
                return False, self._current_limit, self._current_limit, 0.0
            self._events.append(now)
            self._attempt_outcomes.append(True)
            cutoff = now - self.rate_limit_window_seconds
            while self._events and self._events[0] < cutoff:
                self._events.popleft()
            burst_threshold = min(self.rate_limit_burst_threshold, self._current_limit)
            burst_detected = len(self._events) >= burst_threshold
            sustained_events = sum(self._attempt_outcomes)
            sustained_detected = (
                len(self._attempt_outcomes) >= self.sustained_rate_limit_attempt_window
                and sustained_events >= self.sustained_rate_limit_threshold
            )
            if not burst_detected and not sustained_detected:
                return False, self._current_limit, self._current_limit, 0.0
            trigger = "burst" if burst_detected else "sustained"
            event_count = burst_threshold if burst_detected else sustained_events
            window_label = (
                f"{self.rate_limit_window_seconds:.2f}s"
                if burst_detected
                else f"{self.sustained_rate_limit_attempt_window} attempts"
            )
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
        print(
            f"[llm] Adaptive admission decreased: model={model} mode=aimd trigger={trigger} "
            f"events={event_count} window={window_label} "
            f"in_flight_limit={previous}->{reduced} cooldown={cooldown:.2f}s"
        )
        return True, previous, reduced, cooldown

    def record_success(
        self, model: str, admission_generation: int | None = None
    ) -> tuple[bool, int, int]:
        """Use slow start before throttling; use additive recovery afterward."""
        if not self.enabled:
            return False, self.maximum_in_flight, self.maximum_in_flight
        with self._condition:
            if (
                admission_generation is not None
                and admission_generation != self._admission_generation
            ):
                return False, self._current_limit, self._current_limit
            if monotonic() < self._cooldown_until:
                return False, self._current_limit, self._current_limit
            self._attempt_outcomes.append(False)
            self._successes_since_adjustment += 1
            required_successes = (
                self._current_limit if self._slow_start else self.increase_successes_per_step
            )
            if (
                self._successes_since_adjustment < required_successes
                or self._current_limit >= self.maximum_in_flight
            ):
                return False, self._current_limit, self._current_limit
            previous = self._current_limit
            if self._slow_start:
                increased = min(
                    self.maximum_in_flight,
                    max(previous + 1, int(previous * self.slow_start_multiplier)),
                )
                mode = "slow_start"
            else:
                increased = min(self.maximum_in_flight, previous + self.increase_step)
                mode = "aimd"
            self._current_limit = increased
            self._peak_limit = max(self._peak_limit, increased)
            self._successes_since_adjustment = 0
            self._next_cooldown_seconds = self.cooldown_initial_seconds
            self._events.clear()
            self._attempt_outcomes.clear()
            self._condition.notify_all()
        print(
            f"[llm] Adaptive admission increased: model={model} mode={mode} "
            f"in_flight_limit={previous}->{increased} successes={required_successes}"
        )
        return True, previous, increased

    def snapshot(self) -> dict[str, int]:
        with self._condition:
            return {
                "adaptive_current_in_flight_limit": self._current_limit,
                "adaptive_peak_in_flight_limit": self._peak_limit,
                "adaptive_min_in_flight_limit": self._minimum_observed_limit,
            }


SUPPORTED_PROVIDERS = {"openrouter"}
SUPPORTED_OPENROUTER_ROUTING_MODES = frozenset({"auto", "prefer", "strict"})


def _clean_optional_env_string(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


@dataclass(frozen=True)
class OpenRouterRoutingPolicy:
    """Validated OpenRouter provider routing request policy."""

    mode: str = "auto"
    requested_provider: str | None = None

    def provider_preferences(self, *, require_parameters: bool, allow_fallbacks: bool) -> dict[str, Any]:
        preferences: dict[str, Any] = {
            "require_parameters": bool(require_parameters),
        }
        if self.mode == "auto":
            preferences["allow_fallbacks"] = bool(allow_fallbacks)
        elif self.mode == "prefer":
            assert self.requested_provider is not None
            preferences.update(
                {
                    "order": [self.requested_provider],
                    "allow_fallbacks": True,
                }
            )
        elif self.mode == "strict":
            assert self.requested_provider is not None
            preferences.update(
                {
                    "only": [self.requested_provider],
                    "allow_fallbacks": False,
                }
            )
        else:  # pragma: no cover - construction validates this.
            raise ValueError(f"Unsupported OpenRouter routing mode: {self.mode}")
        return preferences

    def metadata(self, *, allow_fallbacks: bool) -> dict[str, Any]:
        return {
            "routing_mode": self.mode,
            "requested_provider": self.requested_provider,
            "allow_fallbacks": bool(allow_fallbacks),
        }


def resolve_openrouter_routing_policy(
    *,
    mode: str | None = None,
    provider: str | None = None,
) -> OpenRouterRoutingPolicy:
    """Resolve OpenRouter routing policy from explicit values or environment."""
    raw_mode = _clean_optional_env_string(mode)
    if raw_mode is None:
        raw_mode = _clean_optional_env_string(os.environ.get("OPENROUTER_ROUTING_MODE")) or "auto"
    normalized_mode = raw_mode.lower()
    if normalized_mode not in SUPPORTED_OPENROUTER_ROUTING_MODES:
        supported = ", ".join(sorted(SUPPORTED_OPENROUTER_ROUTING_MODES))
        raise ValueError(
            f"Unsupported OPENROUTER_ROUTING_MODE '{raw_mode}'. Supported values: {supported}"
        )

    requested_provider = _clean_optional_env_string(
        provider if provider is not None else os.environ.get("OPENROUTER_PROVIDER")
    )
    if normalized_mode in {"prefer", "strict"} and requested_provider is None:
        raise ValueError(
            f"OPENROUTER_PROVIDER is required when OPENROUTER_ROUTING_MODE={normalized_mode}"
        )
    if normalized_mode == "auto" and requested_provider is not None:
        raise ValueError(
            "OPENROUTER_PROVIDER requires OPENROUTER_ROUTING_MODE=prefer or OPENROUTER_ROUTING_MODE=strict"
        )
    return OpenRouterRoutingPolicy(mode=normalized_mode, requested_provider=requested_provider)


class LLMBackend:
    """OpenAI-compatible OpenRouter client for generation paths.

    ``generate_batch`` uses an ``{"items": [...]}`` object contract.
    ``generate_structured_object`` uses strict JSON Schema responses.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        *,
        json_mode: bool = True,
        service_tier: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_request_retries: int = 3,
        max_retryable_request_attempts: int = 20,
        retry_max_elapsed_seconds: float = 1800.0,
        retry_sleep_seconds: float = 0.5,
        retry_backoff_initial_seconds: float = 1.0,
        retry_backoff_max_seconds: float = 30.0,
        retry_backoff_multiplier: float = 2.0,
        retry_jitter_ratio: float = 0.30,
        adaptive_concurrency_enabled: bool = True,
        adaptive_maximum_in_flight: int = 1,
        adaptive_initial_in_flight: int = 8,
        adaptive_minimum_in_flight: int = 1,
        adaptive_slow_start_enabled: bool = True,
        adaptive_slow_start_multiplier: float = 2.0,
        adaptive_increase_successes_per_step: int = 64,
        adaptive_increase_step: int = 16,
        adaptive_rate_limit_burst_threshold: int = 4,
        adaptive_rate_limit_window_seconds: float = 2.0,
        adaptive_rate_limit_decrease_factor: float = 0.50,
        adaptive_sustained_rate_limit_attempt_window: int = 60,
        adaptive_sustained_rate_limit_threshold: int = 20,
        adaptive_cooldown_initial_seconds: float = 5.0,
        adaptive_cooldown_max_seconds: float = 60.0,
        adaptive_cooldown_multiplier: float = 2.0,
        require_parameters: bool = True,
        allow_fallbacks: bool = True,
        openrouter_routing_mode: str | None = None,
        openrouter_provider: str | None = None,
    ):
        provider = str(provider).lower().strip()
        if provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        missing_key = "OPENROUTER_API_KEY"
        default_headers = {
            "HTTP-Referer": "https://github.com/tohio/slm-synthetic-data",
            "X-Title": "SLM grounded synthetic generation",
        }
        if not api_key:
            raise RuntimeError(f"{missing_key} is not set in the environment.")

        self.provider = provider
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.json_mode = bool(json_mode)
        self.service_tier = service_tier
        self.request_timeout = request_timeout
        self.max_request_retries = int(max_request_retries)
        self.max_retryable_request_attempts = int(max_retryable_request_attempts)
        self.retry_max_elapsed_seconds = float(retry_max_elapsed_seconds)
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.retry_backoff_initial_seconds = float(retry_backoff_initial_seconds)
        self.retry_backoff_max_seconds = float(retry_backoff_max_seconds)
        self.retry_backoff_multiplier = float(retry_backoff_multiplier)
        self.retry_jitter_ratio = float(retry_jitter_ratio)
        self.adaptive_controller = AdaptiveRequestController(
            enabled=adaptive_concurrency_enabled,
            maximum_in_flight=adaptive_maximum_in_flight,
            initial_in_flight=adaptive_initial_in_flight,
            minimum_in_flight=adaptive_minimum_in_flight,
            slow_start_enabled=adaptive_slow_start_enabled,
            slow_start_multiplier=adaptive_slow_start_multiplier,
            increase_successes_per_step=adaptive_increase_successes_per_step,
            increase_step=adaptive_increase_step,
            rate_limit_burst_threshold=adaptive_rate_limit_burst_threshold,
            rate_limit_window_seconds=adaptive_rate_limit_window_seconds,
            rate_limit_decrease_factor=adaptive_rate_limit_decrease_factor,
            sustained_rate_limit_attempt_window=adaptive_sustained_rate_limit_attempt_window,
            sustained_rate_limit_threshold=adaptive_sustained_rate_limit_threshold,
            cooldown_initial_seconds=adaptive_cooldown_initial_seconds,
            cooldown_max_seconds=adaptive_cooldown_max_seconds,
            cooldown_multiplier=adaptive_cooldown_multiplier,
        )
        self.require_parameters = bool(require_parameters)
        self.allow_fallbacks = bool(allow_fallbacks)
        self.openrouter_routing_policy = resolve_openrouter_routing_policy(
            mode=openrouter_routing_mode,
            provider=openrouter_provider,
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            default_headers=default_headers,
        )

    def clone_for(
        self,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        json_mode: Optional[bool] = None,
        service_tier: Optional[str] = None,
    ) -> "LLMBackend":
        clone = LLMBackend(
            provider=self.provider,
            model=model or self.model,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            json_mode=self.json_mode if json_mode is None else json_mode,
            service_tier=self.service_tier if service_tier is None else service_tier,
            request_timeout=self.request_timeout,
            max_request_retries=self.max_request_retries,
            max_retryable_request_attempts=self.max_retryable_request_attempts,
            retry_max_elapsed_seconds=self.retry_max_elapsed_seconds,
            retry_sleep_seconds=self.retry_sleep_seconds,
            retry_backoff_initial_seconds=self.retry_backoff_initial_seconds,
            retry_backoff_max_seconds=self.retry_backoff_max_seconds,
            retry_backoff_multiplier=self.retry_backoff_multiplier,
            retry_jitter_ratio=self.retry_jitter_ratio,
            adaptive_concurrency_enabled=self.adaptive_controller.enabled,
            adaptive_maximum_in_flight=self.adaptive_controller.maximum_in_flight,
            adaptive_initial_in_flight=self.adaptive_controller.initial_in_flight,
            adaptive_minimum_in_flight=self.adaptive_controller.minimum_in_flight,
            adaptive_slow_start_enabled=self.adaptive_controller.slow_start_enabled,
            adaptive_slow_start_multiplier=self.adaptive_controller.slow_start_multiplier,
            adaptive_increase_successes_per_step=self.adaptive_controller.increase_successes_per_step,
            adaptive_increase_step=self.adaptive_controller.increase_step,
            adaptive_rate_limit_burst_threshold=self.adaptive_controller.rate_limit_burst_threshold,
            adaptive_rate_limit_window_seconds=self.adaptive_controller.rate_limit_window_seconds,
            adaptive_rate_limit_decrease_factor=self.adaptive_controller.rate_limit_decrease_factor,
            adaptive_sustained_rate_limit_attempt_window=self.adaptive_controller.sustained_rate_limit_attempt_window,
            adaptive_sustained_rate_limit_threshold=self.adaptive_controller.sustained_rate_limit_threshold,
            adaptive_cooldown_initial_seconds=self.adaptive_controller.cooldown_initial_seconds,
            adaptive_cooldown_max_seconds=self.adaptive_controller.cooldown_max_seconds,
            adaptive_cooldown_multiplier=self.adaptive_controller.cooldown_multiplier,
            require_parameters=self.require_parameters,
            allow_fallbacks=self.allow_fallbacks,
            openrouter_routing_mode=self.openrouter_routing_policy.mode,
            openrouter_provider=self.openrouter_routing_policy.requested_provider,
        )
        clone.adaptive_controller = self.adaptive_controller
        return clone

    def _clean(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
        return text

    def _extract_json_candidate(self, raw: str) -> str:
        raw = self._clean(raw)
        if raw.startswith("{") or raw.startswith("["):
            return raw
        object_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if object_match:
            return object_match.group(0)
        array_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if array_match:
            return array_match.group(0)
        raise ValueError(f"Model did not return JSON: {raw[:2000]}")

    def _parse_items(self, raw: str, batch_size: int) -> List[Dict[str, Any]]:
        candidate = self._extract_json_candidate(raw)
        parsed = json.loads(candidate)
        objs = parsed.get("items") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else None
        if not isinstance(objs, list):
            raise ValueError("Expected a JSON object with an 'items' array")
        if len(objs) != batch_size:
            raise ValueError(f"Expected {batch_size} items, got {len(objs)}")
        if not all(isinstance(obj, dict) for obj in objs):
            raise ValueError("Every generated item must be a JSON object")
        return objs

    def _base_kwargs(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON-only synthetic data generator. Return exactly one valid JSON object and no prose.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _provider_extra_body(self) -> dict[str, Any]:
        return {
            "provider": self.openrouter_routing_policy.provider_preferences(
                require_parameters=self.require_parameters,
                allow_fallbacks=self.allow_fallbacks,
            )
        }

    def _routing_metadata(self) -> dict[str, Any]:
        policy = getattr(self, "openrouter_routing_policy", None)
        if not isinstance(policy, OpenRouterRoutingPolicy):
            return {}
        preferences = policy.provider_preferences(
            require_parameters=getattr(self, "require_parameters", True),
            allow_fallbacks=getattr(self, "allow_fallbacks", True),
        )
        return policy.metadata(allow_fallbacks=bool(preferences.get("allow_fallbacks", False)))

    def _create_completion(self, prompt: str):
        kwargs = self._base_kwargs(prompt)
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = self._provider_extra_body()
        return self.client.chat.completions.create(**kwargs)

    def _create_structured_completion(self, prompt: str, schema: dict[str, Any], schema_name: str):
        if self.provider != "openrouter":
            raise ValueError("Strict grounded structured generation requires provider='openrouter'")
        kwargs = self._base_kwargs(prompt)
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        kwargs["extra_body"] = self._provider_extra_body()
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return str(exc).lower()

    @staticmethod
    def _exception_class_names(exc: Exception) -> set[str]:
        return {cls.__name__.lower() for cls in type(exc).__mro__ if cls is not object}

    @staticmethod
    def _coerce_status_code(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _provider_status_code(cls, exc: Exception) -> int | None:
        for value in (getattr(exc, "status_code", None), getattr(exc, "http_status", None)):
            status_code = cls._coerce_status_code(value)
            if status_code is not None:
                return status_code

        response = getattr(exc, "response", None)
        if response is None:
            return None
        for value in (getattr(response, "status_code", None), getattr(response, "status", None)):
            status_code = cls._coerce_status_code(value)
            if status_code is not None:
                return status_code
        return None

    def _is_capacity_or_rate_error(self, exc: Exception) -> bool:
        class_names = self._exception_class_names(exc)
        if "ratelimiterror" in class_names:
            return True

        status_code = self._provider_status_code(exc)
        if status_code is not None:
            return status_code in {429, 498}

        text = self._error_text(exc)
        return any(marker in text for marker in (
            "capacity_exceeded", "rate_limit", "rate limit", "too many requests",
            "error code: 429", "error code: 498", " 429 ", " 498 ",
        ))

    def _is_transient_transport_error(self, exc: Exception) -> bool:
        class_names = self._exception_class_names(exc)
        if class_names.intersection({"apitimeouterror", "apiconnectionerror", "internalservererror"}):
            return True

        status_code = self._provider_status_code(exc)
        if status_code is not None:
            return status_code in {500, 502, 503, 504}

        text = self._error_text(exc)
        return any(marker in text for marker in (
            "timeout", "timed out", "temporarily unavailable", "connection error",
            "connection reset", "remoteprotocolerror", "peer closed connection",
            "incomplete chunked read", "error code: 500", "error code: 502",
            "error code: 503", "error code: 504", " 500 ", " 502 ", " 503 ", " 504 ",
        ))

    def _is_retryable_provider_error(self, exc: Exception) -> bool:
        return self._is_capacity_or_rate_error(exc) or self._is_transient_transport_error(exc)

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        value = headers.get("retry-after") or headers.get("Retry-After")
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

    def _acquire_provider_slot(self) -> tuple[float, int]:
        return self.adaptive_controller.acquire()

    def _release_provider_slot(self) -> None:
        self.adaptive_controller.release()

    @staticmethod
    def _provider_retry_elapsed_seconds(*, started: float, admission_wait_seconds: float) -> float:
        """Return retry-budget elapsed time, excluding local adaptive admission waits."""
        return max(0.0, monotonic() - started - max(0.0, admission_wait_seconds))

    def _can_retry(
        self,
        attempt: int,
        started: float,
        exc: Exception,
        *,
        admission_wait_seconds: float = 0.0,
    ) -> bool:
        if self._is_retryable_provider_error(exc):
            return (
                attempt < self.max_retryable_request_attempts
                and self._provider_retry_elapsed_seconds(
                    started=started, admission_wait_seconds=admission_wait_seconds
                ) < self.retry_max_elapsed_seconds
            )
        return attempt < self.max_request_retries

    def _sleep_before_retry(
        self,
        attempt: int,
        exc: Exception,
        *,
        started: float,
        admission_wait_seconds: float = 0.0,
        admission_generation: int | None = None,
    ) -> tuple[float, int, float]:
        adaptive_window_decreases = 0
        max_adaptive_cooldown_seconds = 0.0
        if self._is_retryable_provider_error(exc):
            if self._is_capacity_or_rate_error(exc):
                decreased, _previous, _current, cooldown = self.adaptive_controller.record_rate_limit(
                    self.model, admission_generation
                )
                if decreased:
                    adaptive_window_decreases = 1
                    max_adaptive_cooldown_seconds = cooldown
            retry_after = self._retry_after_seconds(exc)
            if retry_after is not None:
                delay = retry_after
            else:
                base_delay = min(
                    self.retry_backoff_max_seconds,
                    self.retry_backoff_initial_seconds * (self.retry_backoff_multiplier ** max(0, attempt - 1)),
                )
                delay = base_delay + base_delay * max(0.0, self.retry_jitter_ratio) * random.random()
            retry_elapsed_seconds = self._provider_retry_elapsed_seconds(
                started=started, admission_wait_seconds=admission_wait_seconds
            )
            remaining = max(0.0, self.retry_max_elapsed_seconds - retry_elapsed_seconds)
            delay = min(delay, remaining)
            print(
                f"[llm] Retryable provider failure: model={self.model} attempt={attempt}/"
                f"{self.max_retryable_request_attempts} delay={delay:.2f}s error={str(exc)[:180]!r}"
            )
        else:
            delay = self.retry_sleep_seconds * attempt
        if delay > 0:
            time.sleep(delay)
        return delay, adaptive_window_decreases, max_adaptive_cooldown_seconds

    def generate_batch(self, prompt: str, batch_size: int) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        retry_started: float | None = None
        admission_wait_seconds = 0.0
        attempt = 0
        while True:
            attempt += 1
            if retry_started is None:
                retry_started = monotonic()
            admission_wait, admission_generation = self._acquire_provider_slot()
            admission_wait_seconds += max(0.0, admission_wait)
            try:
                response = self._create_completion(prompt)
            except Exception as exc:
                last_error = exc
            else:
                self.adaptive_controller.record_success(self.model, admission_generation)
                raw = (response.choices[0].message.content or "").strip()
                try:
                    return self._parse_items(raw, batch_size)
                except Exception as exc:
                    last_error = exc
            finally:
                self._release_provider_slot()
            assert retry_started is not None
            if not self._can_retry(
                attempt, retry_started, last_error, admission_wait_seconds=admission_wait_seconds
            ):
                break
            self._sleep_before_retry(
                attempt,
                last_error,
                started=retry_started,
                admission_wait_seconds=admission_wait_seconds,
                admission_generation=admission_generation,
            )
        raise RuntimeError(f"LLM batch failed after {attempt} attempts: {last_error}")

    @staticmethod
    def _usage_dict(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            result = usage.model_dump()
        elif isinstance(usage, dict):
            result = dict(usage)
        else:
            result = {key: getattr(usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens") if hasattr(usage, key)}
        extra = getattr(usage, "model_extra", None) or {}
        if isinstance(extra, dict):
            result.update(extra)
        return result

    @staticmethod
    def _merge_usage(total: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
        """Accumulate billed usage across completed renderer responses, including parse failures."""
        merged = dict(total)
        for key, value in usage.items():
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                merged[key] = int(merged.get(key, 0) or 0) + int(value or 0)
            elif key == "cost":
                merged[key] = float(merged.get(key, 0.0) or 0.0) + float(value or 0.0)
            elif key not in merged:
                merged[key] = value
        return merged

    def generate_structured_object_with_metadata(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Generate a strict object and retain operational telemetry for persisted batches."""
        last_error: Optional[Exception] = None
        overall_started = monotonic()
        retry_started: float | None = None
        attempt = 0
        retryable_provider_retries = 0
        retry_sleep_seconds = 0.0
        adaptive_window_increases = 0
        adaptive_window_decreases = 0
        adaptive_admission_wait_seconds = 0.0
        max_adaptive_cooldown_seconds = 0.0
        accumulated_usage: dict[str, Any] = {}
        last_response: Any | None = None
        last_failure_was_rendered_response = False
        while True:
            attempt += 1
            if retry_started is None:
                retry_started = monotonic()
            admission_wait, admission_generation = self._acquire_provider_slot()
            adaptive_admission_wait_seconds += max(0.0, admission_wait)
            response: Any | None = None
            try:
                response = self._create_structured_completion(prompt, schema, schema_name)
            except Exception as exc:
                last_error = exc
                last_failure_was_rendered_response = False
            else:
                last_response = response
                accumulated_usage = self._merge_usage(accumulated_usage, self._usage_dict(response))
                increased, _previous, _current = self.adaptive_controller.record_success(
                    self.model, admission_generation
                )
                if increased:
                    adaptive_window_increases += 1
                try:
                    raw = (response.choices[0].message.content or "").strip()
                    parsed = json.loads(self._extract_json_candidate(raw))
                    if not isinstance(parsed, dict):
                        raise ValueError("Expected a structured JSON object response")
                    telemetry = {
                        "model": getattr(response, "model", self.model),
                        "provider": (getattr(response, "model_extra", None) or {}).get("provider"),
                        "usage": accumulated_usage,
                        "retry_count": attempt - 1,
                        "retryable_provider_retries": retryable_provider_retries,
                        "retry_sleep_seconds": round(retry_sleep_seconds, 3),
                        **self._routing_metadata(),
                        "adaptive_window_increases": adaptive_window_increases,
                        "adaptive_window_decreases": adaptive_window_decreases,
                        "adaptive_admission_wait_seconds": round(adaptive_admission_wait_seconds, 3),
                        "max_adaptive_cooldown_seconds": round(max_adaptive_cooldown_seconds, 3),
                        **self.adaptive_controller.snapshot(),
                        "elapsed_seconds": round(monotonic() - overall_started, 3),
                    }
                    return {"data": parsed, "telemetry": telemetry}
                except Exception as exc:
                    last_error = exc
                    last_failure_was_rendered_response = True
            finally:
                self._release_provider_slot()
            assert retry_started is not None
            if not self._can_retry(
                attempt,
                retry_started,
                last_error,
                admission_wait_seconds=adaptive_admission_wait_seconds,
            ):
                break
            if self._is_retryable_provider_error(last_error):
                retryable_provider_retries += 1
            delay, decreases, cooldown = self._sleep_before_retry(
                attempt,
                last_error,
                started=retry_started,
                admission_wait_seconds=adaptive_admission_wait_seconds,
                admission_generation=admission_generation,
            )
            retry_sleep_seconds += delay
            adaptive_window_decreases += decreases
            max_adaptive_cooldown_seconds = max(max_adaptive_cooldown_seconds, cooldown)
        telemetry = {
            "model": getattr(last_response, "model", self.model),
            "provider": (getattr(last_response, "model_extra", None) or {}).get("provider")
            if last_response is not None else None,
            "usage": accumulated_usage,
            "retry_count": attempt - 1,
            "retryable_provider_retries": retryable_provider_retries,
            "retry_sleep_seconds": round(retry_sleep_seconds, 3),
            **self._routing_metadata(),
            "adaptive_window_increases": adaptive_window_increases,
            "adaptive_window_decreases": adaptive_window_decreases,
            "adaptive_admission_wait_seconds": round(adaptive_admission_wait_seconds, 3),
            "max_adaptive_cooldown_seconds": round(max_adaptive_cooldown_seconds, 3),
            **self.adaptive_controller.snapshot(),
            "elapsed_seconds": round(monotonic() - overall_started, 3),
        }
        if self._is_retryable_provider_error(last_error):
            raise RetryableProviderExhaustedError(
                f"Retryable structured provider failure exhausted after {attempt} attempts: {last_error}",
                telemetry=telemetry,
            ) from last_error
        if last_failure_was_rendered_response:
            raise StructuredRenderedResponseError(
                f"Structured rendered response unusable after {attempt} attempts: {last_error}",
                telemetry=telemetry,
            ) from last_error
        raise RuntimeError(f"Structured LLM request failed after {attempt} attempts: {last_error}")

    def generate_structured_object(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        return self.generate_structured_object_with_metadata(
            prompt=prompt, schema=schema, schema_name=schema_name
        )["data"]
