from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMBackend:
    """OpenAI-compatible LLM client for legacy and grounded generation paths.

    Legacy Groq sources use ``generate_batch`` with an ``{"items": [...]}``
    object contract. Grounded OpenRouter sources use ``generate_structured_object``
    with strict JSON Schema responses.
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
        require_parameters: bool = True,
        allow_fallbacks: bool = False,
    ):
        if provider not in {"groq", "openrouter"}:
            raise ValueError(f"Unsupported provider: {provider}")

        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
            missing_key = "GROQ_API_KEY"
            default_headers = None
        else:
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
        self.require_parameters = bool(require_parameters)
        self.allow_fallbacks = bool(allow_fallbacks)

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
        return LLMBackend(
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
            require_parameters=self.require_parameters,
            allow_fallbacks=self.allow_fallbacks,
        )

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

    def _create_completion(self, prompt: str):
        kwargs = self._base_kwargs(prompt)
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.service_tier and self.provider == "groq":
            kwargs["service_tier"] = self.service_tier
        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError:
            tier = kwargs.pop("service_tier", None)
            if tier:
                kwargs["extra_body"] = {"service_tier": tier}
                return self.client.chat.completions.create(**kwargs)
            raise

    def _create_structured_completion(self, prompt: str, schema: dict[str, Any], schema_name: str):
        if self.provider != "openrouter":
            raise ValueError("Strict grounded structured generation requires provider='openrouter'")
        kwargs = self._base_kwargs(prompt)
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        kwargs["extra_body"] = {
            "provider": {
                "require_parameters": self.require_parameters,
                "allow_fallbacks": self.allow_fallbacks,
            }
        }
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return str(exc).lower()

    def _is_capacity_or_rate_error(self, exc: Exception) -> bool:
        text = self._error_text(exc)
        return any(marker in text for marker in (
            "capacity_exceeded", "rate_limit", "rate limit", "too many requests",
            "error code: 429", "error code: 498", " 429 ", " 498 ",
        ))

    def _is_transient_transport_error(self, exc: Exception) -> bool:
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

    def _can_retry(self, attempt: int, started: float, exc: Exception) -> bool:
        if self._is_retryable_provider_error(exc):
            return (
                attempt < self.max_retryable_request_attempts
                and monotonic() - started < self.retry_max_elapsed_seconds
            )
        return attempt < self.max_request_retries

    def _sleep_before_retry(self, attempt: int, exc: Exception, *, started: float) -> float:
        if self._is_retryable_provider_error(exc):
            retry_after = self._retry_after_seconds(exc)
            if retry_after is not None:
                delay = retry_after
            else:
                base_delay = min(
                    self.retry_backoff_max_seconds,
                    self.retry_backoff_initial_seconds * (self.retry_backoff_multiplier ** max(0, attempt - 1)),
                )
                delay = base_delay + base_delay * max(0.0, self.retry_jitter_ratio) * random.random()
            remaining = max(0.0, self.retry_max_elapsed_seconds - (monotonic() - started))
            delay = min(delay, remaining)
            print(
                f"[llm] Retryable provider failure: model={self.model} attempt={attempt}/"
                f"{self.max_retryable_request_attempts} delay={delay:.2f}s error={str(exc)[:180]!r}"
            )
        else:
            delay = self.retry_sleep_seconds * attempt
        if delay > 0:
            time.sleep(delay)
        return delay

    def generate_batch(self, prompt: str, batch_size: int) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        started = monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._create_completion(prompt)
                raw = (response.choices[0].message.content or "").strip()
                return self._parse_items(raw, batch_size)
            except Exception as exc:
                last_error = exc
                if not self._can_retry(attempt, started, exc):
                    break
                self._sleep_before_retry(attempt, exc, started=started)
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

    def generate_structured_object_with_metadata(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Generate a strict object and retain operational telemetry for persisted batches."""
        last_error: Optional[Exception] = None
        started = monotonic()
        attempt = 0
        retryable_provider_retries = 0
        retry_sleep_seconds = 0.0
        while True:
            attempt += 1
            try:
                response = self._create_structured_completion(prompt, schema, schema_name)
                raw = (response.choices[0].message.content or "").strip()
                parsed = json.loads(self._extract_json_candidate(raw))
                if not isinstance(parsed, dict):
                    raise ValueError("Expected a structured JSON object response")
                telemetry = {
                    "model": getattr(response, "model", self.model),
                    "provider": (getattr(response, "model_extra", None) or {}).get("provider"),
                    "usage": self._usage_dict(response),
                    "retry_count": attempt - 1,
                    "retryable_provider_retries": retryable_provider_retries,
                    "retry_sleep_seconds": round(retry_sleep_seconds, 3),
                    "elapsed_seconds": round(monotonic() - started, 3),
                }
                return {"data": parsed, "telemetry": telemetry}
            except Exception as exc:
                last_error = exc
                if not self._can_retry(attempt, started, exc):
                    break
                if self._is_retryable_provider_error(exc):
                    retryable_provider_retries += 1
                retry_sleep_seconds += self._sleep_before_retry(attempt, exc, started=started)
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
