from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from distill.generation.adaptive import AdaptiveRequestController, RetryableProviderExhaustedError, backoff_delay, is_capacity_or_rate_error, is_retryable_provider_error, sleep
from distill.providers.base import GenerationRequest, GenerationResponse
from distill.utils.env import get_env_value


@dataclass(frozen=True)
class HostedBackendConfig:
    provider_name: str
    request_timeout_seconds: float | None = 300.0
    json_mode: bool = True
    service_tier: str | None = None
    require_parameters: bool = True
    allow_fallbacks: bool = False
    http_referer: str = 'https://github.com/tohio/slm-distillation'
    x_title: str = 'SLM distillation hosted generation'
    max_request_retries: int = 3
    max_retryable_request_attempts: int = 20
    retry_max_elapsed_seconds: float = 1800.0
    retry_sleep_seconds: float = 0.5
    retry_backoff_initial_seconds: float = 1.0
    retry_backoff_max_seconds: float = 30.0
    retry_backoff_multiplier: float = 2.0
    retry_jitter_ratio: float = 0.30
    adaptive_concurrency_enabled: bool = True
    adaptive_maximum_in_flight: int = 1
    adaptive_initial_in_flight: int = 8
    adaptive_minimum_in_flight: int = 1
    adaptive_slow_start_enabled: bool = True
    adaptive_slow_start_multiplier: float = 2.0
    adaptive_increase_successes_per_step: int = 64
    adaptive_increase_step: int = 16
    adaptive_rate_limit_burst_threshold: int = 4
    adaptive_rate_limit_window_seconds: float = 2.0
    adaptive_rate_limit_decrease_factor: float = 0.50
    adaptive_sustained_rate_limit_attempt_window: int = 60
    adaptive_sustained_rate_limit_threshold: int = 20
    adaptive_cooldown_initial_seconds: float = 5.0
    adaptive_cooldown_max_seconds: float = 60.0
    adaptive_cooldown_multiplier: float = 2.0


class HostedLLMBackend:
    provider_name: str

    def __init__(self, config: HostedBackendConfig, *, env_path: str = '.env') -> None:
        self.config = config
        self.provider_name = config.provider_name
        if self.provider_name not in {'openrouter', 'groq'}:
            raise ValueError(f'Unsupported hosted provider: {self.provider_name}')
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The optimized hosted backend requires the 'openai' package. Install dependencies with `python3 -m pip install -r requirements.txt`.") from exc
        if self.provider_name == 'groq':
            api_key = get_env_value('GROQ_API_KEY', env_path)
            base_url = 'https://api.groq.com/openai/v1'
            missing = 'GROQ_API_KEY'
            default_headers = None
        else:
            api_key = get_env_value('OPENROUTER_API_KEY', env_path)
            base_url = 'https://openrouter.ai/api/v1'
            missing = 'OPENROUTER_API_KEY'
            default_headers = {'HTTP-Referer': config.http_referer, 'X-Title': config.x_title}
        if not api_key:
            raise ValueError(f'{missing} is required in .env for hosted generation')
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.request_timeout_seconds, default_headers=default_headers)
        self.adaptive_controller = AdaptiveRequestController(
            enabled=config.adaptive_concurrency_enabled,
            maximum_in_flight=config.adaptive_maximum_in_flight,
            initial_in_flight=config.adaptive_initial_in_flight,
            minimum_in_flight=config.adaptive_minimum_in_flight,
            slow_start_enabled=config.adaptive_slow_start_enabled,
            slow_start_multiplier=config.adaptive_slow_start_multiplier,
            increase_successes_per_step=config.adaptive_increase_successes_per_step,
            increase_step=config.adaptive_increase_step,
            rate_limit_burst_threshold=config.adaptive_rate_limit_burst_threshold,
            rate_limit_window_seconds=config.adaptive_rate_limit_window_seconds,
            rate_limit_decrease_factor=config.adaptive_rate_limit_decrease_factor,
            sustained_rate_limit_attempt_window=config.adaptive_sustained_rate_limit_attempt_window,
            sustained_rate_limit_threshold=config.adaptive_sustained_rate_limit_threshold,
            cooldown_initial_seconds=config.adaptive_cooldown_initial_seconds,
            cooldown_max_seconds=config.adaptive_cooldown_max_seconds,
            cooldown_multiplier=config.adaptive_cooldown_multiplier,
        )

    @staticmethod
    def from_mapping(provider_name: str, mapping: dict[str, Any] | None) -> 'HostedLLMBackend':
        cfg = mapping or {}
        retry_cfg = cfg.get('retries', {}) or {}
        def get(name: str, default: Any) -> Any:
            return cfg.get(name, retry_cfg.get(name, default))
        parallel = int(get('parallel_requests', get('concurrency', 1)) or 1)
        return HostedLLMBackend(HostedBackendConfig(
            provider_name=provider_name,
            request_timeout_seconds=float(get('request_timeout_seconds', 300.0)),
            json_mode=bool(get('json_mode', True)),
            service_tier=get('service_tier', None),
            require_parameters=bool(get('require_parameters', True)),
            allow_fallbacks=bool(get('allow_fallbacks', False)),
            http_referer=str(get('http_referer', 'https://github.com/tohio/slm-distillation')),
            x_title=str(get('x_title', 'SLM distillation hosted generation')),
            max_request_retries=int(get('max_request_retries', 3)),
            max_retryable_request_attempts=int(get('max_retryable_request_attempts', 20)),
            retry_max_elapsed_seconds=float(get('retry_max_elapsed_seconds', 1800.0)),
            retry_sleep_seconds=float(get('retry_sleep_seconds', 0.5)),
            retry_backoff_initial_seconds=float(get('retry_backoff_initial_seconds', 1.0)),
            retry_backoff_max_seconds=float(get('retry_backoff_max_seconds', 30.0)),
            retry_backoff_multiplier=float(get('retry_backoff_multiplier', 2.0)),
            retry_jitter_ratio=float(get('retry_jitter_ratio', 0.30)),
            adaptive_concurrency_enabled=bool(get('adaptive_concurrency_enabled', True)),
            adaptive_maximum_in_flight=parallel,
            adaptive_initial_in_flight=int(get('adaptive_initial_in_flight', min(8, parallel))),
            adaptive_minimum_in_flight=int(get('adaptive_minimum_in_flight', 1)),
            adaptive_slow_start_enabled=bool(get('adaptive_slow_start_enabled', True)),
            adaptive_slow_start_multiplier=float(get('adaptive_slow_start_multiplier', 2.0)),
            adaptive_increase_successes_per_step=int(get('adaptive_increase_successes_per_step', 64)),
            adaptive_increase_step=int(get('adaptive_increase_step', 16)),
            adaptive_rate_limit_burst_threshold=int(get('adaptive_rate_limit_burst_threshold', 4)),
            adaptive_rate_limit_window_seconds=float(get('adaptive_rate_limit_window_seconds', 2.0)),
            adaptive_rate_limit_decrease_factor=float(get('adaptive_rate_limit_decrease_factor', 0.50)),
            adaptive_sustained_rate_limit_attempt_window=int(get('adaptive_sustained_rate_limit_attempt_window', 60)),
            adaptive_sustained_rate_limit_threshold=int(get('adaptive_sustained_rate_limit_threshold', 20)),
            adaptive_cooldown_initial_seconds=float(get('adaptive_cooldown_initial_seconds', 5.0)),
            adaptive_cooldown_max_seconds=float(get('adaptive_cooldown_max_seconds', 60.0)),
            adaptive_cooldown_multiplier=float(get('adaptive_cooldown_multiplier', 2.0)),
        ))

    def _completion_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            'model': request.model,
            'messages': [
                {'role': 'system', 'content': 'You are a JSON-only distillation data generator. Return exactly one valid JSON object and no prose.'},
                {'role': 'user', 'content': request.prompt},
            ],
            'max_tokens': request.max_output_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
        }
        if self.config.json_mode:
            kwargs['response_format'] = {'type': 'json_object'}
        extra_body: dict[str, Any] = {}
        if self.provider_name == 'openrouter':
            extra_body['provider'] = {'require_parameters': self.config.require_parameters, 'allow_fallbacks': self.config.allow_fallbacks}
        if self.provider_name == 'groq' and self.config.service_tier:
            kwargs['service_tier'] = self.config.service_tier
        if extra_body:
            kwargs['extra_body'] = extra_body
        return kwargs

    @staticmethod
    def _usage_dict(response: Any) -> dict[str, Any]:
        usage = getattr(response, 'usage', None)
        if usage is None:
            return {}
        if hasattr(usage, 'model_dump'):
            result = usage.model_dump()
        elif isinstance(usage, dict):
            result = dict(usage)
        else:
            result = {key: getattr(usage, key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens') if hasattr(usage, key)}
        extra = getattr(usage, 'model_extra', None) or {}
        if isinstance(extra, dict):
            result.update(extra)
        return result

    @staticmethod
    def _raw_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        if isinstance(response, dict):
            return dict(response)
        return {'repr': repr(response)}

    def _create_completion(self, request: GenerationRequest) -> Any:
        kwargs = self._completion_kwargs(request)
        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError:
            tier = kwargs.pop('service_tier', None)
            if tier:
                extra = kwargs.setdefault('extra_body', {})
                extra['service_tier'] = tier
                return self.client.chat.completions.create(**kwargs)
            raise

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        last_error: Exception | None = None
        retry_started: float | None = None
        attempt = 0
        retryable_provider_retries = 0
        retry_sleep_seconds = 0.0
        adaptive_admission_wait_seconds = 0.0
        adaptive_window_increases = 0
        adaptive_window_decreases = 0
        max_adaptive_cooldown_seconds = 0.0
        started = monotonic()
        while True:
            attempt += 1
            admission_wait, admission_generation = self.adaptive_controller.acquire()
            adaptive_admission_wait_seconds += admission_wait
            if retry_started is None:
                retry_started = monotonic()
            try:
                response = self._create_completion(request)
            except Exception as exc:
                last_error = exc
            else:
                increased, _previous, _current = self.adaptive_controller.record_success(request.model, admission_generation)
                if increased:
                    adaptive_window_increases += 1
                content = (response.choices[0].message.content or '').strip()
                usage = self._usage_dict(response)
                raw = self._raw_dict(response)
                raw['telemetry'] = {
                    'retry_count': attempt - 1,
                    'retryable_provider_retries': retryable_provider_retries,
                    'retry_sleep_seconds': round(retry_sleep_seconds, 3),
                    'adaptive_window_increases': adaptive_window_increases,
                    'adaptive_window_decreases': adaptive_window_decreases,
                    'adaptive_admission_wait_seconds': round(adaptive_admission_wait_seconds, 3),
                    'max_adaptive_cooldown_seconds': round(max_adaptive_cooldown_seconds, 3),
                    **self.adaptive_controller.snapshot(),
                    'elapsed_seconds': round(monotonic() - started, 3),
                }
                self.adaptive_controller.release()
                return GenerationResponse(text=content, model=getattr(response, 'model', request.model), provider=self.provider_name, input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'), raw=raw)
            finally:
                # Success path releases before returning; failure path releases here.
                if last_error is not None:
                    self.adaptive_controller.release()
            assert retry_started is not None
            retryable = last_error is not None and is_retryable_provider_error(last_error)
            max_attempts = self.config.max_retryable_request_attempts if retryable else self.config.max_request_retries
            elapsed = monotonic() - retry_started
            if attempt >= max_attempts or elapsed >= self.config.retry_max_elapsed_seconds:
                telemetry = {
                    'retry_count': attempt - 1,
                    'retryable_provider_retries': retryable_provider_retries,
                    'retry_sleep_seconds': round(retry_sleep_seconds, 3),
                    'adaptive_window_increases': adaptive_window_increases,
                    'adaptive_window_decreases': adaptive_window_decreases,
                    'adaptive_admission_wait_seconds': round(adaptive_admission_wait_seconds, 3),
                    'max_adaptive_cooldown_seconds': round(max_adaptive_cooldown_seconds, 3),
                    **self.adaptive_controller.snapshot(),
                    'elapsed_seconds': round(monotonic() - started, 3),
                }
                if retryable:
                    raise RetryableProviderExhaustedError(f'Retryable provider failure exhausted after {attempt} attempts: {last_error}', telemetry=telemetry) from last_error
                raise RuntimeError(f'Hosted request failed after {attempt} attempts: {last_error}') from last_error
            if retryable:
                retryable_provider_retries += 1
                if last_error is not None and is_capacity_or_rate_error(last_error):
                    decreased, _previous, _current, cooldown = self.adaptive_controller.record_rate_limit(request.model, admission_generation)
                    if decreased:
                        adaptive_window_decreases += 1
                        max_adaptive_cooldown_seconds = max(max_adaptive_cooldown_seconds, cooldown)
                remaining = max(0.0, self.config.retry_max_elapsed_seconds - elapsed)
                delay = backoff_delay(attempt=attempt, exc=last_error, initial_seconds=self.config.retry_backoff_initial_seconds, max_seconds=self.config.retry_backoff_max_seconds, multiplier=self.config.retry_backoff_multiplier, jitter_ratio=self.config.retry_jitter_ratio, remaining_seconds=remaining)
                print(f'[hosted] Retryable provider failure: provider={self.provider_name} model={request.model} attempt={attempt}/{max_attempts} delay={delay:.2f}s error={str(last_error)[:180]!r}', flush=True)
            else:
                delay = self.config.retry_sleep_seconds * attempt
            retry_sleep_seconds += delay
            sleep(delay)
