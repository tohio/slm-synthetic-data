from types import SimpleNamespace

import pytest

import slm_synth.llm as llm_module
from slm_synth.llm import (
    AdaptiveRequestController,
    LLMBackend,
    RetryableProviderExhaustedError,
    StructuredRenderedResponseError,
)

from slm_synth.pretrain.generate import run_signal


def test_pretrain_run_signal_rejects_non_grounded_configs(tmp_path):
    cfg = {
        "target_total_tokens": 100,
        "backend": {"model": "openai/gpt-4.1-mini"},
        "mix": {"arithmetic": {"share": 1.0}},
    }

    with pytest.raises(ValueError, match="only supports grounded architecture"):
        run_signal("arithmetic", cfg, tmp_path)


def test_parse_json_object_items_without_api_call():
    backend = LLMBackend.__new__(LLMBackend)
    raw = '{"items":[{"type":"arithmetic_candidate","question":"Q"}]}'
    objs = LLMBackend._parse_items(backend, raw, 1)
    assert objs[0]["type"] == "arithmetic_candidate"


def test_parse_legacy_json_array_without_api_call():
    backend = LLMBackend.__new__(LLMBackend)
    raw = '[{"candidate_id":0,"answer":"A"}]'
    objs = LLMBackend._parse_items(backend, raw, 1)
    assert objs[0]["answer"] == "A"


class RetryableProviderError(RuntimeError):
    def __init__(self, message="Error code: 429 - temporarily rate-limited upstream", retry_after=None):
        super().__init__(message)
        headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
        self.response = SimpleNamespace(headers=headers)


class StatusCodeProviderError(RuntimeError):
    def __init__(self, status_code: int, message: str = "provider request failed"):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers={})


def _backend_for_retry_test():
    backend = LLMBackend.__new__(LLMBackend)
    backend.model = "deepseek/deepseek-v4-flash"
    backend.max_request_retries = 1
    backend.max_retryable_request_attempts = 4
    backend.retry_max_elapsed_seconds = 1800.0
    backend.retry_sleep_seconds = 0.0
    backend.retry_backoff_initial_seconds = 2.0
    backend.retry_backoff_max_seconds = 120.0
    backend.retry_backoff_multiplier = 2.0
    backend.retry_jitter_ratio = 0.0
    backend.adaptive_controller = AdaptiveRequestController(enabled=False, maximum_in_flight=1)
    return backend


def _structured_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": []}'))],
        model="deepseek/deepseek-v4-flash",
        model_extra={"provider": "DeepInfra"},
        usage=None,
    )


def _batch_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"items": [{"ok": true}]}'))],
        model="deepseek/deepseek-v4-flash",
        model_extra={"provider": "DeepInfra"},
        usage=None,
    )


def test_status_code_429_retries_without_message_marker(monkeypatch):
    backend = _backend_for_retry_test()
    calls = []
    sleeps = []

    def create(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise StatusCodeProviderError(429, "provider overloaded")
        return _structured_response()

    backend._create_structured_completion = create
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = backend.generate_structured_object_with_metadata(
        prompt="prompt", schema={}, schema_name="schema"
    )

    assert len(calls) == 2
    assert sleeps == [2.0]
    assert result["telemetry"]["retry_count"] == 1
    assert result["telemetry"]["retryable_provider_retries"] == 1


def test_response_status_code_503_is_transient_without_message_marker():
    backend = _backend_for_retry_test()
    exc = RuntimeError("upstream failed")
    exc.response = SimpleNamespace(status_code=503, headers={})

    assert backend._is_transient_transport_error(exc) is True
    assert backend._is_retryable_provider_error(exc) is True


def test_typed_timeout_and_connection_errors_are_retryable():
    backend = _backend_for_retry_test()
    APITimeoutError = type("APITimeoutError", (RuntimeError,), {})
    APIConnectionError = type("APIConnectionError", (RuntimeError,), {})

    assert backend._is_transient_transport_error(APITimeoutError("request exceeded deadline")) is True
    assert backend._is_transient_transport_error(APIConnectionError("socket closed")) is True


def test_status_code_400_blocks_substring_retry_fallback():
    backend = _backend_for_retry_test()
    exc = StatusCodeProviderError(400, "rate limit timeout error code: 503")

    assert backend._is_capacity_or_rate_error(exc) is False
    assert backend._is_transient_transport_error(exc) is False
    assert backend._is_retryable_provider_error(exc) is False


def test_legacy_substring_retry_fallback_still_works_without_status():
    backend = _backend_for_retry_test()

    assert backend._is_capacity_or_rate_error(RetryableProviderError()) is True
    assert backend._is_transient_transport_error(RuntimeError("connection reset by peer")) is True


def test_structured_generation_extends_retry_budget_for_rate_limits(monkeypatch):
    backend = _backend_for_retry_test()
    calls = []
    sleeps = []

    def create(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise RetryableProviderError(retry_after=7)
        return _structured_response()

    backend._create_structured_completion = create
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = backend.generate_structured_object_with_metadata(
        prompt="prompt", schema={}, schema_name="schema"
    )

    assert len(calls) == 3
    assert sleeps == [7.0, 7.0]
    assert result["telemetry"]["retry_count"] == 2
    assert result["telemetry"]["retryable_provider_retries"] == 2
    assert result["telemetry"]["retry_sleep_seconds"] == 14.0


def test_non_provider_failures_keep_short_retry_budget(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_request_retries = 2
    calls = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)

    def create(*args, **kwargs):
        calls.append(1)
        raise ValueError("bad structured payload")

    backend._create_structured_completion = create
    try:
        backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")
    except RuntimeError as exc:
        assert "after 2 attempts" in str(exc)
    else:
        raise AssertionError("expected structured generation failure")
    assert len(calls) == 2


def test_malformed_structured_responses_raise_droppable_error_with_accumulated_usage(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_request_retries = 2
    calls = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)

    malformed = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"records": ['))],
        model="deepseek/deepseek-v4-flash",
        model_extra={"provider": "DeepInfra"},
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
    )

    def create(*args, **kwargs):
        calls.append(1)
        return malformed

    backend._create_structured_completion = create
    try:
        backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")
    except StructuredRenderedResponseError as exc:
        assert "unusable after 2 attempts" in str(exc)
        assert exc.telemetry["usage"] == {
            "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.02
        }
        assert exc.telemetry["retry_count"] == 1
    else:
        raise AssertionError("expected droppable structured rendered response failure")
    assert len(calls) == 2


def test_retryable_provider_failure_stops_after_bounded_provider_elapsed(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_retryable_request_attempts = 20
    backend.retry_max_elapsed_seconds = 5.0
    calls = []
    clock = {"now": 0.0}
    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)

    def create(*args, **kwargs):
        calls.append(1)
        clock["now"] = 6.0
        raise RetryableProviderError()

    backend._create_structured_completion = create
    with pytest.raises(RetryableProviderExhaustedError, match="after 1 attempts"):
        backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")
    assert len(calls) == 1


def test_initial_admission_wait_does_not_consume_provider_retry_budget(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_retryable_request_attempts = 3
    backend.retry_max_elapsed_seconds = 5.0
    calls = []
    sleeps = []
    clock = {"now": 0.0}

    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])
    backend._acquire_provider_slot = lambda: (100.0, 0)
    backend._release_provider_slot = lambda: None

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(llm_module.time, "sleep", sleep)

    def create(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            clock["now"] = 100.0
            raise RetryableProviderError()
        return _structured_response()

    backend._create_structured_completion = create
    result = backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")
    assert len(calls) == 2
    assert sleeps == [2.0]
    assert result["telemetry"]["retryable_provider_retries"] == 1


def test_later_admission_wait_after_provider_failure_does_not_consume_retry_budget(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_retryable_request_attempts = 3
    backend.retry_max_elapsed_seconds = 5.0
    calls = []
    acquire_calls = []
    sleeps = []
    clock = {"now": 0.0}
    waits = [0.0, 100.0, 0.0]

    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])

    def acquire():
        wait = waits[len(acquire_calls)]
        acquire_calls.append(wait)
        clock["now"] += wait
        return wait, 0

    backend._acquire_provider_slot = acquire
    backend._release_provider_slot = lambda: None

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(llm_module.time, "sleep", sleep)

    def create(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise RetryableProviderError()
        return _structured_response()

    backend._create_structured_completion = create
    result = backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")

    assert len(calls) == 3
    assert sleeps == [2.0, 3.0]
    assert result["telemetry"]["retryable_provider_retries"] == 2
    assert result["telemetry"]["adaptive_admission_wait_seconds"] == 100.0


def test_retry_backoff_sleep_consumes_provider_retry_budget(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_retryable_request_attempts = 20
    backend.retry_max_elapsed_seconds = 3.0
    calls = []
    sleeps = []
    clock = {"now": 0.0}

    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])
    backend._acquire_provider_slot = lambda: (0.0, 0)
    backend._release_provider_slot = lambda: None

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(llm_module.time, "sleep", sleep)

    def create(*args, **kwargs):
        calls.append(1)
        raise RetryableProviderError()

    backend._create_structured_completion = create
    with pytest.raises(RetryableProviderExhaustedError, match="after 3 attempts") as exc_info:
        backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")

    assert len(calls) == 3
    assert sleeps == [2.0, 1.0]
    assert exc_info.value.telemetry["retry_sleep_seconds"] == 3.0
    assert exc_info.value.telemetry["retryable_provider_retries"] == 2


def test_generate_batch_excludes_later_admission_wait_from_retry_budget(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_retryable_request_attempts = 3
    backend.retry_max_elapsed_seconds = 5.0
    calls = []
    acquire_calls = []
    sleeps = []
    clock = {"now": 0.0}
    waits = [0.0, 100.0, 0.0]

    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])

    def acquire():
        wait = waits[len(acquire_calls)]
        acquire_calls.append(wait)
        clock["now"] += wait
        return wait, 0

    backend._acquire_provider_slot = acquire
    backend._release_provider_slot = lambda: None

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(llm_module.time, "sleep", sleep)

    def create(prompt):
        calls.append(1)
        if len(calls) < 3:
            raise RetryableProviderError()
        return _batch_response()

    backend._create_completion = create
    rows = backend.generate_batch("prompt", batch_size=1)

    assert rows == [{"ok": True}]
    assert len(calls) == 3
    assert sleeps == [2.0, 3.0]


def test_adaptive_controller_slow_start_doubles_after_a_successful_window():
    controller = AdaptiveRequestController(
        maximum_in_flight=32, initial_in_flight=4, minimum_in_flight=1,
        slow_start_enabled=True, slow_start_multiplier=2.0,
    )

    assert controller.snapshot()["adaptive_current_in_flight_limit"] == 4
    for _ in range(3):
        assert controller.record_success("model") == (False, 4, 4)
    assert controller.record_success("model") == (True, 4, 8)
    for _ in range(7):
        assert controller.record_success("model") == (False, 8, 8)
    assert controller.record_success("model") == (True, 8, 16)
    assert controller.snapshot()["adaptive_peak_in_flight_limit"] == 16


def test_adaptive_controller_uses_additive_increase_after_a_throttle_burst(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])
    controller = AdaptiveRequestController(
        maximum_in_flight=32, initial_in_flight=8, minimum_in_flight=1,
        slow_start_enabled=True, increase_successes_per_step=2, increase_step=3,
        rate_limit_burst_threshold=2, cooldown_initial_seconds=0.0,
    )

    assert controller.record_rate_limit("model") == (False, 8, 8, 0.0)
    assert controller.record_rate_limit("model") == (True, 8, 4, 0.0)
    assert controller.record_success("model") == (False, 4, 4)
    assert controller.record_success("model") == (True, 4, 7)


def test_adaptive_controller_reduces_window_after_rate_limit_burst(monkeypatch):
    monkeypatch.setattr(llm_module, "monotonic", lambda: 0.0)
    controller = AdaptiveRequestController(
        maximum_in_flight=16, initial_in_flight=8, minimum_in_flight=2,
        rate_limit_burst_threshold=2, rate_limit_window_seconds=2.0,
        rate_limit_decrease_factor=0.5, cooldown_initial_seconds=5.0,
    )

    assert controller.record_rate_limit("model") == (False, 8, 8, 0.0)
    assert controller.record_rate_limit("model") == (True, 8, 4, 5.0)
    assert controller.snapshot()["adaptive_min_in_flight_limit"] == 4


def test_adaptive_controller_reduces_window_after_sustained_rate_limit_pressure(monkeypatch, capsys):
    clock = {"now": 0.0}
    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])
    controller = AdaptiveRequestController(
        maximum_in_flight=64, initial_in_flight=32, minimum_in_flight=1,
        rate_limit_burst_threshold=4, rate_limit_window_seconds=2.0,
        rate_limit_decrease_factor=0.5,
        sustained_rate_limit_attempt_window=10, sustained_rate_limit_threshold=4,
        cooldown_initial_seconds=5.0,
    )

    outcomes = [True, False, False, True, False, True, False, False, False, True]
    result = None
    for is_rate_limit in outcomes:
        clock["now"] += 3.0  # Never permit the fast 2-second burst detector to fire.
        if is_rate_limit:
            result = controller.record_rate_limit("model")
        else:
            controller.record_success("model")

    assert result == (True, 32, 16, 5.0)
    assert controller.snapshot()["adaptive_current_in_flight_limit"] == 16
    assert "trigger=sustained" in capsys.readouterr().out


def test_adaptive_controller_sustained_pressure_requires_full_attempt_window(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(llm_module, "monotonic", lambda: clock["now"])
    controller = AdaptiveRequestController(
        maximum_in_flight=64, initial_in_flight=32, minimum_in_flight=1,
        rate_limit_burst_threshold=4, rate_limit_window_seconds=2.0,
        sustained_rate_limit_attempt_window=10, sustained_rate_limit_threshold=4,
        cooldown_initial_seconds=0.0,
    )

    for _ in range(4):
        clock["now"] += 3.0
        assert controller.record_rate_limit("model") == (False, 32, 32, 0.0)
        clock["now"] += 3.0
        controller.record_success("model")

    assert controller.snapshot()["adaptive_current_in_flight_limit"] == 32


def test_adaptive_controller_ignores_late_429s_from_previous_window(monkeypatch):
    monkeypatch.setattr(llm_module, "monotonic", lambda: 0.0)
    controller = AdaptiveRequestController(
        maximum_in_flight=16, initial_in_flight=8, minimum_in_flight=1,
        rate_limit_burst_threshold=2, cooldown_initial_seconds=0.0,
    )
    _wait, old_generation = controller.acquire()
    controller.release()

    assert controller.record_rate_limit("model", old_generation) == (False, 8, 8, 0.0)
    assert controller.record_rate_limit("model", old_generation) == (True, 8, 4, 0.0)
    # A late response from the old eight-request wave cannot reduce the new window.
    assert controller.record_rate_limit("model", old_generation) == (False, 4, 4, 0.0)
    _wait, new_generation = controller.acquire()
    controller.release()
    assert new_generation != old_generation
    assert controller.record_rate_limit("model", new_generation) == (False, 4, 4, 0.0)
    assert controller.record_rate_limit("model", new_generation) == (True, 4, 2, 0.0)


def test_adaptive_controller_limits_active_provider_calls():
    import threading

    controller = AdaptiveRequestController(maximum_in_flight=2, initial_in_flight=1, minimum_in_flight=1)
    entered = threading.Event()
    controller.acquire()

    def acquire_second_slot():
        controller.acquire()
        entered.set()
        controller.release()

    worker = threading.Thread(target=acquire_second_slot)
    worker.start()
    assert not entered.wait(0.05)
    controller.release()
    assert entered.wait(1.0)
    worker.join(timeout=1.0)
