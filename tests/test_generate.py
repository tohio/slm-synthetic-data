from types import SimpleNamespace

import slm_synth.llm as llm_module
from slm_synth.llm import AdaptiveRequestController, LLMBackend
from slm_synth.sources.arithmetic import ArithmeticGenerator
from slm_synth.sources.educational_qa_mcq_general import EducationalQAMCQGeneralGenerator
from slm_synth.sources.educational_qa_mcq_math import EducationalQAMCQMathGenerator
from slm_synth.sources.factual_restraint import FactualRestraintGenerator
from slm_synth.sources.task_code import TaskCodeGenerator
from slm_synth.sources.two_pass import order_responses


class MockArithmeticCandidateLLM:
    def generate_batch(self, prompt, batch_size):
        assert "arithmetic_candidate" in prompt
        return [{"type": "arithmetic_candidate", "question": "What is 2 + 2?"} for _ in range(batch_size)]


class MockArithmeticResponseLLM:
    def generate_batch(self, prompt, batch_size):
        assert "Candidates to answer" in prompt
        return [
            {
                "candidate_id": i,
                "steps": ["2 + 2 = 4"],
                "answer": "4",
                "verification_expression": "2 + 2",
                "verification_answer": "4",
            }
            for i in range(batch_size)
        ]


class MockMathCandidateLLM:
    def generate_batch(self, prompt, batch_size):
        assert "educational_qa_mcq_math_candidate" in prompt
        return [
            {
                "type": "educational_qa_mcq_math_candidate",
                "question": "What is 8 * 5?",
                "choices": ["30", "35", "40", "45"],
            }
            for _ in range(batch_size)
        ]


class MockMathResponseLLM:
    def generate_batch(self, prompt, batch_size):
        assert "verification_expression" in prompt
        return [
            {
                "candidate_id": i,
                "explanation": "8 * 5 = 40.",
                "verification_expression": "8 * 5",
                "verification_answer": "40",
            }
            for i in range(batch_size)
        ]


class MockGeneralCandidateLLM:
    def generate_batch(self, prompt, batch_size):
        assert "educational_qa_mcq_general_candidate" in prompt
        return [
            {
                "type": "educational_qa_mcq_general_candidate",
                "question": "Which word is an adverb in 'Mina quickly packed the box'?",
                "choices": ["Mina", "quickly", "packed", "box"],
            }
            for _ in range(batch_size)
        ]


class MockGeneralResponseLLM:
    def generate_batch(self, prompt, batch_size):
        assert "Candidates to answer" in prompt
        return [
            {
                "candidate_id": i,
                "answer": "quickly",
                "explanation": "Quickly describes how Mina packed the box.",
            }
            for i in range(batch_size)
        ]


class MockTaskCandidateLLM:
    def generate_batch(self, prompt, batch_size):
        return [{"type": "task_code_candidate", "task": "Return sorted positive values from a list."} for _ in range(batch_size)]


class MockTaskResponseLLM:
    def generate_batch(self, prompt, batch_size):
        return [{"candidate_id": i, "plan": ["Filter positive values", "Sort them"], "code": "def sorted_positive(values):\n    return sorted(v for v in values if v > 0)"} for i in range(batch_size)]


class MockFactualCandidateLLM:
    def generate_batch(self, prompt, batch_size):
        return [{"type": "factual_restraint_candidate", "question": "Who will win the next election?"} for _ in range(batch_size)]


class MockFactualResponseLLM:
    def generate_batch(self, prompt, batch_size):
        return [{"candidate_id": i, "safe_answer": "I cannot predict a future election outcome."} for i in range(batch_size)]


def test_arithmetic_generator_runs_candidate_then_response_pass():
    batch = ArithmeticGenerator(MockArithmeticCandidateLLM(), response_llm=MockArithmeticResponseLLM(), batch_size=2).generate_batch()
    assert len(batch) == 2
    assert batch[0] == {
        "type": "arithmetic",
        "question": "What is 2 + 2?",
        "steps": ["2 + 2 = 4"],
        "answer": "4",
        "verification_expression": "2 + 2",
        "verification_answer": "4",
    }


def test_math_mcq_generator_runs_candidate_then_response_pass():
    batch = EducationalQAMCQMathGenerator(MockMathCandidateLLM(), response_llm=MockMathResponseLLM(), batch_size=1).generate_batch()
    assert batch[0]["type"] == "educational_qa_mcq_math"
    assert batch[0]["verification_answer"] == "40"
    assert batch[0]["choices"].count("40") == 1
    assert batch[0]["choices"][batch[0]["correct_index"]] == "40"


def test_general_mcq_generator_runs_candidate_then_response_pass():
    batch = EducationalQAMCQGeneralGenerator(MockGeneralCandidateLLM(), response_llm=MockGeneralResponseLLM(), batch_size=1).generate_batch()
    assert batch[0]["type"] == "educational_qa_mcq_general"
    assert batch[0]["choices"][batch[0]["correct_index"]] == "quickly"


def test_math_mcq_python_inserts_missing_solved_answer_into_choices():
    class CandidateWithoutAnswer(MockMathCandidateLLM):
        def generate_batch(self, prompt, batch_size):
            return [{"type": "educational_qa_mcq_math_candidate", "question": "What is 24 / 4?", "choices": ["4", "5", "7", "8"]}]

    class SolvedAnswer:
        def generate_batch(self, prompt, batch_size):
            return [{"candidate_id": 0, "explanation": "24 / 4 = 6.", "verification_expression": "24 / 4", "verification_answer": "6"}]

    row = EducationalQAMCQMathGenerator(CandidateWithoutAnswer(), response_llm=SolvedAnswer(), batch_size=1).generate_batch()[0]
    assert len(row["choices"]) == 4
    assert row["choices"].count("6") == 1
    assert row["choices"][row["correct_index"]] == "6"


def test_general_mcq_python_replaces_distractor_when_answer_is_missing():
    class CandidateWithoutAnswer:
        def generate_batch(self, prompt, batch_size):
            return [{"type": "educational_qa_mcq_general_candidate", "question": "Which word is an adverb in 'Mina quickly packed the box'?", "choices": ["Mina", "packed", "box", "the"]}]

    class SolvedAnswer:
        def generate_batch(self, prompt, batch_size):
            return [{"candidate_id": 0, "answer": "quickly", "explanation": "Quickly describes how Mina packed the box."}]

    row = EducationalQAMCQGeneralGenerator(CandidateWithoutAnswer(), response_llm=SolvedAnswer(), batch_size=1).generate_batch()[0]
    assert len(row["choices"]) == 4
    assert row["choices"].count("quickly") == 1
    assert row["choices"][row["correct_index"]] == "quickly"


def test_task_code_generator_runs_candidate_then_response_pass():
    batch = TaskCodeGenerator(MockTaskCandidateLLM(), response_llm=MockTaskResponseLLM(), batch_size=1).generate_batch()
    assert batch[0]["task"].startswith("Return sorted")
    assert "def sorted_positive" in batch[0]["code"]


def test_factual_generator_runs_candidate_then_response_pass():
    batch = FactualRestraintGenerator(MockFactualCandidateLLM(), response_llm=MockFactualResponseLLM(), batch_size=1).generate_batch()
    assert batch[0]["safe_answer"].startswith("I cannot predict")


def test_order_responses_reorders_candidate_ids():
    rows = order_responses([{"candidate_id": 1}, {"candidate_id": 0}], 2)
    assert [row["candidate_id"] for row in rows] == [0, 1]


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


def test_retryable_provider_failure_stops_after_bounded_elapsed(monkeypatch):
    backend = _backend_for_retry_test()
    backend.max_retryable_request_attempts = 20
    backend.retry_max_elapsed_seconds = 5.0
    calls = []
    clock = iter([0.0, 6.0])
    monkeypatch.setattr(llm_module, "monotonic", lambda: next(clock))
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)

    def create(*args, **kwargs):
        calls.append(1)
        raise RetryableProviderError()

    backend._create_structured_completion = create
    try:
        backend.generate_structured_object_with_metadata(prompt="prompt", schema={}, schema_name="schema")
    except RuntimeError as exc:
        assert "after 1 attempts" in str(exc)
    else:
        raise AssertionError("expected bounded provider retry failure")
    assert len(calls) == 1


def test_adaptive_controller_grows_after_sustained_success():
    controller = AdaptiveRequestController(
        maximum_in_flight=16, initial_in_flight=4, minimum_in_flight=2,
        increase_successes_per_step=2, increase_step=4,
    )

    assert controller.snapshot()["adaptive_current_in_flight_limit"] == 4
    assert controller.record_success("model") == (False, 4, 4)
    assert controller.record_success("model") == (True, 4, 8)
    assert controller.snapshot()["adaptive_peak_in_flight_limit"] == 8


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
