from slm_synth.llm import LLMBackend
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
        return [{"candidate_id": i, "steps": ["2 + 2 = 4"], "answer": "4"} for i in range(batch_size)]


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
    assert batch[0] == {"type": "arithmetic", "question": "What is 2 + 2?", "steps": ["2 + 2 = 4"], "answer": "4"}


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
