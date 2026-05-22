from slm_synth.llm import LLMBackend
from slm_synth.sources.arithmetic import ArithmeticGenerator
from slm_synth.sources.educational_qa_mcq_general import EducationalQAMCQGeneralGenerator
from slm_synth.sources.educational_qa_mcq_math import EducationalQAMCQMathGenerator


class MockArithmeticLLM:
    def generate_batch(self, prompt, batch_size):
        assert '"items"' in prompt
        return [
            {
                "type": "arithmetic",
                "question": "What is 2 + 2?",
                "steps": ["2 + 2 = 4"],
                "answer": "4",
            }
            for _ in range(batch_size)
        ]


class MockMathMCQLLM:
    def generate_batch(self, prompt, batch_size):
        assert "verification_expression" in prompt
        return [
            {
                "type": "educational_qa_mcq_math",
                "question": "What is 8 * 5?",
                "choices": ["30", "35", "40", "45"],
                "correct_index": 2,
                "explanation": "8 * 5 = 40.",
                "verification_expression": "8 * 5",
                "verification_answer": "40",
            }
            for _ in range(batch_size)
        ]


class MockGeneralMCQLLM:
    def generate_batch(self, prompt, batch_size):
        assert "non-math educational" in prompt
        return [
            {
                "type": "educational_qa_mcq_general",
                "question": "Which word is an adverb in 'Mina quickly packed the box'?",
                "choices": ["Mina", "quickly", "packed", "box"],
                "correct_index": 1,
                "explanation": "Quickly describes the action.",
            }
            for _ in range(batch_size)
        ]


def test_arithmetic_generator_batch():
    batch = ArithmeticGenerator(MockArithmeticLLM(), batch_size=2).generate_batch()
    assert len(batch) == 2
    assert batch[0]["type"] == "arithmetic"


def test_math_mcq_generator_batch():
    batch = EducationalQAMCQMathGenerator(MockMathMCQLLM(), batch_size=1).generate_batch()
    assert batch[0]["type"] == "educational_qa_mcq_math"


def test_general_mcq_generator_batch():
    batch = EducationalQAMCQGeneralGenerator(MockGeneralMCQLLM(), batch_size=1).generate_batch()
    assert batch[0]["type"] == "educational_qa_mcq_general"


def test_parse_json_object_items_without_api_call():
    backend = LLMBackend.__new__(LLMBackend)
    raw = '{"items":[{"type":"arithmetic","question":"Q","steps":["S"],"answer":"A"}]}'
    objs = LLMBackend._parse_items(backend, raw, 1)
    assert objs[0]["type"] == "arithmetic"


def test_parse_legacy_json_array_without_api_call():
    backend = LLMBackend.__new__(LLMBackend)
    raw = '[{"type":"arithmetic","question":"Q","steps":["S"],"answer":"A"}]'
    objs = LLMBackend._parse_items(backend, raw, 1)
    assert objs[0]["answer"] == "A"
