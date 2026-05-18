from slm_synth.llm import LLMBackend
from slm_synth.sources.arithmetic import ArithmeticGenerator


class MockLLM:
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


def test_arithmetic_generator_batch():
    gen = ArithmeticGenerator(MockLLM(), "prompts/arithmetic.yaml", batch_size=2)
    batch = gen.generate_batch()
    assert len(batch) == 2
    assert batch[0]["type"] == "arithmetic"
    assert batch[0]["answer"] == "4"


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
