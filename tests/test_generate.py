import json
from slm_synth.generate import GENERATOR_MAP
from slm_synth.sources.arithmetic import ArithmeticGenerator

class MockLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response

def test_arithmetic_generator():
    mock = MockLLM('{"type":"arithmetic","question":"Q","steps":["S"],"answer":"A"}')
    gen = ArithmeticGenerator(mock, "prompts/arithmetic.yaml")
    gen.prompt = {
        "system": "",
        "instruction": "",
        "format": ""
    }
    obj = gen.generate()
    assert obj["type"] == "arithmetic"
    assert obj["answer"] == "A"
