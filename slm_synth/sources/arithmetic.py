# slm_synth/sources/arithmetic.py

from prompts.wrapper import build_prompt
from slm_synth.repair import repair_arithmetic


class ArithmeticGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_arithmetic_prompt()

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_arithmetic(obj)
        return obj
