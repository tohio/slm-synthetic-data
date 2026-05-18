# slm_synth/sources/arithmetic.py

from prompts.wrapper import build_prompt
from slm_synth.repair import repair_arithmetic
from slm_synth.schemas import ARITHMETIC_SCHEMA


class ArithmeticGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_prompt(
            schema=ARITHMETIC_SCHEMA,
            task_instruction="",
            prompt_name="arithmetic"
        )

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_arithmetic(obj)
        return obj

