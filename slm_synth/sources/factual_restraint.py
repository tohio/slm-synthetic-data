# slm_synth/sources/factual_restraint.py

from prompts.wrapper import build_prompt
from slm_synth.repair import repair_factual_restraint


class FactualRestraintGenerator:
    def __init__(self, llm, prompt_file: str = None):
        self.llm = llm

    def build_prompt(self) -> str:
        return build_factual_restraint_prompt()

    def generate_one(self):
        obj = self.llm.generate_one(self.build_prompt())
        obj = repair_factual_restraint(obj)
        return obj
